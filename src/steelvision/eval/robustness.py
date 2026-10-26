"""Image perturbations and robustness / cross-domain evaluation utilities.

Evaluates how much detection and anomaly performance degrades under realistic
factory floor shifts:
- Lighting shifts: ``brightness_up``, ``brightness_down``, ``contrast_low``
- Optical blur: ``gaussian_blur``
- Sensor noise: ``gaussian_noise``
- Mechanical fixture misalignment: ``rotate_5deg``
- Cross-domain shift: model trained on NEU-DET evaluated on GC10-DET (and vice versa).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np


def apply_brightness_up(img: np.ndarray) -> np.ndarray:
    """Simulate over-exposure (+45 intensity offset, +15% gain)."""
    return cv2.convertScaleAbs(img, alpha=1.15, beta=45)


def apply_brightness_down(img: np.ndarray) -> np.ndarray:
    """Simulate dim lighting (0.70 gain, -35 intensity offset)."""
    return cv2.convertScaleAbs(img, alpha=0.70, beta=-35)


def apply_contrast_low(img: np.ndarray) -> np.ndarray:
    """Compress contrast around mid-gray (128) with 0.55 factor."""
    out = (img.astype(np.float32) - 128.0) * 0.55 + 128.0
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_gaussian_blur(img: np.ndarray) -> np.ndarray:
    """Simulate out-of-focus / motion blur (5x5 kernel, sigma=1.5)."""
    return cv2.GaussianBlur(img, (5, 5), sigmaX=1.5)


def apply_gaussian_noise(img: np.ndarray, seed: int = 42) -> np.ndarray:
    """Simulate camera sensor noise (additive Gaussian sigma=18)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 18.0, size=img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_rotate_small(img: np.ndarray, angle_deg: float = 5.0) -> np.ndarray:
    """Simulate slight part/camera rotation (+5 degrees, reflect border)."""
    h, w = img.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        img, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )


PERTURBATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "clean": lambda img: img,
    "brightness_up": apply_brightness_up,
    "brightness_down": apply_brightness_down,
    "contrast_low": apply_contrast_low,
    "gaussian_blur": apply_gaussian_blur,
    "gaussian_noise": apply_gaussian_noise,
    "rotate_5deg": apply_rotate_small,
}


@dataclass(frozen=True)
class BoxIoU:
    """Axis-aligned normalized bounding box (x_center, y_center, width, height)."""

    cls_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    conf: float = 1.0

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        x1 = self.x_center - self.width / 2.0
        y1 = self.y_center - self.height / 2.0
        x2 = self.x_center + self.width / 2.0
        y2 = self.y_center + self.height / 2.0
        return x1, y1, x2, y2


def box_iou(a: BoxIoU, b: BoxIoU) -> float:
    """Compute IoU between two normalized YOLO boxes."""
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def compute_ap50(
    predictions: list[tuple[int, list[BoxIoU]]],
    ground_truths: list[tuple[int, list[BoxIoU]]],
    *,
    class_id: int | None = None,
    iou_threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Compute VOC/COCO 101-point interpolated AP@0.5, precision, and recall.

    Parameters
    ----------
    predictions:
        List of ``(image_idx, predicted_boxes)``.
    ground_truths:
        List of ``(image_idx, gt_boxes)``.
    class_id:
        If given, filter both predictions and ground truths to ``cls_id == class_id``.
        If ``None``, evaluate class-agnostically (any defect box matches any GT box).

    Returns
    -------
    ``(ap50, precision_at_max_f1, recall_at_max_f1)``
    """
    gt_by_img: dict[int, list[BoxIoU]] = {}
    n_gt = 0
    for img_idx, boxes in ground_truths:
        filtered = [b for b in boxes if class_id is None or b.cls_id == class_id]
        gt_by_img[img_idx] = filtered
        n_gt += len(filtered)

    if n_gt == 0:
        return 0.0, 0.0, 0.0

    flat_preds: list[tuple[float, int, BoxIoU]] = []
    for img_idx, boxes in predictions:
        for b in boxes:
            if class_id is None or b.cls_id == class_id:
                flat_preds.append((b.conf, img_idx, b))
    flat_preds.sort(key=lambda x: x[0], reverse=True)

    if not flat_preds:
        return 0.0, 0.0, 0.0

    matched: dict[int, set[int]] = {img_idx: set() for img_idx in gt_by_img}
    tp = np.zeros(len(flat_preds), dtype=np.float64)
    fp = np.zeros(len(flat_preds), dtype=np.float64)

    for i, (_conf, img_idx, pred_box) in enumerate(flat_preds):
        gts = gt_by_img.get(img_idx, [])
        best_iou = 0.0
        best_j = -1
        for j, gt_box in enumerate(gts):
            if j in matched[img_idx]:
                continue
            iou = box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_threshold and best_j >= 0:
            matched[img_idx].add(best_j)
            tp[i] = 1.0
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / float(n_gt)
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # 101-point interpolation
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    recall_pts = np.linspace(0.0, 1.0, 101)
    ap50 = float(np.mean([mpre[np.searchsorted(mrec, r, side="left")] for r in recall_pts]))

    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    best_idx = int(np.argmax(f1))
    return ap50, float(prec[best_idx]), float(rec[best_idx])
