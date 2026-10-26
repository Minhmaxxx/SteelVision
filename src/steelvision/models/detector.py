"""YOLOv8 object detector ONNX export and pure ONNX Runtime CPU inference wrapper.

Used in Phase 6 (CPU latency benchmarking) and Phase 7 (FastAPI `/inspect`
service) so the production inspection service can run without PyTorch /
Ultralytics training dependencies.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from steelvision.decision.fusion import Detection

NEU_DET_CLASSES: tuple[str, ...] = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
)


def export_yolo_onnx(
    weights_path: str | Path,
    onnx_path: str | Path | None = None,
    imgsz: int = 640,
    opset: int = 18,
) -> Path:
    """Export an Ultralytics YOLOv8 ``.pt`` checkpoint to a static ``batch=1`` ONNX file."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    from ultralytics import YOLO

    weights_path = Path(weights_path)
    target = Path(onnx_path) if onnx_path is not None else weights_path.with_suffix(".onnx")
    model = YOLO(str(weights_path))
    exported = Path(
        model.export(format="onnx", imgsz=imgsz, opset=opset, dynamic=False, simplify=False)
    )
    if exported.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported), str(target))
        data_file = exported.with_suffix(".onnx.data")
        if data_file.exists():
            shutil.move(str(data_file), str(target.with_suffix(".onnx.data")))
    return target


def _letterbox(
    img_bgr: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize and pad image while meeting stride-multiple constraints (Ultralytics letterbox)."""
    h, w = img_bgr.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = int(round(w * r)), int(round(h * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2.0, (new_shape[0] - new_unpad[1]) / 2.0

    if (w, h) != new_unpad:
        img_bgr = cv2.resize(img_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_bgr = cv2.copyMakeBorder(
        img_bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img_bgr, r, (dw, dh)


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list[int]:
    """Pure-NumPy greedy Non-Maximum Suppression on ``(N, 4)`` ``[x1, y1, x2, y2]`` boxes."""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0.0, inter / union, 0.0)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


class OnnxYoloDetector:
    """Pure ONNX Runtime CPU wrapper for YOLOv8 detection."""

    def __init__(
        self,
        onnx_path: str | Path,
        class_names: tuple[str, ...] | list[str] = NEU_DET_CLASSES,
        imgsz: int = 640,
        intra_op_threads: int = 0,
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        self.class_names = tuple(class_names)
        self.imgsz = imgsz

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads > 0:
            sess_opts.intra_op_num_threads = intra_op_threads

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_opts,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(
        self,
        img_bgr: np.ndarray,
        conf_threshold: float = 0.10,
        iou_threshold: float = 0.45,
    ) -> list[Detection]:
        """Run YOLOv8 ONNX inference on a BGR uint8 image and return ``Detection`` boxes."""
        orig_h, orig_w = img_bgr.shape[:2]
        padded, ratio, (dw, dh) = _letterbox(img_bgr, (self.imgsz, self.imgsz))
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
        batch = np.expand_dims(chw, axis=0)

        raw_out = self.session.run(None, {self.input_name: batch})[0]
        # YOLOv8 output: (1, 4 + C, num_anchors) -> transpose to (num_anchors, 4 + C)
        preds = np.squeeze(raw_out, axis=0).T
        if preds.shape[0] == 0:
            return []

        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
        cls_ids = np.argmax(class_scores, axis=1)
        confs = class_scores[np.arange(len(preds)), cls_ids]

        mask = confs >= conf_threshold
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        confs = confs[mask]
        cls_ids = cls_ids[mask]

        # Convert center xywh (in letterboxed space) to original image xyxy
        x1 = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0 - dw) / ratio
        y1 = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0 - dh) / ratio
        x2 = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0 - dw) / ratio
        y2 = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0 - dh) / ratio
        x1 = np.clip(x1, 0.0, float(orig_w))
        y1 = np.clip(y1, 0.0, float(orig_h))
        x2 = np.clip(x2, 0.0, float(orig_w))
        y2 = np.clip(y2, 0.0, float(orig_h))
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Class-offset NMS (matches Ultralytics batched_nms)
        max_wh = float(max(orig_w, orig_h)) * 2.0
        offset_boxes = boxes_xyxy + (cls_ids[:, None].astype(np.float32) * max_wh)
        keep = _nms_xyxy(offset_boxes, confs, iou_threshold=iou_threshold)

        detections: list[Detection] = []
        for idx in keep:
            cid = int(cls_ids[idx])
            cname = self.class_names[cid] if 0 <= cid < len(self.class_names) else str(cid)
            bx1, by1, bx2, by2 = boxes_xyxy[idx]
            detections.append(
                Detection(
                    class_name=cname,
                    confidence=float(np.clip(confs[idx], 0.0, 1.0)),
                    bbox=(
                        round(float(bx1), 2),
                        round(float(by1), 2),
                        round(float(bx2), 2),
                        round(float(by2), 2),
                    ),
                )
            )
        return detections
