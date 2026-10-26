"""Run Phase 4 QC evaluation, unseen-defect tests, perturbation robustness, and cross-domain tests.

    python -m steelvision.eval.run_qc --config configs/eval.yaml

Outputs structured metrics to ``runs/eval/qc_report.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from steelvision.eval.qc_metrics import (
    ProductSample,
    evaluate_qc,
    select_operating_point,
    sweep_thresholds,
)
from steelvision.eval.robustness import (
    PERTURBATIONS,
    BoxIoU,
    compute_ap50,
)
from steelvision.models.anomaly import OnnxAnomalyDetector


def _load_yolo_gt_boxes(label_path: Path) -> list[BoxIoU]:
    if not label_path.exists():
        return []
    boxes: list[BoxIoU] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            boxes.append(
                BoxIoU(
                    cls_id=int(parts[0]),
                    x_center=float(parts[1]),
                    y_center=float(parts[2]),
                    width=float(parts[3]),
                    height=float(parts[4]),
                    conf=1.0,
                )
            )
    return boxes


def _extract_clean_ok_crops(
    images_dir: Path,
    labels_dir: Path,
    n_crops: int,
    seed: int = 42,
    min_patch_px: int = 64,
) -> list[tuple[str, np.ndarray]]:
    """Extract defect-free background crops disjoint from all ground-truth boxes.

    NEU-DET has at least one annotated defect box per image. To measure
    overkill rate on real hot-rolled steel background from the same camera and
    illumination distribution, we find largest axis-aligned background strips
    strictly outside all ground-truth bounding boxes (with a 6-pixel safety
    margin) and resize them to the native 200x200 image size.
    """
    rng = np.random.default_rng(seed)
    img_paths = sorted(images_dir.glob("*.*"))
    rng.shuffle(img_paths)
    crops: list[tuple[str, np.ndarray]] = []
    margin = 6

    for img_path in img_paths:
        if len(crops) >= n_crops:
            break
        gt_boxes = _load_yolo_gt_boxes(labels_dir / f"{img_path.stem}.txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Compute bounding union of all GT boxes in pixel coordinates
        if not gt_boxes:
            crops.append((f"ok_bg__{img_path.stem}", img))
            continue

        x1s = [max(0, int((b.x_center - b.width / 2) * w) - margin) for b in gt_boxes]
        y1s = [max(0, int((b.y_center - b.height / 2) * h) - margin) for b in gt_boxes]
        x2s = [min(w, int((b.x_center + b.width / 2) * w) + margin) for b in gt_boxes]
        y2s = [min(h, int((b.y_center + b.height / 2) * h) + margin) for b in gt_boxes]

        ux1, uy1, ux2, uy2 = min(x1s), min(y1s), max(x2s), max(y2s)
        # Candidate strips: left, right, top, bottom of the union box
        candidates = [
            (ux1, h, img[:, :ux1]),
            (w - ux2, h, img[:, ux2:]),
            (w, uy1, img[:uy1, :]),
            (w, h - uy2, img[uy2:, :]),
        ]
        candidates.sort(key=lambda c: min(c[0], c[1]), reverse=True)
        bw, bh, sub = candidates[0]
        if min(bw, bh) >= min_patch_px and sub.size > 0:
            resized = cv2.resize(sub, (w, h), interpolation=cv2.INTER_LINEAR)
            crops.append((f"ok_bg__{img_path.stem}", resized))

    return crops


def _compute_binary_auroc(labels: list[int], scores: list[float]) -> float:
    """Exact Mann-Whitney U / Wilcoxon rank-sum AUROC without sklearn dependency."""
    pos = [s for y, s in zip(labels, scores, strict=False) if y == 1]
    neg = [s for y, s in zip(labels, scores, strict=False) if y == 0]
    if not pos or not neg:
        return 0.0
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def evaluate_patchcore_qc_and_robustness(cfg: dict, target_escape: float, seed: int) -> dict:
    """Run PatchCore ONNX on MVTec metal_nut: val/test QC, unseen defect, and robustness."""
    onnx_path = Path(cfg["onnx_path"])
    test_dir = Path(cfg["mvtec_test_dir"])
    input_size = tuple(cfg["input_size"])
    anom_thresholds = [float(x) for x in cfg["anom_thresholds"]]

    detector = OnnxAnomalyDetector(onnx_path, input_size=input_size)

    # 1. Score all clean images in metal_nut/test
    raw_items: list[tuple[str, str, Path, np.ndarray, float]] = []
    for cat_dir in sorted(d for d in test_dir.iterdir() if d.is_dir()):
        cat = cat_dir.name
        for p in sorted(cat_dir.glob("*.png")):
            bgr = cv2.imread(str(p))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            score, _ = detector.predict(rgb)
            raw_items.append((f"{cat}/{p.stem}", cat, p, bgr, score))

    # Stratified 50/50 split into val and test by category
    rng = np.random.default_rng(seed)
    val_samples: list[ProductSample] = []
    test_samples: list[ProductSample] = []
    all_samples: list[ProductSample] = []

    by_cat: dict[str, list[ProductSample]] = {}
    for sid, cat, _p, _bgr, score in raw_items:
        gt_ng = cat != "good"
        # Simulate a detector that detects known defects with high confidence if not held out
        ps = ProductSample(
            sample_id=sid,
            gt_is_ng=gt_ng,
            det_confidences=(0.90,) if gt_ng else (),
            det_classes=(cat,) if gt_ng else (),
            anomaly_score=score,
            defect_category=cat,
        )
        all_samples.append(ps)
        by_cat.setdefault(cat, []).append(ps)

    for _cat, items in sorted(by_cat.items()):
        idxs = list(range(len(items)))
        rng.shuffle(idxs)
        mid = len(idxs) // 2
        for i in idxs[:mid]:
            val_samples.append(items[i])
        for i in idxs[mid:]:
            test_samples.append(items[i])

    # Anomaly-only threshold sweep on val -> evaluate on test
    val_anom_only = [
        ProductSample(s.sample_id, s.gt_is_ng, (), (), s.anomaly_score, s.defect_category)
        for s in val_samples
    ]
    test_anom_only = [
        ProductSample(s.sample_id, s.gt_is_ng, (), (), s.anomaly_score, s.defect_category)
        for s in test_samples
    ]
    all_anom_only = [
        ProductSample(s.sample_id, s.gt_is_ng, (), (), s.anomaly_score, s.defect_category)
        for s in all_samples
    ]

    val_curve = sweep_thresholds(
        val_anom_only, det_thresholds=[0.5], anom_thresholds=anom_thresholds
    )
    chosen_val = select_operating_point(val_curve, max_escape_rate=target_escape)
    chosen_tau_anom = chosen_val.anom_threshold
    test_at_chosen = evaluate_qc(test_anom_only, det_threshold=0.5, anom_threshold=chosen_tau_anom)
    full_curve = sweep_thresholds(
        all_anom_only, det_thresholds=[0.5], anom_thresholds=anom_thresholds
    )

    # 2. Unseen defect class test (leave-one-defect-category-out)
    defect_cats = [c for c in sorted(by_cat) if c != "good"]
    unseen_results: dict[str, dict] = {}
    for held_out in defect_cats:
        det_only = evaluate_qc(
            all_samples,
            det_threshold=0.5,
            anom_threshold=999.0,  # disable anomaly branch
            ignore_det_classes={held_out},
        )
        fused = evaluate_qc(
            all_samples,
            det_threshold=0.5,
            anom_threshold=chosen_tau_anom,
            ignore_det_classes={held_out},
        )
        unseen_results[held_out] = {
            "n_held_out_ng": len(by_cat[held_out]),
            "detector_only_escape_rate": round(det_only.escape_rate, 4),
            "detector_only_fn": det_only.fn,
            "fused_escape_rate": round(fused.escape_rate, 4),
            "fused_fn": fused.fn,
            "fused_overkill_rate": round(fused.overkill_rate, 4),
        }

    # 3. Perturbation robustness on all 115 MVTec metal_nut test images
    robustness_table: dict[str, dict] = {}
    clean_auroc = _compute_binary_auroc(
        [1 if s.gt_is_ng else 0 for s in all_anom_only],
        [float(s.anomaly_score or 0.0) for s in all_anom_only],
    )
    for pert_name, pert_fn in PERTURBATIONS.items():
        if pert_name == "clean":
            pert_samples = all_anom_only
            auroc = clean_auroc
        else:
            pert_samples = []
            labels: list[int] = []
            scores: list[float] = []
            for sid, cat, _p, bgr, _clean_s in raw_items:
                pert_bgr = pert_fn(bgr)
                pert_rgb = cv2.cvtColor(pert_bgr, cv2.COLOR_BGR2RGB)
                sc, _ = detector.predict(pert_rgb)
                gt_ng = cat != "good"
                pert_samples.append(ProductSample(sid, gt_ng, (), (), sc, cat))
                labels.append(1 if gt_ng else 0)
                scores.append(sc)
            auroc = _compute_binary_auroc(labels, scores)

        m = evaluate_qc(pert_samples, det_threshold=0.5, anom_threshold=chosen_tau_anom)
        robustness_table[pert_name] = {
            "image_AUROC": round(auroc, 4),
            "delta_AUROC": round(auroc - clean_auroc, 4),
            "escape_rate": round(m.escape_rate, 4),
            "overkill_rate": round(m.overkill_rate, 4),
            "f1": round(m.f1, 4),
        }

    return {
        "n_val": len(val_anom_only),
        "n_test": len(test_anom_only),
        "chosen_anom_threshold": chosen_tau_anom,
        "val_operating_point": chosen_val.to_dict(),
        "test_at_operating_point": test_at_chosen.to_dict(),
        "full_set_tradeoff_curve": [m.to_dict() for m in full_curve],
        "unseen_defect_leave_one_out": unseen_results,
        "perturbation_robustness": robustness_table,
    }


def _run_yolo_on_images(
    model,
    images: list[tuple[str, np.ndarray, list[BoxIoU], bool]],
    class_names: list[str],
    imgsz: int = 640,
) -> tuple[list[ProductSample], list[tuple[int, list[BoxIoU]]], list[tuple[int, list[BoxIoU]]]]:
    """Run Ultralytics YOLO on a list of ``(sample_id, bgr_img, gt_boxes, gt_is_ng)``."""
    bgr_list = [item[1] for item in images]
    results = model.predict(bgr_list, imgsz=imgsz, conf=0.01, verbose=False, batch=32)

    samples: list[ProductSample] = []
    preds_iou: list[tuple[int, list[BoxIoU]]] = []
    gts_iou: list[tuple[int, list[BoxIoU]]] = []

    for idx, ((sid, _bgr, gt_boxes, gt_ng), res) in enumerate(zip(images, results, strict=False)):
        boxes = res.boxes
        confs: list[float] = []
        clss: list[str] = []
        pred_boxes: list[BoxIoU] = []

        if boxes is not None and len(boxes) > 0:
            xywhn = boxes.xywhn.cpu().numpy()
            conf_arr = boxes.conf.cpu().numpy()
            cls_arr = boxes.cls.cpu().numpy().astype(int)
            for (xc, yc, w, h), c, cid in zip(xywhn, conf_arr, cls_arr, strict=False):
                confs.append(float(c))
                cname = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
                clss.append(cname)
                pred_boxes.append(BoxIoU(cid, float(xc), float(yc), float(w), float(h), float(c)))

        samples.append(
            ProductSample(
                sample_id=sid,
                gt_is_ng=gt_ng,
                det_confidences=tuple(confs),
                det_classes=tuple(clss),
                anomaly_score=None,
            )
        )
        if gt_ng:
            preds_iou.append((idx, pred_boxes))
            gts_iou.append((idx, gt_boxes))

    return samples, preds_iou, gts_iou


def evaluate_yolo_qc_robustness_and_cross_domain(
    cfg: dict, target_escape: float, seed: int
) -> dict:
    """Run YOLOv8n on NEU-DET, perturbations, unseen classes, and GC10 cross-domain."""
    from ultralytics import YOLO

    weights_path = Path(cfg["weights"])
    neu_dir = Path(cfg["neu_det_dir"])
    gc10_dir = Path(cfg["gc10_dir"])
    imgsz = int(cfg.get("imgsz", 640))
    det_thresholds = [float(x) for x in cfg["det_thresholds"]]

    neu_yaml = yaml.safe_load((neu_dir / "data.yaml").read_text(encoding="utf-8"))
    neu_classes = [neu_yaml["names"][i] for i in sorted(neu_yaml["names"])]

    model = YOLO(str(weights_path))

    # Load NEU-DET val and test defective images + clean background OK crops
    def _load_split(split: str, n_ok: int) -> list[tuple[str, np.ndarray, list[BoxIoU], bool]]:
        img_dir = neu_dir / "images" / split
        lbl_dir = neu_dir / "labels" / split
        items: list[tuple[str, np.ndarray, list[BoxIoU], bool]] = []
        for p in sorted(img_dir.glob("*.*")):
            bgr = cv2.imread(str(p))
            gts = _load_yolo_gt_boxes(lbl_dir / f"{p.stem}.txt")
            items.append((p.stem, bgr, gts, True))
        ok_crops = _extract_clean_ok_crops(img_dir, lbl_dir, n_crops=n_ok, seed=seed)
        for sid, crop_bgr in ok_crops:
            items.append((sid, crop_bgr, [], False))
        return items

    val_items = _load_split("val", int(cfg.get("n_ok_crops_val", 80)))
    test_items = _load_split("test", int(cfg.get("n_ok_crops_test", 80)))

    val_samples, _, _ = _run_yolo_on_images(model, val_items, neu_classes, imgsz=imgsz)
    test_samples, clean_preds, clean_gts = _run_yolo_on_images(
        model, test_items, neu_classes, imgsz=imgsz
    )

    val_curve = sweep_thresholds(
        val_samples, det_thresholds=det_thresholds, anom_thresholds=[999.0]
    )
    chosen_val = select_operating_point(val_curve, max_escape_rate=target_escape)
    chosen_tau_det = chosen_val.det_threshold

    test_at_chosen = evaluate_qc(test_samples, det_threshold=chosen_tau_det, anom_threshold=999.0)
    test_curve = sweep_thresholds(
        test_samples, det_thresholds=det_thresholds, anom_thresholds=[999.0]
    )

    # Unseen class simulation on NEU-DET test set (masking out 1 class at a time)
    neu_unseen: dict[str, dict] = {}
    for cls_name in neu_classes:
        m_masked = evaluate_qc(
            test_samples,
            det_threshold=chosen_tau_det,
            anom_threshold=999.0,
            ignore_det_classes={cls_name},
        )
        neu_unseen[cls_name] = {
            "detector_only_escape_rate": round(m_masked.escape_rate, 4),
            "escaped_products_fn": m_masked.fn,
            "delta_escape_vs_all_classes": round(
                m_masked.escape_rate - test_at_chosen.escape_rate, 4
            ),
        }

    # Perturbation robustness on NEU-DET test set
    clean_ap50, clean_prec, clean_rec = compute_ap50(clean_preds, clean_gts, class_id=None)
    robustness_table: dict[str, dict] = {}

    for pert_name, pert_fn in PERTURBATIONS.items():
        if pert_name == "clean":
            p_samples, p_preds, p_gts = test_samples, clean_preds, clean_gts
            ap50, prec, rec = clean_ap50, clean_prec, clean_rec
        else:
            pert_items = [(sid, pert_fn(bgr), gts, gt_ng) for sid, bgr, gts, gt_ng in test_items]
            p_samples, p_preds, p_gts = _run_yolo_on_images(
                model, pert_items, neu_classes, imgsz=imgsz
            )
            ap50, prec, rec = compute_ap50(p_preds, p_gts, class_id=None)

        m = evaluate_qc(p_samples, det_threshold=chosen_tau_det, anom_threshold=999.0)
        robustness_table[pert_name] = {
            "class_agnostic_ap50": round(ap50, 4),
            "delta_ap50": round(ap50 - clean_ap50, 4),
            "box_precision": round(prec, 4),
            "box_recall": round(rec, 4),
            "escape_rate": round(m.escape_rate, 4),
            "overkill_rate": round(m.overkill_rate, 4),
        }

    # Cross-domain test: Train on NEU-DET -> Test on GC10-DET test split (349 images)
    cross_domain: dict = {}
    gc10_test_img_dir = gc10_dir / "images" / "test"
    gc10_test_lbl_dir = gc10_dir / "labels" / "test"
    if gc10_test_img_dir.exists():
        gc10_items: list[tuple[str, np.ndarray, list[BoxIoU], bool]] = []
        for p in sorted(gc10_test_img_dir.glob("*.*")):
            bgr = cv2.imread(str(p))
            gts = _load_yolo_gt_boxes(gc10_test_lbl_dir / f"{p.stem}.txt")
            gc10_items.append((p.stem, bgr, gts, len(gts) > 0))

        gc10_samples, gc10_preds, gc10_gts = _run_yolo_on_images(
            model, gc10_items, neu_classes, imgsz=imgsz
        )
        gc10_ap50, gc10_prec, gc10_rec = compute_ap50(gc10_preds, gc10_gts, class_id=None)

        # Also evaluate shared class 'inclusion' (class_id=1 in NEU-DET, class_id=6 in GC10-DET)
        neu_inc_preds = [
            (
                idx,
                [
                    BoxIoU(0, b.x_center, b.y_center, b.width, b.height, b.conf)
                    for b in bs
                    if b.cls_id == 1
                ],
            )
            for idx, bs in gc10_preds
        ]
        gc10_inc_gts = [
            (
                idx,
                [
                    BoxIoU(0, b.x_center, b.y_center, b.width, b.height, 1.0)
                    for b in bs
                    if b.cls_id == 6
                ],
            )
            for idx, bs in gc10_gts
        ]
        inc_ap50, _, _ = compute_ap50(neu_inc_preds, gc10_inc_gts, class_id=0)

        m_gc10_chosen = evaluate_qc(
            gc10_samples, det_threshold=chosen_tau_det, anom_threshold=999.0
        )
        m_gc10_low = evaluate_qc(gc10_samples, det_threshold=0.15, anom_threshold=999.0)
        cross_domain = {
            "source_dataset": "neu_det",
            "target_dataset": "gc10_test",
            "n_target_images": len(gc10_items),
            "in_domain_class_agnostic_ap50": round(clean_ap50, 4),
            "cross_domain_class_agnostic_ap50": round(gc10_ap50, 4),
            "delta_ap50": round(gc10_ap50 - clean_ap50, 4),
            "shared_class_inclusion_ap50": round(inc_ap50, 4),
            "box_precision": round(gc10_prec, 4),
            "box_recall": round(gc10_rec, 4),
            f"escape_rate_at_tau_{chosen_tau_det}": round(m_gc10_chosen.escape_rate, 4),
            "escape_rate_at_tau_0.15": round(m_gc10_low.escape_rate, 4),
        }

    return {
        "n_val_ng": sum(1 for s in val_samples if s.gt_is_ng),
        "n_val_ok": sum(1 for s in val_samples if not s.gt_is_ng),
        "n_test_ng": sum(1 for s in test_samples if s.gt_is_ng),
        "n_test_ok": sum(1 for s in test_samples if not s.gt_is_ng),
        "chosen_det_threshold": chosen_tau_det,
        "val_operating_point": chosen_val.to_dict(),
        "test_at_operating_point": test_at_chosen.to_dict(),
        "test_tradeoff_curve": [m.to_dict() for m in test_curve],
        "unseen_class_simulation": neu_unseen,
        "perturbation_robustness": robustness_table,
        "cross_domain_neu_to_gc10": cross_domain,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 42))
    target_escape = float(cfg.get("target_max_escape_rate", 0.02))

    t0 = time.perf_counter()
    print("[eval] 1/2 Running PatchCore QC + unseen-defect + robustness on MVTec metal_nut ...")
    patchcore_report = evaluate_patchcore_qc_and_robustness(cfg["patchcore"], target_escape, seed)

    print(
        "[eval] 2/2 Running YOLOv8n QC + perturbation robustness + cross-domain (NEU -> GC10) ..."
    )
    yolo_report = evaluate_yolo_qc_robustness_and_cross_domain(cfg["yolo"], target_escape, seed)

    elapsed = time.perf_counter() - t0
    report = {
        "seed": seed,
        "target_max_escape_rate": target_escape,
        "elapsed_sec": round(elapsed, 2),
        "patchcore_mvtec_metal_nut": patchcore_report,
        "yolov8n_neu_det": yolo_report,
    }

    out_path = Path(cfg["output"]["json_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[eval] Done in {elapsed:.1f}s -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
