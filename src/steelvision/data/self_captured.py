"""Self-captured dataset tooling (Phase 5): manifest validation, ingestion, and QC.

Implements the rules specified in:
- ``docs/capture-protocol.vi.md`` (manifest CSV schema, >=40% OK ratio, panel+position grouping)
- ``docs/labeling-guide.vi.md`` (8 px min box, 10% re-annotation: mean IoU >= 0.75, F1 >= 0.90)
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path

from steelvision.eval.robustness import BoxIoU, box_iou

REQUIRED_MANIFEST_COLUMNS = (
    "file",
    "session",
    "panel",
    "position",
    "lighting",
    "is_defective",
    "label_status",
    "notes",
)
VALID_LIGHTING = {"A", "B", "C"}
VALID_DEFECTIVE = {"ok", "defective", "unknown"}
VALID_STATUS = {"pending", "labeled", "qc-passed", "skip"}

SELF_CAPTURED_CLASSES = (
    "dent",
    "pit",
    "rust",
    "scratch",
    "stain",
    "weld_seam",
)


@dataclass
class ManifestReport:
    n_rows: int = 0
    n_qc_passed: int = 0
    n_ok: int = 0
    n_defective: int = 0
    ok_ratio: float = 0.0
    lighting_counts: dict[str, int] = field(default_factory=dict)
    lighting_ok_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    meets_stopping_criteria: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def validate_manifest(
    manifest_csv: Path,
    *,
    min_total_qc_passed: int = 300,
    min_ok_ratio: float = 0.40,
    min_ok_per_lighting: int = 15,
) -> ManifestReport:
    """Validate a ``manifest.csv`` file against ``docs/capture-protocol.vi.md`` rules."""
    report = ManifestReport()
    if not manifest_csv.exists():
        report.errors.append(f"Manifest file not found: {manifest_csv}")
        return report

    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing_cols = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in fieldnames]
        if missing_cols:
            report.errors.append(f"Missing required columns: {missing_cols}")
            return report

        seen_files: set[str] = set()
        for row_idx, row in enumerate(reader, start=2):
            report.n_rows += 1
            fname = (row.get("file") or "").strip()
            lighting = (row.get("lighting") or "").strip().upper()
            is_def = (row.get("is_defective") or "").strip().lower()
            status = (row.get("label_status") or "").strip().lower()
            panel = (row.get("panel") or "").strip()
            pos = (row.get("position") or "").strip()

            if not fname:
                report.errors.append(f"Row {row_idx}: empty 'file'")
            elif fname in seen_files:
                report.errors.append(f"Row {row_idx}: duplicate file '{fname}'")
            seen_files.add(fname)

            if not panel or not pos:
                report.errors.append(f"Row {row_idx}: 'panel' and 'position' must be non-empty")
            if lighting not in VALID_LIGHTING:
                report.errors.append(
                    f"Row {row_idx}: invalid lighting '{lighting}' (expected A/B/C)"
                )
            if is_def not in VALID_DEFECTIVE:
                report.errors.append(f"Row {row_idx}: invalid is_defective '{is_def}'")
            if status not in VALID_STATUS:
                report.errors.append(f"Row {row_idx}: invalid label_status '{status}'")

            if status == "skip":
                continue

            report.lighting_counts[lighting] = report.lighting_counts.get(lighting, 0) + 1
            if status == "qc-passed":
                report.n_qc_passed += 1
            if is_def == "ok":
                report.n_ok += 1
                report.lighting_ok_counts[lighting] = report.lighting_ok_counts.get(lighting, 0) + 1
            elif is_def == "defective":
                report.n_defective += 1

    labeled_total = report.n_ok + report.n_defective
    report.ok_ratio = report.n_ok / labeled_total if labeled_total > 0 else 0.0

    report.meets_stopping_criteria = (
        len(report.errors) == 0
        and report.n_qc_passed >= min_total_qc_passed
        and report.ok_ratio >= min_ok_ratio
        and all(
            report.lighting_ok_counts.get(lt, 0) >= min_ok_per_lighting for lt in VALID_LIGHTING
        )
    )
    return report


def panel_position_groups(manifest_csv: Path) -> dict[str, list[str]]:
    """Group image filenames by ``(panel, position)`` so lighting shots A/B/C stay in one split."""
    groups: dict[str, list[str]] = {}
    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("label_status") or "").strip().lower() == "skip":
                continue
            key = f"{row['panel'].strip()}__{row['position'].strip()}".lower()
            groups.setdefault(key, []).append(row["file"].strip())
    return groups


@dataclass(frozen=True)
class ConsistencyReport:
    """Inter-annotator / 10% re-annotation consistency metrics (Section 5 of labeling guide)."""

    n_images: int
    mean_box_iou: float
    image_level_f1: float
    image_level_agreement: float
    n_matched_boxes: int
    n_unmatched_boxes: int
    passes_guide_targets: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("mean_box_iou", "image_level_f1", "image_level_agreement"):
            d[k] = round(float(d[k]), 4)
        return d


def _parse_yolo_txt(path: Path) -> list[BoxIoU]:
    if not path.exists():
        return []
    boxes: list[BoxIoU] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            boxes.append(
                BoxIoU(
                    cls_id=int(parts[0]),
                    x_center=float(parts[1]),
                    y_center=float(parts[2]),
                    width=float(parts[3]),
                    height=float(parts[4]),
                )
            )
    return boxes


def compute_label_consistency(
    labels_dir_a: Path,
    labels_dir_b: Path,
    *,
    min_mean_iou: float = 0.75,
    min_image_f1: float = 0.90,
) -> ConsistencyReport:
    """Measure consistency between original labels (A) and 10% blind re-annotation (B).

    Matches boxes of the same class greedily by IoU within each image, and also
    computes image-level OK/NG F1 (where an image is NG iff it has >=1 box).
    """
    stems = sorted(
        {p.stem for p in labels_dir_a.glob("*.txt")} | {p.stem for p in labels_dir_b.glob("*.txt")}
    )
    if not stems:
        raise ValueError("No .txt label files found in either directory.")

    ious: list[float] = []
    n_matched = 0
    n_unmatched = 0
    tp = fp = fn = tn = 0

    for stem in stems:
        boxes_a = _parse_yolo_txt(labels_dir_a / f"{stem}.txt")
        boxes_b = _parse_yolo_txt(labels_dir_b / f"{stem}.txt")

        ng_a = len(boxes_a) > 0
        ng_b = len(boxes_b) > 0
        if ng_a and ng_b:
            tp += 1
        elif ng_a and not ng_b:
            fn += 1
        elif not ng_a and ng_b:
            fp += 1
        else:
            tn += 1

        used_b: set[int] = set()
        for ba in boxes_a:
            best_iou = 0.0
            best_j = -1
            for j, bb in enumerate(boxes_b):
                if j in used_b or bb.cls_id != ba.cls_id:
                    continue
                iou = box_iou(ba, bb)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou > 0.0:
                used_b.add(best_j)
                ious.append(best_iou)
                n_matched += 1
            else:
                ious.append(0.0)
                n_unmatched += 1

        for j in range(len(boxes_b)):
            if j not in used_b:
                ious.append(0.0)
                n_unmatched += 1

    mean_iou = float(sum(ious) / len(ious)) if ious else 1.0
    denom = 2 * tp + fp + fn
    image_f1 = (2 * tp / denom) if denom > 0 else 1.0
    agreement = (tp + tn) / len(stems)

    return ConsistencyReport(
        n_images=len(stems),
        mean_box_iou=mean_iou,
        image_level_f1=image_f1,
        image_level_agreement=agreement,
        n_matched_boxes=n_matched,
        n_unmatched_boxes=n_unmatched,
        passes_guide_targets=(mean_iou >= min_mean_iou and image_f1 >= min_image_f1),
    )
