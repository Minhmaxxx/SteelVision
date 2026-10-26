"""Product-level QC metrics (escape rate, overkill rate) and threshold selection.

In manufacturing QC, every inspected image represents one product:
- **Escape rate** (False Negative Rate on defective products): fraction of
  defective (NG) products that pass inspection as OK. Primary safety metric.
- **Overkill rate** (False Positive Rate on good products): fraction of
  defect-free (OK) products that get rejected as NG. Primary cost/yield metric.

Thresholds ``(τ_det, τ_anom)`` are chosen on a validation set to satisfy a target
maximum escape rate (default ``<= 2%``) while minimizing overkill rate, then
evaluated once on the held-out test set.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from steelvision.decision.fusion import FusionThresholds, decide


@dataclass(frozen=True)
class ProductSample:
    """Single product (image) with ground-truth label and model outputs."""

    sample_id: str
    gt_is_ng: bool
    det_confidences: tuple[float, ...] = ()
    det_classes: tuple[str, ...] = ()
    anomaly_score: float | None = None
    defect_category: str = ""


@dataclass(frozen=True)
class QCMetrics:
    """Product-level QC evaluation metrics at a specific operating point."""

    det_threshold: float
    anom_threshold: float
    n_total: int
    n_gt_ng: int
    n_gt_ok: int
    tp: int  # Defective product correctly rejected (True NG)
    fn: int  # Defective product passed as OK (Escape)
    fp: int  # Good product rejected as NG (Overkill)
    tn: int  # Good product passed as OK (True OK)
    escape_rate: float
    overkill_rate: float
    precision: float
    recall: float
    f1: float
    reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("escape_rate", "overkill_rate", "precision", "recall", "f1"):
            d[k] = round(float(d[k]), 6)
        return d


def evaluate_qc(
    samples: Sequence[ProductSample],
    det_threshold: float = 0.5,
    anom_threshold: float = 0.5,
    *,
    ignore_det_classes: set[str] | None = None,
) -> QCMetrics:
    """Compute product-level QC metrics for ``samples`` at ``(det_threshold, anom_threshold)``.

    Parameters
    ----------
    samples:
        Product samples with ground-truth ``gt_is_ng``, detector confidences,
        and optional anomaly scores.
    det_threshold, anom_threshold:
        Operating thresholds ``(τ_det, τ_anom)``.
    ignore_det_classes:
        Optional set of defect class names to mask out from detector predictions
        (used to simulate an *unseen* defect class that the detector was never
        trained to recognize).
    """
    tau = FusionThresholds(det_threshold=det_threshold, anom_threshold=anom_threshold)
    tp = fn = fp = tn = 0
    reasons = {"ok": 0, "detection": 0, "anomaly": 0, "both": 0}

    for s in samples:
        if ignore_det_classes and s.det_classes:
            confs = [
                c
                for c, cls in zip(s.det_confidences, s.det_classes, strict=False)
                if cls not in ignore_det_classes
            ]
        else:
            confs = list(s.det_confidences)

        res = decide(confs, s.anomaly_score, thresholds=tau)
        reasons[res.reason] = reasons.get(res.reason, 0) + 1

        if s.gt_is_ng:
            if res.is_ng:
                tp += 1
            else:
                fn += 1
        else:
            if res.is_ng:
                fp += 1
            else:
                tn += 1

    n_gt_ng = tp + fn
    n_gt_ok = fp + tn
    escape_rate = fn / n_gt_ng if n_gt_ng > 0 else 0.0
    overkill_rate = fp / n_gt_ok if n_gt_ok > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / n_gt_ng if n_gt_ng > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return QCMetrics(
        det_threshold=det_threshold,
        anom_threshold=anom_threshold,
        n_total=len(samples),
        n_gt_ng=n_gt_ng,
        n_gt_ok=n_gt_ok,
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        escape_rate=escape_rate,
        overkill_rate=overkill_rate,
        precision=precision,
        recall=recall,
        f1=f1,
        reasons=reasons,
    )


def sweep_thresholds(
    samples: Sequence[ProductSample],
    det_thresholds: Sequence[float] = (0.25, 0.35, 0.50, 0.65, 0.80),
    anom_thresholds: Sequence[float] = (0.35, 0.45, 0.50, 0.55, 0.65),
    *,
    ignore_det_classes: set[str] | None = None,
) -> list[QCMetrics]:
    """Evaluate a grid of ``(det_threshold, anom_threshold)`` pairs."""
    curve: list[QCMetrics] = []
    for dt in det_thresholds:
        for at in anom_thresholds:
            curve.append(
                evaluate_qc(
                    samples,
                    det_threshold=float(dt),
                    anom_threshold=float(at),
                    ignore_det_classes=ignore_det_classes,
                )
            )
    return curve


def select_operating_point(
    curve: Sequence[QCMetrics],
    max_escape_rate: float = 0.02,
) -> QCMetrics:
    """Choose the best operating point from a validation threshold sweep.

    Rule:
    1. Filter candidates that achieve ``escape_rate <= max_escape_rate``.
       Among those, minimize ``overkill_rate``, tie-breaking by lowest
       ``escape_rate``, highest ``f1``, and highest ``(det_threshold, anom_threshold)``.
    2. If no candidate meets ``max_escape_rate``, pick the candidate with the
       minimum ``escape_rate`` (and lowest ``overkill_rate`` on ties).
    """
    if not curve:
        raise ValueError("Cannot select an operating point from an empty curve.")

    feasible = [m for m in curve if m.escape_rate <= max_escape_rate + 1e-9]
    if feasible:
        return min(
            feasible,
            key=lambda m: (
                round(m.overkill_rate, 9),
                round(m.escape_rate, 9),
                -round(m.f1, 9),
                -m.det_threshold,
                -m.anom_threshold,
            ),
        )
    return min(
        curve,
        key=lambda m: (
            round(m.escape_rate, 9),
            round(m.overkill_rate, 9),
            -round(m.f1, 9),
        ),
    )
