"""Product-level OK/NG decision rule fusing object detection and anomaly scores.

A product (image) is judged **NG** if:
    1. any predicted bounding box has ``confidence >= det_threshold`` (``τ_det``), OR
    2. the image-level anomaly score has ``anomaly_score >= anom_threshold`` (``τ_anom``).

The detector identifies *known* defect classes and locations; the anomaly model
(PatchCore, trained on defect-free images only) acts as a safety net for
*unseen* defect types or subtle textures below the detector threshold.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

Verdict = Literal["OK", "NG"]
Reason = Literal["ok", "detection", "anomaly", "both"]


@dataclass(frozen=True)
class Detection:
    """Single bounding-box prediction from the object detector."""

    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if math.isnan(self.confidence) or not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Detection confidence must be in [0, 1], got {self.confidence}")


@dataclass(frozen=True)
class FusionThresholds:
    """Operating thresholds (τ_det, τ_anom) chosen on a validation set."""

    det_threshold: float = 0.5
    anom_threshold: float = 0.5

    def __post_init__(self) -> None:
        if math.isnan(self.det_threshold) or not (0.0 <= self.det_threshold <= 1.0):
            raise ValueError(f"det_threshold must be in [0, 1], got {self.det_threshold}")
        if not math.isfinite(self.anom_threshold):
            raise ValueError(f"anom_threshold must be finite, got {self.anom_threshold}")


@dataclass(frozen=True)
class DecisionResult:
    """Structured product-level inspection verdict."""

    verdict: Verdict
    reason: Reason
    det_triggered: bool
    anom_triggered: bool
    max_det_confidence: float
    anomaly_score: float | None
    thresholds: FusionThresholds
    active_detections: list[Detection] = field(default_factory=list)

    @property
    def is_ng(self) -> bool:
        return self.verdict == "NG"

    @property
    def defect_classes(self) -> list[str]:
        """Unique defect classes among active detections (ordered by confidence)."""
        seen: list[str] = []
        for det in self.active_detections:
            if det.class_name not in seen:
                seen.append(det.class_name)
        return seen


def decide(
    detections: Sequence[Detection | float] = (),
    anomaly_score: float | None = None,
    *,
    det_threshold: float = 0.5,
    anom_threshold: float = 0.5,
    thresholds: FusionThresholds | None = None,
) -> DecisionResult:
    """Apply the OR-fusion rule to detector predictions and an optional anomaly score.

    Parameters
    ----------
    detections:
        Predicted bounding boxes (:class:`Detection`) or raw box confidence
        floats (convenient for threshold sweeps during QC evaluation).
    anomaly_score:
        Image-level anomaly score from PatchCore, or ``None`` when running the
        detector standalone.
    det_threshold, anom_threshold:
        Operating thresholds ``τ_det`` and ``τ_anom``. Ignored if ``thresholds``
        is passed explicitly.
    thresholds:
        Optional pre-constructed :class:`FusionThresholds` instance.
    """
    tau = thresholds if thresholds is not None else FusionThresholds(det_threshold, anom_threshold)

    if anomaly_score is not None and not math.isfinite(anomaly_score):
        raise ValueError(f"anomaly_score must be finite or None, got {anomaly_score}")

    normalized: list[Detection] = [
        d if isinstance(d, Detection) else Detection(class_name="defect", confidence=float(d))
        for d in detections
    ]
    max_det_conf = max((d.confidence for d in normalized), default=0.0)
    active = sorted(
        (d for d in normalized if d.confidence >= tau.det_threshold),
        key=lambda d: d.confidence,
        reverse=True,
    )

    det_triggered = len(active) > 0
    anom_triggered = anomaly_score is not None and anomaly_score >= tau.anom_threshold

    if det_triggered and anom_triggered:
        verdict: Verdict = "NG"
        reason: Reason = "both"
    elif det_triggered:
        verdict = "NG"
        reason = "detection"
    elif anom_triggered:
        verdict = "NG"
        reason = "anomaly"
    else:
        verdict = "OK"
        reason = "ok"

    return DecisionResult(
        verdict=verdict,
        reason=reason,
        det_triggered=det_triggered,
        anom_triggered=anom_triggered,
        max_det_confidence=max_det_conf,
        anomaly_score=anomaly_score,
        thresholds=tau,
        active_detections=active,
    )
