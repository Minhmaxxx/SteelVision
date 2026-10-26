import pytest

from steelvision.decision.fusion import (
    Detection,
    FusionThresholds,
    decide,
)


def test_ok_when_both_below_threshold():
    res = decide(
        detections=[Detection("crazing", 0.25), Detection("scratches", 0.49)],
        anomaly_score=0.35,
        det_threshold=0.50,
        anom_threshold=0.60,
    )
    assert res.verdict == "OK"
    assert not res.is_ng
    assert res.reason == "ok"
    assert not res.det_triggered
    assert not res.anom_triggered
    assert res.max_det_confidence == pytest.approx(0.49)
    assert res.active_detections == []
    assert res.defect_classes == []


def test_ng_by_detection_only():
    d_low = Detection("crazing", 0.20, (0, 0, 10, 10))
    d_high = Detection("scratches", 0.82, (5, 5, 50, 50))
    d_mid = Detection("patches", 0.55, (10, 10, 30, 30))
    res = decide(
        detections=[d_low, d_mid, d_high],
        anomaly_score=0.30,
        det_threshold=0.50,
        anom_threshold=0.70,
    )
    assert res.verdict == "NG"
    assert res.is_ng
    assert res.reason == "detection"
    assert res.det_triggered
    assert not res.anom_triggered
    # Active detections are filtered and sorted by confidence descending.
    assert res.active_detections == [d_high, d_mid]
    assert res.defect_classes == ["scratches", "patches"]


def test_ng_by_anomaly_only_catches_unseen_defect():
    # Detector misses an unseen defect class (no boxes or low confidence),
    # but PatchCore anomaly score exceeds tau_anom.
    res = decide(
        detections=[Detection("crazing", 0.12)],
        anomaly_score=0.78,
        det_threshold=0.50,
        anom_threshold=0.65,
    )
    assert res.verdict == "NG"
    assert res.reason == "anomaly"
    assert not res.det_triggered
    assert res.anom_triggered
    assert res.active_detections == []


def test_ng_by_both_signals():
    res = decide(
        detections=[Detection("inclusion", 0.91)],
        anomaly_score=0.88,
        det_threshold=0.50,
        anom_threshold=0.70,
    )
    assert res.verdict == "NG"
    assert res.reason == "both"
    assert res.det_triggered
    assert res.anom_triggered


def test_boundary_equality_triggers_ng():
    # Rule: confidence >= tau_det OR anomaly_score >= tau_anom.
    res_det = decide(detections=[0.50], anomaly_score=0.10, det_threshold=0.50, anom_threshold=0.60)
    assert res_det.verdict == "NG"
    assert res_det.reason == "detection"

    res_anom = decide(detections=[], anomaly_score=0.60, det_threshold=0.50, anom_threshold=0.60)
    assert res_anom.verdict == "NG"
    assert res_anom.reason == "anomaly"


def test_detector_standalone_without_anomaly_score():
    res_ok = decide(detections=[], anomaly_score=None, det_threshold=0.40)
    assert res_ok.verdict == "OK"
    assert res_ok.anomaly_score is None
    assert res_ok.max_det_confidence == 0.0

    res_ng = decide(detections=[0.45], anomaly_score=None, det_threshold=0.40)
    assert res_ng.verdict == "NG"
    assert res_ng.reason == "detection"


def test_explicit_thresholds_object_overrides_kwargs():
    tau = FusionThresholds(det_threshold=0.80, anom_threshold=0.90)
    res = decide(
        detections=[0.70],
        anomaly_score=0.85,
        det_threshold=0.10,
        anom_threshold=0.10,
        thresholds=tau,
    )
    assert res.verdict == "OK"
    assert res.thresholds == tau


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"det_threshold": -0.1},
        {"det_threshold": 1.1},
        {"det_threshold": float("nan")},
        {"anom_threshold": float("nan")},
        {"anom_threshold": float("inf")},
    ],
)
def test_invalid_thresholds_raise(bad_kwargs):
    with pytest.raises(ValueError):
        FusionThresholds(**bad_kwargs)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        Detection("scratches", 1.2)
    with pytest.raises(ValueError):
        Detection("scratches", float("nan"))
    with pytest.raises(ValueError):
        decide(detections=[], anomaly_score=float("nan"))
