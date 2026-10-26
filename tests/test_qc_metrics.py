import pytest

from steelvision.eval.qc_metrics import (
    ProductSample,
    evaluate_qc,
    select_operating_point,
    sweep_thresholds,
)


def _toy_samples() -> list[ProductSample]:
    return [
        # 4 good products (OK)
        ProductSample("ok_1", gt_is_ng=False, det_confidences=(), anomaly_score=0.20),
        ProductSample(
            "ok_2",
            gt_is_ng=False,
            det_confidences=(0.15,),
            det_classes=("crazing",),
            anomaly_score=0.35,
        ),
        ProductSample(
            "ok_3",
            gt_is_ng=False,
            det_confidences=(0.42,),
            det_classes=("scratches",),
            anomaly_score=0.45,
        ),
        ProductSample("ok_4", gt_is_ng=False, det_confidences=(), anomaly_score=0.52),
        # 5 defective products (NG) — 4 caught by detector, 1 unseen class caught by anomaly
        ProductSample(
            "ng_1",
            gt_is_ng=True,
            det_confidences=(0.91,),
            det_classes=("scratches",),
            anomaly_score=0.80,
        ),
        ProductSample(
            "ng_2",
            gt_is_ng=True,
            det_confidences=(0.75,),
            det_classes=("inclusion",),
            anomaly_score=0.40,
        ),
        ProductSample(
            "ng_3",
            gt_is_ng=True,
            det_confidences=(0.68,),
            det_classes=("patches",),
            anomaly_score=0.62,
        ),
        ProductSample(
            "ng_4",
            gt_is_ng=True,
            det_confidences=(0.55,),
            det_classes=("crazing",),
            anomaly_score=0.58,
        ),
        ProductSample(
            "ng_unseen",
            gt_is_ng=True,
            det_confidences=(0.10,),
            det_classes=("crazing",),
            anomaly_score=0.72,
        ),
    ]


def test_evaluate_qc_counts_and_rates():
    samples = _toy_samples()
    m = evaluate_qc(samples, det_threshold=0.50, anom_threshold=0.55)
    assert m.n_total == 9
    assert m.n_gt_ng == 5
    assert m.n_gt_ok == 4
    assert m.tp == 5
    assert m.fn == 0
    assert m.fp == 0
    assert m.tn == 4
    assert m.escape_rate == pytest.approx(0.0)
    assert m.overkill_rate == pytest.approx(0.0)
    assert m.f1 == pytest.approx(1.0)
    assert m.reasons["anomaly"] == 1  # ng_unseen caught solely by anomaly


def test_unseen_class_simulation_masks_detector_and_anomaly_saves_escape():
    samples = _toy_samples()
    # Without anomaly (anom_threshold=999), masking 'crazing' lets ng_4 & ng_unseen escape:
    m_det_only = evaluate_qc(
        samples,
        det_threshold=0.50,
        anom_threshold=999.0,
        ignore_det_classes={"crazing"},
    )
    assert m_det_only.fn == 2
    assert m_det_only.escape_rate == pytest.approx(2 / 5)

    # With anomaly enabled at 0.55, PatchCore catches both ng_4 (0.58) and ng_unseen (0.72):
    m_fused = evaluate_qc(
        samples,
        det_threshold=0.50,
        anom_threshold=0.55,
        ignore_det_classes={"crazing"},
    )
    assert m_fused.fn == 0
    assert m_fused.escape_rate == pytest.approx(0.0)


def test_sweep_and_select_operating_point():
    samples = _toy_samples()
    curve = sweep_thresholds(
        samples,
        det_thresholds=[0.30, 0.50, 0.80],
        anom_thresholds=[0.40, 0.55, 0.85],
    )
    assert len(curve) == 9
    best = select_operating_point(curve, max_escape_rate=0.02)
    assert best.escape_rate == pytest.approx(0.0)
    assert best.overkill_rate == pytest.approx(0.0)
    assert (best.det_threshold, best.anom_threshold) == (0.50, 0.55)


def test_select_operating_point_fallback_when_target_infeasible():
    samples = _toy_samples()
    # Very high thresholds so escape_rate > 0 everywhere
    curve = sweep_thresholds(
        samples,
        det_thresholds=[0.85, 0.95],
        anom_thresholds=[0.85, 0.95],
    )
    best = select_operating_point(curve, max_escape_rate=0.0)
    assert best.escape_rate == min(m.escape_rate for m in curve)
