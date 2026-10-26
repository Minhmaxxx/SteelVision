import numpy as np
import pytest

from steelvision.eval.robustness import (
    PERTURBATIONS,
    BoxIoU,
    box_iou,
    compute_ap50,
)


def test_all_perturbations_preserve_shape_and_dtype():
    rng = np.random.default_rng(7)
    img = rng.integers(20, 230, size=(64, 80, 3), dtype=np.uint8)
    for name, fn in PERTURBATIONS.items():
        out = fn(img)
        assert out.shape == img.shape, name
        assert out.dtype == np.uint8, name
        if name != "clean":
            assert not np.array_equal(out, img), name


def test_box_iou_and_compute_ap50():
    a = BoxIoU(cls_id=0, x_center=0.5, y_center=0.5, width=0.4, height=0.4, conf=0.9)
    b = BoxIoU(cls_id=0, x_center=0.5, y_center=0.5, width=0.4, height=0.4, conf=1.0)
    c = BoxIoU(cls_id=0, x_center=0.1, y_center=0.1, width=0.1, height=0.1, conf=0.8)
    assert box_iou(a, b) == pytest.approx(1.0)
    assert box_iou(a, c) == pytest.approx(0.0)

    gts = [(0, [b]), (1, [c])]
    preds_perfect = [(0, [a]), (1, [BoxIoU(0, 0.1, 0.1, 0.1, 0.1, conf=0.85)])]
    ap, prec, rec = compute_ap50(preds_perfect, gts)
    assert ap == pytest.approx(1.0)
    assert prec == pytest.approx(1.0)
    assert rec == pytest.approx(1.0)
