import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from steelvision.eval.benchmark import benchmark_callable, get_process_rss_mb
from steelvision.models.detector import OnnxYoloDetector, _letterbox, _nms_xyxy


def _build_tiny_yolo_onnx(path, imgsz: int = 64, num_classes: int = 6) -> None:
    """Create a tiny valid YOLOv8-shaped ONNX model: (1, 3, H, W) -> (1, 4+C, 10)."""
    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, imgsz, imgsz])
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4 + num_classes, 10])

    # Constant output with 1 high-confidence box at anchor 0 for class 2 ('patches')
    arr = np.zeros((1, 4 + num_classes, 10), dtype=np.float32)
    arr[0, :4, 0] = [32.0, 32.0, 20.0, 20.0]  # xc, yc, w, h in 64x64 space
    arr[0, 4 + 2, 0] = 0.88  # class 2 confidence

    const_tensor = helper.make_tensor(
        "const_out", TensorProto.FLOAT, arr.shape, arr.flatten().tolist()
    )
    node_const = helper.make_node("Constant", inputs=[], outputs=["const_val"], value=const_tensor)
    # Add 0 * reduce_mean(x) so input 'images' is part of the computation graph
    zero_tensor = helper.make_tensor("zero", TensorProto.FLOAT, [1, 1, 1], [0.0])
    node_zero = helper.make_node("Constant", inputs=[], outputs=["zero_val"], value=zero_tensor)
    axes = helper.make_tensor("axes", TensorProto.INT64, [3], [1, 2, 3])
    node_axes = helper.make_node("Constant", inputs=[], outputs=["axes_val"], value=axes)
    node_mean = helper.make_node(
        "ReduceMean", inputs=["images", "axes_val"], outputs=["x_mean"], keepdims=1
    )
    node_mul = helper.make_node("Mul", inputs=["x_mean", "zero_val"], outputs=["x_zero"])
    node_reshape = helper.make_node(
        "Reshape",
        inputs=["x_zero", "shape_val"],
        outputs=["x_zero_3d"],
    )
    shape_t = helper.make_tensor("shape_t", TensorProto.INT64, [3], [1, 1, 1])
    node_shape = helper.make_node("Constant", inputs=[], outputs=["shape_val"], value=shape_t)
    node_add = helper.make_node("Add", inputs=["const_val", "x_zero_3d"], outputs=["output0"])

    graph = helper.make_graph(
        [node_const, node_zero, node_axes, node_shape, node_mean, node_mul, node_reshape, node_add],
        "tiny_yolo",
        [x],
        [out],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)
    onnx.save(model, str(path))


def test_letterbox_and_nms():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    padded, ratio, (dw, dh) = _letterbox(img, (640, 640))
    assert padded.shape == (640, 640, 3)
    assert ratio == pytest.approx(3.2)
    assert dw == pytest.approx(0.0)
    assert dh == pytest.approx(160.0)

    boxes = np.array(
        [
            [10.0, 10.0, 50.0, 50.0],
            [12.0, 12.0, 52.0, 52.0],  # high IoU overlap with box 0
            [100.0, 100.0, 140.0, 140.0],  # disjoint
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = _nms_xyxy(boxes, scores, iou_threshold=0.45)
    assert keep == [0, 2]


def test_onnx_yolo_detector_decode(tmp_path):
    onnx_path = tmp_path / "tiny_yolo.onnx"
    _build_tiny_yolo_onnx(onnx_path, imgsz=64)

    det = OnnxYoloDetector(onnx_path, imgsz=64, intra_op_threads=1)
    bgr = np.zeros((64, 64, 3), dtype=np.uint8)
    boxes = det.predict(bgr, conf_threshold=0.25)
    assert len(boxes) == 1
    assert boxes[0].class_name == "patches"
    assert boxes[0].confidence == pytest.approx(0.88, abs=1e-4)
    assert boxes[0].bbox == (22.0, 22.0, 42.0, 42.0)


def test_benchmark_callable_and_rss():
    rss = get_process_rss_mb()
    assert rss >= 0.0
    stats = benchmark_callable(
        "noop",
        lambda i: i * 2,
        n_warmup=2,
        n_runs=5,
        model_size_mb=1.5,
    )
    assert stats.n_warmup == 2
    assert stats.n_runs == 5
    assert stats.min_ms <= stats.p50_ms <= stats.p95_ms <= stats.max_ms
    assert stats.model_size_mb == 1.5
