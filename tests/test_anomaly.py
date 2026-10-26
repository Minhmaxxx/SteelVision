import numpy as np
import onnx
from onnx import TensorProto, helper

from steelvision.models.anomaly import OnnxAnomalyDetector, PatchCoreConfig, verify_onnx


def _build_tiny_anomaly_onnx(path, height: int = 32, width: int = 32) -> None:
    """Create a tiny valid ONNX model: input (1, 3, H, W) -> (pred_score, anomaly_map)."""
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, height, width])
    score = helper.make_tensor_value_info("pred_score", TensorProto.FLOAT, [1, 1])
    amap = helper.make_tensor_value_info("anomaly_map", TensorProto.FLOAT, [1, 1, height, width])

    axes_c = helper.make_tensor("axes_c", TensorProto.INT64, [1], [1])
    axes_hw = helper.make_tensor("axes_hw", TensorProto.INT64, [2], [2, 3])

    node_map = helper.make_node(
        "ReduceMean", inputs=["input", "axes_c"], outputs=["anomaly_map"], keepdims=1
    )
    node_score = helper.make_node(
        "ReduceMax", inputs=["anomaly_map", "axes_hw"], outputs=["pred_score"], keepdims=0
    )

    graph = helper.make_graph(
        [node_map, node_score],
        "tiny_patchcore",
        [x],
        [score, amap],
        initializer=[axes_c, axes_hw],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=9,
    )
    onnx.save(model, str(path))


def test_patchcore_config_defaults():
    cfg = PatchCoreConfig()
    assert cfg.category == "metal_nut"
    assert cfg.backbone == "wide_resnet50_2"
    assert cfg.layers == ("layer2", "layer3")
    assert cfg.coreset_sampling_ratio == 0.01
    assert cfg.image_size == (256, 256)


def test_onnx_anomaly_detector_preprocess_and_predict(tmp_path):
    onnx_path = tmp_path / "tiny.onnx"
    _build_tiny_anomaly_onnx(onnx_path, height=32, width=32)

    detector = OnnxAnomalyDetector(onnx_path, input_size=(32, 32))
    # Create an RGB uint8 image of a different size to exercise resize + normalize
    rgb = np.full((64, 48, 3), 128, dtype=np.uint8)
    batch = detector.preprocess(rgb)
    assert batch.shape == (1, 3, 32, 32)
    assert batch.dtype == np.float32
    assert np.allclose(batch, 128.0 / 255.0)

    score, amap = detector.predict(rgb)
    assert abs(score - (128.0 / 255.0)) < 1e-4
    assert amap is not None
    assert amap.shape == (32, 32)

    info = verify_onnx(onnx_path, input_size=(32, 32))
    assert info["verified"] is True
    assert info["anomaly_map_shape"] == [32, 32]
