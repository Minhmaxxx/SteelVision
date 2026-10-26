"""PatchCore anomaly detection pipeline (MVTec AD validation + ONNX export/inference).

Usage:
    python -m steelvision.models.anomaly --category metal_nut --skip-download

Trains PatchCore (memory-bank of multi-scale CNN patch features, fitted on
defect-free ``train/good`` images only), evaluates image-level and pixel-level
AUROC on the test split, exports an ONNX model, and verifies ONNX Runtime
inference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from steelvision.data import download

MVTEC_KAGGLE_HANDLE = "ipythonx/mvtec-ad"


@dataclass(frozen=True)
class PatchCoreConfig:
    """Configuration for a PatchCore training + evaluation run."""

    category: str = "metal_nut"
    backbone: str = "wide_resnet50_2"
    layers: tuple[str, ...] = ("layer2", "layer3")
    coreset_sampling_ratio: float = 0.01
    num_neighbors: int = 9
    image_size: tuple[int, int] = (256, 256)
    train_batch_size: int = 32
    eval_batch_size: int = 32
    num_workers: int = 0
    seed: int = 42


class OnnxAnomalyDetector:
    """Lightweight CPU inference wrapper around an exported PatchCore ONNX model."""

    def __init__(
        self,
        onnx_path: str | Path,
        input_size: tuple[int, int] = (256, 256),
        intra_op_threads: int = 0,
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        self.input_size = input_size
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
        self.output_names = [o.name for o in self.session.get_outputs()]

    def preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        """Convert an ``(H, W, 3)`` uint8 RGB image to a ``(1, 3, H, W)`` float32 tensor."""
        import cv2

        h, w = self.input_size
        if image_rgb.shape[:2] != (h, w):
            image_rgb = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_AREA)
        chw = np.transpose(image_rgb.astype(np.float32) / 255.0, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def predict(self, image_rgb: np.ndarray) -> tuple[float, np.ndarray | None]:
        """Return ``(anomaly_score, anomaly_map)`` for an RGB uint8 image."""
        batch = self.preprocess(image_rgb)
        return self.predict_batch(batch)

    def predict_batch(self, batch_nchw: np.ndarray) -> tuple[float, np.ndarray | None]:
        """Run ONNX Runtime on a ``(1, 3, H, W)`` float32 tensor in ``[0, 1]``."""
        outputs = self.session.run(None, {self.input_name: batch_nchw.astype(np.float32)})
        # Anomalib's InferenceBatch returns (pred_score, pred_label, anomaly_map, pred_mask)
        # or (pred_score, anomaly_map) depending on post_processor configuration.
        score: float | None = None
        amap: np.ndarray | None = None
        for out in outputs:
            arr = np.asarray(out)
            if arr.dtype == bool:
                continue
            if arr.ndim <= 2 and arr.size == batch_nchw.shape[0] and score is None:
                score = float(arr.reshape(-1)[0])
            elif arr.ndim >= 3 and amap is None:
                amap = arr.squeeze()
        if score is None:
            score = float(np.max(outputs[0]))
        return score, amap


def verify_onnx(onnx_path: Path, input_size: tuple[int, int]) -> dict:
    """Verify the exported ONNX graph runs cleanly in ONNX Runtime on a dummy input."""
    detector = OnnxAnomalyDetector(onnx_path, input_size=input_size)
    rng = np.random.default_rng(0)
    dummy = rng.random((1, 3, input_size[0], input_size[1]), dtype=np.float32)
    # Warm-up + timed single pass
    detector.predict_batch(dummy)
    t0 = time.perf_counter()
    score, amap = detector.predict_batch(dummy)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "verified": True,
        "output_names": detector.output_names,
        "dummy_score": round(score, 6),
        "anomaly_map_shape": list(amap.shape) if amap is not None else None,
        "single_pass_ms": round(latency_ms, 2),
    }


def train_and_evaluate(
    cfg: PatchCoreConfig,
    data_root: Path = Path("data"),
    out_dir: Path = Path("runs/patchcore"),
    *,
    export_onnx: bool = True,
    skip_download: bool = False,
    ckpt_path: Path | None = None,
) -> dict:
    """Fit PatchCore on MVTec AD ``cfg.category``, evaluate on test, and export ONNX."""
    import lightning.pytorch as pl
    from anomalib.data import MVTecAD
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    pl.seed_everything(cfg.seed, workers=True)

    mvtec_root = data_root / "raw" / "mvtec_ad"
    category_dir = mvtec_root / cfg.category
    if not category_dir.exists():
        if skip_download:
            raise FileNotFoundError(
                f"{category_dir} not found; rerun without --skip-download to fetch MVTec AD."
            )
        print(f"[mvtec_ad] downloading kagglehub://{MVTEC_KAGGLE_HANDLE} -> {mvtec_root} ...")
        download.copy_from_kagglehub(MVTEC_KAGGLE_HANDLE, mvtec_root)

    run_dir = out_dir / cfg.category
    run_dir.mkdir(parents=True, exist_ok=True)

    datamodule = MVTecAD(
        root=mvtec_root,
        category=cfg.category,
        train_batch_size=cfg.train_batch_size,
        eval_batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    datamodule.setup()
    n_train = len(datamodule.train_data)
    n_test = len(datamodule.test_data)
    print(f"[patchcore:{cfg.category}] train={n_train} (good only) | test={n_test}")

    pre_processor = Patchcore.configure_pre_processor(image_size=cfg.image_size)
    model = Patchcore(
        backbone=cfg.backbone,
        layers=list(cfg.layers),
        pre_trained=True,
        coreset_sampling_ratio=cfg.coreset_sampling_ratio,
        num_neighbors=cfg.num_neighbors,
        pre_processor=pre_processor,
        visualizer=False,
    )

    engine = Engine(
        default_root_dir=run_dir,
        logger=False,
        enable_progress_bar=True,
    )

    t_fit0 = time.perf_counter()
    if ckpt_path is not None:
        print(f"[patchcore:{cfg.category}] loading fitted checkpoint -> {ckpt_path}")
        model = Patchcore.load_from_checkpoint(
            ckpt_path,
            pre_processor=pre_processor,
            visualizer=False,
        )
        fit_sec = 0.0
    else:
        engine.fit(model=model, datamodule=datamodule)
        fit_sec = time.perf_counter() - t_fit0

    t_test0 = time.perf_counter()
    test_results = engine.test(model=model, datamodule=datamodule)
    test_sec = time.perf_counter() - t_test0

    raw_metrics = test_results[0] if test_results else {}
    metrics = {k: round(float(v), 6) for k, v in raw_metrics.items()}
    mb_shape = list(model.model.memory_bank.shape)
    print(
        f"[patchcore:{cfg.category}] image_AUROC={metrics.get('image_AUROC')} "
        f"pixel_AUROC={metrics.get('pixel_AUROC')} "
        f"memory_bank={mb_shape} fit={fit_sec:.1f}s test={test_sec:.1f}s"
    )

    onnx_info: dict = {"exported": False}
    if export_onnx:
        onnx_path = engine.export(
            model=model,
            export_type=ExportType.ONNX,
            export_root=run_dir,
            model_file_name="patchcore",
            input_size=cfg.image_size,
            onnx_kwargs={
                "opset_version": 18,
                "dynamic_axes": {},
                "dynamic_shapes": None,
            },
        )
        if onnx_path is not None and Path(onnx_path).exists():
            onnx_path = Path(onnx_path)
            total_bytes = sum(
                p.stat().st_size for p in onnx_path.parent.glob(f"{onnx_path.name}*") if p.is_file()
            )
            onnx_info = {
                "exported": True,
                "path": onnx_path.as_posix(),
                "size_mb": round(total_bytes / (1024 * 1024), 2),
                **verify_onnx(onnx_path, cfg.image_size),
            }
            print(
                f"[patchcore:{cfg.category}] ONNX exported -> "
                f"{onnx_path} ({onnx_info['size_mb']} MB)"
            )

    summary = {
        "config": {
            **asdict(cfg),
            "layers": list(cfg.layers),
            "image_size": list(cfg.image_size),
        },
        "dataset": {
            "root": mvtec_root.as_posix(),
            "category": cfg.category,
            "n_train_good": n_train,
            "n_test": n_test,
        },
        "memory_bank_shape": mb_shape,
        "metrics": metrics,
        "timing_sec": {
            "fit": round(fit_sec, 2),
            "test": round(test_sec, 2),
        },
        "onnx": onnx_info,
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[patchcore:{cfg.category}] summary written -> {metrics_path}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument(
        "--category", default=None, help="MVTec AD category (default from configs/datasets.yaml)"
    )
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--coreset-ratio", type=float, default=0.01)
    parser.add_argument("--num-neighbors", type=int, default=9)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default="runs/patchcore")
    parser.add_argument("--no-onnx", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--ckpt-path", default=None, help="Load existing .ckpt instead of fitting")
    args = parser.parse_args(argv)

    category = args.category
    if category is None and Path(args.config).exists():
        ds_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        category = ds_cfg.get("datasets", {}).get("mvtec_ad", {}).get("category", "metal_nut")
    category = category or "metal_nut"

    cfg = PatchCoreConfig(
        category=category,
        backbone=args.backbone,
        coreset_sampling_ratio=args.coreset_ratio,
        num_neighbors=args.num_neighbors,
        image_size=(args.imgsz, args.imgsz),
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
    )
    train_and_evaluate(
        cfg,
        data_root=Path(args.data_root),
        out_dir=Path(args.out_dir),
        export_onnx=not args.no_onnx,
        skip_download=args.skip_download,
        ckpt_path=Path(args.ckpt_path) if args.ckpt_path else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
