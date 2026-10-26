"""CPU latency (p50/p95/mean), FPS, and RAM benchmark before and after ONNX optimization.

    python -m steelvision.eval.benchmark --config configs/benchmark.yaml

Measures batch=1 single-product inference latency with explicit warm-up across:
1. YOLOv8n detector (PyTorch CPU vs ONNX Runtime CPU default vs tuned threads)
2. PatchCore anomaly detector (PyTorch CPU vs ONNX Runtime CPU default vs tuned threads)
3. End-to-end fused inspection pipeline (PyTorch E2E vs ONNX E2E Full vs ONNX E2E Early-Exit)
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from steelvision.decision.fusion import decide
from steelvision.models.anomaly import OnnxAnomalyDetector
from steelvision.models.detector import OnnxYoloDetector, export_yolo_onnx


def get_process_rss_mb() -> float:
    """Return current process Resident Set Size (Working Set) in MiB using stdlib only."""
    if sys.platform.startswith("win"):

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.wintypes.DWORD),
                ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            ctypes.wintypes.DWORD,
        ]
        k32.K32GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL
        handle = k32.GetCurrentProcess()
        if k32.K32GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
            return round(pmc.WorkingSetSize / (1024.0 * 1024.0), 2)
    else:
        status_path = Path("/proc/self/status")
        if status_path.exists():
            for line in status_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    kb = float(line.split()[1])
                    return round(kb / 1024.0, 2)
    return 0.0


def file_size_mb(path: Path) -> float:
    """Total size in MiB of ``path`` plus any external ``.data`` file alongside it."""
    if not path.exists():
        return 0.0
    total = sum(p.stat().st_size for p in path.parent.glob(f"{path.name}*") if p.is_file())
    return round(total / (1024.0 * 1024.0), 2)


@dataclass(frozen=True)
class LatencyStats:
    """Summary statistics for a batch=1 latency benchmark."""

    name: str
    n_warmup: int
    n_runs: int
    mean_ms: float
    std_ms: float
    min_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float
    fps: float
    rss_ram_mb: float
    model_size_mb: float

    def to_dict(self) -> dict:
        return asdict(self)


def benchmark_callable(
    name: str,
    fn: Callable[[int], object],
    *,
    n_warmup: int = 10,
    n_runs: int = 50,
    model_size_mb: float = 0.0,
) -> LatencyStats:
    """Run ``fn(i)`` for ``n_warmup`` untimed iterations followed by ``n_runs`` timed iterations."""
    for i in range(n_warmup):
        fn(i)

    latencies_ms: list[float] = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        fn(i)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(latencies_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    p50_ms = float(np.percentile(arr, 50))
    p95_ms = float(np.percentile(arr, 95))
    rss_mb = get_process_rss_mb()

    return LatencyStats(
        name=name,
        n_warmup=n_warmup,
        n_runs=n_runs,
        mean_ms=round(mean_ms, 2),
        std_ms=round(float(np.std(arr)), 2),
        min_ms=round(float(np.min(arr)), 2),
        p50_ms=round(p50_ms, 2),
        p95_ms=round(p95_ms, 2),
        max_ms=round(float(np.max(arr)), 2),
        fps=round(1000.0 / p50_ms, 2) if p50_ms > 0 else 0.0,
        rss_ram_mb=rss_mb,
        model_size_mb=model_size_mb,
    )


def run_benchmark_suite(cfg: dict) -> dict:
    """Execute the full Phase 6 CPU benchmark suite."""
    n_warmup = int(cfg.get("n_warmup", 10))
    n_runs = int(cfg.get("n_runs", 50))
    threads = int(cfg.get("intra_op_threads", 4))

    yolo_cfg = cfg["yolo"]
    pc_cfg = cfg["patchcore"]

    pt_yolo_path = Path(yolo_cfg["pt_weights"])
    onnx_yolo_path = Path(yolo_cfg["onnx_weights"])
    imgsz = int(yolo_cfg.get("imgsz", 640))
    det_tau = float(yolo_cfg.get("det_threshold", 0.25))

    if not onnx_yolo_path.exists():
        print(f"[benchmark] Exporting YOLOv8n {pt_yolo_path} -> {onnx_yolo_path} ...")
        export_yolo_onnx(pt_yolo_path, onnx_yolo_path, imgsz=imgsz)

    # Load test images into memory so disk I/O does not pollute inference timing
    neu_imgs = [
        cv2.imread(str(p))
        for p in sorted(Path(yolo_cfg["test_images_dir"]).glob("*.*"))[: max(n_runs, 20)]
    ]
    mvtec_bgrs = [cv2.imread(str(p)) for p in sorted(Path(pc_cfg["test_images_dir"]).glob("*.png"))]
    mvtec_rgbs = [cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for b in mvtec_bgrs]

    pc_ckpt = Path(pc_cfg["ckpt_path"])
    pc_onnx = Path(pc_cfg["onnx_weights"])
    pc_input_size = tuple(pc_cfg["input_size"])
    anom_tau = float(pc_cfg.get("anom_threshold", 0.53))

    pt_yolo_size = file_size_mb(pt_yolo_path)
    onnx_yolo_size = file_size_mb(onnx_yolo_path)
    onnx_pc_size = file_size_mb(pc_onnx)

    # 1. Pure ONNX Runtime Benchmarks FIRST (before importing torch/ultralytics/anomalib)
    print("[benchmark] 1/4 Timing pure ONNX Runtime CPU (YOLOv8n, PatchCore, E2E) ...")
    yolo_ort_default = OnnxYoloDetector(onnx_yolo_path, imgsz=imgsz, intra_op_threads=0)
    yolo_ort_tuned = OnnxYoloDetector(onnx_yolo_path, imgsz=imgsz, intra_op_threads=threads)

    stats_yolo_ort_def = benchmark_callable(
        "YOLOv8n (ONNX Runtime CPU - default threads)",
        lambda i: yolo_ort_default.predict(neu_imgs[i % len(neu_imgs)], conf_threshold=det_tau),
        n_warmup=n_warmup,
        n_runs=n_runs,
        model_size_mb=onnx_yolo_size,
    )
    stats_yolo_ort_tuned = benchmark_callable(
        f"YOLOv8n (ONNX Runtime CPU - {threads} threads)",
        lambda i: yolo_ort_tuned.predict(neu_imgs[i % len(neu_imgs)], conf_threshold=det_tau),
        n_warmup=n_warmup,
        n_runs=n_runs,
        model_size_mb=onnx_yolo_size,
    )

    pc_ort_default = OnnxAnomalyDetector(pc_onnx, input_size=pc_input_size, intra_op_threads=0)
    pc_ort_tuned = OnnxAnomalyDetector(pc_onnx, input_size=pc_input_size, intra_op_threads=threads)

    stats_pc_ort_def = benchmark_callable(
        "PatchCore (ONNX Runtime CPU - default threads)",
        lambda i: pc_ort_default.predict(mvtec_rgbs[i % len(mvtec_rgbs)]),
        n_warmup=n_warmup,
        n_runs=n_runs,
        model_size_mb=onnx_pc_size,
    )
    stats_pc_ort_tuned = benchmark_callable(
        f"PatchCore (ONNX Runtime CPU - {threads} threads)",
        lambda i: pc_ort_tuned.predict(mvtec_rgbs[i % len(mvtec_rgbs)]),
        n_warmup=n_warmup,
        n_runs=n_runs,
        model_size_mb=onnx_pc_size,
    )

    best_yolo_ort = (
        yolo_ort_default
        if stats_yolo_ort_def.p50_ms <= stats_yolo_ort_tuned.p50_ms
        else yolo_ort_tuned
    )
    best_pc_ort = (
        pc_ort_default if stats_pc_ort_def.p50_ms <= stats_pc_ort_tuned.p50_ms else pc_ort_tuned
    )

    def _run_e2e_ort_full(i: int):
        bgr = neu_imgs[i % len(neu_imgs)]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dets = best_yolo_ort.predict(bgr, conf_threshold=det_tau)
        anom_score, _ = best_pc_ort.predict(rgb)
        return decide(dets, anom_score, det_threshold=det_tau, anom_threshold=anom_tau)

    def _run_e2e_ort_early_exit(i: int):
        bgr = neu_imgs[i % len(neu_imgs)]
        dets = best_yolo_ort.predict(bgr, conf_threshold=det_tau)
        if any(d.confidence >= det_tau for d in dets):
            return decide(dets, None, det_threshold=det_tau, anom_threshold=anom_tau)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        anom_score, _ = best_pc_ort.predict(rgb)
        return decide(dets, anom_score, det_threshold=det_tau, anom_threshold=anom_tau)

    stats_e2e_ort_full = benchmark_callable(
        "E2E Pipeline (ONNX Runtime CPU - Full YOLO + PatchCore)",
        _run_e2e_ort_full,
        n_warmup=n_warmup,
        n_runs=n_runs,
        model_size_mb=round(onnx_yolo_size + onnx_pc_size, 2),
    )
    stats_e2e_ort_early = benchmark_callable(
        "E2E Pipeline (ONNX Runtime CPU - Early-Exit on Known Defect)",
        _run_e2e_ort_early_exit,
        n_warmup=n_warmup,
        n_runs=n_runs,
        model_size_mb=round(onnx_yolo_size + onnx_pc_size, 2),
    )

    # 2. PyTorch CPU Benchmarks (YOLOv8n, PatchCore, E2E)
    print("[benchmark] 2/4 Timing PyTorch CPU (YOLOv8n, PatchCore, E2E) ...")
    import torch
    from anomalib.models import Patchcore
    from ultralytics import YOLO

    yolo_pt = YOLO(str(pt_yolo_path))
    stats_yolo_pt = benchmark_callable(
        "YOLOv8n (PyTorch CPU)",
        lambda i: yolo_pt.predict(
            neu_imgs[i % len(neu_imgs)], imgsz=imgsz, conf=det_tau, verbose=False
        ),
        n_warmup=n_warmup,
        n_runs=n_runs,
        model_size_mb=pt_yolo_size,
    )

    parity_matches = 0
    max_conf_diffs: list[float] = []
    for img in neu_imgs[:20]:
        pt_res = yolo_pt.predict(img, imgsz=imgsz, conf=det_tau, verbose=False)[0]
        pt_confs = (
            pt_res.boxes.conf.cpu().numpy()
            if pt_res.boxes is not None and len(pt_res.boxes)
            else np.array([0.0])
        )
        pt_max = float(np.max(pt_confs))

        ort_dets = best_yolo_ort.predict(img, conf_threshold=det_tau)
        ort_max = max((d.confidence for d in ort_dets), default=0.0)
        max_conf_diffs.append(abs(pt_max - ort_max))
        if (pt_max >= det_tau) == (ort_max >= det_tau):
            parity_matches += 1

    stats_pc_pt = None
    stats_e2e_pt = None
    if pc_ckpt.exists():
        pre_proc = Patchcore.configure_pre_processor(image_size=pc_input_size)
        pc_pt_model = Patchcore.load_from_checkpoint(
            pc_ckpt,
            pre_processor=pre_proc,
            visualizer=False,
        ).eval()

        def _run_pc_pt(i: int) -> float:
            rgb = mvtec_rgbs[i % len(mvtec_rgbs)]
            batch_np = best_pc_ort.preprocess(rgb)
            with torch.no_grad():
                out = pc_pt_model(torch.from_numpy(batch_np))
            return float(out.pred_score.reshape(-1)[0])

        stats_pc_pt = benchmark_callable(
            "PatchCore (PyTorch CPU)",
            _run_pc_pt,
            n_warmup=n_warmup,
            n_runs=n_runs,
            model_size_mb=file_size_mb(pc_ckpt),
        )

        def _run_e2e_pt(i: int):
            bgr = neu_imgs[i % len(neu_imgs)]
            pt_res = yolo_pt.predict(bgr, imgsz=imgsz, conf=det_tau, verbose=False)[0]
            confs = (
                pt_res.boxes.conf.cpu().numpy().tolist()
                if pt_res.boxes is not None and len(pt_res.boxes)
                else []
            )
            anom_score = _run_pc_pt(i)
            return decide(confs, anom_score, det_threshold=det_tau, anom_threshold=anom_tau)

        stats_e2e_pt = benchmark_callable(
            "E2E Pipeline (PyTorch CPU - YOLO + PatchCore)",
            _run_e2e_pt,
            n_warmup=n_warmup,
            n_runs=n_runs,
            model_size_mb=round(pt_yolo_size + file_size_mb(pc_ckpt), 2),
        )

    yolo_best = min([stats_yolo_ort_def, stats_yolo_ort_tuned], key=lambda s: s.p50_ms)
    pc_best = min([stats_pc_ort_def, stats_pc_ort_tuned], key=lambda s: s.p50_ms)

    report = {
        "hardware": {
            "platform": sys.platform,
            "cpu_count_logical": os.cpu_count(),
            "n_warmup": n_warmup,
            "n_runs": n_runs,
            "tuned_intra_op_threads": threads,
        },
        "yolov8n_640": {
            "pytorch_cpu": stats_yolo_pt.to_dict(),
            "onnx_cpu_default": stats_yolo_ort_def.to_dict(),
            "onnx_cpu_tuned": stats_yolo_ort_tuned.to_dict(),
            "speedup_p50_x": round(stats_yolo_pt.p50_ms / yolo_best.p50_ms, 2),
            "parity_20_test_imgs": {
                "verdict_agreement_rate": round(parity_matches / 20.0, 4),
                "mean_abs_max_conf_diff": round(float(np.mean(max_conf_diffs)), 6),
                "max_abs_max_conf_diff": round(float(np.max(max_conf_diffs)), 6),
            },
        },
        "patchcore_256": {
            "pytorch_cpu": stats_pc_pt.to_dict() if stats_pc_pt else None,
            "onnx_cpu_default": stats_pc_ort_def.to_dict(),
            "onnx_cpu_tuned": stats_pc_ort_tuned.to_dict(),
            "speedup_p50_x": round(stats_pc_pt.p50_ms / pc_best.p50_ms, 2) if stats_pc_pt else None,
        },
        "end_to_end_pipeline": {
            "pytorch_e2e": stats_e2e_pt.to_dict() if stats_e2e_pt else None,
            "onnx_e2e_full": stats_e2e_ort_full.to_dict(),
            "onnx_e2e_early_exit": stats_e2e_ort_early.to_dict(),
            "full_speedup_p50_x": (
                round(stats_e2e_pt.p50_ms / stats_e2e_ort_full.p50_ms, 2) if stats_e2e_pt else None
            ),
            "early_exit_speedup_p50_x": (
                round(stats_e2e_pt.p50_ms / stats_e2e_ort_early.p50_ms, 2) if stats_e2e_pt else None
            ),
        },
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark.yaml")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    report = run_benchmark_suite(cfg)

    out_path = Path(cfg["output"]["json_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[benchmark] 4/4 Saved benchmark report -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
