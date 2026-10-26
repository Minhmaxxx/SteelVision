# SteelVision

End-to-end quality-control (QC) system for metal surface defect inspection.
Given an image of a metal product, the system decides **OK / NG**, reports the
**defect type and location**, highlights **unseen anomalies**, and records a
complete **SQLite inspection history** for factory traceability.

It combines two complementary models behind a pre-inference **Image Quality Gate**:

- **Object detection** (`YOLOv8n` ONNX) — localizes and classifies 6 known steel defect types.
- **Anomaly detection** (`PatchCore` ONNX, trained on defect-free images only) — a safety net that catches *unseen* defect types and explicitly defines what "OK" looks like.

The final OK/NG product rule merges both signals ($\text{NG} \iff \text{det} \ge \tau_{\text{det}}^* \lor \text{anom} \ge \tau_{\text{anom}}^*$), with operating thresholds selected on a validation set against QC objectives (**escape rate $\le 2\%$**), not by guesswork.

---

## Highlights

- **QC-language evaluation:** **escape rate** (defective products that pass) and **overkill rate** (good products rejected) with threshold trade-off curves and leave-one-class-out unseen defect simulation.
- **Pre-inference Image Quality Gate:** flags `too_dark`, `too_bright`, `low_contrast`, and `blurry` frames before inference to prevent false-alarm overkill under lighting drops.
- **CPU-friendly ONNX deployment:** `4.38x` end-to-end latency speedup (`279.60 ms -> 63.78 ms` p50, `15.68 FPS`) and `-51.7%` RSS RAM (`1,297 MB -> 626 MB`) on CPU (`batch=1`).
- **Production-ready service:** FastAPI (`/inspect`, `/inspections`, `/stats`), SQLite traceability + automatic NG image archiving, interactive Streamlit QC dashboard, and Docker Compose.

---

## Project Status

| Stage | Status | Key Result |
|---|---|---|
| Repo + CI + deterministic data pipeline (`pHash` dedup + stratified split) | ✅ Done | `1,799` NEU-DET, `2,286` GC10-DET, `335` MVTec `metal_nut` |
| YOLOv8 baseline + 3 ablations (Kaggle T4 GPU) | ✅ Done | NEU-DET test `mAP@50 = 0.7836` (`mosaic=0.0`), GC10 `0.6793` |
| PatchCore anomaly detection (`metal_nut` + ONNX + OR-fusion) | ✅ Done | Image AUROC **`99.80%`**, Pixel AUROC **`98.69%`** |
| QC metrics (`escape`/`overkill`) + unseen defect + robustness | ✅ Done | NEU escape **`0.75%`** (vs `9.40%` at `0.50`), MVTec escape **`0.00%`** |
| Self-captured dataset protocol, labeling guide & QC/manifest tooling | ✅ Done | Protocol + 10% re-annotation IoU/F1 checker ready |
| ONNX Runtime CPU optimization & benchmark (p50/p95, RAM, FPS) | ✅ Done | E2E `p50`: **`279.60 ms -> 63.78 ms` (`4.38x`)**, RAM **`-51.7%`** |
| FastAPI + SQLite traceability + Streamlit dashboard + Docker + Demo | ✅ Done | `39/39` unit tests pass, `runs/demo_steelvision.mp4` generated |

---

## Architecture

```
        Image / camera / upload
                 │
                 ▼
  Image Quality Gate (brightness, contrast, Laplacian focus)
                 │
         ┌───────┴────────┐
         ▼                ▼
   YOLOv8n (ONNX)   PatchCore (ONNX)
   (type + bbox)    (anomaly score + heatmap)
         └───────┬────────┘
                 ▼
  OR-Gate Fusion (det >= 0.25 OR anom >= 0.53) → OK / NG + reason
                 │
         ┌───────┴────────────────┐
         ▼                        ▼
  FastAPI /inspect         SQLite Traceability DB + NG Image Archive
         └───────┬────────────────┘
                 ▼
       Streamlit QC Dashboard
```

---

## Experimental Results Summary

### 1. Object Detection & Ablation Study (NEU-DET & GC10-DET Test Splits)

Full details in [`docs/reports/experiments.vi.md`](docs/reports/experiments.vi.md):

| Experiment | Dataset | Model | ImgSz | Mosaic | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | NEU-DET | YOLOv8n | 640 | `1.0` | 0.6778 | 0.7280 | 0.7641 | 0.4272 |
| **Ablation 2 (Selected)** | **NEU-DET** | **YOLOv8n** | **640** | **`0.0`** | **0.6967** | **0.7429** | **0.7836** | **0.4404** |
| Ablation 1 (Scale) | NEU-DET | YOLOv8s | 640 | `1.0` | 0.7082 | 0.7007 | 0.7662 | 0.4314 |
| Ablation 3 (Resolution) | NEU-DET | YOLOv8n | 800 | `0.0` | 0.7001 | 0.7002 | 0.7510 | 0.4193 |
| Baseline | GC10-DET | YOLOv8n | 640 | `1.0` | 0.6808 | 0.6352 | **0.6793** | **0.3512** |

### 2. Product-Level QC Evaluation (`escape_rate <= 2%` Target)

Full details in [`docs/reports/qc_robustness.vi.md`](docs/reports/qc_robustness.vi.md):

| Pipeline / Dataset | Operating Point (from `val`) | Test Escape Rate | Test Overkill Rate | Test Precision | Test Recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|
| **PatchCore (`MVTec metal_nut`)** | $\tau_{\text{anom}}^* = 0.53$ | **0.00%** (`0/47`) | **0.00%** (`0/11` test, `4.55%` full) | **100.00%** | **100.00%** | **1.0000** |
| **YOLOv8n (`NEU-DET` + OK bg)** | $\tau_{\text{det}}^* = 0.25$ | **0.75%** (`2/266`) | **22.50%** (`18/80`) | **93.62%** | **99.25%** | **0.9635** |
| *YOLOv8n naive threshold* | $\tau_{\text{det}} = 0.50$ | *9.40%* (`25/266`) | *2.50%* (`2/80`) | *99.18%* | *90.60%* | *0.9469* |
| **Dual-Branch Fusion (Unseen defect test)** | $\tau_{\text{det}}^* = 0.25, \tau_{\text{anom}}^* = 0.53$ | **0.00%** (vs `23.7%–26.9%` det-only) | **4.55%** | **98.94%** | **100.00%** | **0.9947** |

### 3. CPU Latency & Memory Benchmark (`AMD Ryzen 7 5800H`, `batch=1`)

Full details in [`docs/reports/performance.vi.md`](docs/reports/performance.vi.md):

| Stage / Mode | Backend | Latency p50 (ms) | Latency p95 (ms) | FPS | RSS RAM (MB) | Speedup |
|---|---|---:|---:|---:|---:|---:|
| YOLOv8n (`640x640`) | PyTorch CPU | 66.65 | 72.39 | 15.00 | 1,166.22 | `1.00x` |
| YOLOv8n (`640x640`) | ONNX Runtime CPU | **58.37** | **67.60** | **17.13** | **241.38** | **`1.14x`** |
| PatchCore (`256x256`) | PyTorch CPU | 210.89 | 237.22 | 4.74 | 1,295.18 | `1.00x` |
| PatchCore (`256x256`) | ONNX Runtime CPU | **114.47** | **123.53** | **8.74** | **570.81** | **`1.84x`** |
| E2E Full Dual-Branch | PyTorch CPU | 279.60 | 361.33 | 3.58 | 1,297.18 | `1.00x` |
| E2E Full Dual-Branch | ONNX Runtime CPU | 236.23 | 282.66 | 4.23 | 626.36 | `1.18x` |
| **E2E Early-Exit** | **ONNX Runtime CPU** | **63.78** | **100.22** | **15.68** | **626.37** | **`4.38x`** |

---

## Quickstart

### Local Installation & Commands

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -e ".[dev]"

make data        # download + convert + dedup + split public datasets
make patchcore   # train & export PatchCore ONNX on MVTec metal_nut
make eval        # run QC evaluation (escape/overkill) & robustness suite
make benchmark   # run CPU latency (p50/p95), FPS, and RAM benchmark
make serve       # start FastAPI inspection service on http://localhost:8000
make dashboard   # start Streamlit QC dashboard on http://localhost:8501
make demo        # generate 60s MP4 demo video at runs/demo_steelvision.mp4
make test        # run pytest unit test suite (39 tests)
make lint        # run ruff linter
```

### Docker Compose Deployment

```bash
docker compose up --build
```

- **FastAPI Swagger UI:** `http://localhost:8000/docs` (`POST /inspect`, `GET /inspections`, `GET /stats`, `GET /health`)
- **Streamlit QC Dashboard:** `http://localhost:8501`

---

## Limitations

1. **Hairline Crack vs. Rolled-Steel Grain Confusion (`crazing`):** At `200x200` native resolution on NEU-DET, fine `crazing` micro-cracks reach `mAP@50 = 0.491` due to low local contrast against hot-rolled surface texture. Sub-millimeter cracks require higher optical resolution or tiled inference (SAHI).
2. **Sensitivity to Severe Under-Exposure (`brightness_down`):** While PatchCore preserves `image_AUROC >= 0.9785` across all 6 synthetic perturbations, a `40%` drop in illumination shifts feature distances upward and triggers `100%` overkill at a fixed threshold. The pre-inference **Image Quality Gate (`check_image_quality`)** catches and flags `too_dark`, `too_bright`, `low_contrast`, and `blurry` camera frames before inference, but physical LED ring-light stabilization remains essential.
3. **Cross-Domain Scale Shift (`NEU-DET -> GC10-DET`):** Zero-shot transfer of `inclusion` from `200x200` macro crops (`NEU-DET`) to `2048x1000` wide-sheet images (`GC10-DET`) drops `AP@50` to `0.0686`. Deploying to a new camera geometry requires matching physical millimeter-per-pixel scale or fine-tuning on local line images.

---

## Documentation Index

- [`docs/model-card.md`](docs/model-card.md) — Model Card (architecture, datasets, metrics, limitations)
- [`docs/dataset-card.md`](docs/dataset-card.md) — Dataset Card (sources, licenses, deduplication, splits)
- [`docs/lessons-learned.vi.md`](docs/lessons-learned.vi.md) — Engineering lessons learned (Vietnamese)
- [`docs/reports/experiments.vi.md`](docs/reports/experiments.vi.md) — Phase 2 & 3 YOLOv8 + PatchCore experiment report
- [`docs/reports/qc_robustness.vi.md`](docs/reports/qc_robustness.vi.md) — Phase 4 QC (`escape`/`overkill`), unseen defect & robustness report
- [`docs/reports/performance.vi.md`](docs/reports/performance.vi.md) — Phase 6 ONNX CPU latency & memory benchmark report
- [`docs/capture-protocol.vi.md`](docs/capture-protocol.vi.md) & [`docs/labeling-guide.vi.md`](docs/labeling-guide.vi.md) — Self-captured dataset protocol & labeling guide
- [`docs/journal.vi.md`](docs/journal.vi.md) — Full chronological engineering journal (Vietnamese)

## License

Code is released under the MIT License. Public datasets (`NEU-DET`, `GC10-DET`, `MVTec AD`) retain their original licenses as documented in [`docs/dataset-card.md`](docs/dataset-card.md).
