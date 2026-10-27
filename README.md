# SteelVision: End-to-End Metal Surface Defect Inspection System

SteelVision is an end-to-end industrial quality-control (QC) system for metal surface defect inspection. Given an input image of a steel strip or machined metal part, the system performs a pre-inference camera quality check, executes a dual-branch inference pipeline (supervised object detection + unsupervised anomaly detection), outputs a product-level **OK / NG** verdict with localized defect coordinates and anomaly heatmaps, and logs every inspection to a relational database for manufacturing traceability.

---

## 1. Project Overview and Problem Statement

Standard academic models trained on surface defect datasets typically frame inspection as closed-set image classification and report top-1 accuracy. On a real manufacturing line, that approach fails for five practical reasons:

1. **Localization is required:** QC operators and downstream sorting mechanisms need to know *where* a defect is located and *how large* it is, not only a class label.
2. **Closed-set detectors cannot define "OK" or catch novel defects:** A supervised detector trained on $K$ known defect categories assigns low confidence to an *unseen* defect type (such as chemical discoloration or an inverted stamping) and silently passes the defective part as OK.
3. **Asymmetric error costs (Escape vs. Overkill):** Passing a defective part to a customer (**escape rate** / false negative) is orders of magnitude more costly than flagging a good part for manual re-check (**overkill rate** / false positive). Operating thresholds must be calibrated on a validation set against explicit QC escape constraints rather than left at a naive `0.50` default.
4. **Sensitivity to optical and domain shifts:** Changes in LED ring-light intensity, motion blur, sensor noise, or camera field-of-view can shift model scores and trigger false rejections unless guarded by an upfront image quality check.
5. **CPU-only edge deployment constraints:** Many factory inspection stations run on industrial PCs without discrete GPUs, requiring low-latency single-image (`batch=1`) inference on CPU and a self-contained service with persistent audit logs.

SteelVision addresses all five requirements in a single reproducible codebase.

---

## 2. System Architecture

```text
                  Input Image (Camera Frame / File Upload)
                                     |
                                     v
           +---------------------------------------------------+
           |            Pre-Inference Image Quality Gate       |
           |  (Mean Brightness, Contrast Std, Laplacian Focus) |
           +-------------------------+-------------------------+
                                     |
                      +--------------+--------------+
                      |                             |
                      v                             v
        +---------------------------+ +---------------------------+
        |   Branch 1: YOLOv8n ONNX  | |  Branch 2: PatchCore ONNX |
        |   Supervised Detection    | |   Unsupervised Anomaly    |
        |  (Class + Bounding Boxes) | | (Score + Anomaly Heatmap) |
        +-------------+-------------+ +-------------+-------------+
                      |                             |
                      +--------------+--------------+
                                     |
                                     v
           +---------------------------------------------------+
           |              OR-Gate Product Fusion               |
           |    NG if max(det_conf) >= 0.25 OR anom >= 0.53    |
           +-------------------------+-------------------------+
                                     |
                      +--------------+--------------+
                      |                             |
                      v                             v
        +---------------------------+ +---------------------------+
        |    FastAPI REST Service   | | SQLite Traceability Store |
        |  (/inspect, /inspections) | |  + NG Evidence Archiving  |
        +-------------+-------------+ +-------------+-------------+
                      |                             |
                      +--------------+--------------+
                                     |
                                     v
                      Streamlit QC Inspection Console
```

### Decision Rule

Let $\max_i c_i$ be the highest bounding-box confidence predicted by YOLOv8n and $s_{\text{anom}}$ be the image-level anomaly score computed by PatchCore. The product-level decision rule is a logical OR gate:

$$\text{Verdict} = \text{NG} \iff \left(\max_i c_i \ge \tau_{\text{det}}^*\right) \lor \left(s_{\text{anom}} \ge \tau_{\text{anom}}^*\right)$$

- **Known defect localization:** When $\max_i c_i \ge \tau_{\text{det}}^*$, YOLOv8n provides the exact defect class name and bounding box coordinates.
- **Unseen defect safety net:** When a novel defect occurs that YOLOv8n was never trained to recognize, PatchCore detects the deviation from defect-free feature embeddings ($s_{\text{anom}} \ge \tau_{\text{anom}}^*$) and flags the part as NG.
- **Early-Exit CPU optimization:** Because the fusion rule is a logical OR, when `early_exit=True` and YOLOv8n (`58.37 ms` median CPU latency) already detects a defect above $\tau_{\text{det}}^*$, the system can immediately return `NG (reason="detection")` and skip PatchCore (`114.47 ms`), achieving a `4.38x` end-to-end speedup with zero change in OK/NG accuracy.

---

## 3. Datasets and Data Integrity Pipeline

All public datasets are processed through a deterministic pipeline (`python -m steelvision.data.prepare --all`) that verifies bounding boxes, removes near-duplicate images via perceptual hashing (`pHash`, Hamming distance $\le 4$) before splitting to prevent train/test leakage, and performs multi-label stratified `70 / 15 / 15` splits (`seed = 42`):

| Dataset | Role | Raw Images | Near-Duplicate Pairs Removed | Clean Images | Split (`train / val / test`) | Categories |
|---|---|---:|---:|---:|---:|---|
| **NEU-DET** | Primary supervised steel defect detection | 1,800 | 1 | 1,799 | `1,259 / 270 / 270` | 6 (`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`) |
| **GC10-DET** | Multi-class detection & cross-domain test | 2,294 | 8 (35 boxes dropped for invalid coords) | 2,286 | `1,600 / 343 / 343` | 10 (`crease`, `crescent_gap`, `inclusion`, `oil_spot`, `punching_hole`, `rolled_pit`, `silk_spot`, `waist_folding`, `water_spot`, `welding_line`) |
| **MVTec AD (`metal_nut`)** | Unsupervised anomaly detection & OK/NG test | 335 | 0 | 335 | `220` (`train/good`) / `115` (`test`: 22 OK + 93 NG) | `good` + 4 defect types (`bent`, `color`, `flip`, `scratch`) |

In addition, the repository includes a complete protocol ([`docs/capture-protocol.vi.md`](docs/capture-protocol.vi.md)), labeling guide ([`docs/labeling-guide.vi.md`](docs/labeling-guide.vi.md)), and automated manifest/consistency validator ([`src/steelvision/data/self_captured.py`](src/steelvision/data/self_captured.py)) for self-captured metal surface datasets across three controlled lighting setups (`A/B/C`) with a 10% re-annotation quality check (`mean IoU >= 0.75`, `image F1 >= 0.90`).

---

## 4. Experimental Results

Full experimental logs and analysis are documented in [`docs/reports/experiments.vi.md`](docs/reports/experiments.vi.md), [`docs/reports/qc_robustness.vi.md`](docs/reports/qc_robustness.vi.md), and [`docs/reports/performance.vi.md`](docs/reports/performance.vi.md).

### 4.1 Supervised Object Detection and Ablation Study (Test Split)

All models were trained for 80 epochs (`seed = 42`) on NVIDIA Tesla T4 GPUs. One factor was varied at a time:

| Experiment | Dataset | Model | Input Size | Mosaic Augmentation | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | NEU-DET | YOLOv8n | 640 | `1.0` | 0.6778 | 0.7280 | 0.7641 | 0.4272 |
| **Ablation 2 (Selected)** | **NEU-DET** | **YOLOv8n** | **640** | **`0.0` (Disabled)** | **0.6967** | **0.7429** | **0.7836** | **0.4404** |
| Ablation 1 (Model Scale) | NEU-DET | YOLOv8s | 640 | `1.0` | 0.7082 | 0.7007 | 0.7662 | 0.4314 |
| Ablation 3 (Resolution) | NEU-DET | YOLOv8n | 800 | `0.0` | 0.7001 | 0.7002 | 0.7510 | 0.4193 |
| Multi-Class Baseline | GC10-DET | YOLOv8n | 640 | `1.0` | 0.6808 | 0.6352 | **0.6793** | **0.3512** |

**Key Finding:** Disabling `Mosaic` augmentation (`mosaic=0.0`) on `NEU-DET` improved test `mAP@50` from `0.7641` to `0.7836` (`+1.95%`) and lifted the hardest micro-crack class (`crazing`) from `0.388` to `0.491` (`+10.3%`), because artificial 4-image tile boundaries in Mosaic mimic linear surface defects and disrupt local defect-to-background contrast.

### 4.2 Unsupervised Anomaly Detection (PatchCore on MVTec AD `metal_nut`)

Trained exclusively on `220` defect-free (`train/good`) images (`wide_resnet50_2` backbone, `layers=[layer2, layer3]`, `coreset_sampling_ratio=0.01`, `256x256` input):

| Metric | Value |
|---|---:|
| **Image-level AUROC** | **0.9980 (99.80%)** |
| **Image-level F1 Score** | **0.9840 (98.40%)** |
| **Pixel-level AUROC** | **0.9869 (98.69%)** |
| **Pixel-level F1 Score** | **0.8445 (84.45%)** |

### 4.3 Product-Level QC Evaluation (`escape_rate <= 2.0%` Target)

Operating thresholds were selected on the validation split (`val`) to satisfy `escape_rate <= 2.0%` while minimizing `overkill_rate`, then locked and evaluated once on the held-out `test` split:

| Pipeline / Dataset | Selected Threshold (`val`) | Test Escape Rate (FN / Total NG) | Test Overkill Rate (FP / Total OK) | Test Precision | Test Recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|
| **PatchCore (`MVTec metal_nut`)** | $\tau_{\text{anom}}^* = 0.53$ | **0.00%** (`0/47`) | **0.00%** (`0/11` test, `4.55%` full) | **100.00%** | **100.00%** | **1.0000** |
| **YOLOv8n (`NEU-DET` + OK bg)** | $\tau_{\text{det}}^* = 0.25$ | **0.75%** (`2/266`) | **22.50%** (`18/80`) | **93.62%** | **99.25%** | **0.9635** |
| *YOLOv8n naive threshold* | $\tau_{\text{det}} = 0.50$ | *9.40%* (`25/266`) | *2.50%* (`2/80`) | *99.18%* | *90.60%* | *0.9469* |
| **Dual-Branch Fusion (Unseen defect test)** | $\tau_{\text{det}}^* = 0.25, \tau_{\text{anom}}^* = 0.53$ | **0.00%** (across all 4 held-out classes) | **4.55%** | **98.94%** | **100.00%** | **0.9947** |

**Unseen Defect Leave-One-Class-Out Simulation:** When hiding one defect class at a time to simulate a novel defect on the production line, a detector-only pipeline suffered an escape rate of `14.66%–17.29%` on `NEU-DET` and `23.66%–26.88%` on `metal_nut`. Combining the detector with PatchCore via OR-gate fusion reduced the escape rate on `metal_nut` back to **`0.00%`** across all four held-out defect categories (`bent`, `color`, `flip`, `scratch`).

### 4.4 CPU Latency, Throughput, and Memory Benchmark

Measured on an `AMD Ryzen 7 5800H` CPU (`batch=1`, 10 warm-up runs + 50 timed iterations per stage):

| Stage / Mode | Backend | Latency p50 (ms) | Latency p95 (ms) | Mean (ms) | FPS | RSS RAM (MB) | Speedup vs PyTorch |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n (`640x640`) | PyTorch CPU | 66.65 | 72.39 | 66.22 | 15.00 | 1,166.22 | `1.00x` |
| YOLOv8n (`640x640`) | ONNX Runtime CPU | **58.37** | **67.60** | **58.65** | **17.13** | **241.38** | **`1.14x`** |
| PatchCore (`256x256`) | PyTorch CPU | 210.89 | 237.22 | 213.71 | 4.74 | 1,295.18 | `1.00x` |
| PatchCore (`256x256`) | ONNX Runtime CPU | **114.47** | **123.53** | **114.33** | **8.74** | **570.81** | **`1.84x`** |
| E2E Full Dual-Branch | PyTorch CPU | 279.60 | 361.33 | 290.89 | 3.58 | 1,297.18 | `1.00x` |
| E2E Full Dual-Branch | ONNX Runtime CPU | 236.23 | 282.66 | 240.31 | 4.23 | 626.36 | `1.18x` |
| **E2E Early-Exit** | **ONNX Runtime CPU** | **63.78** | **100.22** | **73.32** | **15.68** | **626.37** | **`4.38x`** |

---

## 5. Repository Structure

```text
SteelVision/
├── assets/samples/                  # 11 curated test images (NEU-DET + MVTec) for instant UI testing
├── configs/
│   ├── baseline_neu.yaml            # YOLOv8n baseline training configuration
│   ├── eval.yaml                    # Phase 4 QC & robustness evaluation configuration
│   └── benchmark.yaml               # Phase 6 CPU latency & memory benchmark configuration
├── docs/
│   ├── dataset-card.md              # Dataset sources, licenses, pHash deduplication & splits
│   ├── model-card.md                # Model architecture, metrics, operating points & limitations
│   ├── lessons-learned.vi.md        # Technical lessons learned across Phases 0-7 (Vietnamese)
│   ├── capture-protocol.vi.md       # Self-captured dataset collection protocol
│   ├── labeling-guide.vi.md         # Bounding-box annotation standard & QC rules
│   ├── journal.vi.md                # Chronological engineering log
│   └── reports/                     # Detailed experimental, QC/robustness, and CPU benchmark reports
├── notebooks/
│   ├── 01_train_yolo_baseline.ipynb # Reproducible GPU training notebook (NEU-DET & GC10-DET)
│   └── 02_train_patchcore.ipynb     # Reproducible PatchCore training & ONNX export notebook
├── scripts/
│   ├── build_kernel.py              # Packages src/steelvision into a self-contained Kaggle kernel
│   └── generate_demo_video.py       # Renders the 60-second 720p MP4 showcase video
├── src/steelvision/
│   ├── api/                         # FastAPI service (app.py), Image Quality Gate (quality.py), SQLite store (store.py)
│   ├── dashboard/                   # Streamlit interactive QC console (app.py)
│   ├── data/                        # Dataset download, VOC-to-YOLO conversion, pHash split, self-captured QC
│   ├── decision/                    # OR-gate product fusion logic (fusion.py)
│   ├── eval/                        # QC escape/overkill metrics, perturbations, cross-domain & CPU benchmark
│   └── models/                      # Pure ONNX Runtime wrappers for YOLOv8 (detector.py) and PatchCore (anomaly.py)
├── tests/                           # 39 automated unit tests covering data, fusion, QC, ONNX, and FastAPI
├── Dockerfile                       # CPU-only Python 3.11 serving container
├── docker-compose.yml               # Orchestrates FastAPI (:8000) and Streamlit (:8501) with shared SQLite volume
├── Makefile                         # Standard automation targets
└── pyproject.toml                   # Package metadata, dependencies, ruff and pytest configuration
```

---

## 6. Quickstart and Deployment

### Local Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (use `source .venv/bin/activate` on Linux/macOS)
pip install -e ".[dev]"
```

### Standard Workflow Commands

```bash
make data        # Download, deduplicate (pHash), and split NEU-DET, GC10-DET, MVTec AD
make patchcore   # Train PatchCore on MVTec metal_nut and export static batch=1 ONNX
make eval        # Run QC threshold calibration, unseen defect simulation & robustness suite
make benchmark   # Run PyTorch vs ONNX Runtime CPU latency (p50/p95), FPS, and RAM benchmark
make serve       # Launch FastAPI service at http://localhost:8000 (Swagger UI at /docs)
make dashboard   # Launch Streamlit QC console at http://localhost:8501 (includes 11 sample images)
make demo        # Generate 60-second MP4 showcase video at runs/demo_steelvision.mp4
make test        # Run the 39 unit tests via pytest
make lint        # Run ruff code linter
```

### Docker Compose Deployment

```bash
docker compose up --build
```

- **FastAPI Inspection API:** `http://localhost:8000/docs` (`POST /inspect`, `GET /inspections`, `GET /inspections/{id}`, `GET /stats`, `GET /health`)
- **Streamlit QC Dashboard:** `http://localhost:8501`

---

## 7. Known Limitations

1. **Hairline Crack vs. Background Grain Confusion (`crazing`):** At `200x200` native resolution on `NEU-DET`, fine `crazing` micro-cracks reach `mAP@50 = 0.491` because thin dark fissures blend into hot-rolled steel grain. Detecting sub-millimeter cracks requires higher optical magnification or tiled inference (SAHI).
2. **Threshold Shift Under Severe Under-Exposure (`brightness_down`):** While PatchCore preserves ranking capability (`image_AUROC >= 0.9785`) across all six synthetic perturbations, a `40%` drop in illumination shifts feature distances away from the clean coreset and pushes `overkill_rate` to `100%` at a fixed threshold. The pre-inference **Image Quality Gate (`check_image_quality`)** mitigates this by flagging `too_dark`, `too_bright`, `low_contrast`, and `blurry` camera frames before inference, but physical lighting stability remains important.
3. **Cross-Domain Scale Mismatch (`NEU-DET -> GC10-DET`):** Zero-shot evaluation of `inclusion` from `200x200` close-up crops (`NEU-DET`) onto `2048x1000` wide-sheet images (`GC10-DET`) drops `AP@50` to `0.0686`. Deploying on a new camera line requires matching physical millimeter-per-pixel scale or fine-tuning on local production samples.

---

## 8. Documentation Index

- [`SteelVision_Project_Overview.md`](SteelVision_Project_Overview.md) — Comprehensive project overview and executive summary (Vietnamese)
- [`docs/model-card.md`](docs/model-card.md) — Model Card (architecture, datasets, metrics, operating thresholds, limitations)
- [`docs/dataset-card.md`](docs/dataset-card.md) — Dataset Card (sources, licenses, deduplication, split statistics)
- [`docs/lessons-learned.vi.md`](docs/lessons-learned.vi.md) — Six core engineering lessons learned (Vietnamese)
- [`docs/reports/experiments.vi.md`](docs/reports/experiments.vi.md) — YOLOv8 baseline, ablations, error analysis, and PatchCore report
- [`docs/reports/qc_robustness.vi.md`](docs/reports/qc_robustness.vi.md) — Product-level QC (`escape`/`overkill`), unseen defect, perturbation, and cross-domain report
- [`docs/reports/performance.vi.md`](docs/reports/performance.vi.md) — CPU latency (`p50`/`p95`), throughput (`FPS`), and memory (`RSS RAM`) benchmark report
- [`docs/capture-protocol.vi.md`](docs/capture-protocol.vi.md) & [`docs/labeling-guide.vi.md`](docs/labeling-guide.vi.md) — Self-captured dataset collection protocol and annotation standard
- [`docs/journal.vi.md`](docs/journal.vi.md) — Chronological engineering journal

## License

Code is released under the MIT License. Public datasets (`NEU-DET`, `GC10-DET`, `MVTec AD`) retain their original licenses as documented in [`docs/dataset-card.md`](docs/dataset-card.md).
