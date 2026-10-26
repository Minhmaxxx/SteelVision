# Model Card — SteelVision Dual-Branch QC Pipeline

## 1. Model Details

- **System Name:** SteelVision Dual-Branch QC Inspection Pipeline (`v0.1.0`)
- **Architecture:**
  1. **Pre-Inference Image Quality Gate (`steelvision.api.quality`):** Evaluates grayscale mean brightness (`[20, 240]`), contrast standard deviation (`>= 8.0`), and Laplacian focus variance (`>= 15.0`) prior to neural inference.
  2. **Branch 1 — Supervised Defect Detector (`steelvision.models.detector`):** YOLOv8n (`3,006,818` parameters, `640x640` input, exported to ONNX opset 18, `12.16 MB`) trained to localize and classify 6 known steel surface defect categories.
  3. **Branch 2 — Unsupervised Anomaly Detector (`steelvision.models.anomaly`):** PatchCore with `wide_resnet50_2` backbone (`layers = [layer2, layer3]`, `coreset_sampling_ratio = 0.01`, `num_neighbors = 9`, `256x256` input, exported to ONNX opset 18, `108.27 MB`) trained exclusively on defect-free (`train/good`) images.
  4. **Product Decision Rule (`steelvision.decision.fusion`):** Logical OR gate:
     $$\text{Verdict} = \text{NG} \iff (\max_i c_i \ge \tau_{\text{det}}) \lor (s_{\text{anom}} \ge \tau_{\text{anom}})$$
     with operating thresholds calibrated on validation sets (`v1-val-2026-10-26`: $\tau_{\text{det}}^* = 0.25$, $\tau_{\text{anom}}^* = 0.53$).
- **License:** MIT (code and exported configuration).

---

## 2. Intended Use

- **Primary Use Case:** Automated quality control (QC) inspection of flat steel strips and small machined/stamped metal components on an industrial inspection station, providing:
  - Product-level **OK / NG** verdict and triggering reason (`ok`, `detection`, `anomaly`, `both`).
  - Defect bounding boxes and class labels for known defect types.
  - Pixel-level anomaly heatmaps for novel/unseen surface defects.
  - Full SQLite traceability (`runs/traceability/inspections.db`) and automatic archiving of NG images.
- **Intended Users:** Manufacturing QC engineers, computer vision engineers, and MLOps practitioners.
- **Out-of-Scope Uses:** Safety-critical structural/internal weld inspection (e.g., ultrasonic or X-ray subsurface crack detection) without domain-specific retraining and human-in-the-loop verification.

---

## 3. Training & Evaluation Data

All datasets are documented in [`docs/dataset-card.md`](dataset-card.md) with deterministic pHash deduplication (`threshold = 4`) and stratified `70 / 15 / 15` splits (`seed = 42`):

| Dataset | Role | Images (After Dedup) | Split (`train / val / test`) | Classes |
|---|---|---:|---:|---|
| **NEU-DET** | Primary supervised detection benchmark | `1,799` | `1,259 / 270 / 270` | 6 (`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`) |
| **GC10-DET** | Multi-class & cross-domain benchmark | `2,286` | `1,600 / 343 / 343` | 10 (`crease`, `crescent_gap`, `inclusion`, `oil_spot`, `punching_hole`, `rolled_pit`, `silk_spot`, `waist_folding`, `water_spot`, `welding_line`) |
| **MVTec AD (`metal_nut`)** | Unsupervised anomaly detection & OK/NG benchmark | `335` | `220` (`train/good`) / `115` (`test`: `22` OK + `93` NG) | `good` + 4 defect categories (`bent`, `color`, `flip`, `scratch`) |

---

## 4. Quantitative Performance

### 4.1 Object Detection (YOLOv8 — Test Split)

| Experiment | Dataset | Model | Resolution | Augmentation | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---:|---|---:|---:|---:|---:|
| Baseline | NEU-DET | YOLOv8n | 640 | Default (`mosaic=1.0`) | 0.6778 | 0.7280 | 0.7641 | 0.4272 |
| **Ablation 2 (Selected)** | **NEU-DET** | **YOLOv8n** | **640** | **No Mosaic (`mosaic=0.0`)** | **0.6967** | **0.7429** | **0.7836** | **0.4404** |
| Ablation 1 (Scale) | NEU-DET | YOLOv8s | 640 | Default (`mosaic=1.0`) | 0.7082 | 0.7007 | 0.7662 | 0.4314 |
| Ablation 3 (Resolution) | NEU-DET | YOLOv8n | 800 | No Mosaic (`mosaic=0.0`) | 0.7001 | 0.7002 | 0.7510 | 0.4193 |
| Baseline | GC10-DET | YOLOv8n | 640 | Default (`mosaic=1.0`) | 0.6808 | 0.6352 | **0.6793** | **0.3512** |

### 4.2 Anomaly Detection (PatchCore on MVTec AD `metal_nut`)

| Metric | Score |
|---|---:|
| **Image-level AUROC** | **0.9980 (99.80%)** |
| **Image-level F1 Score** | **0.9840 (98.40%)** |
| **Pixel-level AUROC** | **0.9869 (98.69%)** |
| **Pixel-level F1 Score** | **0.8445 (84.45%)** |

### 4.3 Product-Level QC Metrics (`escape_rate <= 2.0%` Objective)

Thresholds selected on validation sets (`val`) and evaluated once on held-out `test`:

| Branch / Dataset | Selected Threshold (`val`) | Test Escape Rate (FN / Total NG) | Test Overkill Rate (FP / Total OK) | Test Precision | Test Recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|
| **PatchCore (`metal_nut`)** | $\tau_{\text{anom}}^* = 0.53$ | **0.00%** (`0/47`) | **0.00%** (`0/11` test, `4.55%` full) | **100.00%** | **100.00%** | **1.0000** |
| **YOLOv8n (`NEU-DET` + OK bg)** | $\tau_{\text{det}}^* = 0.25$ | **0.75%** (`2/266`) | **22.50%** (`18/80`) | **93.62%** | **99.25%** | **0.9635** |
| *YOLOv8n naive default* | $\tau_{\text{det}} = 0.50$ | *9.40%* (`25/266`) | *2.50%* (`2/80`) | *99.18%* | *90.60%* | *0.9469* |
| **Dual-Branch Fusion (Unseen defect test)** | $\tau_{\text{det}}^* = 0.25, \tau_{\text{anom}}^* = 0.53$ | **0.00%** (across all 4 held-out classes) | **4.55%** | **98.94%** | **100.00%** | **0.9947** |

### 4.4 CPU Latency, Throughput & Memory (`AMD Ryzen 7 5800H`, `batch=1`)

| Stage / Mode | Backend | Latency p50 (ms) | Latency p95 (ms) | FPS | RSS RAM (MB) | Speedup vs PyTorch |
|---|---|---:|---:|---:|---:|---:|
| YOLOv8n (`640x640`) | PyTorch CPU | 66.65 | 72.39 | 15.00 | 1,166.22 | `1.00x` |
| YOLOv8n (`640x640`) | ONNX Runtime CPU | **58.37** | **67.60** | **17.13** | **241.38** | **`1.14x`** |
| PatchCore (`256x256`) | PyTorch CPU | 210.89 | 237.22 | 4.74 | 1,295.18 | `1.00x` |
| PatchCore (`256x256`) | ONNX Runtime CPU | **114.47** | **123.53** | **8.74** | **570.81** | **`1.84x`** |
| E2E Pipeline (Full Dual-Branch) | PyTorch CPU | 279.60 | 361.33 | 3.58 | 1,297.18 | `1.00x` |
| E2E Pipeline (Full Dual-Branch) | ONNX Runtime CPU | 236.23 | 282.66 | 4.23 | 626.36 | `1.18x` |
| **E2E Pipeline (Early-Exit)** | **ONNX Runtime CPU** | **63.78** | **100.22** | **15.68** | **626.37** | **`4.38x`** |

---

## 5. Limitations & Failure Modes

1. **Micro-crack vs. Background Grain Confusion (`crazing` & `rolled-in_scale`):** On NEU-DET (`200x200` native resolution), hairline `crazing` cracks achieve `mAP@50 = 0.491` because thin dark lines blend with normal rolled-steel texture. Higher optical magnification or SAHI tiled inference is required for sub-millimeter cracks.
2. **PatchCore Overkill Under Severe Lighting Drop (`brightness_down`) or Sensor Noise:** While PatchCore maintains `image_AUROC >= 0.9785` across all 6 synthetic perturbations, severe under-exposure (`0.6x` brightness) shifts feature embeddings away from the clean coreset, pushing `overkill_rate` to `100%` (`escape_rate = 0%`). The pre-inference **Image Quality Gate** mitigates this by flagging `too_dark`, `too_bright`, `low_contrast`, and `blurry` camera frames explicitly.
3. **Cross-Domain Generalization (`NEU-DET -> GC10-DET`):** Direct zero-shot evaluation of `inclusion` from `NEU-DET` (`200x200` hot-rolled crops) onto `GC10-DET` (`2048x1000` cold-rolled wide sheets) drops `inclusion` AP@50 to `0.0686` due to a `10x` field-of-view scale mismatch. Deploying on a new camera line requires re-normalizing patch scale or fine-tuning on local samples.
