# SteelVision

End-to-end quality-control (QC) system for metal surface defect inspection.
Given an image of a metal product, the system decides **OK / NG**, reports the
**defect type and location**, and keeps an **inspection history** for
traceability.

It combines two complementary models:

- **Object detection** (YOLOv8) — finds known defect types and where they are.
- **Anomaly detection** (PatchCore, trained on defect-free images only) — a
  safety net that also catches *unseen* defect types.

The final OK/NG rule merges both signals; thresholds are chosen on a
validation set against QC objectives (escape rate / overkill rate), not by
feel.

## Highlights

- QC-language evaluation: **escape rate** (defective products that pass) and
  **overkill rate** (good products that get rejected) instead of accuracy only.
- Robustness analysis: brightness, blur, noise, small rotations, and
  cross-dataset (cross-domain) testing.
- CPU-friendly deployment: ONNX / ONNX Runtime benchmarking with latency
  p50/p95 before and after optimization.
- Delivered as a service: FastAPI (`/inspect`), SQLite traceability,
  Streamlit dashboard, Docker Compose.

## Status

Work in progress — results tables are filled as experiments land (tracked in
`docs/reports/`).

| Stage | Status |
|---|---|
| Repo + CI + data pipeline | ✅ |
| Public data (NEU-DET, GC10-DET) — converted, deduped, split | ✅ |
| MVTec AD (metal_nut) download | ⬜ Phase 3 |
| YOLOv8 baseline + ablations (Kaggle/Colab) | ⬜ notebooks ready |
| PatchCore anomaly detection | ⬜ |
| QC metrics + robustness | ⬜ |
| Self-captured dataset (protocol + guides ready) | ⬜ |
| ONNX CPU optimization | ⬜ |
| FastAPI + Docker + dashboard | ⬜ |

## Architecture

```
        Image / camera / video
                │
                ▼
        Preprocessing (resize, normalize, quality check)
                │
        ┌───────┴────────┐
        ▼                ▼
  YOLO detector    PatchCore anomaly
  (type + box)     (anomaly score)
        └────────┬───────┘
                 ▼
      Decision rule → OK / NG + reason
                 ▼
   FastAPI /inspect      SQLite traceability (NG images kept)
                 ▼
        Streamlit dashboard
```

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -e ".[dev]"

make data    # download + convert + split public datasets
make test    # unit tests
make serve   # run the inspection API (after weights exist)
```

Training runs on Kaggle/Colab GPUs — see `notebooks/`.

## Documentation

- `docs/labeling-guide.vi.md` — defect definitions and labeling rules (Vietnamese)
- `docs/capture-protocol.vi.md` — how the self-captured dataset is collected
- `docs/reports/` — experiment, QC and robustness reports
- `docs/model-card.md` — model card
- `SteelVision_Project_Overview.md` — original project brief (Vietnamese)

## License & data

Code: MIT. Public datasets (NEU-DET, GC10-DET, MVTec AD) keep their own
licenses, documented in `docs/dataset-card.md` with reproduction steps.
