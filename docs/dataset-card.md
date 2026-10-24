# Dataset Card — public detection datasets

## NEU-DET (hot-rolled steel strip surface defects)

| | |
|---|---|
| Source | [NEU Surface Defect Database](https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list) (Northeastern University, China) |
| Kaggle mirror used | `kagglehub://kaustubhdikshit/neu-surface-defect-database` |
| Content | 1,800 grayscale 200×200 JPGs, 6 defect classes, PASCAL VOC XML boxes |
| Classes | crazing, inclusion, patches, pitted_surface, rolled_in_scale, scratches |
| Boxes | 4,189 (post-conversion; 0 degenerate, 0 dropped) |
| License | For academic research only (no explicit redistribution license) |
| Notes | Mirror ships the original train/validation folders (1440/360); we re-split everything ourselves — see below |

## GC10-DET (metallic sheet surface defects)

| | |
|---|---|
| Source | GC10-DET — "Deep Metallic Surface Defect Detection: The New Benchmark" (Lv et al., 2020) |
| Kaggle mirror used | `kagglehub://alex000kim/gc10det` |
| Content | 2,294 annotated images (6 images in the mirror have no XML), PASCAL VOC XML boxes |
| Classes | punching, weld_line, crescent_gap, water_spot, oil_spot, silk_spot, inclusion, rolled_pit, crease, waist_folding |
| Boxes | 3,563 (1 dropped: a single box labeled `d`, a source-data typo) |
| License | For academic research only |
| Notes | The mirror labels classes with Chinese pinyin (`1_chongkong`…`10_yaozhed`); `configs/datasets.yaml` carries the rename table to the canonical English names of the paper. `10_yaozhe` and `10_yaozhed` are the same class (waist folding) — a naming inconsistency in the source data. |

## MVTec AD (anomaly detection validation)

| | |
|---|---|
| Source | [MVTec AD](https://www.mvtec.com/company/research/datasets-mvtec) (MVTec Software GmbH) |
| Planned Kaggle mirror | `ipythonx/mvtec-ad` (not yet verified/downloaded — Phase 3) |
| Used categories | `metal_nut` primarily (has pixel-precise anomaly masks for PatchCore AUROC validation) |
| License | CC BY-NC-SA 4.0 |

## Reproduction

```bash
make data                                    # neu_det + gc10: download -> convert -> dedupe -> split
python -m steelvision.data.prepare --datasets neu_det --skip-download   # from cached raw
```

Pipeline facts (see `data/processed/<name>/summary.json` after a run):

- Conversion: VOC XML → YOLO `class x_c y_c w h` normalized; boxes clipped to image bounds; degenerate (<1 px) boxes dropped and counted.
- Class policy: pinyin renames (GC10) and junk-class drops are explicit in `configs/datasets.yaml`, printed on every run, and fail loudly if an unknown class appears.
- Near-duplicate handling: 16-bit perceptual hash (pHash); images within Hamming distance 8 form one group; groups are assigned atomically to a single split (no train/test leakage via near-duplicates).
- Splitting: group-aware greedy stratification by class (train 0.7 / val 0.15 / test 0.15, seed 42). Every run writes a `split_fingerprint` into `summary.json`; rerunning with the same config and raw data reproduces it exactly.
- Both datasets are re-split uniformly (rather than using NEU-DET's original train/validation folders) so that val/test are held out consistently and cross-dataset evaluation is clean.
