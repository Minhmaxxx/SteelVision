"""Prepare public detection datasets end-to-end.

    python -m steelvision.data.prepare --datasets neu_det gc10

For each dataset: download from Kaggle into ``data/raw/<name>/``, discover
(image, xml) pairs, group near-duplicates by perceptual hash, split into
train/val/test at the group level, and materialize a YOLO-format dataset under
``data/processed/<name>/`` with an Ultralytics-compatible ``data.yaml``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

from steelvision.data import download
from steelvision.data.split import SplitConfig, duplicate_groups, phash_matrix, split_groups
from steelvision.data.voc_to_yolo import ParseReport, discover_samples


def sample_id(image: Path, raw_root: Path) -> str:
    """Unique, collision-free id from the path below raw_root (folders joined)."""
    rel = image.relative_to(raw_root).with_suffix("")
    return "__".join(rel.parts).lower()


def apply_class_policy(
    samples, rename: dict[str, str], ignore: set[str]
) -> tuple[Counter, Counter]:
    """Rename observed class names to canonical ones; drop ignored classes.

    Returns (observed counts after policy, per-name counts of dropped boxes).
    """
    dropped: Counter[str] = Counter()
    if not rename and not ignore:
        return Counter(b.class_name for s in samples for b in s.boxes), dropped
    for sample in samples:
        kept = []
        for box in sample.boxes:
            box.class_name = rename.get(box.class_name, box.class_name)
            if box.class_name in ignore:
                dropped[box.class_name] += 1
            else:
                kept.append(box)
        sample.boxes = kept
    return Counter(b.class_name for s in samples for b in s.boxes), dropped


def check_class_map(observed: Counter[str], configured: list[str]) -> None:
    missing = sorted(set(observed) - set(configured))
    if missing:
        raise ValueError(
            "XML annotations contain class names absent from configs/datasets.yaml: "
            f"{missing}\nObserved: {dict(observed)}\nConfigured: {configured}\n"
            "Add them (normalized: lowercase, spaces/hyphens -> underscores) and rerun."
        )


def materialize(
    samples, assignment: dict[str, list[int]], class_names: list[str], out_dir: Path, raw_root: Path
) -> None:
    """Write images + YOLO labels for ``assignment``: split -> sample indices."""
    class_ids = {name: i for i, name in enumerate(class_names)}
    for split, sample_indices in assignment.items():
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for si in sample_indices:
            sample = samples[si]
            sid = sample_id(sample.image, raw_root)
            shutil.copy2(
                sample.image, out_dir / "images" / split / f"{sid}{sample.image.suffix.lower()}"
            )
            lines = [box.to_line(class_ids[box.class_name]) for box in sample.boxes]
            label_path = out_dir / "labels" / split / f"{sid}.txt"
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )


def prepare_detection(
    name: str, cfg: dict, data_root: Path, scfg: SplitConfig, skip_download: bool
) -> dict:
    raw_root = data_root / "raw" / name
    if not skip_download:
        print(f"[{name}] downloading kagglehub://{cfg['kaggle']} ...")
        download.copy_from_kagglehub(cfg["kaggle"], raw_root)
    elif not raw_root.exists():
        raise FileNotFoundError(f"{raw_root} not found; rerun without --skip-download")

    report = ParseReport()
    samples = discover_samples(raw_root, report)
    observed, dropped = apply_class_policy(
        samples, cfg.get("rename", {}), set(cfg.get("ignore", []))
    )
    print(
        f"[{name}] samples={len(samples)} boxes={sum(len(s.boxes) for s in samples)} "
        f"degenerate={report.n_degenerate} missing_image={report.n_missing_image} "
        f"unannotated={report.n_unannotated_images}"
    )
    print(f"[{name}] observed classes: {dict(sorted(observed.items()))}")
    if dropped:
        print(f"[{name}] dropped boxes by class: {dict(sorted(dropped.items()))}")
    check_class_map(observed, cfg["classes"])

    print(f"[{name}] perceptual-hash duplicate grouping ({len(samples)} images) ...")
    hashes = phash_matrix([s.image for s in samples], scfg.hash_size)
    groups = duplicate_groups(hashes, scfg.max_distance)
    n_dupes = sum(len(g) - 1 for g in groups)
    print(f"[{name}] {len(groups)} groups ({n_dupes} near-duplicate images kept in-group)")

    class_of = [s.class_names for s in samples]
    assignment = split_groups(groups, class_of, scfg)
    # split_groups returns group indices; flatten to sample indices once.
    sample_assignment = {
        split: [si for gi in gis for si in groups[gi]] for split, gis in assignment.items()
    }
    for split, indices in sample_assignment.items():
        per_class = Counter()
        for si in indices:
            per_class.update(samples[si].class_names)
        print(f"[{name}] {split}: {len(indices)} images | {dict(sorted(per_class.items()))}")

    out_dir = data_root / "processed" / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    materialize(samples, sample_assignment, cfg["classes"], out_dir, raw_root)
    for split, indices in sample_assignment.items():
        n_img = len(list((out_dir / "images" / split).iterdir()))
        n_lbl = len(list((out_dir / "labels" / split).iterdir()))
        if (n_img, n_lbl) != (len(indices), len(indices)):
            raise RuntimeError(
                f"[{name}] {split}: wrote {n_img} images / {n_lbl} labels, expected {len(indices)}"
            )
    fingerprint = hashlib.md5(
        repr(
            sorted(
                (sample_id(samples[si].image, raw_root), split)
                for split, indices in sample_assignment.items()
                for si in indices
            )
        ).encode()
    ).hexdigest()
    print(f"[{name}] split fingerprint: {fingerprint[:12]}")
    summary = {
        "dataset": name,
        "raw_root": str(raw_root),
        "classes": cfg["classes"],
        "n_samples": len(samples),
        "n_boxes": sum(len(s.boxes) for s in samples),
        "n_dropped_boxes": sum(dropped.values()),
        "n_degenerate_boxes": report.n_degenerate,
        "n_near_duplicate_groups": len(groups),
        "split_fingerprint": fingerprint,
        "splits": {
            split: {
                "images": len(indices),
                "boxes": sum(len(samples[si].boxes) for si in indices),
            }
            for split, indices in sample_assignment.items()
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{name}] done -> {out_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--datasets", nargs="+", default=["neu_det", "gc10"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    scfg = SplitConfig(ratios=config["splits"], seed=config["seed"])
    for name in args.datasets:
        cfg = config["datasets"][name]
        if cfg.get("type") == "anomaly":
            print(f"[{name}] anomaly dataset: download-only (no VOC conversion)")
            if not args.skip_download:
                download.copy_from_kagglehub(cfg["kaggle"], Path(args.data_root) / "raw" / name)
            continue
        prepare_detection(name, cfg, Path(args.data_root), scfg, args.skip_download)
    return 0


if __name__ == "__main__":
    sys.exit(main())
