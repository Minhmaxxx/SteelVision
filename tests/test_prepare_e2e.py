"""End-to-end regression test: group->sample index mapping in prepare.

Catches index-shift bugs where group indices are used as sample indices:
with a near-duplicate pair present, every group after it used to materialize
the WRONG image/label pairing.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from steelvision.data.prepare import prepare_detection
from steelvision.data.split import SplitConfig

XML_TEMPLATE = """<annotation>
  <filename>{stem}.jpg</filename>
  <size><width>64</width><height>64</height><depth>3</depth></size>
  <object>
    <name>Defect</name>
    <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>26</ymax></bndbox>
  </object>
</annotation>
"""


def _make_raw(raw_root: Path, n: int) -> None:
    rng = np.random.default_rng(0)
    raw_root.mkdir(parents=True, exist_ok=True)
    patterns = []
    for _ in range(n - 1):  # distinct patterns; the last repeats pattern 0
        patterns.append(rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8))
    patterns.append(patterns[0].copy())  # exact near-duplicate pair: s01 & s06
    for i, arr in enumerate(patterns, start=1):
        stem = f"s{i:02d}"
        Image.fromarray(arr).save(raw_root / f"{stem}.jpg")
        (raw_root / f"{stem}.xml").write_text(XML_TEMPLATE.format(stem=stem), encoding="utf-8")


def test_prepare_end_to_end_groups_and_pairing(tmp_path):
    raw_root = tmp_path / "raw" / "testds"
    _make_raw(raw_root, n=6)
    cfg = {"kaggle": "example/testds", "classes": ["defect"]}
    scfg = SplitConfig(ratios={"train": 0.6, "val": 0.2, "test": 0.2}, seed=42)

    prepare_detection("testds", cfg, tmp_path, scfg, skip_download=True)

    out = tmp_path / "processed" / "testds"
    all_images = {
        split: {p.stem for p in (out / "images" / split).iterdir()}
        for split in ("train", "val", "test")
    }
    total = sum(len(v) for v in all_images.values())
    assert total == 6, f"expected all 6 samples materialized, got {total}"

    # every image has a label file in the same split, and vice versa
    for split, stems in all_images.items():
        labels = {p.stem for p in (out / "labels" / split).iterdir()}
        assert stems == labels, f"{split}: image/label mismatch: {stems ^ labels}"

    # the exact-duplicate pair must sit in ONE split
    home = {s for s, stems in all_images.items() if {"s01", "s06"} & stems}
    assert len(home) == 1, f"near-duplicate pair straddles splits: {all_images}"
    pair_split = home.pop()
    assert {"s01", "s06"} <= all_images[pair_split]

    # label content pairs with the right image: class id 0 and valid coords
    for split, stems in all_images.items():
        for stem in stems:
            line = (out / "labels" / split / f"{stem}.txt").read_text().strip()
            cls, xc, yc, w, h = map(float, line.split())
            assert cls == 0
            assert all(0 <= v <= 1 for v in (xc, yc, w, h))
