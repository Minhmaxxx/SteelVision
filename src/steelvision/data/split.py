"""Near-duplicate grouping and group-aware stratified splitting.

Images that look alike (perceptual hash within a small Hamming distance) are
treated as one group and are always assigned to the same split, so that
near-identical images never leak between train/val/test.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image


@dataclass
class SplitConfig:
    ratios: dict[str, float] = field(
        default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15}
    )
    seed: int = 42
    hash_size: int = 16
    max_distance: int = 8


def phash_matrix(paths: list[Path], hash_size: int = 16, desc: str = "hashing") -> np.ndarray:
    """Stack perceptual hashes of images into a boolean matrix (n, hash_size**2)."""
    hashes = []
    for path in paths:
        with Image.open(path) as image:
            hashes.append(imagehash.phash(image, hash_size=hash_size).hash.reshape(-1))
    return np.stack(hashes)


def duplicate_groups(hashes: np.ndarray, max_distance: int) -> list[list[int]]:
    """Group row indices whose pairwise Hamming distance <= ``max_distance``."""
    n = len(hashes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    bits = hashes.reshape(n, -1).astype(np.uint8)
    chunk = 128  # keeps the (chunk, n, dim) intermediate a few hundred MB at most
    for start in range(0, n, chunk):
        distances = np.count_nonzero(
            bits[start : start + chunk, None, :] != bits[None, :, :], axis=2
        )
        rows, cols = np.where(distances <= max_distance)
        for i, j in zip(rows, cols, strict=False):
            union(start + int(i), int(j))

    grouped: dict[int, list[int]] = {}
    for i in range(n):
        grouped.setdefault(find(i), []).append(i)
    return list(grouped.values())


def split_groups(
    groups: list[list[int]], class_of: list[set[str]], cfg: SplitConfig
) -> dict[str, list[int]]:
    """Assign whole groups to splits, greedily balancing image and class counts.

    Groups are visited largest-first (ties shuffled with ``cfg.seed``); each is
    placed in the split with the largest remaining deficit for its classes,
    then by remaining image count. Group-level assignment guarantees that
    near-duplicates never straddle two splits.
    """
    splits = list(cfg.ratios)
    rng = random.Random(cfg.seed)
    order = list(range(len(groups)))
    rng.shuffle(order)
    order.sort(key=lambda gi: -len(groups[gi]))

    total = sum(len(groups[gi]) for gi in order)
    total_cls: Counter[str] = Counter()
    for gi in order:
        total_cls.update(class_of[gi])

    target_n = {s: cfg.ratios[s] * total for s in splits}
    target_cls = {s: {c: cfg.ratios[s] * n for c, n in total_cls.items()} for s in splits}
    counts_n = {s: 0 for s in splits}
    counts_cls: dict[str, Counter[str]] = {s: Counter() for s in splits}
    assignment: dict[str, list[int]] = {s: [] for s in splits}

    for gi in order:
        classes = class_of[gi]
        best = max(
            splits,
            key=lambda s: (
                sum(target_cls[s].get(c, 0) - counts_cls[s].get(c, 0) for c in classes),
                target_n[s] - counts_n[s],
            ),
        )
        assignment[best].append(gi)
        counts_n[best] += len(groups[gi])
        counts_cls[best].update(classes)
    return assignment
