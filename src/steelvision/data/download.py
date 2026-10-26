"""Download public datasets from Kaggle into ``data/raw/`` via kagglehub."""

from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub


def copy_from_kagglehub(handle: str, dest: Path) -> Path:
    """Download a public Kaggle dataset (no credentials needed) and copy it to ``dest``.

    kagglehub caches downloads under the user cache dir; the copy gives the
    rest of the pipeline a stable path independent of that cache.
    """
    cached = kagglehub.dataset_download(handle)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(cached, dest)
    return dest
