"""Bundle src/steelvision into notebooks/steelvision_src.zip for kernel upload."""

import zipfile
from pathlib import Path

SRC = Path("src/steelvision")
OUT = Path("notebooks/steelvision_src.zip")


def main() -> None:
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(SRC.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
                zf.write(f, Path("steelvision") / f.relative_to(SRC))
    n_files = len(zipfile.ZipFile(OUT).namelist())
    print(f"{OUT} rewritten: {n_files} files, {OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
