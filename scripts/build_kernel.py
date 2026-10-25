"""Build the Kaggle kernel notebook with the steelvision package embedded.

`kaggle kernels push` (CLI 2.x) uploads ONLY the code_file, so the package
source is base64-embedded into an import cell instead of cloned from GitHub
(the repo may stay private). Output goes to notebooks/dist/ (gitignored).

Ablations override the template's training hyperparameters by string
replacement, e.g.:

    python scripts/build_kernel.py --imgsz 1024 --run-name yolov8n-neu-det-imgsz1024
"""

import argparse
import base64
import copy
import io
import json
import zipfile
from pathlib import Path

TEMPLATE = Path("notebooks/01_train_yolo_baseline.ipynb")
METADATA = Path("notebooks/kernel-metadata.json")
DIST = Path("notebooks/dist")

DEFAULT_RUN_NAME = "yolov8n-neu-det-baseline"
DEFAULT_IMGSZ = 640

CELL_TEMPLATE = """\
import base64
import io
import sys
import zipfile
from pathlib import Path

_src = (
{lines}
)
zipfile.ZipFile(io.BytesIO(base64.b64decode("".join(_src)))).extractall("steelvision_src")
sys.path.insert(0, str(Path("steelvision_src").resolve()))

import steelvision
print("steelvision", steelvision.__version__)
"""


def zip_source() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(Path("src/steelvision").rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
                zf.write(f, Path("steelvision") / f.relative_to("src/steelvision"))
        cfg = Path("configs/datasets.yaml")
        if cfg.exists():
            zf.write(cfg, "configs/datasets.yaml")
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--extra-train-arg",
        action="append",
        default=[],
        help="extra model.train() kwarg inserted into the kernel, e.g. mosaic=0.0",
    )
    args = parser.parse_args()

    data = zip_source()
    b64 = base64.b64encode(data).decode()
    lines = "\n".join(f'    "{b64[i:i + 120]}",' for i in range(0, len(b64), 120))

    nb = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    nb = copy.deepcopy(nb)
    nb["cells"][2] = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": CELL_TEMPLATE.format(lines=lines).splitlines(keepends=True),
    }
    text = json.dumps(nb, indent=1)
    if args.imgsz != DEFAULT_IMGSZ:
        assert text.count("imgsz=640,") == 1, "template train cell changed; update --imgsz mapping"
        text = text.replace("imgsz=640,", f"imgsz={args.imgsz},")
    if args.extra_train_arg:
        anchor = f"imgsz={args.imgsz},"
        assert text.count(anchor) == 1
        extras = ",\\n    ".join(args.extra_train_arg)  # JSON-escaped newline
        text = text.replace(anchor, f"{anchor}{extras},")
    if args.run_name != DEFAULT_RUN_NAME:
        n = text.count(DEFAULT_RUN_NAME)
        # train name, val name, zip filename, FileLink — all follow the run name.
        assert n == 4, f"expected 4 run-name occurrences in template, found {n}"
        text = text.replace(DEFAULT_RUN_NAME, args.run_name)
    nb = json.loads(text)

    DIST.mkdir(exist_ok=True)
    (DIST / TEMPLATE.name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (DIST / METADATA.name).write_text(METADATA.read_text(encoding="utf-8"), encoding="utf-8")
    print(
        f"{DIST / TEMPLATE.name}: {len(nb['cells'])} cells, embedded source {len(data)} bytes, "
        f"imgsz={args.imgsz}, run-name={args.run_name}"
    )


if __name__ == "__main__":
    main()
