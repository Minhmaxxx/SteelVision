"""Build the Kaggle kernel notebook with the steelvision package embedded.

`kaggle kernels push` (CLI 2.x) uploads ONLY the code_file, so the package
source is base64-embedded into an import cell instead of cloned from GitHub
(the repo may stay private). Output goes to notebooks/dist/ (gitignored).
"""

import base64
import copy
import io
import json
import zipfile
from pathlib import Path

TEMPLATE = Path("notebooks/01_train_yolo_baseline.ipynb")
METADATA = Path("notebooks/kernel-metadata.json")
DIST = Path("notebooks/dist")

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
    return buf.getvalue()


def main() -> None:
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
    DIST.mkdir(exist_ok=True)
    (DIST / TEMPLATE.name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (DIST / METADATA.name).write_text(METADATA.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"{DIST / TEMPLATE.name}: {len(nb['cells'])} cells, embedded source {len(data)} bytes ({len(b64)} b64)")


if __name__ == "__main__":
    main()
