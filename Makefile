.PHONY: help data train patchcore eval benchmark serve dashboard demo test lint kaggle-push

help:
	@echo "SteelVision targets:"
	@echo "  make data      - download + convert + split datasets into data/processed"
	@echo "  make train     - print where/how training runs (Kaggle/Colab notebooks)"
	@echo "  make patchcore - train + evaluate PatchCore on MVTec metal_nut and export ONNX"
	@echo "  make eval      - run QC evaluation (escape/overkill) and robustness tests"
	@echo "  make benchmark - run CPU latency (p50/p95), FPS, and RAM benchmark (PyTorch vs ONNX)"
	@echo "  make serve     - run the FastAPI inspection service locally"
	@echo "  make dashboard - run the Streamlit QC traceability dashboard locally"
	@echo "  make demo      - generate 60s MP4 demo video in runs/demo_steelvision.mp4"
	@echo "  make test      - run pytest"
	@echo "  make lint      - ruff check"
	@echo "  make kaggle-push - zip src/steelvision and push/run the Kaggle notebook (needs KAGGLE_API_TOKEN)"

data:
	python -m steelvision.data.prepare --all

train:
	@echo "Training runs on Kaggle/Colab (GPU). Open notebooks/01_train_yolo_baseline.ipynb"
	@echo "Downloaded weights go to runs/ and evaluation uses make eval."

patchcore:
	python -m steelvision.models.anomaly --category metal_nut --skip-download

eval:
	python -m steelvision.eval.run_qc --config configs/eval.yaml

benchmark:
	python -m steelvision.eval.benchmark --config configs/benchmark.yaml

serve:
	uvicorn steelvision.api.app:app --host 0.0.0.0 --port 8000

dashboard:
	streamlit run src/steelvision/dashboard/app.py --server.port 8501

demo:
	python scripts/generate_demo_video.py

test:
	pytest -q

lint:
	ruff check src tests

kaggle-push:
	python scripts/build_kernel.py
	.venv/Scripts/kaggle.exe kernels push -p notebooks/dist --accelerator NvidiaTeslaT4 --timeout 7200
