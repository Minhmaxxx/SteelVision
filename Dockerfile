FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY assets/ ./assets/

# Install lightweight CPU serving runtime + editable package
RUN pip install --upgrade pip && \
    pip install --no-deps -e . && \
    pip install \
        "numpy>=1.26,<3" \
        "opencv-python-headless>=4.9" \
        "pillow>=10" \
        "pyyaml>=6" \
        "pandas>=2.1" \
        "onnxruntime>=1.17" \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.29" \
        "python-multipart>=0.0.9" \
        "streamlit>=1.35"

EXPOSE 8000 8501

CMD ["uvicorn", "steelvision.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
