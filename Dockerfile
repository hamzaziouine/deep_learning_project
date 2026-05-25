# Multi-Agent Product Review Intelligence — reproducibility container.
# Slim Python 3.11 base; CPU-only inference target (the trained model lives
# at runtime via volume mount or download). For GPU training, run on host
# with CUDA 12.4 wheels — see README.

FROM python:3.11-slim AS base

LABEL org.opencontainers.image.title="Multi-Agent Product Review Intelligence"
LABEL org.opencontainers.image.description="Hierarchical CrewAI crew + RoBERTa sentiment classifier"
LABEL org.opencontainers.image.source="https://github.com/hamzaziouine/deep_learning_project"
LABEL org.opencontainers.image.licenses="Educational"

WORKDIR /app

# System dependencies (minimal — pandas/torch wheels include what they need)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python deps first (cache-friendly layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Application code
COPY config/ ./config/
COPY src/ ./src/
COPY data/sample/ ./data/sample/
COPY data/processed/split_indices.json ./data/processed/split_indices.json
COPY tests/ ./tests/
COPY pytest.ini .

# At runtime, the user must mount:
#   - models/sentiment_roberta/  (the trained checkpoint, ~500MB)
#   - .env                        (with GEMINI_API_KEY=...)
# Example:
#   docker run --rm -it \
#     -v $(pwd)/models:/app/models \
#     -v $(pwd)/.env:/app/.env \
#     dlp:latest python -m src.main --niche "wireless earbuds"
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV LITELLM_CACHE=disk

CMD ["python", "-m", "src.main", "--validate-key"]
