FROM python:3.11-slim

WORKDIR /app

# System deps for spacy + sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Download spacy model for entity extraction
RUN python -m spacy download en_core_web_sm

# Copy source
COPY src/ ./src/
COPY data/ ./data/
COPY scripts/ ./scripts/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
