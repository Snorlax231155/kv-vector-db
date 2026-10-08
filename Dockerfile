# Backend image. Mock mode by default; pass MEDMEMORY_MODE=real plus keys to go live.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
# rocksdb extra so the container exercises the RocksDB backend too; real adapters are
# installed but stay idle in mock mode. torch/transformers are intentionally excluded
# (multi-GB); the LoRA router falls back to rules inside the container.
RUN pip install ".[rocksdb,faiss]" "pinecone==10.0.0" "langchain-anthropic==1.7.5"

COPY data ./data
# committed evaluation/benchmark results so the dashboard has data in Docker too
COPY eval/results ./eval/results
COPY bench/results ./bench/results
RUN useradd --create-home app && mkdir -p /app/var && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"
CMD ["python", "-m", "medmemory", "serve", "--host", "0.0.0.0", "--port", "8000"]
