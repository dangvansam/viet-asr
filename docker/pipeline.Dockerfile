# Lightweight orchestrator image. NO torch / nemo / funasr — every model call goes
# out over REST to a service. Installs only the HTTP-client + crawl runtime deps and
# the project package WITHOUT its heavy core deps (--no-deps).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements-pipeline.txt /tmp/requirements-pipeline.txt
RUN pip install --no-cache-dir -r /tmp/requirements-pipeline.txt

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
RUN pip install --no-cache-dir --no-deps .

CMD ["python", "scripts/run_pipeline.py", "--config", "configs/pipeline_crawl_services.yaml"]
