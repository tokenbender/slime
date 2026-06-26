FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    git \
    patch \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir pytest aiohttp httpx

WORKDIR /workspace
COPY . /workspace/slime

ENV PYTHONPATH=/workspace/slime

RUN git config --system --add safe.directory '*' \
    && cd /workspace/slime \
    && git status --short >/dev/null
