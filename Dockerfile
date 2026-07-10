ARG CUDA_VERSION=13.3.0-cudnn-devel-ubuntu24.04
FROM nvidia/cuda:${CUDA_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        python3-dev \
        curl \
        git \
        build-essential \
        && ln -sf /usr/bin/python3 /usr/bin/python \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --break-system-packages --ignore-installed pip wheel

# vLLM 0.24.0 + nvidia-cutlass-dsl 4.5.2 required for NVFP4
# Unsloth: "Do NOT use CUDA 13.2 — use below 13.2 or CUDA 13.3"
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        'vllm==0.24.0' \
        'nvidia-cutlass-dsl==4.5.2' \
        hf_transfer \
        huggingface_hub

# RunPod serverless SDK (for Queue-based handler mode)
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        runpod \
        requests

RUN mkdir -p /data/models /data/logs /src

# --- Caddy: /ping shim + SSE-friendly reverse proxy for RunPod LB ---------
ARG CADDY_VERSION=2.8.4
RUN curl -fsSL "https://github.com/caddyserver/caddy/releases/download/v${CADDY_VERSION}/caddy_${CADDY_VERSION}_linux_amd64.tar.gz" \
        | tar -xz -C /usr/local/bin caddy \
    && chmod +x /usr/local/bin/caddy \
    && mkdir -p /etc/caddy

COPY Caddyfile /etc/caddy/Caddyfile

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY runpod-entrypoint.sh /usr/local/bin/runpod-entrypoint.sh
COPY queue-entrypoint.sh /usr/local/bin/queue-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh /usr/local/bin/runpod-entrypoint.sh /usr/local/bin/queue-entrypoint.sh

# Handler for Queue-based serverless mode
COPY src/handler.py /src/handler.py

ENV MODEL_DIR=/data/models
ENV LOG_DIR=/data/logs
ENV MODEL_REPO=unsloth/Qwen3.6-35B-A3B-NVFP4-Fast
ENV PORT=1234
ENV SERVED_MODEL_NAME=qwen3.6-35b-nvfp4
ENV MAX_MODEL_LEN=32768
ENV MAX_NUM_SEQS=256
ENV MAX_NUM_BATCHED_TOKENS=8192
ENV GPU_MEMORY_UTIL=0.94
ENV TEMPERATURE=0.6
ENV TOP_P=0.95
ENV TOP_K=20
ENV MIN_P=0.0
ENV PRESENCE_PENALTY=0
ENV REPETITION_PENALTY=1.0
ENV REASONING_PARSER=qwen3
ENV MODEL_DOWNLOAD=1
ENV PUBLIC_PORT=8000
ENV KV_CACHE_DTYPE=fp8
ENV MTP_SPECULATIVE_TOKENS=2

# Default: Queue-based mode (works with RunPod's default serverless endpoint type)
# Override with runpod-entrypoint.sh for Load Balancer mode
ENTRYPOINT ["/usr/local/bin/queue-entrypoint.sh"]
CMD ["serve"]
