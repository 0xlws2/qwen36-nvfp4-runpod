# Qwen3.6-35B-A3B NVFP4 — RunPod Serverless

Docker image for serving `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` on RunPod Serverless (Load Balancer mode).

## Requirements

- vLLM 0.24.0 + nvidia-cutlass-dsl 4.5.2
- CUDA 13.3 (NOT 13.2 — causes gibberish outputs per Unsloth)
- 32GB VRAM GPU (RTX 5090, RTX 4090, A100, etc.)

## Architecture

```
[RunPod LB] ──:8000──► [caddy] ──/ping──► 200 if upstream /v1/models=200, else 204
                          │
                          └──/* (incl. /v1/*) ──► [vLLM :1234 on 127.0.0.1]
```

Caddy provides the `/ping` health probe RunPod requires and proxies the OpenAI-compatible API with SSE streaming preserved.

## Deploy on RunPod

### Option 1: Import from GitHub

RunPod console → Serverless → New Endpoint → Import from GitHub:
- Repository: `<your-username>/qwen36-nvfp4-runpod`
- Branch: `main`
- Endpoint Type: **Load Balancer**
- Internal port: `8000`

### Option 2: Use pre-built GHCR image

```
ghcr.io/<your-username>/qwen36-nvfp4-runpod:latest
```

### Configuration

| Setting | Value |
|---------|-------|
| Endpoint Type | Load Balancer |
| Port | 8000 |
| GPU | RTX 5090 (32GB) or RTX 4090 (24GB) |
| Container Disk | 200 GB |
| Workers | min 0, max 2-3 |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_REPO` | `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` | HuggingFace model repo |
| `MODEL_DOWNLOAD` | `1` | Auto-download model on cold start |
| `MAX_MODEL_LEN` | `32768` | Max context length |
| `MAX_NUM_SEQS` | `256` | Concurrent sequences |
| `GPU_MEMORY_UTIL` | `0.94` | GPU memory fraction |
| `KV_CACHE_DTYPE` | `fp8` | KV cache quantization |
| `MTP_SPECULATIVE_TOKENS` | `2` | MTP speculative decoding tokens |

### Network Volume (recommended)

Pre-populate a network volume to avoid downloading the model on every cold start:

1. Create a ~50GB network volume in your target datacenter
2. Mount at `/data/models`
3. Set `MODEL_DOWNLOAD=0` after initial population
