"""
RunPod Serverless Handler — proxies Queue-based jobs to vLLM's OpenAI API.

In Queue-based mode, RunPod calls handler(event) for each job.
This handler forwards requests to the local vLLM HTTP server (port 1234)
and returns the OpenAI-compatible response.
"""

import os
import time
import subprocess
import threading
import requests
import runpod

VLLM_URL = f"http://127.0.0.1:{os.getenv('PORT', '1234')}"
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "qwen3.6-35b-nvfp4")

_vllm_ready = False


def wait_for_vllm(timeout=600):
    """Block until vLLM's /v1/models responds 200."""
    global _vllm_ready
    if _vllm_ready:
        return True

    print("[handler] Waiting for vLLM to be ready...", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{VLLM_URL}/v1/models", timeout=5)
            if resp.status_code == 200:
                models = resp.json()
                print(f"[handler] vLLM ready! Models: {models}", flush=True)
                _vllm_ready = True
                return True
            else:
                print(f"[handler] vLLM /v1/models returned {resp.status_code}, retrying...", flush=True)
        except Exception as e:
            elapsed = int(time.time() - start)
            if elapsed % 10 == 0:
                print(f"[handler] vLLM not ready yet ({elapsed}s): {e}", flush=True)
        time.sleep(2)

    raise RuntimeError(f"vLLM not ready within {timeout}s")


def handler(job):
    """
    Forward a RunPod job to vLLM's OpenAI-compatible API.

    Supports three input formats:

    1. Simple prompt:  {"input": {"prompt": "Hello"}}
    2. OpenAI-style:   {"input": {"messages": [...], "max_tokens": 2048}}
    3. Raw route:      {"input": {"openai_route": true, "path": "/v1/chat/completions", "body": {...}}}
    """
    wait_for_vllm()

    job_input = job.get("input", {})

    # --- Raw OpenAI route forwarding ---
    if job_input.get("openai_route"):
        method = job_input.get("method", "POST")
        path = job_input.get("path", "/v1/chat/completions")
        body = job_input.get("body", {})
        headers = job_input.get("headers", {})

        resp = requests.request(
            method=method,
            url=f"{VLLM_URL}{path}",
            json=body,
            headers=headers,
            timeout=300,
        )
        return resp.json()

    # --- Build OpenAI chat completion request ---
    messages = job_input.get("messages")
    if not messages and "prompt" in job_input:
        messages = [{"role": "user", "content": job_input["prompt"]}]

    payload = {
        "model": job_input.get("model", SERVED_MODEL_NAME),
        "messages": messages or [{"role": "user", "content": "Hello"}],
        "max_tokens": job_input.get("max_tokens", 2048),
        "temperature": job_input.get("temperature", float(os.getenv("TEMPERATURE", "0.6"))),
        "top_p": job_input.get("top_p", float(os.getenv("TOP_P", "0.95"))),
        "stream": job_input.get("stream", False),
    }

    # Forward optional generation parameters
    for key in [
        "tools", "tool_choice", "response_format", "seed", "stop",
        "presence_penalty", "frequency_penalty", "top_k", "min_p",
        "repetition_penalty",
    ]:
        if key in job_input:
            payload[key] = job_input[key]

    # Add reasoning/thinking config if supported
    if os.getenv("REASONING_PARSER"):
        payload.setdefault("chat_template_kwargs", {})
        if "preserve_thinking" in job_input:
            payload["chat_template_kwargs"]["preserve_thinking"] = job_input["preserve_thinking"]

    try:
        resp = requests.post(
            f"{VLLM_URL}/v1/chat/completions",
            json=payload,
            timeout=300,
        )
        return resp.json()
    except requests.exceptions.ConnectionError as e:
        return {"error": f"vLLM connection error: {str(e)}"}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}


def _start_vllm_subprocess():
    """Start vLLM serve as a subprocess (must be called before runpod.serverless.start)."""
    model_dir = os.getenv("MODEL_DIR", "/data/models")
    port = os.getenv("PORT", "1234")
    served_name = os.getenv("SERVED_MODEL_NAME", "qwen3-0.6b")
    max_model_len = os.getenv("MAX_MODEL_LEN", "32768")
    gpu_util = os.getenv("GPU_MEMORY_UTIL", "0.90")
    
    cmd = [
        "vllm", "serve", model_dir,
        "--served-model-name", served_name,
        "--port", port,
        "--host", "0.0.0.0",
        "--max-model-len", max_model_len,
        "--gpu-memory-utilization", gpu_util,
        "--trust-remote-code",
    ]
    
    print(f"[handler] Starting vLLM: {' '.join(cmd[:6])}...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(f"[handler] vLLM PID: {proc.pid}")
    return proc


if __name__ == "__main__":
    # Start vLLM in subprocess FIRST (before RunPod takes over)
    _vllm_proc = _start_vllm_subprocess()
    
    # Wait for vLLM to be ready
    # Wait for vLLM before accepting any jobs
    wait_for_vllm()

    # Start RunPod serverless worker (blocks forever)
    runpod.serverless.start({
        "handler": handler,
        "concurrency_modifier": lambda x: int(os.getenv("MAX_NUM_SEQS", "256")),
        "return_aggregate_stream": True,
    })
