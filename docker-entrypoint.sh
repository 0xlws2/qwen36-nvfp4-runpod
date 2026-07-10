#!/usr/bin/env bash
set -euo pipefail

# --- Auto-detect GPU count & set TP if not overridden --------------------
if [[ -z "${TENSOR_PARALLEL:-}" ]]; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    if (( GPU_COUNT == 0 )); then
        echo "ERROR: No GPUs detected. Pass --gpus to docker run." >&2
        exit 1
    fi
    TENSOR_PARALLEL=$GPU_COUNT
    echo "Detected ${GPU_COUNT} GPU(s), setting TENSOR_PARALLEL=${TENSOR_PARALLEL}"
fi
export TENSOR_PARALLEL

# --- Cachedir setup -------------------------------------------------------
export HF_HOME="${MODEL_DIR}/.hf_cache"
export TMPDIR="${MODEL_DIR}/.tmp"
export PIP_CACHE_DIR="${MODEL_DIR}/.pip_cache"
mkdir -p "$HF_HOME" "$TMPDIR" "$PIP_CACHE_DIR"

# --- Logs setup -----------------------------------------------------------
mkdir -p "$LOG_DIR"

# --- Commands --------------------------------------------------------------
CMD="${1:-serve}"

case "$CMD" in
    download)
        if [[ -f "${MODEL_DIR}/config.json" ]]; then
            echo "Model already present at ${MODEL_DIR} — skipping download."
            exit 0
        fi
        echo "Downloading ${MODEL_REPO} to ${MODEL_DIR} ..."
        mkdir -p "$MODEL_DIR"
        HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$MODEL_REPO" \
            --local-dir "$MODEL_DIR"
        echo "Download complete."
        exit 0
        ;;

    serve)
        if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
            if [[ "${MODEL_DOWNLOAD:-0}" == "1" ]]; then
                echo "Model not found and MODEL_DOWNLOAD=1 — downloading now ..."
                HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$MODEL_REPO" \
                    --local-dir "$MODEL_DIR"
                echo "Download complete."
            else
                echo "ERROR: Model not found at ${MODEL_DIR}." >&2
                echo "Run 'docker compose run --rm qwen36 download' or set MODEL_DOWNLOAD=1." >&2
                exit 1
            fi
        fi

        # --- Build generation config JSON from env vars ---------------------
        GEN_CONFIG="{\"temperature\": ${TEMPERATURE}, \"top_p\": ${TOP_P}, \"top_k\": ${TOP_K}, \"min_p\": ${MIN_P}, \"presence_penalty\": ${PRESENCE_PENALTY}, \"repetition_penalty\": ${REPETITION_PENALTY}}"

        # --- Optional reasoning parser --------------------------------------
        REASONING_PARSER_FLAG=""
        [[ -n "${REASONING_PARSER:-}" ]] && REASONING_PARSER_FLAG="--reasoning-parser ${REASONING_PARSER}"

        # --- MTP speculative decoding (NVFP4 Fast variant supports MTP) -----
        MTP_TOKENS="${MTP_SPECULATIVE_TOKENS:-2}"
        SPEC_CONFIG="{\"method\": \"mtp\", \"num_speculative_tokens\": ${MTP_TOKENS}}"

        echo "Starting vLLM server (NVFP4):"
        echo "  Model              : ${MODEL_DIR}"
        echo "  Served as          : ${SERVED_MODEL_NAME}"
        echo "  Port               : ${PORT}"
        echo "  Tensor parallelism : ${TENSOR_PARALLEL}"
        echo "  Max model len      : ${MAX_MODEL_LEN}"
        echo "  Max num seqs       : ${MAX_NUM_SEQS}"
        echo "  Max batched tokens : ${MAX_NUM_BATCHED_TOKENS}"
        echo "  GPU memory util    : ${GPU_MEMORY_UTIL}"
        echo "  KV cache dtype     : ${KV_CACHE_DTYPE:-fp8}"
        echo "  MTP tokens         : ${MTP_TOKENS}"
        echo "  Generation config  : ${GEN_CONFIG}"
        echo ""

        vllm serve "$MODEL_DIR" \
            --served-model-name "$SERVED_MODEL_NAME" \
            --override-generation-config "$GEN_CONFIG" \
            --port "$PORT" \
            --dtype bfloat16 \
            --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}" \
            --enable-prefix-caching \
            --enable-chunked-prefill \
            --tensor-parallel-size "$TENSOR_PARALLEL" \
            --max-model-len "$MAX_MODEL_LEN" \
            --max-num-seqs "$MAX_NUM_SEQS" \
            --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
            --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
            --disable-custom-all-reduce \
            --enable-auto-tool-choice \
            --tool-call-parser qwen3_coder \
            --trust-remote-code \
            ${REASONING_PARSER_FLAG} \
            --speculative-config "$SPEC_CONFIG" \
            2>&1 | tee -a "${LOG_DIR}/vllm.log"
        ;;

    *)
        echo "Unknown command: ${CMD}" >&2
        echo "Usage: docker-entrypoint.sh {download|serve}" >&2
        exit 1
        ;;
esac
