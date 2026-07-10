#!/usr/bin/env bash
# queue-entrypoint.sh — Entry point for RunPod Queue-based serverless mode.
#
# Starts vLLM serve in the background, then runs the RunPod handler
# (handler.py) which blocks and processes jobs from the queue.
#
# If either process dies, the container exits.

set -euo pipefail

echo "[queue-entrypoint] Starting vLLM serve in background..."
/usr/local/bin/docker-entrypoint.sh serve &
VLLM_PID=$!

# Give vLLM a moment to start before launching the handler
sleep 2

echo "[queue-entrypoint] Starting RunPod handler (blocks)..."
python /src/handler.py &
HANDLER_PID=$!

# Wait for either process to exit; propagate its status
wait -n "$VLLM_PID" "$HANDLER_PID"
EXIT_CODE=$?

echo "[queue-entrypoint] Process exited (code $EXIT_CODE) — shutting down"
kill -TERM "$VLLM_PID" "$HANDLER_PID" 2>/dev/null || true
wait 2>/dev/null || true
exit "$EXIT_CODE"
