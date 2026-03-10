#!/usr/bin/env bash
# =============================================================================
# docker_run.sh
# Convenience wrapper — runs any pipeline command inside the container
# with ALL output directories mounted to the host so nothing is lost
# when the container exits.
#
# Usage (from PTQ/ root):
#   bash scripts/docker_run.sh <command>
#
# Examples:
#   bash scripts/docker_run.sh python scripts/00_download_data.py
#   bash scripts/docker_run.sh python scripts/01_quantize.py --model facebook/layerskip-llama2-7B --method awq
#   bash scripts/docker_run.sh python scripts/02_run_benchmark.py --model facebook/layerskip-llama2-7B --ptq_method awq --task cnn_dm_summarization --num_samples 50
#   bash scripts/docker_run.sh python scripts/03_collect_results.py
#   bash scripts/docker_run.sh python scripts/04_plot_results.py
#   bash scripts/docker_run.sh bash scripts/run_pipeline.sh
# =============================================================================

set -euo pipefail

IMAGE="ptq-layerskip:latest"
PTQ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# Host-side output directories — created here so Docker doesn't create them
# as root-owned on first mount
# ---------------------------------------------------------------------------
mkdir -p \
    "$PTQ_ROOT/logs" \
    "$PTQ_ROOT/results" \
    "$PTQ_ROOT/figures" \
    "$PTQ_ROOT/quantized_models" \
    "$PTQ_ROOT/model_cache"

# ---------------------------------------------------------------------------
# Require HuggingFace token
# ---------------------------------------------------------------------------
if [ -z "${HUGGINGFACE_TOKEN:-}" ]; then
    echo "ERROR: HUGGINGFACE_TOKEN is not set."
    echo "  Export it first:  export HUGGINGFACE_TOKEN=hf_..."
    exit 1
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "Running in container: $IMAGE"
echo "  PTQ root mounted at: $PTQ_ROOT → /app"
echo "  Command: $*"
echo ""

docker run --gpus all --rm -it \
    -e HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
    \
    -v "$PTQ_ROOT/logs":/app/logs \
    -v "$PTQ_ROOT/results":/app/results \
    -v "$PTQ_ROOT/figures":/app/figures \
    -v "$PTQ_ROOT/quantized_models":/app/quantized_models \
    -v "$PTQ_ROOT/model_cache":/app/model_cache \
    -v "$PTQ_ROOT/model_cache":/root/.cache/huggingface \
    \
    "$IMAGE" "$@"
