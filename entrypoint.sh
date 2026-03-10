#!/bin/bash
# =============================================================================
# entrypoint.sh — PTQ × LayerSkip container entrypoint
# Follows the same pattern as LayerSkip/entrypoint.sh
# =============================================================================

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate ptq_layerskip

# Login to HuggingFace if token is provided
export HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential 2>/dev/null || true
fi

# Tell 02_run_benchmark.py where LayerSkip lives
export LAYERSKIP_DIR=/app/LayerSkip

# Execute whatever command was passed to docker run
exec "$@"
