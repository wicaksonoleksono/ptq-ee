#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh
# Full benchmark pipeline: download → quantize → benchmark → collect → plot
#
# Run from PTQ/ root:
#   bash scripts/run_pipeline.sh
#
# To benchmark only specific models / methods, edit the arrays below.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTQ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LAYERSKIP_DIR="$PTQ_DIR/LayerSkip"
LOGS_DIR="$PTQ_DIR/logs"
RESULTS_DIR="$PTQ_DIR/results"
FIGURES_DIR="$PTQ_DIR/figures"

export LAYERSKIP_DIR

# ---------------------------------------------------------------------------
# Configuration — edit these as needed
# ---------------------------------------------------------------------------

# Models to benchmark (comment out lines you don't want to run)
MODELS=(
  "facebook/layerskip-llama2-7B"
  "facebook/layerskip-llama2-13B"
  # "facebook/layerskip-llama2-70B"      # uncomment when ready (needs ~35GB VRAM with W4)
  "facebook/layerskip-llama3-8B"
  "facebook/layerskip-llama3.2-1B"
)

# PTQ methods to apply
PTQ_METHODS=("fp16" "int8_bnb" "awq" "gptq" "smoothquant")

# Decoding strategies
STRATEGIES=("autoregressive" "self_speculative")

# Tasks (LayerSkip dataset format names)
TASKS=("cnn_dm_summarization" "xsum_summarization")

NUM_SAMPLES=200
MAX_STEPS=256
EXIT_LAYER=20        # for self_speculative — adjust after running sweep
NUM_SPECULATIONS=6

# ---------------------------------------------------------------------------
# Phase 0: Download datasets
# ---------------------------------------------------------------------------
echo "========================================"
echo "Phase 0: Downloading datasets"
echo "========================================"
python "$SCRIPT_DIR/00_download_data.py"

# ---------------------------------------------------------------------------
# Phase 1: Quantize all models with all methods
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Phase 1: Quantizing models"
echo "========================================"

for MODEL in "${MODELS[@]}"; do
  for METHOD in "${PTQ_METHODS[@]}"; do
    echo ""
    echo "--- Quantizing $MODEL with $METHOD ---"
    python "$SCRIPT_DIR/01_quantize.py" \
      --model "$MODEL" \
      --method "$METHOD"
  done
done

# ---------------------------------------------------------------------------
# Phase 2: Sweep LayerSkip hyperparameters (find best exit_layer)
# Run for the 7B model first since it's fastest
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Phase 2: Sweeping LayerSkip hyperparameters"
echo "========================================"

cd "$LAYERSKIP_DIR"
LOCAL_RANK=0 python sweep.py \
  --model facebook/layerskip-llama2-7B \
  --dataset cnn_dm_summarization \
  --generation_strategy self_speculative \
  --num_samples 50 \
  --max_steps 128 \
  --output_dir "$LOGS_DIR/sweep/" \
  --sample False

echo "Sweep done. Check $LOGS_DIR/sweep/ for optimal exit_layer and num_speculations."
echo "Update EXIT_LAYER and NUM_SPECULATIONS in this script accordingly, then re-run Phase 3."
cd "$PTQ_DIR"

# ---------------------------------------------------------------------------
# Phase 3: Run all benchmarks
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Phase 3: Running benchmarks"
echo "========================================"

mkdir -p "$LOGS_DIR"

for MODEL in "${MODELS[@]}"; do
  for METHOD in "${PTQ_METHODS[@]}"; do
    for TASK in "${TASKS[@]}"; do
      for STRAT in "${STRATEGIES[@]}"; do

        echo ""
        echo "--- Benchmarking: model=$MODEL ptq=$METHOD task=$TASK strategy=$STRAT ---"

        if [ "$STRAT" = "self_speculative" ]; then
          LOCAL_RANK=0 python "$SCRIPT_DIR/02_run_benchmark.py" \
            --model "$MODEL" \
            --ptq_method "$METHOD" \
            --task "$TASK" \
            --generation_strategy "$STRAT" \
            --exit_layer "$EXIT_LAYER" \
            --num_speculations "$NUM_SPECULATIONS" \
            --num_samples "$NUM_SAMPLES" \
            --max_steps "$MAX_STEPS" \
            --sample False \
            --output_dir "$LOGS_DIR"
        else
          LOCAL_RANK=0 python "$SCRIPT_DIR/02_run_benchmark.py" \
            --model "$MODEL" \
            --ptq_method "$METHOD" \
            --task "$TASK" \
            --generation_strategy "$STRAT" \
            --num_samples "$NUM_SAMPLES" \
            --max_steps "$MAX_STEPS" \
            --sample False \
            --output_dir "$LOGS_DIR"
        fi

      done
    done
  done
done

# ---------------------------------------------------------------------------
# Phase 4: Collect results into CSV + JSON
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Phase 4: Collecting results"
echo "========================================"

python "$SCRIPT_DIR/03_collect_results.py" \
  --logs_dir "$LOGS_DIR" \
  --output_dir "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Phase 5: Plot figures
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Phase 5: Generating figures"
echo "========================================"

python "$SCRIPT_DIR/04_plot_results.py" \
  --results_json "$RESULTS_DIR/results_summary.json" \
  --output_dir "$FIGURES_DIR"

echo ""
echo "========================================"
echo "Pipeline complete!"
echo "  Results: $RESULTS_DIR/results_table.csv"
echo "  Figures: $FIGURES_DIR/"
echo "========================================"
