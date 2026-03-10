# =============================================================================
# PTQ × LayerSkip — Makefile
# =============================================================================
# From PTQ/ root:  make <target>
#
# Mirrors the original run_pipeline.sh flow exactly:
#   Phase 0: download → Phase 1: quantize → Phase 2: sweep →
#   Phase 3: benchmark → Phase 4: collect → Phase 5: plot
#
# Model: facebook/layerskip-llama2-70B (LayerSkip early-exit trained)
# =============================================================================

SHELL := /bin/bash

# --- Defaults (override: make benchmark METHOD=gptq STRATEGY=self_speculative) ---
IMAGE          := ptq-layerskip:latest
MODEL          := facebook/layerskip-llama2-70B
METHOD         := awq
TASK           := cnn_dm_summarization
STRATEGY       := autoregressive
EXIT_LAYER     := 20
NUM_SPEC       := 6
NUM_SAMPLES    := 200
MAX_STEPS      := 256

ALL_METHODS    := fp16 int8_bnb awq gptq smoothquant
ALL_TASKS      := cnn_dm_summarization xsum_summarization

# --- Docker: all output dirs mounted to host so results survive container exit ---
DOCKER := docker run --gpus all --rm -it \
	-e HUGGINGFACE_TOKEN="$$HUGGINGFACE_TOKEN" \
	-v "$(CURDIR)/logs":/app/logs \
	-v "$(CURDIR)/results":/app/results \
	-v "$(CURDIR)/figures":/app/figures \
	-v "$(CURDIR)/quantized_models":/app/quantized_models \
	-v "$(CURDIR)/model_cache":/app/model_cache \
	-v "$(CURDIR)/model_cache":/root/.cache/huggingface \
	$(IMAGE)

.PHONY: help build dry-run dirs download \
        quantize quantize-all quantize-fp16 quantize-int8 quantize-awq quantize-gptq quantize-smoothquant \
        benchmark benchmark-ar benchmark-ss sweep \
        collect plot pipeline shell \
        clean clean-logs clean-results clean-figures clean-models

# ---------------------------------------------------------------------------
help:
	@echo ""
	@echo "PTQ x LayerSkip Benchmark (llama2-70B)"
	@echo "======================================="
	@echo ""
	@echo "Setup:"
	@echo "  make build         Build Docker image"
	@echo "  make dry-run       Syntax check (no GPU)"
	@echo ""
	@echo "Pipeline (matches run_pipeline.sh exactly):"
	@echo "  make download      Phase 0 — download all datasets"
	@echo "  make quantize      Phase 1 — quantize MODEL with METHOD"
	@echo "  make quantize-all  Phase 1 — quantize 70B with ALL methods"
	@echo "  make sweep         Phase 2 — find best exit_layer + num_speculations"
	@echo "  make benchmark     Phase 3 — run one benchmark"
	@echo "  make benchmark-ar  Phase 3 — autoregressive only"
	@echo "  make benchmark-ss  Phase 3 — self-speculative only"
	@echo "  make collect       Phase 4 — aggregate logs/ -> results/"
	@echo "  make plot          Phase 5 — generate figures/"
	@echo "  make pipeline      ALL phases end-to-end (70B × 5 methods × 2 strategies × 2 tasks)"
	@echo ""
	@echo "Shortcuts:"
	@echo "  make quantize-fp16 / quantize-int8 / quantize-awq / quantize-gptq / quantize-smoothquant"
	@echo ""
	@echo "Utilities:"
	@echo "  make shell         Interactive bash in container"
	@echo "  make clean         Remove logs + results + figures"
	@echo ""
	@echo "Defaults:"
	@echo "  MODEL=$(MODEL)  METHOD=$(METHOD)"
	@echo "  TASK=$(TASK)  MAX_STEPS=$(MAX_STEPS)"
	@echo "  STRATEGY=$(STRATEGY)  EXIT_LAYER=$(EXIT_LAYER)  NUM_SPEC=$(NUM_SPEC)"
	@echo ""

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
build:
	docker build -f scripts/Dockerfile -t $(IMAGE) .

dry-run:
	python scripts/dry_run.py

dirs:
	@mkdir -p logs results figures quantized_models model_cache

# ---------------------------------------------------------------------------
# Phase 0: Download datasets
# ---------------------------------------------------------------------------
download: dirs
	@echo "========================================"
	@echo "Phase 0: Downloading datasets"
	@echo "========================================"
	$(DOCKER) python scripts/00_download_data.py

# ---------------------------------------------------------------------------
# Phase 1: Quantize
# ---------------------------------------------------------------------------
quantize: dirs
	$(DOCKER) python scripts/01_quantize.py --model $(MODEL) --method $(METHOD)

quantize-fp16: dirs
	$(DOCKER) python scripts/01_quantize.py --model $(MODEL) --method fp16

quantize-int8: dirs
	$(DOCKER) python scripts/01_quantize.py --model $(MODEL) --method int8_bnb

quantize-awq: dirs
	$(DOCKER) python scripts/01_quantize.py --model $(MODEL) --method awq

quantize-gptq: dirs
	$(DOCKER) python scripts/01_quantize.py --model $(MODEL) --method gptq

quantize-smoothquant: dirs
	$(DOCKER) python scripts/01_quantize.py --model $(MODEL) --method smoothquant

quantize-all: dirs
	@echo "========================================"
	@echo "Phase 1: Quantizing $(MODEL) with all methods"
	@echo "========================================"
	@for method in $(ALL_METHODS); do \
		echo ""; \
		echo "--- Quantizing $(MODEL) with $$method ---"; \
		$(DOCKER) python scripts/01_quantize.py --model $(MODEL) --method $$method; \
	done

# ---------------------------------------------------------------------------
# Phase 2: Sweep (find best exit_layer + num_speculations for 70B)
# ---------------------------------------------------------------------------
sweep: dirs
	@echo "========================================"
	@echo "Phase 2: Sweeping LayerSkip hyperparameters"
	@echo "========================================"
	$(DOCKER) bash -c "cd /app/LayerSkip && LOCAL_RANK=0 python sweep.py \
		--model $(MODEL) \
		--dataset cnn_dm_summarization \
		--generation_strategy self_speculative \
		--num_samples 50 \
		--max_steps 128 \
		--output_dir /app/logs/sweep \
		--sample False"
	@echo ""
	@echo "Sweep done. Check logs/sweep/ for optimal exit_layer and num_speculations."
	@echo "Update EXIT_LAYER and NUM_SPEC if needed, then run benchmarks."

# ---------------------------------------------------------------------------
# Phase 3: Benchmark
# ---------------------------------------------------------------------------
benchmark: dirs
	$(DOCKER) python scripts/02_run_benchmark.py \
		--model $(MODEL) \
		--ptq_method $(METHOD) \
		--task $(TASK) \
		--generation_strategy $(STRATEGY) \
		--exit_layer $(EXIT_LAYER) \
		--num_speculations $(NUM_SPEC) \
		--num_samples $(NUM_SAMPLES) \
		--max_steps $(MAX_STEPS) \
		--sample False \
		--output_dir /app/logs

benchmark-ar: dirs
	$(DOCKER) python scripts/02_run_benchmark.py \
		--model $(MODEL) \
		--ptq_method $(METHOD) \
		--task $(TASK) \
		--generation_strategy autoregressive \
		--num_samples $(NUM_SAMPLES) \
		--max_steps $(MAX_STEPS) \
		--sample False \
		--output_dir /app/logs

benchmark-ss: dirs
	$(DOCKER) python scripts/02_run_benchmark.py \
		--model $(MODEL) \
		--ptq_method $(METHOD) \
		--task $(TASK) \
		--generation_strategy self_speculative \
		--exit_layer $(EXIT_LAYER) \
		--num_speculations $(NUM_SPEC) \
		--num_samples $(NUM_SAMPLES) \
		--max_steps $(MAX_STEPS) \
		--sample False \
		--output_dir /app/logs

# ---------------------------------------------------------------------------
# Phase 4+5: Collect + Plot
# ---------------------------------------------------------------------------
collect: dirs
	@echo "========================================"
	@echo "Phase 4: Collecting results"
	@echo "========================================"
	$(DOCKER) python scripts/03_collect_results.py \
		--logs_dir /app/logs --output_dir /app/results

plot: dirs
	@echo "========================================"
	@echo "Phase 5: Generating figures"
	@echo "========================================"
	$(DOCKER) python scripts/04_plot_results.py \
		--results_json /app/results/results_summary.json --output_dir /app/figures

# ---------------------------------------------------------------------------
# Full pipeline — mirrors run_pipeline.sh exactly
# Model: 70B only × 5 PTQ methods × 2 strategies × 2 tasks = 20 benchmark runs
# ---------------------------------------------------------------------------
pipeline: dirs
	@echo "========================================"
	@echo "Phase 0: Downloading datasets"
	@echo "========================================"
	$(DOCKER) python scripts/00_download_data.py
	@echo ""
	@echo "========================================"
	@echo "Phase 1: Quantizing $(MODEL) with all methods"
	@echo "========================================"
	@for method in $(ALL_METHODS); do \
		echo ""; \
		echo "--- Quantizing $(MODEL) with $$method ---"; \
		$(DOCKER) python scripts/01_quantize.py --model $(MODEL) --method $$method; \
	done
	@echo ""
	@echo "========================================"
	@echo "Phase 2: Sweeping LayerSkip hyperparameters"
	@echo "========================================"
	$(DOCKER) bash -c "cd /app/LayerSkip && LOCAL_RANK=0 python sweep.py \
		--model $(MODEL) \
		--dataset cnn_dm_summarization \
		--generation_strategy self_speculative \
		--num_samples 50 \
		--max_steps 128 \
		--output_dir /app/logs/sweep \
		--sample False"
	@echo ""
	@echo "========================================"
	@echo "Phase 3: Running benchmarks"
	@echo "========================================"
	@for method in $(ALL_METHODS); do \
		for task in $(ALL_TASKS); do \
			echo ""; \
			echo "--- $(MODEL) / $$method / autoregressive / $$task ---"; \
			$(DOCKER) python scripts/02_run_benchmark.py \
				--model $(MODEL) \
				--ptq_method $$method \
				--task $$task \
				--generation_strategy autoregressive \
				--num_samples $(NUM_SAMPLES) \
				--max_steps $(MAX_STEPS) \
				--sample False \
				--output_dir /app/logs; \
			echo ""; \
			echo "--- $(MODEL) / $$method / self_speculative / $$task ---"; \
			$(DOCKER) python scripts/02_run_benchmark.py \
				--model $(MODEL) \
				--ptq_method $$method \
				--task $$task \
				--generation_strategy self_speculative \
				--exit_layer $(EXIT_LAYER) \
				--num_speculations $(NUM_SPEC) \
				--num_samples $(NUM_SAMPLES) \
				--max_steps $(MAX_STEPS) \
				--sample False \
				--output_dir /app/logs; \
		done; \
	done
	@echo ""
	@echo "========================================"
	@echo "Phase 4: Collecting results"
	@echo "========================================"
	$(DOCKER) python scripts/03_collect_results.py \
		--logs_dir /app/logs --output_dir /app/results
	@echo ""
	@echo "========================================"
	@echo "Phase 5: Generating figures"
	@echo "========================================"
	$(DOCKER) python scripts/04_plot_results.py \
		--results_json /app/results/results_summary.json --output_dir /app/figures
	@echo ""
	@echo "========================================"
	@echo "Pipeline complete!"
	@echo "  Results: results/results_table.csv"
	@echo "  Figures: figures/"
	@echo "========================================"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
shell: dirs
	$(DOCKER) bash

clean: clean-logs clean-results clean-figures

clean-logs:
	rm -rf logs/ && echo "logs/ removed"

clean-results:
	rm -rf results/ && echo "results/ removed"

clean-figures:
	rm -rf figures/ && echo "figures/ removed"

clean-models:
	rm -rf quantized_models/ && echo "quantized_models/ removed"
