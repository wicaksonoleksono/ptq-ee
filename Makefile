# =============================================================================
# PTQ × LayerSkip — Makefile
# =============================================================================
# Run: make <target>    (from scripts/ aka project root)
#
# Model: facebook/layerskip-llama2-13B
# =============================================================================

SHELL := /bin/bash

MODEL          := facebook/layerskip-llama2-13B
METHOD         := awq
TASK           := cnn_dm_summarization
STRATEGY       := autoregressive
EXIT_LAYER     := 8
NUM_SPEC       := 6
NUM_SAMPLES    := 200
MAX_STEPS      := 256

ALL_METHODS    := fp16 int8_bnb awq gptq smoothquant
ALL_TASKS      := cnn_dm_summarization xsum_summarization

.PHONY: help dry-run dirs download \
        quantize quantize-all quantize-fp16 quantize-int8 quantize-awq quantize-gptq quantize-smoothquant \
        benchmark benchmark-ar benchmark-ss benchmark-all sweep \
        collect plot pipeline \
        clean clean-logs clean-results clean-figures clean-models

# ---------------------------------------------------------------------------
help:
	@echo ""
	@echo "PTQ x LayerSkip Benchmark (llama2-13B)"
	@echo "======================================="
	@echo ""
	@echo "Pipeline:"
	@echo "  make download      Phase 0 — download all datasets"
	@echo "  make quantize      Phase 1 — quantize MODEL with METHOD"
	@echo "  make quantize-all  Phase 1 — quantize 13B with ALL methods"
	@echo "  make sweep         Phase 2 — find best exit_layer + num_speculations"
	@echo "  make benchmark     Phase 3 — run one benchmark"
	@echo "  make benchmark-ar  Phase 3 — autoregressive only"
	@echo "  make benchmark-ss  Phase 3 — self-speculative only"
	@echo "  make benchmark-all Phase 3 — ALL methods × strategies × tasks"
	@echo "  make collect       Phase 4 — aggregate logs/ -> results/"
	@echo "  make plot          Phase 5 — generate figures/"
	@echo "  make pipeline      ALL phases end-to-end"
	@echo ""
	@echo "Shortcuts:"
	@echo "  make quantize-fp16 / quantize-int8 / quantize-awq / quantize-gptq / quantize-smoothquant"
	@echo ""
	@echo "Utilities:"
	@echo "  make dry-run       Syntax check (no GPU)"
	@echo "  make clean         Remove logs + results + figures"
	@echo ""
	@echo "Defaults:"
	@echo "  MODEL=$(MODEL)  METHOD=$(METHOD)  TASK=$(TASK)"
	@echo "  STRATEGY=$(STRATEGY)  EXIT_LAYER=$(EXIT_LAYER)  NUM_SPEC=$(NUM_SPEC)"
	@echo ""

# ---------------------------------------------------------------------------
dry-run:
	python dry_run.py

dirs:
	@mkdir -p logs results figures quantized_models model_cache

# ---------------------------------------------------------------------------
# Phase 0: Download
# ---------------------------------------------------------------------------
download: dirs
	@echo "========================================"
	@echo "Phase 0: Downloading datasets"
	@echo "========================================"
	python 00_download_data.py

# ---------------------------------------------------------------------------
# Phase 1: Quantize
# ---------------------------------------------------------------------------
quantize: dirs
	python 01_quantize.py --model $(MODEL) --method $(METHOD)

quantize-fp16: dirs
	python 01_quantize.py --model $(MODEL) --method fp16

quantize-int8: dirs
	python 01_quantize.py --model $(MODEL) --method int8_bnb

quantize-awq: dirs
	python 01_quantize.py --model $(MODEL) --method awq

quantize-gptq: dirs
	python 01_quantize.py --model $(MODEL) --method gptq

quantize-smoothquant: dirs
	python 01_quantize.py --model $(MODEL) --method smoothquant

quantize-all: dirs
	@echo "========================================"
	@echo "Phase 1: Quantizing $(MODEL) with all methods"
	@echo "========================================"
	@for method in $(ALL_METHODS); do \
		echo ""; \
		echo "--- Quantizing $(MODEL) with $$method ---"; \
		python 01_quantize.py --model $(MODEL) --method $$method || echo "WARNING: $$method failed, continuing..."; \
	done

# ---------------------------------------------------------------------------
# Phase 2: Sweep (find best exit_layer + num_speculations)
# ---------------------------------------------------------------------------
sweep: dirs
	@echo "========================================"
	@echo "Phase 2: Sweeping LayerSkip hyperparameters"
	@echo "========================================"
	LOCAL_RANK=0 python sweep.py \
		--model $(MODEL) \
		--dataset cnn_dm_summarization \
		--generation_strategy self_speculative \
		--num_samples 50 \
		--max_steps 128 \
		--output_dir ./logs/sweep \
		--sample False
	@echo ""
	@echo "Sweep done. Check logs/sweep/"

# ---------------------------------------------------------------------------
# Phase 3: Benchmark
# ---------------------------------------------------------------------------
benchmark: dirs
	python 02_run_benchmark.py \
		--model $(MODEL) \
		--ptq_method $(METHOD) \
		--task $(TASK) \
		--generation_strategy $(STRATEGY) \
		--exit_layer $(EXIT_LAYER) \
		--num_speculations $(NUM_SPEC) \
		--num_samples $(NUM_SAMPLES) \
		--max_steps $(MAX_STEPS) \
		--sample False \
		--output_dir ./logs

benchmark-ar: dirs
	python 02_run_benchmark.py \
		--model $(MODEL) \
		--ptq_method $(METHOD) \
		--task $(TASK) \
		--generation_strategy autoregressive \
		--num_samples $(NUM_SAMPLES) \
		--max_steps $(MAX_STEPS) \
		--sample False \
		--output_dir ./logs

benchmark-ss: dirs
	python 02_run_benchmark.py \
		--model $(MODEL) \
		--ptq_method $(METHOD) \
		--task $(TASK) \
		--generation_strategy self_speculative \
		--exit_layer $(EXIT_LAYER) \
		--num_speculations $(NUM_SPEC) \
		--num_samples $(NUM_SAMPLES) \
		--max_steps $(MAX_STEPS) \
		--sample False \
		--output_dir ./logs

benchmark-all: dirs
	@echo "========================================"
	@echo "Phase 3: Running ALL benchmarks"
	@echo "========================================"
	@for method in $(ALL_METHODS); do \
		for task in $(ALL_TASKS); do \
			echo ""; \
			echo "--- $(MODEL) / $$method / autoregressive / $$task ---"; \
			python 02_run_benchmark.py \
				--model $(MODEL) \
				--ptq_method $$method \
				--task $$task \
				--generation_strategy autoregressive \
				--num_samples $(NUM_SAMPLES) \
				--max_steps $(MAX_STEPS) \
				--sample False \
				--output_dir ./logs || echo "WARNING: $$method/ar/$$task failed, continuing..."; \
			echo ""; \
			echo "--- $(MODEL) / $$method / self_speculative / $$task ---"; \
			python 02_run_benchmark.py \
				--model $(MODEL) \
				--ptq_method $$method \
				--task $$task \
				--generation_strategy self_speculative \
				--exit_layer $(EXIT_LAYER) \
				--num_speculations $(NUM_SPEC) \
				--num_samples $(NUM_SAMPLES) \
				--max_steps $(MAX_STEPS) \
				--sample False \
				--output_dir ./logs || echo "WARNING: $$method/ss/$$task failed, continuing..."; \
		done; \
	done

# ---------------------------------------------------------------------------
# Phase 4+5: Collect + Plot
# ---------------------------------------------------------------------------
collect: dirs
	@echo "========================================"
	@echo "Phase 4: Collecting results"
	@echo "========================================"
	python 03_collect_results.py --logs_dir ./logs --output_dir ./results

plot: dirs
	@echo "========================================"
	@echo "Phase 5: Generating figures"
	@echo "========================================"
	python 04_plot_results.py --results_json ./results/results_summary.json --output_dir ./figures

# ---------------------------------------------------------------------------
# Full pipeline — 13B × 5 methods × 2 strategies × 2 tasks = 20 benchmark runs
# ---------------------------------------------------------------------------
pipeline: download quantize-all benchmark-all collect plot
	@echo ""
	@echo "========================================"
	@echo "Pipeline complete!"
	@echo "  Results: ./results/results_table.csv"
	@echo "  Figures: ./figures/"
	@echo "========================================"

# ---------------------------------------------------------------------------
clean: clean-logs clean-results clean-figures

clean-logs:
	rm -rf logs/ && echo "logs/ removed"

clean-results:
	rm -rf results/ && echo "results/ removed"

clean-figures:
	rm -rf figures/ && echo "figures/ removed"

clean-models:
	rm -rf quantized_models/ && echo "quantized_models/ removed"
