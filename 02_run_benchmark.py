"""
02_run_benchmark.py
-------------------
Main benchmark runner. Wraps LayerSkip's benchmark() with:
  - Energy measurement (pynvml)
  - GPU memory tracking
  - Structured JSON output

Supports all PTQ methods (fp16, int8_bnb, awq, gptq, smoothquant) and
both decoding strategies (autoregressive, self_speculative).

Must be run from the LayerSkip directory (or with LAYERSKIP_DIR set), because
it imports from the LayerSkip codebase.

Usage (single GPU):
    cd PTQ/LayerSkip
    LOCAL_RANK=0 python ../scripts/02_run_benchmark.py \\
        --model facebook/layerskip-llama2-70B \\
        --ptq_method fp16 \\
        --task cnn_dm_summarization \\
        --generation_strategy autoregressive \\
        --num_samples 200 \\
        --output_dir ../logs

    # Self-speculative decoding:
    LOCAL_RANK=0 python ../scripts/02_run_benchmark.py \\
        --model facebook/layerskip-llama2-70B \\
        --ptq_method awq \\
        --task cnn_dm_summarization \\
        --generation_strategy self_speculative \\
        --exit_layer 30 \\
        --num_speculations 6 \\
        --num_samples 200 \\
        --output_dir ../logs

    # With torchrun (multi-GPU or cleaner single-GPU):
    torchrun --nproc_per_node=1 ../scripts/02_run_benchmark.py [args...]
"""

import argparse
import json
import os
import sys
import time
import datetime
from pathlib import Path
import time

# ---------------------------------------------------------------------------
# Bootstrap: add LayerSkip to sys.path
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent

# Load experiment config
CONFIG_PATH = SCRIPT_DIR / "experiment_config.json"
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

sys.path.insert(0, str(SCRIPT_DIR / "LayerSkip"))

QUANT_DIR = (SCRIPT_DIR / CFG["paths"]["quantized_models_dir"]).resolve()
MODEL_CACHE = (SCRIPT_DIR / CFG["paths"]["model_cache_dir"]).resolve()

# ---------------------------------------------------------------------------
# Imports from LayerSkip (after sys.path is set)
# ---------------------------------------------------------------------------
import torch
import transformers
from arguments import Arguments
from self_speculation.generator_base import GenerationConfig
from benchmark import BenchmarkArguments, benchmark


# ---------------------------------------------------------------------------
# Model loading (handles all PTQ methods)
# ---------------------------------------------------------------------------


def load_model_for_ptq(model_id_or_path: str, ptq_method: str, device: str = "auto"):
    """
    Load a model according to the PTQ method.
    Returns (model, tokenizer).
    """
    print(f"\n[Loader] model={model_id_or_path}  ptq={ptq_method}")

    if ptq_method == "fp16":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id_or_path, cache_dir=str(MODEL_CACHE)
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.float16,
            device_map=device,
            cache_dir=str(MODEL_CACHE),
        )

    elif ptq_method == "fp32":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id_or_path, cache_dir=str(MODEL_CACHE)
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.float32,
            device_map=device,
            cache_dir=str(MODEL_CACHE),
        )

    elif ptq_method == "int8_bnb":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id_or_path, cache_dir=str(MODEL_CACHE)
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            load_in_8bit=True,
            device_map=device,
            cache_dir=str(MODEL_CACHE),
        )

    elif ptq_method == "awq":
        # AWQ models saved by llm-compressor are standard HF format
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id_or_path, 
            fix_mistral_regex=True
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            device_map=device,
        )

    elif ptq_method == "gptq":
        # GPTQ models saved by llm-compressor are standard HF format
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id_or_path, 
            fix_mistral_regex=True
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            device_map=device,
        )

    elif ptq_method == "smoothquant":
        # SmoothQuant model is saved as a standard HF model with W8A8Linear layers
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id_or_path)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.float16,
            device_map=device,
        )

    else:
        raise ValueError(f"Unknown ptq_method: {ptq_method}")

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# GPU memory helpers
# ---------------------------------------------------------------------------


def get_vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0.0


def reset_vram_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(args):
    # Resolve which model path to load
    if args.ptq_method in ["fp16", "fp32"]:
        model_path = args.model  # load directly from HF hub / local path
    elif args.ptq_method == "int8_bnb":
        # Check if a quantized dir with load_config.json exists
        quant_dir = QUANT_DIR / f"{args.model.split('/')[-1]}-int8_bnb"
        if quant_dir.exists():
            load_cfg_path = quant_dir / "load_config.json"
            if load_cfg_path.exists():
                with open(load_cfg_path) as f:
                    load_cfg = json.load(f)
                model_path = load_cfg.get("model_id", args.model)
            else:
                model_path = args.model
        else:
            model_path = args.model  # load from HF directly
    else:
        # Quantized checkpoint in quantized_models/<slug>-<method>
        slug = args.model.split("/")[-1]
        quant_dir = QUANT_DIR / f"{slug}-{args.ptq_method}"
        if not quant_dir.exists():
            print(f"ERROR: Quantized model not found at {quant_dir}")
            print(
                f"  Run: python scripts/01_quantize.py --model {args.model} --method {args.ptq_method}"
            )
            sys.exit(1)
        model_path = str(quant_dir)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Generate a unique run ID for this specific configuration
    # Add exit_layer and num_speculations to the ID for self_speculative so sweeps don't overwrite each other
    if args.generation_strategy == "self_speculative":
        config_suffix = f"_L{args.exit_layer}_K{args.num_speculations}"
    elif args.generation_strategy == "adaptive":
        config_suffix = f"_T{args.adaptive_threshold}"
    else:
        config_suffix = ""
        
    run_id = f"{args.run_type}__{args.model.split('/')[-1]}__{args.ptq_method}__{args.generation_strategy}{config_suffix}__{args.task}"
    
    # 1. Skip check & Resume Logic
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if FINAL result exists
    existing_results = list(out_dir.glob(f"{run_id}__*.json"))
    if existing_results:
        print(f"\n[Skip] Final results for {run_id} already exist. Skipping.")
        return

    # Check for partial progress file to resume
    progress_file = Path(f"progress_{run_id}.json")
    start_index = 0
    if progress_file.exists():
        try:
            with open(progress_file, "r") as f:
                data = json.load(f)
                if data:
                    start_index = data[-1]["index"] + 1
                    print(f"\n[Resume] Found partial progress ({len(data)} samples). Resuming from index {start_index}...")
        except Exception as e:
            print(f"Warning: Could not read progress file: {e}. Starting from scratch.")

    # Init distributed (required by LayerSkip setup)
    if not torch.distributed.is_initialized():
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12355"
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = "1"
        if "RANK" not in os.environ:
            os.environ["RANK"] = "0"
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = "0"

        backend = "cpu:gloo" if device == "cpu" else "cuda:nccl,cpu:gloo"
        torch.distributed.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(hours=48),
        )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        exit()  # LayerSkip only runs on rank 0

    # Load model
    reset_vram_stats()
    t_load_start = time.perf_counter()
    model, tokenizer = load_model_for_ptq(model_path, args.ptq_method, device=device)
    t_load_end = time.perf_counter()
    
    load_time_s = t_load_end - t_load_start
    vram_after_load_gb = get_vram_gb()
    print(
        f"[Loader] Model loaded in {load_time_s:.1f}s  |  VRAM: {vram_after_load_gb:.2f} GB"
    )

    # Build LayerSkip config objects
    generation_config = GenerationConfig(
        generation_strategy=args.generation_strategy,
        max_steps=args.max_steps,
        adaptive_threshold=args.adaptive_threshold, # Pass from CLI
        exit_layer=(
            args.exit_layer if args.generation_strategy == "self_speculative" else -1
        ),
        num_speculations=(
            args.num_speculations
            if args.generation_strategy == "self_speculative"
            else -1
        ),
        sample=args.sample,
    )
    benchmark_arguments = BenchmarkArguments(
        dataset=args.task,
        num_samples=args.num_samples,
        n_shot=args.n_shot,
        random_shuffle=True,
    )

    # Start energy meter
    sys.path.insert(0, str(SCRIPT_DIR))
    from energy_meter import EnergyMeter

    meter = EnergyMeter(
        device_idx=CFG["energy"]["gpu_device_idx"],
        sample_interval=CFG["energy"]["sample_interval_s"],
    )

    # Warmup
    print("[Benchmark] Warming up ...")
    reset_vram_stats()
    with torch.no_grad():
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        warmup_inputs = tokenizer(
            "Warmup prompt for benchmarking.", return_tensors="pt"
        )
        warmup_inputs = {k: v.to(device) for k, v in warmup_inputs.items()}
        model.generate(**warmup_inputs, max_new_tokens=10)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Run benchmark with energy measurement
    print(
        f"[Benchmark] Running {args.task} | {args.generation_strategy} | {args.num_samples} samples ..."
    )

    meter.start()
    t_bench_start = time.perf_counter()

    metric_result = benchmark(
        model=model,
        tokenizer=tokenizer,
        benchmark_arguments=benchmark_arguments,
        generation_config=generation_config,
        seed=42,
        run_id=run_id,  # Pass the ID for organized temp saving
        meter=meter,  # Pass the energy meter
        start_index=start_index, # RESUME SUPPORT
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_bench_end = time.perf_counter()
    meter.stop()

    bench_time_s = t_bench_end - t_bench_start
    peak_vram_gb = get_vram_gb()
    energy_summary = meter.summary()

    # Calculate total tokens from metrics
    # EvaluationMetrics uses Mean() for total_time and tokens_per_second.
    # Total tokens = avg_tps * avg_total_time * actual_samples_run
    avg_tps = metric_result.get("tokens_per_second", {}).get("mean", 0.0)
    avg_total_time = metric_result.get("total_time", {}).get("mean", 0.0)
    
    # We should use the number of samples actually processed
    num_processed = args.num_samples - start_index
    total_tokens_est = avg_tps * avg_total_time * num_processed

    joules_per_token = (
        (energy_summary["total_joules"] / total_tokens_est)
        if total_tokens_est > 0
        else 0.0
    )

    # Calculate final averages for efficiency
    tps = metric_result.get("tokens_per_second", {}).get("mean", 0.0)
    ms_per_tok = metric_result.get("time_per_token", {}).get("mean", 0.0) * 1000
    acceptance_rate = metric_result.get("acceptance_rate", {}).get("mean", None)

    # Build output JSON
    result = {
        "run_id": run_id,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "config": {
            "model": args.model,
            "model_path": str(model_path),
            "ptq_method": args.ptq_method,
            "ptq_description": CFG["ptq_methods"][args.ptq_method]["description"],
            "bits_weights": CFG["ptq_methods"][args.ptq_method]["bits_weights"],
            "bits_activations": CFG["ptq_methods"][args.ptq_method]["bits_activations"],
            "generation_strategy": args.generation_strategy,
            "exit_layer": (
                args.exit_layer
                if args.generation_strategy == "self_speculative"
                else None
            ),
            "num_speculations": (
                args.num_speculations
                if args.generation_strategy == "self_speculative"
                else None
            ),
            "task": args.task,
            "num_samples": args.num_samples,
            "n_shot": args.n_shot,
            "max_steps": args.max_steps,
            "sample": args.sample,
        },
        "quality_metrics": {
            "rouge_l": metric_result.get("predicted_text", {}).get("rouge-l"),
            "rouge_1": metric_result.get("predicted_text", {}).get("rouge-1"),
            "rouge_2": metric_result.get("predicted_text", {}).get("rouge-2"),
            "bleu": metric_result.get("predicted_text", {}).get("bleu_score"),
            "edit_distance": metric_result.get("predicted_text", {}).get("exact_match"),
        },
        "efficiency_metrics": {
            "tokens_per_second": round(tps, 3),
            "time_per_token_ms": round(ms_per_tok, 3),
            "prefill_tps": round(
                metric_result.get("prefill_tps", {}).get("mean", 0.0), 3
            ),
            "decode_tps": round(
                metric_result.get("decode_tps", {}).get("mean", 0.0), 3
            ),
            "total_benchmark_time_s": round(bench_time_s, 2),
            "acceptance_rate": round(acceptance_rate, 4) if acceptance_rate else None,
            "peak_vram_gb": round(peak_vram_gb, 3),
            "model_load_time_s": round(load_time_s, 2),
            "vram_after_load_gb": round(vram_after_load_gb, 3),
        },
        "energy_metrics": {
            "total_joules": energy_summary["total_joules"],
            "joules_per_token": round(joules_per_token, 4),
            "avg_power_watts": energy_summary["avg_power_watts"],
            "avg_gpu_util_percent": energy_summary["avg_gpu_util_percent"],
            "avg_cpu_util_percent": energy_summary["avg_cpu_util_percent"],
            "peak_power_watts": energy_summary["peak_power_watts"],
            "num_power_samples": energy_summary["num_power_samples"],
            "pynvml_available": energy_summary["pynvml_available"],
        },
        "raw_layerskip_metrics": metric_result,
    }

    # Save JSON
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"{run_id}__{ts_str}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved → {out_file}")
    print(f"  ROUGE-L:           {result['quality_metrics']['rouge_l']}")
    print(f"  Tokens/sec:        {result['efficiency_metrics']['tokens_per_second']}")
    print(f"  ms/token:          {result['efficiency_metrics']['time_per_token_ms']}")
    print(f"  Peak VRAM (GB):    {result['efficiency_metrics']['peak_vram_gb']}")
    print(f"  Energy (J):        {result['energy_metrics']['total_joules']}")
    print(f"  Energy (J/token):  {result['energy_metrics']['joules_per_token']}")
    print(f"  Avg Power (W):     {result['energy_metrics']['avg_power_watts']}")
    print(f"  Avg GPU Util:      {result['energy_metrics']['avg_gpu_util_percent']}%")
    print(f"  Avg CPU Util:      {result['energy_metrics']['avg_cpu_util_percent']}%")
    if acceptance_rate:
        print(f"  Acceptance rate:   {acceptance_rate:.2%}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Run PTQ + LayerSkip benchmark")

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/layerskip-llama2-70B",
        help="HuggingFace model ID (unquantized) or local path",
    )
    parser.add_argument(
        "--ptq_method",
        type=str,
        default="fp16",
        choices=list(CFG["ptq_methods"].keys()),
        help="PTQ method to use",
    )

    # Task
    parser.add_argument(
        "--task",
        type=str,
        default="cnn_dm_summarization",
        choices=[
            "cnn_dm_summarization",
            "xsum_summarization",
            "cnn_dm_lm",
            "human_eval",
        ],
        help="Benchmark task (LayerSkip dataset format)",
    )
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--n_shot", type=int, default=0)

    # Decoding
    parser.add_argument(
        "--generation_strategy",
        type=str,
        default="autoregressive",
        choices=["autoregressive", "self_speculative", "adaptive", "hf_native"],
    )
    parser.add_argument(
        "--adaptive_threshold",
        type=float,
        default=0.9,
        help="Confidence threshold for adaptive early exit",
    )
    parser.add_argument(
        "--exit_layer",
        type=int,
        default=30,
        help="Early exit layer for self-speculative decoding",
    )
    parser.add_argument(
        "--num_speculations",
        type=int,
        default=6,
        help="Number of draft tokens for self-speculative decoding",
    )
    parser.add_argument("--max_steps", type=int, default=256)
    parser.add_argument(
        "--sample",
        type=bool,
        default=False,
        help="Enable sampling (False = greedy, matches paper results)",
    )

    # Output
    parser.add_argument("--output_dir", type=str, default="../logs")
    parser.add_argument(
        "--run_type",
        type=str,
        default="evaluation",
        choices=["calibration", "evaluation"],
        help="Tag for the run (calibration or evaluation)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
