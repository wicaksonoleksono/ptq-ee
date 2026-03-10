"""
01_quantize.py
--------------
Unified quantization script for all PTQ methods.
Loads a base HuggingFace model, applies quantization, and saves the result.

Supported methods:
  fp16        — save model as float16 (reference baseline, no real quantization)
  int8_bnb    — bitsandbytes LLM.int8() naive quantization (W8A8)
  awq         — Activation-aware Weight Quantization W4A16
  gptq        — GPTQ second-order weight quantization W4A16
  smoothquant — SmoothQuant W8A8 (requires calibration data)

Usage:
    python scripts/01_quantize.py \\
        --model facebook/layerskip-llama2-70B \\
        --method awq \\
        --output_dir ./quantized_models

    # Quantize ALL models listed in experiment_config.json:
    python scripts/01_quantize.py --method awq --all_models
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "experiment_config.json"
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

QUANT_BASE = (Path(__file__).parent / CFG["paths"]["quantized_models_dir"]).resolve()
MODEL_CACHE = (Path(__file__).parent / CFG["paths"]["model_cache_dir"]).resolve()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def model_slug(model_id: str) -> str:
    """facebook/layerskip-llama2-70B → layerskip-llama2-70B"""
    return model_id.split("/")[-1]


def output_path(model_id: str, method: str) -> Path:
    return QUANT_BASE / f"{model_slug(model_id)}-{method}"


def save_metadata(out_dir: Path, model_id: str, method: str, extra: dict = None):
    meta = {
        "model": model_id,
        "ptq_method": method,
        "method_config": CFG["ptq_methods"][method],
        "quantized_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "output_dir": str(out_dir),
    }
    if extra:
        meta.update(extra)
    with open(out_dir / "quantization_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved → {out_dir / 'quantization_meta.json'}")


# ---------------------------------------------------------------------------
# fp16 — just re-save as float16 for a clean baseline checkpoint
# ---------------------------------------------------------------------------

def quantize_fp16(model_id: str, out_dir: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[fp16] Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(MODEL_CACHE))
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu",   # keep on CPU for saving
        cache_dir=str(MODEL_CACHE),
    )

    print(f"[fp16] Saving to {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    save_metadata(out_dir, model_id, "fp16")
    print("[fp16] Done.")


# ---------------------------------------------------------------------------
# int8_bnb — bitsandbytes LLM.int8() naive quantization
# NOTE: bitsandbytes quantizes on-the-fly at load time; we just save a marker
#       so the benchmark runner knows to load with load_in_8bit=True.
# ---------------------------------------------------------------------------

def quantize_int8_bnb(model_id: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # BnB int8 is load-time quantization — no offline quantization step.
    # We save a marker JSON so run_benchmark.py knows how to load this model.
    print(f"[int8_bnb] bitsandbytes quantization is applied at load time.")
    print(f"  Model to load: {model_id}")
    marker = {
        "model_id": model_id,
        "load_in_8bit": True,
        "note": "Load with: AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto')"
    }
    with open(out_dir / "load_config.json", "w") as f:
        json.dump(marker, f, indent=2)
    save_metadata(out_dir, model_id, "int8_bnb")
    print(f"[int8_bnb] Marker saved to {out_dir}")


# ---------------------------------------------------------------------------
# AWQ — Activation-aware Weight Quantization W4A16
# ---------------------------------------------------------------------------

def quantize_awq(model_id: str, out_dir: Path):
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.awq import AWQModifier
    except ImportError:
        print("ERROR: llmcompressor not installed. Run: pip install llmcompressor")
        sys.exit(1)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    method_cfg = CFG["ptq_methods"]["awq"]

    print(f"[AWQ] Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(MODEL_CACHE))
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=str(MODEL_CACHE),
    )

    recipe = [
        AWQModifier(
            ignore=["lm_head"],
            scheme="W4A16",
            targets=["Linear"],
            duo_scaling=False,
        ),
    ]

    print(f"[AWQ] Running quantization (W{method_cfg['bits_weights']}A{method_cfg['bits_activations']}) ...")
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        recipe=recipe,
        max_seq_length=512,
        num_calibration_samples=128,
    )

    print(f"[AWQ] Saving to {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    save_metadata(out_dir, model_id, "awq")
    print("[AWQ] Done.")


# ---------------------------------------------------------------------------
# GPTQ — W4A16
# ---------------------------------------------------------------------------

def quantize_gptq(model_id: str, out_dir: Path):
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier
    except ImportError:
        print("ERROR: llmcompressor not installed. Run: pip install llmcompressor")
        sys.exit(1)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    method_cfg = CFG["ptq_methods"]["gptq"]

    print(f"[GPTQ] Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(MODEL_CACHE))
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=str(MODEL_CACHE),
    )

    recipe = [
        GPTQModifier(
            ignore=["lm_head"],
            scheme="W4A16",
            targets=["Linear"],
        ),
    ]

    print(f"[GPTQ] Running quantization (W{method_cfg['bits_weights']}A{method_cfg['bits_activations']}) ...")
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        recipe=recipe,
        max_seq_length=512,
        num_calibration_samples=128,
    )

    print(f"[GPTQ] Saving to {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    save_metadata(out_dir, model_id, "gptq", {"bits": method_cfg["bits_weights"], "group_size": method_cfg["group_size"]})
    print("[GPTQ] Done.")


# ---------------------------------------------------------------------------
# SmoothQuant — W8A8
# ---------------------------------------------------------------------------

def quantize_smoothquant(model_id: str, out_dir: Path):
    try:
        from smoothquant.smooth import smooth_lm
        from smoothquant.fake_quant import W8A8Linear
    except ImportError:
        print("ERROR: smoothquant not installed. Run: pip install smoothquant")
        sys.exit(1)

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    method_cfg = CFG["ptq_methods"]["smoothquant"]
    alpha = method_cfg["alpha"]

    print(f"[SmoothQuant] Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(MODEL_CACHE))
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=str(MODEL_CACHE),
    )
    model.eval()

    # Collect activation stats on calibration data
    print("[SmoothQuant] Collecting activation statistics ...")
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    act_scales = {}

    def stat_hook(name):
        def hook(_, inp, out):
            hidden = inp[0].detach()
            # Track per-channel max absolute value
            scales = hidden.abs().view(-1, hidden.shape[-1]).max(dim=0).values
            if name not in act_scales:
                act_scales[name] = scales
            else:
                act_scales[name] = torch.max(act_scales[name], scales)
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(stat_hook(name)))

    samples = [s for s in data["text"] if len(s.strip()) > 50][:64]
    with torch.no_grad():
        for text in samples:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            model(**inputs)

    for h in hooks:
        h.remove()

    # Apply SmoothQuant migration
    print(f"[SmoothQuant] Smoothing with alpha={alpha} ...")
    smooth_lm(model, act_scales, alpha=alpha)

    # Quantize to W8A8
    print("[SmoothQuant] Applying W8A8 quantization ...")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            new_module = W8A8Linear.from_float(module, weight_quant="per_channel", act_quant="per_token")
            setattr(parent, child_name, new_module)

    print(f"[SmoothQuant] Saving to {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    save_metadata(out_dir, model_id, "smoothquant", {"alpha": alpha})
    print("[SmoothQuant] Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

METHODS = {
    "fp16": quantize_fp16,
    "int8_bnb": quantize_int8_bnb,
    "awq": quantize_awq,
    "gptq": quantize_gptq,
    "smoothquant": quantize_smoothquant,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize a model using a specified PTQ method")
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model ID (e.g. facebook/layerskip-llama2-70B)")
    parser.add_argument("--method", type=str, required=True, choices=list(METHODS.keys()),
                        help="PTQ method to apply")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save quantized model (default: ../quantized_models/<model>-<method>)")
    parser.add_argument("--all_models", action="store_true",
                        help="Quantize ALL models listed in experiment_config.json")
    return parser.parse_args()


def run_one(model_id: str, method: str, out_dir: Path):
    print(f"\n{'='*60}")
    print(f"Model  : {model_id}")
    print(f"Method : {method}")
    print(f"Output : {out_dir}")
    print(f"{'='*60}")

    if out_dir.exists() and (out_dir / "quantization_meta.json").exists():
        print(f"Already quantized. Skipping. (Delete {out_dir} to re-run)")
        return

    t0 = time.time()
    METHODS[method](model_id, out_dir)
    elapsed = time.time() - t0
    print(f"Finished in {elapsed/60:.1f} min")


def main():
    args = parse_args()

    models = CFG["models"]["all_layerskip"] if args.all_models else [args.model]
    if not args.all_models and args.model is None:
        print("ERROR: provide --model or use --all_models")
        sys.exit(1)

    for model_id in models:
        out = Path(args.output_dir) if args.output_dir else output_path(model_id, args.method)
        run_one(model_id, args.method, out)

    print("\nAll quantization tasks complete.")


if __name__ == "__main__":
    main()
