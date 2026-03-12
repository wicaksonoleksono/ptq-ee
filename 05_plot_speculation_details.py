"""
05_plot_speculation_details.py
------------------------------
Generates detailed speculation figures from progress_*.json files.

Figures produced:
  1. acceptance_ratio_dist.png       — Distribution of per-step acceptance ratios
  2. token_origin_exit_layers.png    — Early-exit vs full-model token % per run
  3. per_token_exit_timeline.png     — Exit layer per token (first sample, first 100 tokens)
  4. hardware_profile_timeline.png   — GPU/CPU util and VRAM across samples

Usage:
    python 05_plot_speculation_details.py \\
        --logs_dir . \\
        --output_dir ./figures
"""

import json
import argparse
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})

DISPLAY_NAMES = {
    "fp16":        "Unquantized",
    "int8_bnb":    "INT8 (BnB)",
    "awq":         "AWQ",
    "gptq":        "GPTQ",
    "smoothquant": "SmoothQuant",
}

PTQ_COLORS = {
    "fp16":        "#4878CF",
    "int8_bnb":    "#6ACC65",
    "awq":         "#D65F5F",
    "gptq":        "#B47CC7",
    "smoothquant": "#C4AD66",
}

TASK_SHORT = {
    "cnn_dm_summarization": "CNN/DM",
    "xsum_summarization":   "XSum",
}


def get_ptq_method(filename: str) -> str:
    for m in PTQ_COLORS:
        if m in filename:
            return m
    return "fp16"


def make_label(p_file: Path) -> str:
    """Build a short readable label: 'Unquantized (CNN/DM)'."""
    stem = p_file.stem  # progress_layerskip-llama2-13B__fp16__self_speculative__cnn_dm_summarization
    stem = stem.replace("progress_layerskip-llama2-13B__", "").replace("__self_speculative", "")
    parts = stem.split("__")
    ptq = parts[0] if parts else "fp16"
    task = parts[1] if len(parts) > 1 else ""
    return f"{DISPLAY_NAMES.get(ptq, ptq)} ({TASK_SHORT.get(task, task)})"


# ---------------------------------------------------------------------------
# Figure 1: Distribution of per-step acceptance ratios
# ---------------------------------------------------------------------------

def plot_acceptance_distribution(progress_files: list, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    for p_file in progress_files:
        with open(p_file) as f:
            data = json.load(f)
        if not data:
            continue

        all_rates = []
        for sample in data:
            # Field is acceptance_rates_per_step (list of ratios per speculation step)
            rates = sample.get("acceptance_rates_per_step")
            if rates:
                all_rates.extend(rates)
        if not all_rates:
            continue

        ptq = get_ptq_method(p_file.name)
        ax.hist(all_rates, bins=20,
                label=make_label(p_file),
                color=PTQ_COLORS.get(ptq, "#999"),
                alpha=0.5, density=True)

    ax.set_xlabel("Acceptance Ratio per Speculation Step (0.0 to 1.0)")
    ax.set_ylabel("Density")
    ax.set_title("Self-Speculation Acceptance Ratio Distribution")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.35, 1))

    path = out_dir / "acceptance_ratio_dist.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 2: Early-exit vs full-model token percentage
# ---------------------------------------------------------------------------

def plot_exit_layers(progress_files: list, out_dir: Path):
    """Percentage of tokens from early exit (draft) vs full model per run."""
    fig, ax = plt.subplots(figsize=(11, 6))

    methods, early_pcts, full_pcts = [], [], []

    for p_file in progress_files:
        with open(p_file) as f:
            data = json.load(f)
        if not data:
            continue

        # exit_layers_per_token: layer index per token (e.g. 30=draft exit, 40=full model)
        all_layers = []
        for s in data:
            layers = s.get("exit_layers_per_token")
            if layers:
                all_layers.extend(layers)
        if not all_layers:
            continue

        counts = Counter(all_layers)
        max_layer = max(counts.keys())
        early_sum = sum(v for k, v in counts.items() if k < max_layer)
        full_sum = counts[max_layer]
        total = early_sum + full_sum

        methods.append(make_label(p_file))
        early_pcts.append(early_sum / total * 100)
        full_pcts.append(full_sum / total * 100)

    if not methods:
        print("  [exit layers] No exit_layers_per_token data, skipping.")
        return

    x = np.arange(len(methods))
    ax.bar(x, early_pcts, label="Early Exit Tokens (Draft Accepted)",
           color="#6ACC65", edgecolor="black", alpha=0.85)
    ax.bar(x, full_pcts, bottom=early_pcts, label="Full Model Tokens (Verification)",
           color="#4878CF", edgecolor="black", alpha=0.85)

    ax.set_ylabel("% of Total Tokens")
    ax.set_title("Token Origin: Early Exit vs Full Model Verification")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    path = out_dir / "token_origin_exit_layers.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 3: Per-token exit layer timeline (first sample, first 100 tokens)
# ---------------------------------------------------------------------------

def plot_token_timeline(progress_files: list, out_dir: Path):
    """Step plot of exit layer per token for the first sample of each run."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for p_file in progress_files:
        with open(p_file) as f:
            data = json.load(f)
        if not data:
            continue

        sample = data[0]
        layers = sample.get("exit_layers_per_token")
        if not layers:
            continue

        layers = layers[:100]
        x = np.arange(len(layers))
        ptq = get_ptq_method(p_file.name)
        color = PTQ_COLORS.get(ptq, "#999")
        ax.step(x, layers, where="post", label=make_label(p_file),
                color=color, alpha=0.8, linewidth=1.5)

    ax.set_xlabel("Token Index in Sequence")
    ax.set_ylabel("Exit Layer Index")
    ax.set_title("Per-Token Exit Layer Timeline (First Sample, First 100 Tokens)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))

    path = out_dir / "per_token_exit_timeline.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 4: Hardware utilization timeline across samples
# ---------------------------------------------------------------------------

def plot_hardware_timeline(progress_files: list, out_dir: Path):
    """GPU/CPU utilization and VRAM usage over the sample sequence."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for p_file in progress_files:
        with open(p_file) as f:
            data = json.load(f)
        if not data:
            continue

        indices = [s.get("index", i) for i, s in enumerate(data)]
        gpu_utils = [s.get("gpu_util_percent") for s in data]
        cpu_utils = [s.get("cpu_util_percent") for s in data]
        vram_mb = [s.get("gpu_mem_used_mb") for s in data]

        ptq = get_ptq_method(p_file.name)
        color = PTQ_COLORS.get(ptq, "#999")
        label = make_label(p_file)

        ax1.plot(indices, gpu_utils, label=f"{label} GPU",
                 color=color, linestyle="-", alpha=0.7)
        ax1.plot(indices, cpu_utils, label=f"{label} CPU",
                 color=color, linestyle="--", alpha=0.4)
        ax2.plot(indices, vram_mb, label=label, color=color, alpha=0.8)

    ax1.set_ylabel("Utilization %")
    ax1.set_title("Hardware Utilization Profile (Sample-by-Sample)")
    ax1.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2.set_ylabel("VRAM Used (MB)")
    ax2.set_xlabel("Sample Index")
    ax2.set_title("VRAM Memory Profile")
    ax2.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    ax2.grid(True, linestyle="--", alpha=0.3)

    path = out_dir / "hardware_profile_timeline.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", type=str, default=".",
                        help="Directory containing progress_*.json files")
    parser.add_argument("--output_dir", type=str, default="./figures")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_files = sorted(Path(args.logs_dir).glob("progress_*.json"))
    if not progress_files:
        print(f"No progress_*.json files found in {args.logs_dir}")
        return

    print(f"Found {len(progress_files)} progress file(s). Generating plots...")
    plot_acceptance_distribution(progress_files, out_dir)
    plot_exit_layers(progress_files, out_dir)
    plot_token_timeline(progress_files, out_dir)
    plot_hardware_timeline(progress_files, out_dir)
    print("\nDetailed speculation and hardware plots generated.")


if __name__ == "__main__":
    main()
