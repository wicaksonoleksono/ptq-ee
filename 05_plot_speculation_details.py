"""
05_plot_speculation_details.py
------------------------------
Generates detailed speculation figures from progress_*.json backup files.
"""

import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Style matching 04_plot_results.py
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})

PTQ_COLORS = {
    "fp16":        "#4878CF",
    "int8_bnb":    "#6ACC65",
    "awq":         "#D65F5F",
    "gptq":        "#B47CC7",
    "smoothquant": "#C4AD66",
}

def get_ptq_method(filename):
    for m in PTQ_COLORS.keys():
        if m in filename:
            return m
    return "other"

def plot_acceptance_distribution(progress_files, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for p_file in progress_files:
        with open(p_file) as f:
            data = json.load(f)
        if not data: continue
        all_rates = []
        for sample in data:
            rates = sample.get("acceptance_rates")
            if rates: all_rates.extend(rates)
        if not all_rates: continue
        
        # Plot distribution of acceptance ratios
        ax.hist(all_rates, bins=20, label=p_file.stem.replace("progress_", ""), 
                color=PTQ_COLORS.get(get_ptq_method(p_file.name), "#999"), alpha=0.5, density=True)

    ax.set_xlabel("Acceptance Ratio per Step (0.0 to 1.0)")
    ax.set_ylabel("Density")
    ax.set_title("Self-Speculation Acceptance Ratio Distribution")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1))
    path = out_dir / "acceptance_ratio_dist.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_exit_layers(progress_files, out_dir):
    """Visualizes the percentage of tokens coming from early exits vs full model."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods, early_counts, full_counts = [], [], []
    
    for p_file in progress_files:
        with open(p_file) as f:
            data = json.load(f)
        if not data: continue
        
        layers = []
        for s in data:
            if s.get("exit_layers"): layers.extend(s["exit_layers"])
        if not layers: continue
        
        counts = Counter(layers)
        # Early is anything that isn't the max layer found
        max_layer = max(counts.keys())
        early_sum = sum(v for k, v in counts.items() if k < max_layer)
        full_sum = counts[max_layer]
        total = early_sum + full_sum
        
        methods.append(p_file.stem.replace("progress_", "")[:20])
        early_counts.append(early_sum / total * 100)
        full_counts.append(full_sum / total * 100)

    x = np.arange(len(methods))
    ax.bar(x, early_counts, label="Early Exit Tokens", color="#6ACC65", edgecolor="black", alpha=0.8)
    ax.bar(x, full_counts, bottom=early_counts, label="Full Model Tokens", color="#4878CF", edgecolor="black", alpha=0.8)

    ax.set_ylabel("% of Total Tokens")
    ax.set_title("Token Origin: Early Exit vs Full Model Verification")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    path = out_dir / "token_origin_exit_layers.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def plot_token_timeline(progress_files, out_dir):
    """
    Plots a scatter/line timeline of which layer produced each token 
    for the first 100 tokens of a representative sample.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    for p_file in progress_files:
        with open(p_file) as f:
            data = json.load(f)
        if not data or len(data) == 0: continue
        
        # Pick the first sample
        sample = data[0]
        layers = sample.get("exit_layers_per_token")
        if not layers: continue
        
        # Limit to first 100 tokens for readability
        layers = layers[:100]
        x = np.arange(len(layers))
        
        ptq = get_ptq_method(p_file.name)
        color = PTQ_COLORS.get(ptq, "#999")
        
        ax.step(x, layers, where='post', label=p_file.stem.replace("progress_", "")[:25], 
                color=color, alpha=0.8, linewidth=1.5)

    ax.set_xlabel("Token Index in Sequence")
    ax.set_ylabel("Exit Layer Index")
    ax.set_title("Per-Token Exit Layer Timeline (Representative Sample)")
    ax.grid(True, linestyle="--", alpha=0.3)
    # Put legend outside
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))

    path = out_dir / "per_token_exit_timeline.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="./figures")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_files = list(Path(args.logs_dir).glob("progress_*.json"))
    
    if progress_files:
        plot_acceptance_distribution(progress_files, out_dir)
        plot_exit_layers(progress_files, out_dir)
        plot_token_timeline(progress_files, out_dir)
        print("Detailed speculation plots (including Per-Token Timeline) generated.")

if __name__ == "__main__":
    main()
