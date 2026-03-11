"""
05_plot_speculation_details.py
------------------------------
Generates detailed speculation figures from progress_*.json backup files.
Shows:
  1. acceptance_distribution.png  — Histogram of tokens accepted per step
  2. acceptance_over_time.png      — How acceptance changes during a single generation
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
    """Plots a histogram of how many tokens are accepted per step."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for p_file in progress_files:
        with open(p_file) as f:
            data = json.load(f)
        
        if not data: continue
        
        all_rates = []
        for sample in data:
            rates = sample.get("acceptance_rates")
            if rates:
                all_rates.extend(rates)
        
        if not all_rates: continue
        
        counts = Counter(all_rates)
        max_k = max(counts.keys()) if counts else 0
        x = np.arange(max_k + 1)
        y = [counts.get(i, 0) for i in x]
        # Normalize to probability
        total = sum(y)
        y_prob = [val / total for val in y]
        
        ptq = get_ptq_method(p_file.name)
        color = PTQ_COLORS.get(ptq, "#999")
        
        ax.plot(x, y_prob, marker='o', label=p_file.stem.replace("progress_", ""), color=color, alpha=0.7)
        ax.fill_between(x, y_prob, alpha=0.1, color=color)

    ax.set_xlabel("Tokens Accepted per Step (0 to K)")
    ax.set_ylabel("Probability")
    ax.set_title("Self-Speculation Acceptance Distribution")
    ax.set_xticks(x)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1))

    path = out_dir / "acceptance_distribution.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def plot_acceptance_over_tokens(progress_files, out_dir):
    """Plots how acceptance changes as more tokens are generated in a single sample."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for p_file in progress_files:
        with open(p_file) as f:
            data = json.load(f)
        
        if not data: continue
        
        # We'll take the first sample that has a decent length to show the trend
        target_sample = None
        for s in data:
            if s.get("acceptance_rates") and len(s.get("acceptance_rates")) > 20:
                target_sample = s
                break
        
        if not target_sample: 
            # If no long sample, take the first one available
            for s in data:
                if s.get("acceptance_rates"):
                    target_sample = s
                    break
        
        if not target_sample: continue
        
        rates = target_sample.get("acceptance_rates")
        x = np.arange(len(rates))
        
        ptq = get_ptq_method(p_file.name)
        color = PTQ_COLORS.get(ptq, "#999")
        
        # Smoothing for trend
        window = 5
        if len(rates) > window:
            smoothed = np.convolve(rates, np.ones(window)/window, mode='valid')
            ax.plot(x[window-1:], smoothed, label=p_file.stem.replace("progress_", ""), color=color, alpha=0.9)
        else:
            ax.plot(x, rates, label=p_file.stem.replace("progress_", ""), color=color, alpha=0.5)

    ax.set_xlabel("Step Number in Generation")
    ax.set_ylabel("Tokens Accepted (Moving Avg)")
    ax.set_title("Speculation Performance during Single Sample Generation")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1))

    path = out_dir / "acceptance_over_time.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="./figures")
    args = parser.parse_args()

    logs_path = Path(args.logs_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_files = list(logs_path.glob("progress_*.json"))
    if not progress_files:
        print(f"No progress_*.json files found in {logs_path}")
        return

    print(f"Found {len(progress_files)} progress files. Generating detailed plots...")
    
    plot_acceptance_distribution(progress_files, out_dir)
    plot_acceptance_over_tokens(progress_files, out_dir)
    
    print("\nDetailed figures saved.")

if __name__ == "__main__":
    main()
