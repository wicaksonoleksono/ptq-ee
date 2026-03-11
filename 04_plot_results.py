"""
04_plot_results.py
------------------
Generates all benchmark figures from results_summary.json.

Figures produced:
  1. pareto_quality_vs_speed.png       — ROUGE-L vs tokens/sec (Pareto curve)
  2. energy_per_token_bar.png          — Joules/token grouped by PTQ method
  3. vram_footprint_bar.png            — Peak VRAM by PTQ method
  4. speedup_bar.png                   — Speedup over fp16+autoregressive
  5. acceptance_rate_scatter.png       — Self-speculative acceptance rate by config
  6. quality_radar.png                 — Multi-metric radar chart

Usage:
    python scripts/04_plot_results.py \\
        --results_json ./results/results_summary.json \\
        --output_dir ./figures
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe for servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Color palette for PTQ methods
PTQ_COLORS = {
    "fp16":        "#4878CF",   # blue
    "int8_bnb":    "#6ACC65",   # green
    "awq":         "#D65F5F",   # red
    "gptq":        "#B47CC7",   # purple
    "smoothquant": "#C4AD66",   # gold
}

STRATEGY_MARKERS = {
    "autoregressive":          "o",
    "self_speculative_small":  "^",
    "self_speculative_medium": "s",
    "self_speculative_large":  "D",
}
STRATEGY_SHORT = {
    "autoregressive":          "AR",
    "self_speculative_small":  "SS-20",
    "self_speculative_medium": "SS-30",
    "self_speculative_large":  "SS-40",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})


def load_results(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    return data["runs"]


def ptq_label(r: dict) -> str:
    m = r.get("ptq_method", "")
    w = r.get("bits_w", "?")
    a = r.get("bits_a", "?")
    return f"{m}\n(W{w}A{a})"


def strategy_label(r: dict) -> str:
    s = r.get("strategy", "")
    el = r.get("exit_layer")
    ns = r.get("num_speculations")
    if s == "self_speculative" and el and ns:
        return f"SS(L{el},K{ns})"
    return s


# ---------------------------------------------------------------------------
# Figure 1: Pareto — Quality vs Speed
# ---------------------------------------------------------------------------

def plot_pareto(runs: list, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 6))

    for r in runs:
        tps = r.get("tokens_per_sec")
        rl = r.get("rouge_l")
        if tps is None or rl is None:
            continue
        ptq = r.get("ptq_method", "fp16")
        strat = r.get("strategy", "autoregressive")

        color = PTQ_COLORS.get(ptq, "#999999")
        marker = "o" if strat == "autoregressive" else "^"
        ax.scatter(tps, rl, c=color, marker=marker, s=90, alpha=0.85, edgecolors="black", linewidths=0.5)

        label = f"{ptq}\n{strategy_label(r)}"
        ax.annotate(label, (tps, rl), textcoords="offset points", xytext=(6, 4),
                    fontsize=7, alpha=0.8)

    # Legend for PTQ colors
    patches = [mpatches.Patch(color=c, label=m) for m, c in PTQ_COLORS.items()]
    patches += [
        plt.scatter([], [], marker="o", color="grey", label="Autoregressive"),
        plt.scatter([], [], marker="^", color="grey", label="Self-Speculative"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8)

    ax.set_xlabel("Tokens / second  (higher = faster)")
    ax.set_ylabel("ROUGE-L  (higher = better quality)")
    ax.set_title("Quality–Efficiency Pareto: PTQ × LayerSkip Decoding Strategy")
    ax.grid(True, linestyle="--", alpha=0.4)

    path = out_dir / "pareto_quality_vs_speed.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 2: Energy per Token Bar Chart
# ---------------------------------------------------------------------------

def plot_energy_bar(runs: list, out_dir: Path):
    # Group by (ptq_method, strategy), average joules_per_token
    from collections import defaultdict
    groups = defaultdict(list)
    for r in runs:
        j = r.get("joules_per_token")
        if j is not None:
            key = (r.get("ptq_method", "fp16"), r.get("strategy", "autoregressive"))
            groups[key].append(j)

    if not groups:
        print("  [energy bar] No joules_per_token data, skipping.")
        return

    labels, values, colors = [], [], []
    for (ptq, strat), vals in sorted(groups.items()):
        labels.append(f"{ptq}\n{strat[:3].upper()}")
        values.append(np.mean(vals))
        colors.append(PTQ_COLORS.get(ptq, "#aaaaaa"))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, fmt="%.3f J", padding=3, fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Joules / token  (lower = more efficient)")
    ax.set_title("Energy per Token by PTQ Method × Decoding Strategy")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    path = out_dir / "energy_per_token_bar.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 3: VRAM Footprint
# ---------------------------------------------------------------------------

def plot_vram_bar(runs: list, out_dir: Path):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in runs:
        v = r.get("peak_vram_gb")
        if v and v > 0:
            groups[r.get("ptq_method", "fp16")].append(v)

    if not groups:
        print("  [vram bar] No peak_vram_gb data, skipping.")
        return

    ptq_methods = sorted(groups.keys())
    values = [np.mean(groups[m]) for m in ptq_methods]
    colors = [PTQ_COLORS.get(m, "#aaaaaa") for m in ptq_methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(ptq_methods, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, fmt="%.1f GB", padding=3, fontsize=9)
    ax.set_ylabel("Peak GPU Memory (GB)  (lower = better)")
    ax.set_title("GPU Memory Footprint by PTQ Method")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    path = out_dir / "vram_footprint_bar.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 4: Speedup over fp16 + autoregressive baseline
# ---------------------------------------------------------------------------

def plot_speedup_bar(runs: list, out_dir: Path):
    labeled = [(r.get("ptq_method", "?"), strategy_label(r), r.get("speedup_vs_fp16_ar"))
               for r in runs if r.get("speedup_vs_fp16_ar") is not None]
    if not labeled:
        print("  [speedup bar] No speedup data, skipping.")
        return

    from collections import defaultdict
    groups = defaultdict(list)
    for ptq, strat, spd in labeled:
        groups[(ptq, strat)].append(spd)

    labels = [f"{p}\n{s}" for p, s in groups]
    values = [np.mean(v) for v in groups.values()]
    colors = [PTQ_COLORS.get(p, "#aaaaaa") for p, _ in groups]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, fmt="%.2fx", padding=3, fontsize=8)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Baseline (fp16 AR)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Speedup (×)  over fp16 + autoregressive")
    ax.set_title("Inference Speedup by PTQ × Decoding Strategy")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    path = out_dir / "speedup_bar.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 5: Acceptance Rate scatter (self-speculative only)
# ---------------------------------------------------------------------------

def plot_acceptance_rate(runs: list, out_dir: Path):
    spec_runs = [r for r in runs
                 if r.get("strategy") == "self_speculative"
                 and r.get("acceptance_rate") is not None
                 and r.get("exit_layer") is not None]
    if not spec_runs:
        print("  [acceptance rate] No self-speculative data, skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for r in spec_runs:
        ptq = r.get("ptq_method", "fp16")
        el = r.get("exit_layer")
        ar = r.get("acceptance_rate")
        ax.scatter(el, ar, c=PTQ_COLORS.get(ptq, "#999"), s=80,
                   edgecolors="black", linewidths=0.5, label=ptq)

    # De-dup legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.set_xlabel("Exit Layer")
    ax.set_ylabel("Acceptance Rate")
    ax.set_title("Self-Speculative Acceptance Rate vs Exit Layer")
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.4)

    path = out_dir / "acceptance_rate_scatter.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 6: Heatmap — Quality (ROUGE-L) × PTQ × Decoding
# ---------------------------------------------------------------------------

def plot_heatmap(runs: list, out_dir: Path):
    from collections import defaultdict
    cell = defaultdict(list)
    ptq_methods = []
    strategies = []
    for r in runs:
        rl = r.get("rouge_l")
        if rl is None:
            continue
        p = r.get("ptq_method", "fp16")
        s = strategy_label(r)
        cell[(p, s)].append(rl)
        if p not in ptq_methods:
            ptq_methods.append(p)
        if s not in strategies:
            strategies.append(s)

    if not cell:
        print("  [heatmap] No ROUGE-L data, skipping.")
        return

    ptq_methods = sorted(set(ptq_methods))
    strategies = sorted(set(strategies))
    data = np.zeros((len(ptq_methods), len(strategies)))
    for i, p in enumerate(ptq_methods):
        for j, s in enumerate(strategies):
            vals = cell.get((p, s), [])
            data[i, j] = np.mean(vals) if vals else np.nan

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=30, ha="right")
    ax.set_yticks(range(len(ptq_methods)))
    ax.set_yticklabels(ptq_methods)
    ax.set_title("ROUGE-L Heatmap: PTQ Method × Decoding Strategy")
    fig.colorbar(im, ax=ax, label="ROUGE-L")

    for i in range(len(ptq_methods)):
        for j in range(len(strategies)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8,
                        color="black" if val > 0.3 else "white")

    path = out_dir / "quality_heatmap.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def plot_energy_spikes(logs_dir: Path, out_dir: Path):
    \"\"\"
    Reads progress_*.json backup files to plot cumulative Joules across the run
    to show energy 'spikes' and slopes for different methods.
    \"\"\"
    fig, ax = plt.subplots(figsize=(10, 6))

    found_files = list(logs_dir.glob(\"progress_*.json\"))
    if not found_files:
        # Check current dir as well
        found_files = list(Path(\".\").glob(\"progress_*.json\"))

    if not found_files:
        print(\"  [energy spikes] No progress backup files found, skipping spike plot.\")
        return

    for p_file in found_files:
        with open(p_file) as f:
            data = json.load(f)

        if not data: continue

        indices = [i.get(\"index\") for i in data]
        joules = [i.get(\"joules\") for i in data]

        # Determine method from filename
        # progress_llama2-13B__awq__self_speculative__cnn_dm_summarization.json
        label = p_file.stem.replace(\"progress_\", \"\").replace(\"__cnn_dm_summarization\", \"\")

        ax.plot(indices, joules, label=label, alpha=0.8, linewidth=1.5)

    ax.set_xlabel(\"Sample Index\")
    ax.set_ylabel(\"Cumulative Energy (Joules)\")
    ax.set_title(\"Energy Consumption Profile: PTQ × Decoding (Sample-by-Sample)\")
    ax.legend(fontsize=7, loc=\"upper left\", bbox_to_anchor=(1, 1))
    ax.grid(True, linestyle=\"--\", alpha=0.3)

    path = out_dir / \"energy_spikes_profile.png\"
    fig.tight_layout()
    fig.savefig(path, bbox_inches=\"tight\")
    plt.close(fig)
    print(f\"  Saved: {path}\")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--results_json\", type=str, default=\"../results/results_summary.json\")
    parser.add_argument(\"--logs_dir\", type=str, default=\"../logs\")
    parser.add_argument(\"--output_dir\", type=str, default=\"../figures\")
    args = parser.parse_args()

    res_path = Path(args.results_json)
    if not res_path.exists():
        print(f\"ERROR: {res_path} not found. Run 03_collect_results.py first.\")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(args.logs_dir)

    runs = load_results(str(res_path))
    print(f\"Plotting results from {len(runs)} runs...\")

    plot_pareto(runs, out_dir)
    plot_energy_bar(runs, out_dir)
    plot_vram_bar(runs, out_dir)
    plot_speedup_bar(runs, out_dir)
    plot_acceptance_rate(runs, out_dir)
    plot_heatmap(runs, out_dir)

    # New energy spike plot
    plot_energy_spikes(logs_dir, out_dir)

    print(\"\\nAll figures saved.\")



if __name__ == "__main__":
    main()
