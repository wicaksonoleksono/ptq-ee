"""
04_plot_results.py
------------------
Generates all benchmark figures from results/results_summary.json.

Figures produced:
  1. pareto_quality_vs_speed.png       — ROUGE-L vs tokens/sec, colored by PTQ, marker by task
  2. energy_per_token_bar.png          — Joules/token grouped by PTQ × task
  3. vram_footprint_bar.png            — Peak VRAM by PTQ method
  4. speedup_bar.png                   — Inference throughput (tokens/sec) by PTQ × task
  5. acceptance_rate_scatter.png       — Acceptance rate by PTQ × task
  6. acceptance_sweep_lines.png        — Acceptance rate vs exit layer per PTQ method
  7. quality_heatmap.png               — ROUGE-L heatmap: PTQ × task
  8. hardware_util_bar.png             — GPU/CPU utilization by PTQ method
  9. energy_spikes_profile.png         — Cumulative energy over samples from progress_*.json

Usage:
    python 04_plot_results.py \\
        --results_json ./results/results_summary.json \\
        --scripts_dir . \\
        --output_dir ./figures
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


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

TASK_MARKERS = {
    "cnn_dm_summarization": "o",
    "xsum_summarization":   "s",
}
TASK_SHORT = {
    "cnn_dm_summarization": "CNN/DM",
    "xsum_summarization":   "XSum",
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


def display_name(ptq: str) -> str:
    return DISPLAY_NAMES.get(ptq, ptq)


def load_results(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    runs = data["runs"]
    # Deduplicate by run_id (keep first occurrence)
    seen = set()
    deduped = []
    for r in runs:
        rid = r.get("run_id")
        if rid not in seen:
            seen.add(rid)
            deduped.append(r)
    if len(deduped) < len(runs):
        print(f"  [load] Removed {len(runs) - len(deduped)} duplicate run(s).")
    return deduped


def ptq_label(r: dict) -> str:
    m = r.get("ptq_method", "")
    w = r.get("bits_w", "?")
    a = r.get("bits_a", "?")
    return f"{display_name(m)}\n(W{w}A{a})"


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
        task = r.get("task", "")
        color = PTQ_COLORS.get(ptq, "#999999")
        marker = TASK_MARKERS.get(task, "o")
        ax.scatter(tps, rl, c=color, marker=marker, s=90, alpha=0.85,
                   edgecolors="black", linewidths=0.5)
        ax.annotate(display_name(ptq), (tps, rl),
                    textcoords="offset points", xytext=(6, 4), fontsize=7, alpha=0.8)

    ptq_patches = [mpatches.Patch(color=c, label=display_name(m))
                   for m, c in PTQ_COLORS.items()]
    task_handles = [
        plt.scatter([], [], marker=TASK_MARKERS[t], color="grey", label=TASK_SHORT[t])
        for t in TASK_MARKERS
    ]
    ax.legend(handles=ptq_patches + task_handles, loc="lower right", fontsize=8)
    ax.set_xlabel("Tokens / second  (higher = faster)")
    ax.set_ylabel("ROUGE-L  (higher = better quality)")
    ax.set_title("Quality–Efficiency Pareto: PTQ Method × Task")
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
    groups = defaultdict(list)
    for r in runs:
        j = r.get("joules_per_token")
        if j is not None:
            key = (r.get("ptq_method", "fp16"), r.get("task", ""))
            groups[key].append(j)

    if not groups:
        print("  [energy bar] No joules_per_token data, skipping.")
        return

    labels, values, colors = [], [], []
    for (ptq, task), vals in sorted(groups.items()):
        labels.append(f"{display_name(ptq)}\n{TASK_SHORT.get(task, task)}")
        values.append(np.mean(vals))
        colors.append(PTQ_COLORS.get(ptq, "#aaaaaa"))

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, fmt="%.3f J", padding=3, fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Joules / token  (lower = more efficient)")
    ax.set_title("Energy per Token by PTQ Method × Task")
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
    groups = defaultdict(list)
    for r in runs:
        v = r.get("gpu_mem_used_mb")
        if v and v > 0:
            groups[r.get("ptq_method", "fp16")].append(v / 1024.0)

    if not groups:
        print("  [vram bar] No gpu_mem_used_mb data, skipping.")
        return

    ptq_methods = sorted(groups.keys())
    values = [np.mean(groups[m]) for m in ptq_methods]
    colors = [PTQ_COLORS.get(m, "#aaaaaa") for m in ptq_methods]
    labels = [display_name(m) for m in ptq_methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, fmt="%.1f GB", padding=3, fontsize=9)
    ax.set_ylabel("Avg GPU Memory (GB) used during inference")
    ax.set_title("VRAM Footprint by PTQ Method")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    path = out_dir / "vram_footprint_bar.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 4: Inference Throughput (tokens/sec) by PTQ × Task
# Note: speedup_vs_fp16_ar is 1.0 for all runs (single-strategy experiment),
#       so we visualize raw throughput instead.
# ---------------------------------------------------------------------------

def plot_speedup_bar(runs: list, out_dir: Path):
    groups = defaultdict(list)
    for r in runs:
        tps = r.get("tokens_per_sec")
        if tps is not None:
            key = (r.get("ptq_method", "fp16"), r.get("task", ""))
            groups[key].append(tps)

    if not groups:
        print("  [throughput bar] No tokens_per_sec data, skipping.")
        return

    labels, values, colors = [], [], []
    for (ptq, task), vals in sorted(groups.items()):
        labels.append(f"{display_name(ptq)}\n{TASK_SHORT.get(task, task)}")
        values.append(np.mean(vals))
        colors.append(PTQ_COLORS.get(ptq, "#aaaaaa"))

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Tokens / second  (higher = faster)")
    ax.set_title("Inference Throughput by PTQ Method × Task")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    path = out_dir / "speedup_bar.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 5: Acceptance Rate by PTQ × Task
# ---------------------------------------------------------------------------

def plot_acceptance_rate(runs: list, out_dir: Path):
    groups = defaultdict(list)
    for r in runs:
        ar = r.get("acceptance_rate")
        if ar is not None:
            key = (r.get("ptq_method", "fp16"), r.get("task", ""))
            groups[key].append(ar)

    if not groups:
        print("  [acceptance rate] No acceptance_rate data, skipping.")
        return

    labels, values, colors = [], [], []
    for (ptq, task), vals in sorted(groups.items()):
        labels.append(f"{display_name(ptq)}\n{TASK_SHORT.get(task, task)}")
        values.append(np.mean(vals))
        colors.append(PTQ_COLORS.get(ptq, "#aaaaaa"))

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Acceptance Rate  (higher = more tokens accepted from draft)")
    ax.set_title("Self-Speculative Acceptance Rate by PTQ Method × Task")
    ax.set_ylim(0, 0.75)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    path = out_dir / "acceptance_rate_scatter.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 6: Quality Heatmap — ROUGE-L × PTQ × Task
# ---------------------------------------------------------------------------

def plot_heatmap(runs: list, out_dir: Path):
    cell = defaultdict(list)
    ptq_set, task_set = set(), set()
    for r in runs:
        rl = r.get("rouge_l")
        if rl is None:
            continue
        p = r.get("ptq_method", "fp16")
        t = r.get("task", "")
        cell[(p, t)].append(rl)
        ptq_set.add(p)
        task_set.add(t)

    if not cell:
        print("  [heatmap] No ROUGE-L data, skipping.")
        return

    ptq_methods = sorted(ptq_set)
    tasks = sorted(task_set)
    data = np.zeros((len(ptq_methods), len(tasks)))
    for i, p in enumerate(ptq_methods):
        for j, t in enumerate(tasks):
            vals = cell.get((p, t), [])
            data[i, j] = np.mean(vals) if vals else np.nan

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.08, vmax=0.16)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_SHORT.get(t, t) for t in tasks])
    ax.set_yticks(range(len(ptq_methods)))
    ax.set_yticklabels([display_name(m) for m in ptq_methods])
    ax.set_title("ROUGE-L Heatmap: PTQ Method × Task")
    fig.colorbar(im, ax=ax, label="ROUGE-L")

    for i in range(len(ptq_methods)):
        for j in range(len(tasks)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=9)

    path = out_dir / "quality_heatmap.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 7: Hardware Utilization (GPU / CPU %)
# ---------------------------------------------------------------------------

def plot_utility_bar(runs: list, out_dir: Path):
    groups = defaultdict(lambda: {"gpu": [], "cpu": []})
    for r in runs:
        g = r.get("gpu_util_percent")
        c = r.get("cpu_util_percent")
        if g is not None or c is not None:
            ptq = r.get("ptq_method", "fp16")
            groups[ptq]["gpu"].append(g or 0)
            groups[ptq]["cpu"].append(c or 0)

    if not groups:
        print("  [utility bar] No util data, skipping.")
        return

    methods = sorted(groups.keys())
    gpu_means = [np.mean(groups[m]["gpu"]) for m in methods]
    cpu_means = [np.mean(groups[m]["cpu"]) for m in methods]
    labels = [display_name(m) for m in methods]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, gpu_means, width, label="GPU Util %",
           color="#6ACC65", edgecolor="black", linewidth=0.6)
    ax.bar(x + width/2, cpu_means, width, label="CPU Util %",
           color="#4878CF", edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Utilization %")
    ax.set_title("Average Hardware Utilization by PTQ Method")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    path = out_dir / "hardware_util_bar.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 8: Acceptance Rate vs Exit Layer (sweep)
# ---------------------------------------------------------------------------

def plot_acceptance_sweep(runs: list, out_dir: Path):
    data = defaultdict(lambda: defaultdict(list))

    for r in runs:
        ptq = r.get("ptq_method", "fp16")
        el = r.get("exit_layer")
        ar = r.get("acceptance_rate")
        if el is not None and ar is not None:
            data[ptq][el].append(ar)

    if not data:
        print("  [acceptance sweep] No sweep data, skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for ptq, layers in sorted(data.items()):
        sorted_layers = sorted(layers.keys())
        avg_ar = [np.mean(layers[l]) for l in sorted_layers]
        color = PTQ_COLORS.get(ptq, "#999")
        ax.plot(sorted_layers, avg_ar, marker="o", label=display_name(ptq),
                color=color, linewidth=2, markersize=8)

    ax.set_xlabel("Exit Layer Index")
    ax.set_ylabel("Mean Acceptance Rate")
    ax.set_title("Acceptance Rate by Exit Layer and PTQ Method")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    path = out_dir / "acceptance_sweep_lines.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 9: Cumulative Energy Profile from progress_*.json
# ---------------------------------------------------------------------------

def _make_progress_label(p_file: Path) -> str:
    """Build a short readable label from a progress filename."""
    stem = p_file.stem  # e.g. progress_layerskip-llama2-13B__fp16__self_speculative__cnn_dm_summarization
    stem = stem.replace("progress_layerskip-llama2-13B__", "").replace("__self_speculative", "")
    # stem is now e.g. "fp16__cnn_dm_summarization"
    parts = stem.split("__")
    ptq = parts[0] if parts else stem
    task = parts[1] if len(parts) > 1 else ""
    task_short = TASK_SHORT.get(task, task)
    return f"{display_name(ptq)} ({task_short})"


def plot_energy_spikes(scripts_dir: Path, out_dir: Path):
    """Cumulative energy (Joules) over sample index from progress_*.json files."""
    found_files = sorted(scripts_dir.glob("progress_*.json"))
    if not found_files:
        print("  [energy spikes] No progress_*.json files found, skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for p_file in found_files:
        with open(p_file) as f:
            data = json.load(f)
        if not data:
            continue

        indices = [s.get("index", i) for i, s in enumerate(data)]
        # joules_this_sample is per sample; compute cumulative sum
        sample_joules = [s.get("joules_this_sample", 0) or 0 for s in data]
        cumulative = list(np.cumsum(sample_joules))

        label = _make_progress_label(p_file)
        ptq_key = p_file.stem.split("__")[1] if "__" in p_file.stem else "fp16"
        color = PTQ_COLORS.get(ptq_key, "#999999")
        linestyle = "-" if "cnn_dm" in p_file.name else "--"
        ax.plot(indices, cumulative, label=label, color=color,
                alpha=0.85, linewidth=1.5, linestyle=linestyle)

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Cumulative Energy (Joules)")
    ax.set_title("Energy Consumption Profile: PTQ Method × Task\n(solid = CNN/DM, dashed = XSum)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(True, linestyle="--", alpha=0.3)

    path = out_dir / "energy_spikes_profile.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_quantization_early_exit_effects(logs_dir: Path, out_dir: Path):
    """
    Plots the effect of Quantization on Early Exit dynamics using the calibration sweep logs.
    Shows Exit Layer vs: Acceptance Rate, Joules per Token, and Inference Speed (Decode TPS).
    """
    calib_dir = logs_dir / "calibration"
    if not calib_dir.exists():
        print("  [early exit sweep] Calibration directory not found, skipping sweep plots.")
        return
        
    found_files = list(calib_dir.glob("*.json"))
    if not found_files:
        return
        
    # Group data by (ptq_method, exit_layer)
    from collections import defaultdict
    data = defaultdict(lambda: defaultdict(lambda: {"ar": [], "jpt": [], "tps": []}))
    
    for p_file in found_files:
        with open(p_file) as f:
            d = json.load(f)
        
        cfg = d.get("config", {})
        if cfg.get("generation_strategy") != "self_speculative":
            continue
            
        ptq = cfg.get("ptq_method", "fp16")
        el = cfg.get("exit_layer")
        
        eff = d.get("efficiency_metrics", {})
        en = d.get("energy_metrics", {})
        
        ar = eff.get("acceptance_rate")
        tps = eff.get("decode_tps")
        jpt = en.get("joules_per_token")
        
        if el is not None:
            if ar is not None: data[ptq][el]["ar"].append(ar)
            if tps is not None: data[ptq][el]["tps"].append(tps)
            if jpt is not None: data[ptq][el]["jpt"].append(jpt)

    if not data:
        return

    # Create 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ptq, layers in sorted(data.items()):
        sorted_layers = sorted(layers.keys())
        color = PTQ_COLORS.get(ptq, "#999")
        
        y_ar = [np.mean(layers[l]["ar"]) for l in sorted_layers if layers[l]["ar"]]
        y_tps = [np.mean(layers[l]["tps"]) for l in sorted_layers if layers[l]["tps"]]
        y_jpt = [np.mean(layers[l]["jpt"]) for l in sorted_layers if layers[l]["jpt"]]
        
        # Only plot if we have full matching data (simplification)
        if len(y_ar) == len(sorted_layers):
            axes[0].plot(sorted_layers, y_ar, marker='o', label=ptq, color=color, linewidth=2)
        if len(y_tps) == len(sorted_layers):
            axes[1].plot(sorted_layers, y_tps, marker='s', label=ptq, color=color, linewidth=2)
        if len(y_jpt) == len(sorted_layers):
            axes[2].plot(sorted_layers, y_jpt, marker='^', label=ptq, color=color, linewidth=2)

    axes[0].set_title("Acceptance Rate vs Exit Layer")
    axes[0].set_ylabel("Acceptance Rate")
    axes[0].set_xlabel("Exit Layer")
    
    axes[1].set_title("Inference Speed vs Exit Layer")
    axes[1].set_ylabel("Decode TPS")
    axes[1].set_xlabel("Exit Layer")
    
    axes[2].set_title("Energy vs Exit Layer")
    axes[2].set_ylabel("Joules / Token")
    axes[2].set_xlabel("Exit Layer")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        
    plt.suptitle("The Effect of PTQ on Early Exit Dynamics", fontsize=16, y=1.05)
    
    path = out_dir / "quantization_early_exit_effects.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def plot_calibration_sweeps(logs_dir: Path, out_dir: Path):
    """
    Plots Score vs. Exit Layer for every method/task to show the calibration frontier.
    Uses the CSV files in logs/sweeps/.
    """
    import pandas as pd
    sweep_dir = logs_dir / "sweeps"
    if not sweep_dir.exists():
        print("  [calibration sweep] Sweep directory not found, skipping frontier plots.")
        return
        
    csv_files = list(sweep_dir.glob("sweep_*.csv"))
    if not csv_files:
        return

    # Group by task
    tasks_data = defaultdict(list)
    for f in csv_files:
        df = pd.read_csv(f)
        task = df['task'].iloc[0] if 'task' in df.columns else "unknown"
        tasks_data[task].append(df)

    for task, dfs in tasks_data.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for df in dfs:
            ptq = df['method'].iloc[0]
            color = PTQ_COLORS.get(ptq, "#999")
            
            # Sort by exit_layer
            df = df.sort_values('exit_layer')
            
            ax.plot(df['exit_layer'], df['score'], marker='o', label=display_name(ptq), color=color, linewidth=2)
            
        ax.set_title(f"Calibration Frontier: Quality vs. Exit Layer ({task})")
        ax.set_xlabel("Exit Layer Index")
        ax.set_ylabel("Validation Score (ROUGE/Acc)")
        ax.axhline(y=0.0, color='black', alpha=0.1) # base
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        
        # Save per task
        safe_task = task.replace("/", "_")
        path = out_dir / f"calibration_frontier_{safe_task}.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")

def plot_efficiency_dashboard(runs: list, out_dir: Path):
    """
    Creates a 4-panel dashboard showing the most critical research insights.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Quality vs TPS (Pareto)
    for r in runs:
        tps, rl = r.get("tokens_per_sec"), r.get("rouge_l")
        if tps and rl:
            axes[0,0].scatter(tps, rl, color=PTQ_COLORS.get(r['ptq_method'], "#999"), 
                             marker=TASK_MARKERS.get(r['task'], "o"), s=100, alpha=0.7)
    axes[0,0].set_title("Quality vs. Throughput")
    axes[0,0].set_xlabel("Tokens / Sec")
    axes[0,0].set_ylabel("ROUGE-L")

    # 2. TPS vs Energy
    for r in runs:
        tps, jpt = r.get("tokens_per_sec"), r.get("joules_per_token")
        if tps and jpt:
            axes[0,1].scatter(tps, jpt, color=PTQ_COLORS.get(r['ptq_method'], "#999"), 
                             marker=TASK_MARKERS.get(r['task'], "o"), s=100, alpha=0.7)
    axes[0,1].set_title("Throughput vs. Energy Cost")
    axes[0,1].set_xlabel("Tokens / Sec")
    axes[0,1].set_ylabel("Joules / Token")

    # 3. Acceptance Rate by Method
    groups = defaultdict(list)
    for r in runs:
        ar = r.get("acceptance_rate")
        if ar: groups[r['ptq_method']].append(ar)
    
    if groups:
        labels = [display_name(m) for m in sorted(groups.keys())]
        values = [np.mean(groups[m]) for m in sorted(groups.keys())]
        colors = [PTQ_COLORS.get(m, "#999") for m in sorted(groups.keys())]
        axes[1,0].bar(labels, values, color=colors, edgecolor='black')
        axes[1,0].set_title("Mean Acceptance Rate by Method")
        axes[1,0].set_ylabel("Acceptance %")

    # 4. VRAM vs Energy
    for r in runs:
        vram, jpt = r.get("gpu_mem_used_mb"), r.get("joules_per_token")
        if vram and jpt:
            axes[1,1].scatter(vram/1024, jpt, color=PTQ_COLORS.get(r['ptq_method'], "#999"), 
                             marker=TASK_MARKERS.get(r['task'], "o"), s=100, alpha=0.7)
    axes[1,1].set_title("VRAM Footprint vs. Energy Cost")
    axes[1,1].set_xlabel("Peak VRAM (GB)")
    axes[1,1].set_ylabel("Joules / Token")

    for ax in axes.flat:
        ax.grid(True, linestyle="--", alpha=0.3)
    
    plt.suptitle("Post-Training Quantization x LayerSkip Efficiency Dashboard", fontsize=20, y=0.95)
    path = out_dir / "model_efficiency_dashboard.png"
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_json", type=str,
                        default="./logs/results/results_summary.json")
    parser.add_argument("--scripts_dir", type=str, default="./logs",
                        help="Directory containing progress_*.json files")
    parser.add_argument("--output_dir", type=str, default="./logs/figures")
    args = parser.parse_args()

    res_path = Path(args.results_json)
    if not res_path.exists():
        print(f"ERROR: {res_path} not found. Run 03_collect_results.py first.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = Path(args.scripts_dir)

    runs = load_results(str(res_path))
    print(f"Plotting results from {len(runs)} runs...")

    plot_pareto(runs, out_dir)
    plot_energy_bar(runs, out_dir)
    plot_vram_bar(runs, out_dir)
    plot_speedup_bar(runs, out_dir)
    plot_acceptance_rate(runs, out_dir)
    plot_acceptance_sweep(runs, out_dir)
    plot_heatmap(runs, out_dir)
    plot_utility_bar(runs, out_dir)
    plot_energy_spikes(scripts_dir, out_dir)
    plot_quantization_early_exit_effects(scripts_dir, out_dir)
    plot_calibration_sweeps(scripts_dir, out_dir)
    plot_efficiency_dashboard(runs, out_dir)

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
