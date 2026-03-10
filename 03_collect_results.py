"""
03_collect_results.py
---------------------
Aggregates all JSON benchmark outputs from the logs directory into:
  - results_summary.json   (machine-readable, full detail)
  - results_table.csv      (easy import into Excel / pandas)

Usage:
    python scripts/03_collect_results.py --logs_dir ./logs --output_dir ./results
"""

import argparse
import csv
import json
from pathlib import Path


# Flat fields to extract from each run JSON
FLAT_FIELDS = [
    # identifiers
    ("run_id",           "run_id"),
    ("timestamp",        "timestamp"),
    # config
    ("model",            "config.model"),
    ("ptq_method",       "config.ptq_method"),
    ("bits_w",           "config.bits_weights"),
    ("bits_a",           "config.bits_activations"),
    ("strategy",         "config.generation_strategy"),
    ("exit_layer",       "config.exit_layer"),
    ("num_speculations", "config.num_speculations"),
    ("task",             "config.task"),
    ("num_samples",      "config.num_samples"),
    # quality
    ("rouge_l",          "quality_metrics.rouge_l"),
    ("rouge_1",          "quality_metrics.rouge_1"),
    ("rouge_2",          "quality_metrics.rouge_2"),
    ("bleu",             "quality_metrics.bleu"),
    # efficiency
    ("tokens_per_sec",   "efficiency_metrics.tokens_per_second"),
    ("ms_per_token",     "efficiency_metrics.time_per_token_ms"),
    ("total_time_s",     "efficiency_metrics.total_benchmark_time_s"),
    ("acceptance_rate",  "efficiency_metrics.acceptance_rate"),
    ("peak_vram_gb",     "efficiency_metrics.peak_vram_gb"),
    ("load_time_s",      "efficiency_metrics.model_load_time_s"),
    # energy
    ("total_joules",     "energy_metrics.total_joules"),
    ("joules_per_token", "energy_metrics.joules_per_token"),
    ("avg_power_w",      "energy_metrics.avg_power_watts"),
    ("peak_power_w",     "energy_metrics.peak_power_watts"),
]


def get_nested(d: dict, dotted_key: str):
    """Resolve 'config.model' into d['config']['model']."""
    keys = dotted_key.split(".")
    val = d
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return None
    return val


def load_run(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def flatten_run(run: dict) -> dict:
    row = {}
    for col_name, dotted_key in FLAT_FIELDS:
        row[col_name] = get_nested(run, dotted_key)
    return row


def compute_relative_speedup(rows: list) -> list:
    """
    Add a 'speedup_vs_fp16_autoregressive' column.
    Baseline = same model + fp16 + autoregressive, same task.
    """
    baselines = {}
    for r in rows:
        key = (r.get("model"), r.get("task"))
        if r.get("ptq_method") == "fp16" and r.get("strategy") == "autoregressive":
            baselines[key] = r.get("tokens_per_sec", 0)

    for r in rows:
        key = (r.get("model"), r.get("task"))
        base_tps = baselines.get(key)
        this_tps = r.get("tokens_per_sec", 0)
        if base_tps and base_tps > 0 and this_tps:
            r["speedup_vs_fp16_ar"] = round(this_tps / base_tps, 3)
        else:
            r["speedup_vs_fp16_ar"] = None

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", type=str, default="../logs",
                        help="Directory containing benchmark JSON files")
    parser.add_argument("--output_dir", type=str, default="../results",
                        help="Where to write results_summary.json and results_table.csv")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(logs_dir.glob("**/*.json"))
    if not json_files:
        print(f"No JSON files found in {logs_dir}")
        return

    all_runs = []
    errors = []
    for jf in json_files:
        try:
            run = load_run(jf)
            # Check it looks like a benchmark result (has run_id)
            if "run_id" not in run:
                continue
            run["_source_file"] = str(jf)
            all_runs.append(run)
        except Exception as e:
            errors.append({"file": str(jf), "error": str(e)})

    print(f"Loaded {len(all_runs)} runs from {logs_dir}")
    if errors:
        print(f"  {len(errors)} files failed to parse:")
        for e in errors:
            print(f"    {e['file']}: {e['error']}")

    # Flatten
    rows = [flatten_run(r) for r in all_runs]
    rows = compute_relative_speedup(rows)

    # Sort by model → ptq_method → strategy → task
    rows.sort(key=lambda r: (
        r.get("model") or "",
        r.get("ptq_method") or "",
        r.get("strategy") or "",
        r.get("task") or "",
    ))

    # Write JSON summary
    summary_path = output_dir / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "num_runs": len(rows),
            "runs": rows,
            "errors": errors,
        }, f, indent=2)
    print(f"JSON summary → {summary_path}")

    # Write CSV
    csv_path = output_dir / "results_table.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"CSV table    → {csv_path}")

    # Print quick summary table to stdout
    print(f"\n{'Model':<35} {'PTQ':<12} {'Strategy':<20} {'Task':<25} {'tok/s':>7} {'J/tok':>8} {'ROUGE-L':>8}")
    print("-" * 120)
    for r in rows:
        model_short = (r.get("model") or "")[-34:]
        print(
            f"{model_short:<35} "
            f"{str(r.get('ptq_method','')):<12} "
            f"{str(r.get('strategy','')):<20} "
            f"{str(r.get('task','')):<25} "
            f"{r.get('tokens_per_sec') or 0:>7.1f} "
            f"{r.get('joules_per_token') or 0:>8.3f} "
            f"{r.get('rouge_l') or 0:>8.4f}"
        )


if __name__ == "__main__":
    main()
