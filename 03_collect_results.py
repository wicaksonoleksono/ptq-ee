"""
03_collect_results.py
---------------------
Aggregates granular telemetry into run-level averages and PRESERVES full audit trail.
"""

import argparse
import csv
import json
import numpy as np
from pathlib import Path

def load_run(path):
    try:
        with open(path) as f: return json.load(f)
    except: return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Look for any JSON file in the logs directory
    benchmark_files = sorted(logs_dir.glob("**/*.json"))
    # Filter out progress files if they are in the same folder
    benchmark_files = [f for f in benchmark_files if "progress_" not in f.name and "results_summary" not in f.name]
    
    runs_data = []
    for f in benchmark_files:
        data = load_run(f)
        # Ensure it's a valid benchmark file by checking for run_id
        if data and "run_id" in data:
            runs_data.append(data)

    rows = []
    for r in runs_data:
        config = r.get("config", {})
        quality = r.get("quality_metrics", {})
        efficiency = r.get("efficiency_metrics", {})
        energy = r.get("energy_metrics", {})
        
        row = {
            "run_id": r.get("run_id"),
            "model": config.get("model"),
            "ptq_method": config.get("ptq_method"),
            "strategy": config.get("generation_strategy"),
            "task": config.get("task"),
            "bits_w": config.get("bits_weights"),
            "bits_a": config.get("bits_activations"),
            "exit_layer": config.get("exit_layer"),
            "num_speculations": config.get("num_speculations"),
            
            "tokens_per_sec": efficiency.get("tokens_per_second"),
            "decode_tps": efficiency.get("decode_tps"),
            "prefill_tps": efficiency.get("prefill_tps"),
            "ms_per_token": efficiency.get("time_per_token_ms"),
            
            "rouge_l": quality.get("rouge_l"),
            "rouge_1": quality.get("rouge_1"),
            "rouge_2": quality.get("rouge_2"),
            "bleu": quality.get("bleu"),
            
            "joules_per_token": energy.get("joules_per_token"),
            "total_joules": energy.get("total_joules"),
            "avg_power_w": energy.get("avg_power_watts"),
            "gpu_util": energy.get("avg_gpu_util_percent"),
            "cpu_util": energy.get("avg_cpu_util_percent"),
            "gpu_mem_used_mb": efficiency.get("peak_vram_gb", 0) * 1024, # convert GB to MB for the plotter
            # Full token logs if needed
            "full_audit_trail": r.get("raw_layerskip_metrics")
        }
        rows.append(row)

    with open(output_dir / "results_summary.json", "w") as f:
        json.dump({"runs": rows}, f, indent=2)

    if rows:
        csv_rows = []
        for r in rows:
            c = r.copy()
            del c["full_audit_trail"]
            csv_rows.append(c)
        with open(output_dir / "results_table.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"Aggregation complete. Full token-by-token logs preserved in results_summary.json.")

if __name__ == "__main__":
    main()
