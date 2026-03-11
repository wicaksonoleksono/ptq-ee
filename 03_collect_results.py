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

    benchmark_files = sorted(logs_dir.glob("**/benchmark_*.json"))
    runs_data = []
    for f in benchmark_files:
        data = load_run(f)
        if data: runs_data.append(data)

    progress_files = list(logs_dir.glob("**/progress_*.json"))
    progress_map = {}
    for pf in progress_files:
        data = load_run(pf)
        if data:
            key = pf.stem.replace("progress_", "")
            progress_map[key] = data

    rows = []
    for r in runs_data:
        model_name = r.get("model", "").split("/")[-1]
        prog_key = f"{model_name}__{r.get('ptq_method')}__{r.get('generation_strategy')}__" \
                   f"{r.get('task')}"
        
        prog_samples = progress_map.get(prog_key, [])
        
        # Aggregates for CSV
        avg_joules = np.mean([s.get("joules_per_token", 0) for s in prog_samples]) if prog_samples else 0
        avg_gpu = np.mean([s.get("gpu_util_percent", 0) for s in prog_samples]) if prog_samples else 0
        avg_cpu = np.mean([s.get("cpu_util_percent", 0) for s in prog_samples]) if prog_samples else 0
        
        row = {
            "run_id": r.get("run_id"),
            "model": r.get("model"),
            "ptq_method": r.get("ptq_method"),
            "strategy": r.get("generation_strategy"),
            "task": r.get("task"),
            "tokens_per_sec": r.get("results", {}).get("tokens_per_second", {}).get("mean"),
            "rouge_l": r.get("results", {}).get("predicted_text", {}).get("rouge-l"),
            "acceptance_rate": r.get("results", {}).get("acceptance_rate", {}).get("mean"),
            "joules_per_token": round(avg_joules, 4),
            "gpu_util_percent": round(avg_gpu, 1),
            "cpu_util_percent": round(avg_cpu, 1),
            "gpu_mem_used_mb": np.mean([s.get("gpu_mem_used_mb", 0) for s in prog_samples]) if prog_samples else 0,
            # KEEP THE FULL SHIT HERE
            "full_audit_trail": prog_samples 
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
