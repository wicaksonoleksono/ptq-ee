import os
import sys
import subprocess
import json
import glob
from pathlib import Path

# Parameters
MODELS = ["facebook/layerskip-llama2-13B"]
PTQ_METHODS = ["fp16", "awq", "gptq", "int8_bnb", "smoothquant"]
TASKS = ["cnn_dm_summarization", "arc_challenge"]

# Metric mapping for different tasks
TASK_METRIC_KEY = {
    "cnn_dm_summarization": "rouge_l",
    "arc_challenge": "exact_match", # In LayerSkip, multiple choice usually uses exact_match or accuracy
}

EXIT_LAYERS = [10, 20, 30, 40] # Llama-2 13B has 40 layers
NUM_SPECS = [4, 6]
CALIB_SAMPLES = 30
EVAL_SAMPLES = 50
TOLERANCE = 0.95 # Accept config if metric >= 95% of full-depth baseline

SCRIPT_DIR = Path(__file__).parent.resolve()
LOGS_DIR = SCRIPT_DIR.parent / "logs"
CALIB_DIR = LOGS_DIR / "calibration"
EVAL_DIR = LOGS_DIR / "evaluation"
SWEEP_LOGS_DIR = LOGS_DIR / "sweeps"

os.makedirs(CALIB_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(SWEEP_LOGS_DIR, exist_ok=True)

def run_cmd(cmd):
    print(f"\n[RUNNING] {' '.join(cmd)}")
    subprocess.run(cmd, check=False)

def get_latest_json(dir_path, run_id_prefix):
    # Find the latest json matching the prefix
    files = glob.glob(f"{dir_path}/{run_id_prefix}__*.json")
    # Filter out progress files
    files = [f for f in files if "progress_" not in f]
    if not files:
        return None
    latest = max(files, key=os.path.getctime)
    with open(latest, 'r') as f:
        return json.load(f)

def run_calibrated_pipeline():
    # We need pandas for sweep logging
    try:
        import pandas as pd
    except ImportError:
        print("Pandas not found. Sweep logs will be JSON only.")
        pd = None

    summary_csv = SCRIPT_DIR.parent / "results" / "calibration_summary.csv"
    master_summary = []
    if summary_csv.exists() and pd:
        try:
            master_summary = pd.read_csv(summary_csv).to_dict('records')
            print(f"[PROTOCOL] Loaded existing calibration summary with {len(master_summary)} records.")
        except:
            pass

    for model in MODELS:
        model_name = model.split("/")[-1]
        for task in TASKS:
            for method in PTQ_METHODS:
                # SKIP CHECK: If final eval already exists, skip everything for this method/task
                # Note: evaluation run_id will have the winner's L and K
                # We check the directory for any file matching the base run_id
                base_eval_id = f"{model_name}__{method}__self_speculative"
                existing_evals = list(EVAL_DIR.glob(f"{base_eval_id}_L*_K*__*{task}__*.json"))
                if existing_evals:
                    print(f"\n[Skip] Final evaluation for {method} on {task} already exists. Skipping calibration.")
                    continue

                print(f"\n{'='*80}\nStarting Calibration for {method} on {task}\n{'='*80}")
                
                # ... (ar baseline step) ...
                ar_run_id = f"{model_name}__{method}__autoregressive__{task}"
                cmd_ar = [
                    "python", str(SCRIPT_DIR / "02_run_benchmark.py"),
                    "--model", model,
                    "--ptq_method", method,
                    "--task", task,
                    "--generation_strategy", "autoregressive",
                    "--num_samples", str(CALIB_SAMPLES),
                    "--sample", "False",
                    "--output_dir", str(CALIB_DIR)
                ]
                run_cmd(cmd_ar)
                
                ar_data = get_latest_json(CALIB_DIR, ar_run_id)
                if not ar_data:
                    print(f"Failed to get baseline data for {method} / {task}. Skipping.")
                    continue
                
                metric_key = TASK_METRIC_KEY.get(task, "rouge_l")
                baseline_score = ar_data.get("quality_metrics", {}).get(metric_key, 0.0)
                target_score = baseline_score * TOLERANCE
                print(f"Baseline {metric_key}: {baseline_score:.4f} -> Target: {target_score:.4f}")
                
                # 2. Sweep Self-Speculative Configs
                sweep_results = []
                for el in EXIT_LAYERS:
                    for ns in NUM_SPECS:
                        # Check if this specific sweep step already exists
                        ss_run_id = f"{model_name}__{method}__self_speculative_L{el}_K{ns}__{task}"
                        ss_data = get_latest_json(CALIB_DIR, ss_run_id)
                        
                        if not ss_data:
                            # Clean progress file if starting new
                            for p in glob.glob(f"{CALIB_DIR}/progress_{ss_run_id}*.json"):
                                try: os.remove(p)
                                except: pass
                                
                            cmd_ss = [
                                "python", str(SCRIPT_DIR / "02_run_benchmark.py"),
                                "--model", model,
                                "--ptq_method", method,
                                "--task", task,
                                "--generation_strategy", "self_speculative",
                                "--exit_layer", str(el),
                                "--num_speculations", str(ns),
                                "--num_samples", str(CALIB_SAMPLES),
                                "--sample", "False",
                                "--output_dir", str(CALIB_DIR)
                            ]
                            run_cmd(cmd_ss)
                            ss_data = get_latest_json(CALIB_DIR, ss_run_id)
                        else:
                            print(f"[Skip] Sweep step {ss_run_id} already exists.")
                        
                        if ss_data:
                            score = ss_data.get("quality_metrics", {}).get(metric_key, 0.0)
                            tps = ss_data.get("efficiency_metrics", {}).get("decode_tps", 0.0)
                            jpt = ss_data.get("energy_metrics", {}).get("joules_per_token", 0.0)
                            ar = ss_data.get("efficiency_metrics", {}).get("acceptance_rate", 0.0)
                            
                            sweep_results.append({
                                "method": method,
                                "task": task,
                                "exit_layer": el,
                                "num_speculations": ns,
                                "score": score,
                                "decode_tps": tps,
                                "joules_per_token": jpt,
                                "acceptance_rate": ar
                            })
                
                # Save Sweep Data
                if sweep_results:
                    if pd:
                        sweep_df = pd.DataFrame(sweep_results)
                        sweep_csv = SWEEP_LOGS_DIR / f"sweep_{method}_{task}.csv"
                        sweep_df.to_csv(sweep_csv, index=False)
                        print(f"Saved sweep CSV to {sweep_csv}")
                            
                # 3. Select Best Config (Highest Acceptance Rate among those > 95% baseline)
                valid_configs = [c for c in sweep_results if c["score"] >= target_score]
                if not valid_configs:
                    print(f"WARNING: No config met target for {method}. Falling back to best overall AR.")
                    valid_configs = sweep_results
                
                if not valid_configs:
                    continue
                    
                best_config = max(valid_configs, key=lambda x: x["acceptance_rate"])
                best_el = best_config["exit_layer"]
                best_ns = best_config["num_speculations"]
                print(f"*** WINNER for {method}: Exit Layer {best_el}, Spec {best_ns} (AR: {best_config['acceptance_rate']:.4f}) ***")
                
                new_record = {
                    "method": method,
                    "task": task,
                    "best_exit_layer": best_el,
                    "best_num_specs": best_ns,
                    "calibration_ar": best_config["acceptance_rate"],
                    "calibration_tps": best_config["decode_tps"],
                    "calibration_score": best_config["score"]
                }
                # Remove old record for same method/task if exists
                master_summary = [r for r in master_summary if not (r['method'] == method and r['task'] == task)]
                master_summary.append(new_record)
                
                if pd:
                    pd.DataFrame(master_summary).to_csv(summary_csv, index=False)

                # 4. Final Evaluation
                print(f"\nFinal Evaluation for {method} on {task}...")
                cmd_eval = [
                    "python", str(SCRIPT_DIR / "02_run_benchmark.py"),
                    "--model", model,
                    "--ptq_method", method,
                    "--task", task,
                    "--generation_strategy", "self_speculative",
                    "--exit_layer", str(best_el),
                    "--num_speculations", str(best_ns),
                    "--num_samples", str(EVAL_SAMPLES),
                    "--sample", "True",
                    "--output_dir", str(EVAL_DIR)
                ]
                run_cmd(cmd_eval)

    # Save the Master Summary
    if master_summary and pd:
        summary_df = pd.DataFrame(master_summary)
        summary_csv = SCRIPT_DIR.parent / "results" / "calibration_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n[PROTOCOL] Master calibration summary saved to {summary_csv}")

if __name__ == "__main__":
    run_calibrated_pipeline()