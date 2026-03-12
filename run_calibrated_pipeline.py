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
NUM_SPECS = [6] # Fixed at 6 for research consistency
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
    summary_csv.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
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
                
                # 1. Sweep Self-Speculative Configs
                sweep_results = []
                for el in EXIT_LAYERS:
                    for ns in NUM_SPECS:
                        # Add 'calibration__' prefix to distinguish from final eval
                        ss_run_id = f"calibration__{model_name}__{method}__self_speculative_L{el}_K{ns}__{task}"
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
                                "--output_dir", str(CALIB_DIR),
                                "--run_type", "calibration" # Pass tag if 02_run_benchmark.py supports it
                            ]
                            run_cmd(cmd_ss)
                            ss_data = get_latest_json(CALIB_DIR, ss_run_id)
                        else:
                            print(f"[Skip] Sweep step {ss_run_id} already exists.")
                        
                        # ... (rest of logic) ...
                            
                # 4. Final Evaluation
                print(f"\nFinal Evaluation for {method} on {task}...")
                eval_run_id = f"evaluation__{model_name}__{method}__self_speculative_L{best_el}_K{best_ns}__{task}"
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
                    "--output_dir", str(EVAL_DIR),
                    "--run_type", "evaluation"
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