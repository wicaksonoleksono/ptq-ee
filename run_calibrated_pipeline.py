import os
import sys
import subprocess
import json
import glob
from pathlib import Path

# Parameters
MODELS = ["facebook/layerskip-llama2-13B"]
PTQ_METHODS = ["fp16", "awq", "gptq", "int8_bnb", "smoothquant"]
TASKS = ["cnn_dm_summarization", "xsum_summarization"]

EXIT_LAYERS = [20, 30]
NUM_SPECS = [4, 6]
CALIB_SAMPLES = 30
EVAL_SAMPLES = 50
TOLERANCE = 0.95 # Accept config if ROUGE >= 95% of full-depth baseline

SCRIPT_DIR = Path(__file__).parent.resolve()
LOGS_DIR = SCRIPT_DIR.parent / "logs"
CALIB_DIR = LOGS_DIR / "calibration"
EVAL_DIR = LOGS_DIR / "evaluation"

os.makedirs(CALIB_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

def run_cmd(cmd):
    print(f"\n[RUNNING] {' '.join(cmd)}")
    subprocess.run(cmd, check=False)

def get_latest_json(dir_path, run_id_prefix):
    # Find the latest json matching the prefix
    files = glob.glob(f"{dir_path}/{run_id_prefix}__*.json")
    files = [f for f in files if "progress_" not in f]
    if not files:
        return None
    latest = max(files, key=os.path.getctime)
    with open(latest, 'r') as f:
        return json.load(f)

def run_calibrated_pipeline():
    for model in MODELS:
        model_name = model.split("/")[-1]
        for task in TASKS:
            for method in PTQ_METHODS:
                print(f"\n{'='*80}\nStarting Calibration for {method} on {task}\n{'='*80}")
                
                # 1. Run AR Baseline on Calibration Set
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
                
                baseline_rouge = ar_data.get("quality_metrics", {}).get("rouge_l", 0.0)
                target_rouge = baseline_rouge * TOLERANCE
                print(f"Baseline ROUGE-L: {baseline_rouge:.4f} -> Target ROUGE-L: {target_rouge:.4f}")
                
                # 2. Sweep Self-Speculative Configs
                sweep_results = []
                for el in EXIT_LAYERS:
                    for ns in NUM_SPECS:
                        ss_run_id = f"{model_name}__{method}__self_speculative__{task}"
                        # Delete previous sweep progress files to avoid pollution
                        for p in glob.glob(f"{CALIB_DIR}/progress_{ss_run_id}*.json"):
                            os.remove(p)
                            
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
                        if ss_data:
                            rouge = ss_data.get("quality_metrics", {}).get("rouge_l", 0.0)
                            tps = ss_data.get("efficiency_metrics", {}).get("decode_tps", 0.0)
                            sweep_results.append({
                                "exit_layer": el,
                                "num_speculations": ns,
                                "rouge_l": rouge,
                                "decode_tps": tps
                            })
                            
                # 3. Select Best Config
                valid_configs = [c for c in sweep_results if c["rouge_l"] >= target_rouge]
                if not valid_configs:
                    print(f"WARNING: No config met the target ROUGE for {method}. Falling back to fastest overall.")
                    valid_configs = sweep_results
                
                if not valid_configs:
                    print(f"Sweep failed completely for {method}. Skipping final eval.")
                    continue
                    
                best_config = max(valid_configs, key=lambda x: x["decode_tps"])
                best_el = best_config["exit_layer"]
                best_ns = best_config["num_speculations"]
                print(f"Selected Best Config for {method}: Exit Layer {best_el}, Speculations {best_ns} (TPS: {best_config['decode_tps']:.2f}, ROUGE: {best_config['rouge_l']:.4f})")
                
                # 4. Final Evaluation on 50 Samples (Sample=True to avoid greedy loops)
                print(f"\n{'='*80}\nRunning Final Evaluation for {method} on {task}\n{'='*80}")
                eval_run_id = f"{model_name}__{method}__self_speculative__{task}"
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

if __name__ == "__main__":
    run_calibrated_pipeline()