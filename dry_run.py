"""
dry_run.py
----------
Checks all pipeline scripts for syntax errors without running anything heavy.
No GPU, no model downloads, no imports of torch/transformers required.

Usage:
    python scripts/dry_run.py
"""

import ast
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

PY_SCRIPTS = [
    "00_download_data.py",
    "01_quantize.py",
    "02_run_benchmark.py",
    "03_collect_results.py",
    "04_plot_results.py",
    "energy_meter.py",
]

CONFIG_FILES = [
    "experiment_config.json",
]


def check_python_syntax(path: Path) -> list[str]:
    errors = []
    try:
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
    except SyntaxError as e:
        errors.append(f"SyntaxError at line {e.lineno}: {e.msg}")
    except Exception as e:
        errors.append(f"Error reading file: {e}")
    return errors


def check_json(path: Path) -> list[str]:
    errors = []
    try:
        with open(path) as f:
            json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"JSONDecodeError at line {e.lineno}: {e.msg}")
    except Exception as e:
        errors.append(f"Error reading file: {e}")
    return errors


def check_config_paths(config: dict) -> list[str]:
    """Check that path keys in experiment_config.json are present and well-formed."""
    errors = []
    paths = config.get("paths", {})
    required_path_keys = ["layerskip_dir", "quantized_models_dir", "logs_dir",
                          "results_dir", "figures_dir", "model_cache_dir"]
    for k in required_path_keys:
        if k not in paths:
            errors.append(f"Missing key in paths: '{k}'")

    required_top_keys = ["models", "ptq_methods", "decoding_strategies", "tasks", "energy"]
    for k in required_top_keys:
        if k not in config:
            errors.append(f"Missing top-level key: '{k}'")

    ptq_methods = config.get("ptq_methods", {})
    for method, cfg in ptq_methods.items():
        for field in ["bits_weights", "bits_activations", "description"]:
            if field not in cfg:
                errors.append(f"ptq_methods.{method} missing field: '{field}'")

    return errors


def check_benchmark_runner_imports():
    """
    Check that 02_run_benchmark.py's LayerSkip bootstrap logic is correct —
    i.e., the path it constructs for LAYERSKIP_DIR actually exists.
    """
    errors = []
    layerskip_dir = (SCRIPT_DIR.parent / "LayerSkip").resolve()
    if not layerskip_dir.exists():
        errors.append(f"LayerSkip directory not found at: {layerskip_dir}")
    else:
        for required in ["benchmark.py", "generate.py", "arguments.py", "data.py"]:
            if not (layerskip_dir / required).exists():
                errors.append(f"LayerSkip/{required} not found")
    return errors


def main():
    all_ok = True

    print("=" * 60)
    print("Dry run — syntax + config check")
    print("=" * 60)

    # 1. Python syntax
    print("\n[1] Python syntax check")
    for name in PY_SCRIPTS:
        path = SCRIPT_DIR / name
        if not path.exists():
            print(f"  MISSING  {name}")
            all_ok = False
            continue
        errs = check_python_syntax(path)
        if errs:
            print(f"  FAIL     {name}")
            for e in errs:
                print(f"           {e}")
            all_ok = False
        else:
            print(f"  OK       {name}")

    # 2. JSON config
    print("\n[2] JSON config check")
    config = None
    for name in CONFIG_FILES:
        path = SCRIPT_DIR / name
        if not path.exists():
            print(f"  MISSING  {name}")
            all_ok = False
            continue
        errs = check_json(path)
        if errs:
            print(f"  FAIL     {name}")
            for e in errs:
                print(f"           {e}")
            all_ok = False
        else:
            with open(path) as f:
                config = json.load(f)
            print(f"  OK       {name}")

    # 3. Config schema
    print("\n[3] experiment_config.json schema check")
    if config:
        errs = check_config_paths(config)
        if errs:
            for e in errs:
                print(f"  FAIL     {e}")
            all_ok = False
        else:
            print("  OK       all required keys present")

    # 4. LayerSkip directory
    print("\n[4] LayerSkip directory check")
    errs = check_benchmark_runner_imports()
    if errs:
        for e in errs:
            print(f"  FAIL     {e}")
        all_ok = False
    else:
        print("  OK       LayerSkip files found")

    # 5. run_pipeline.sh exists
    print("\n[5] Shell script check")
    sh = SCRIPT_DIR / "run_pipeline.sh"
    if sh.exists():
        print(f"  OK       run_pipeline.sh")
    else:
        print(f"  MISSING  run_pipeline.sh")
        all_ok = False

    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("All checks passed. Safe to run the pipeline.")
    else:
        print("Some checks FAILED. Fix the issues above before running.")
        sys.exit(1)


if __name__ == "__main__":
    main()
