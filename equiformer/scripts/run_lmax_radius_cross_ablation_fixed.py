#!/usr/bin/env python3
"""
Eqv2-Lite lmax × radius Cross Ablation Experiment (Fixed num_layers=3)

Purpose: Re-run the cross ablation with correct variable control (num_layers=3 fixed)
to verify the interaction effect between lmax and max_radius.

Configurations:
- Fixed: num_layers=3, sphere_channels=128, num_heads=4, ffn_hidden=128, edge_channels=128, num_gaussians=256
- Variables: lmax=[3,4], max_radius=[8.0, 12.0, 16.0]
- Total: 6 configurations

Expected runtime: ~30 min per config, ~3 hours total
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import threading
import re

# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def log(msg: str, color: str = ""):
    """Print colored log message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{color}[{timestamp}]{msg}{Colors.ENDC}")
    sys.stdout.flush()

def log_header(msg: str):
    log(msg, Colors.HEADER + Colors.BOLD)

def log_progress(msg: str):
    log(msg, Colors.CYAN)

def log_success(msg: str):
    log(msg, Colors.GREEN)

def log_warning(msg: str):
    log(msg, Colors.YELLOW)

def log_error(msg: str):
    log(msg, Colors.RED)

# Experiment configuration
BASE_DIR = Path(__file__).parent.parent.absolute()
EXPERIMENT_DIR = BASE_DIR / "experiments" / "ablation"
RESULTS_DIR = BASE_DIR / "result" / "eq_lite_ablation" / "cross_lmax_radius_fixed"
SCRIPT_DIR = BASE_DIR / "src"

# Optimal fixed parameters (from previous ablation studies)
FIXED_PARAMS = {
    "num_layers": 3,           # Fixed at optimal value
    "sphere_channels": 128,     # Best from channels ablation
    "num_heads": 4,
    "ffn_hidden_channels": 128,
    "edge_channels": 128,
    "num_gaussians": 256,      # Best from radial ablation
    "attn_hidden_channels": 32,
    "attn_alpha_channels": 16,
    "attn_value_channels": 8,
    "max_neighbors": 20,
    "batch_size": 16,
    "lr": 0.0002,
    "num_epochs": 100,
    "seed": 42,
    "patience": 20,
}

# Variable parameters to test
LMAX_VALUES = [3, 4]
RADIUS_VALUES = [8.0, 12.0, 16.0]

# Training script
TRAIN_SCRIPT = SCRIPT_DIR / "train_enhanced_equiformer_v2.py"

def get_config_name(lmax: int, radius: float) -> str:
    return f"lmax{lmax}_r{radius}"

def build_experiment_name(lmax: int, radius: float) -> str:
    return f"cross_fixed_layers3_lmax{lmax}_r{radius}"

def check_gpu_available() -> bool:
    """Check if GPU is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def run_single_experiment(
    lmax: int,
    radius: float,
    exp_tag: str
) -> Dict:
    """
    Run a single experiment configuration.
    Returns dict with metrics: mae, rmse, r2, loss, params, time
    """
    config_name = get_config_name(lmax, radius)
    exp_name = build_experiment_name(lmax, radius)

    log_progress(f"  Starting: lmax={lmax}, radius={radius}")

    # Build command
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--exp_name", exp_name,
        "--tag", exp_tag,
        "--num_layers", str(FIXED_PARAMS["num_layers"]),
        "--sphere_channels", str(FIXED_PARAMS["sphere_channels"]),
        "--num_heads", str(FIXED_PARAMS["num_heads"]),
        "--attn_hidden_channels", str(FIXED_PARAMS["attn_hidden_channels"]),
        "--attn_alpha_channels", str(FIXED_PARAMS["attn_alpha_channels"]),
        "--attn_value_channels", str(FIXED_PARAMS["attn_value_channels"]),
        "--ffn_hidden_channels", str(FIXED_PARAMS["ffn_hidden_channels"]),
        "--lmax_list", str(lmax),
        "--max_radius", str(radius),
        "--max_neighbors", str(FIXED_PARAMS["max_neighbors"]),
        "--edge_channels", str(FIXED_PARAMS["edge_channels"]),
        "--num_gaussians", str(FIXED_PARAMS["num_gaussians"]),
        "--batch_size", str(FIXED_PARAMS["batch_size"]),
        "--lr", str(FIXED_PARAMS["lr"]),
        "--num_epochs", str(FIXED_PARAMS["num_epochs"]),
        "--seed", str(FIXED_PARAMS["seed"]),
        "--patience", str(FIXED_PARAMS["patience"]),
    ]

    start_time = time.time()

    # Run training
    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        elapsed = time.time() - start_time

        if result.returncode != 0:
            log_error(f"  Failed with return code {result.returncode}")
            log_error(f"  stderr: {result.stderr[-500:]}")
            return {
                "lmax": lmax,
                "radius": radius,
                "mae": None,
                "rmse": None,
                "r2": None,
                "loss": None,
                "params": None,
                "time": elapsed,
                "status": "failed",
                "error": result.stderr[-500:]
            }

        # Parse output for metrics
        output = result.stdout + result.stderr

        # Try to find metrics in output
        mae, rmse, r2, loss, params = parse_metrics(output)

        log_success(f"  Completed: MAE={mae:.4f}, R2={r2:.4f}, Time={elapsed/60:.1f}min")

        return {
            "lmax": lmax,
            "radius": radius,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "loss": loss,
            "params": params,
            "time": elapsed,
            "status": "success"
        }

    except subprocess.TimeoutExpired:
        log_error(f"  Timeout after 1 hour")
        return {
            "lmax": lmax,
            "radius": radius,
            "mae": None,
            "rmse": None,
            "r2": None,
            "loss": None,
            "params": None,
            "time": time.time() - start_time,
            "status": "timeout"
        }
    except Exception as e:
        log_error(f"  Exception: {str(e)}")
        return {
            "lmax": lmax,
            "radius": radius,
            "mae": None,
            "rmse": None,
            "r2": None,
            "loss": None,
            "params": None,
            "time": time.time() - start_time,
            "status": "error",
            "error": str(e)
        }

def parse_metrics(output: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[int]]:
    """
    Parse metrics from training output.
    Looks for patterns like:
    - "test_mae: 0.0868"
    - "MAE: 0.0868"
    - "R2 Score: 0.9334"
    """
    mae = rmse = r2 = loss = params = None

    # Try to find MAE
    for pattern in [r"test_mae[:\s]+([0-9.]+)", r"MAE[:\s]+([0-9.]+)", r"mae[:\s]+([0-9.]+)"]:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            mae = float(match.group(1))

    # Try to find RMSE
    for pattern in [r"test_rmse[:\s]+([0-9.]+)", r"RMSE[:\s]+([0-9.]+)", r"rmse[:\s]+([0-9.]+)"]:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            rmse = float(match.group(1))

    # Try to find R2
    for pattern in [r"test_r2[:\s]+([0-9.]+)", r"R2[:\s]+([0-9.]+)", r"r2[:\s]+([0-9.]+)"]:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            r2 = float(match.group(1))

    # Try to find loss
    for pattern in [r"test_loss[:\s]+([0-9.]+)", r"loss[:\s]+([0-9.]+)"]:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            loss = float(match.group(1))

    # Try to find params
    for pattern in [r"params[:\s]+([0-9]+)", r"parameters[:\s]+([0-9]+)"]:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            params = int(match.group(1))

    return mae, rmse, r2, loss, params

def save_results(results: List[Dict], timestamp: str):
    """Save results to CSV and JSON"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = RESULTS_DIR / f"summary_{timestamp}.csv"
    with open(csv_path, "w") as f:
        f.write("lmax,radius,mae,rmse,r2,loss,params,time,status\n")
        for r in results:
            f.write(f"{r['lmax']},{r['radius']},{r['mae'] or ''},{r['rmse'] or ''},"
                   f"{r['r2'] or ''},{r['loss'] or ''},{r['params'] or ''},"
                   f"{r['time']:.1f},{r['status']}\n")

    # Save JSON
    json_path = RESULTS_DIR / f"results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "fixed_params": FIXED_PARAMS,
            "results": results
        }, f, indent=2)

    log_success(f"Results saved to {csv_path}")
    return csv_path, json_path

def print_progress_table(results: List[Dict], total: int, current: int):
    """Print a progress table"""
    print("\n" + "="*80)
    print(f"{Colors.BOLD}Cross Ablation Progress: {current}/{total} configurations{Colors.ENDC}")
    print("="*80)
    print(f"{'lmax':<8} {'radius':<10} {'MAE':<12} {'R2':<12} {'Status':<10}")
    print("-"*80)

    for r in results:
        mae_str = f"{r['mae']:.4f}" if r['mae'] else "N/A"
        r2_str = f"{r['r2']:.4f}" if r['r2'] else "N/A"
        status_color = Colors.GREEN if r['status'] == 'success' else Colors.RED
        print(f"{r['lmax']:<8} {r['radius']:<10.1f} {mae_str:<12} {r2_str:<12} {status_color}{r['status']:<10}{Colors.ENDC}")

    print("="*80 + "\n")

def print_final_summary(results: List[Dict]):
    """Print final summary table"""
    print("\n" + "="*100)
    print(f"{Colors.BOLD}{'FINAL RESULTS: lmax × radius Cross Ablation (num_layers=3 fixed)'}{Colors.ENDC}")
    print("="*100)

    # Sort by R2 descending
    sorted_results = sorted([r for r in results if r['status'] == 'success'],
                          key=lambda x: x['r2'] or 0, reverse=True)

    print(f"\n{'Rank':<6} {'lmax':<8} {'radius':<10} {'MAE':<12} {'RMSE':<12} {'R2':<12} {'Time':<10}")
    print("-"*100)

    for i, r in enumerate(sorted_results, 1):
        marker = " ★" if i == 1 else ""
        print(f"{i:<6} {r['lmax']:<8} {r['radius']:<10.1f} "
              f"{r['mae']:<12.4f} {r['rmse']:<12.4f} {r['r2']:<12.4f} "
              f"{r['time']/60:<10.1f}min{marker}")

    print("-"*100)

    # Find best configuration
    if sorted_results:
        best = sorted_results[0]
        print(f"\n{Colors.BOLD}Best Configuration:{Colors.ENDC}")
        print(f"  lmax = {best['lmax']}, radius = {best['radius']}")
        print(f"  MAE = {best['mae']:.4f} eV, R2 = {best['r2']:.4f}")
        print(f"  Parameters = {best['params']:,}" if best['params'] else "")

    # Comparison with previous results
    print(f"\n{Colors.BOLD}Comparison with Single-Factor Ablation Best:{Colors.ENDC}")
    print(f"  Previous best (lmax=4, radius=12.0, num_layers=6): MAE=0.0996, R2=0.9280")
    if sorted_results:
        best = sorted_results[0]
        print(f"  New best (lmax={best['lmax']}, radius={best['radius']}, num_layers=3): MAE={best['mae']:.4f}, R2={best['r2']:.4f}")

        # Check if improvement
        prev_mae = 0.0996
        if best['mae'] < prev_mae:
            improvement = (prev_mae - best['mae']) / prev_mae * 100
            print(f"  {Colors.GREEN}✓ Improvement: {improvement:.1f}% reduction in MAE{Colors.ENDC}")
        else:
            degradation = (best['mae'] - prev_mae) / prev_mae * 100
            print(f"  {Colors.YELLOW}✗ No improvement: {degradation:.1f}% higher MAE{Colors.ENDC}")

    print("="*100 + "\n")

def main():
    """Main function"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_tag = f"cross_fixed_{timestamp}"

    log_header("="*80)
    log_header("Eqv2-Lite lmax × radius Cross Ablation (num_layers=3 fixed)")
    log_header("="*80)

    # Check GPU
    has_gpu = check_gpu_available()
    if has_gpu:
        log_success("GPU is available")
    else:
        log_warning("GPU not detected, will run on CPU (SLOW)")

    # Check training script exists
    if not TRAIN_SCRIPT.exists():
        log_error(f"Training script not found: {TRAIN_SCRIPT}")
        sys.exit(1)

    log(f"Base directory: {BASE_DIR}")
    log(f"Results directory: {RESULTS_DIR}")
    log(f"Fixed parameters: num_layers={FIXED_PARAMS['num_layers']}, sphere_channels={FIXED_PARAMS['sphere_channels']}")

    # Generate configurations
    configs = []
    for lmax in LMAX_VALUES:
        for radius in RADIUS_VALUES:
            configs.append((lmax, radius))

    total = len(configs)
    log(f"Total configurations: {total}")
    log(f"  lmax values: {LMAX_VALUES}")
    log(f"  radius values: {RADIUS_VALUES}")
    log(f"Estimated runtime: ~{total * 30} minutes")

    # Run experiments
    results = []
    completed = 0

    log_header("\nStarting experiments...")

    for lmax, radius in configs:
        completed += 1
        log_progress(f"\n[{completed}/{total}] Running: lmax={lmax}, radius={radius}")

        result = run_single_experiment(lmax, radius, exp_tag)
        results.append(result)

        print_progress_table(results, total, completed)

    # Save results
    csv_path, json_path = save_results(results, timestamp)

    # Print final summary
    print_final_summary(results)

    # Save summary to memory file for thesis
    summary_path = RESULTS_DIR / f"thesis_summary_{timestamp}.md"
    with open(summary_path, "w") as f:
        f.write("# Eqv2-Lite lmax × radius Cross Ablation Results (num_layers=3 fixed)\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Fixed Parameters\n")
        for k, v in FIXED_PARAMS.items():
            f.write(f"- {k}: {v}\n\n")
        f.write("## Results\n\n")
        f.write("| lmax | radius | MAE (eV) | RMSE (eV) | R2 | Status |\n")
        f.write("|------|--------|-----------|-----------|-----|--------|\n")
        for r in sorted(results, key=lambda x: (x['lmax'], x['radius'])):
            f.write(f"| {r['lmax']} | {r['radius']} | {r['mae'] or 'N/A':.4f} | "
                    f"{r['rmse'] or 'N/A':.4f} | {r['r2'] or 'N/A':.4f} | {r['status']} |\n\n")

        best = sorted([r for r in results if r['status'] == 'success'],
                     key=lambda x: x['mae'] or 999)[0] if results else None
        if best:
            f.write(f"## Best Configuration\n\n")
            f.write(f"- lmax: {best['lmax']}\n")
            f.write(f"- radius: {best['radius']}\n")
            f.write(f"- MAE: {best['mae']:.4f} eV\n")
            f.write(f"- RMSE: {best['rmse']:.4f} eV\n")
            f.write(f"- R2: {best['r2']:.4f}\n")

    log_success(f"\nThesis summary saved to {summary_path}")
    log_success("Experiment completed!")

if __name__ == "__main__":
    main()
