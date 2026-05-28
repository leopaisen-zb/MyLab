#!/usr/bin/env python3
"""
Eqv2-Lite lmax × radius Cross Ablation - Single Config Quick Test
Run one configuration with minimal epochs to verify everything works.
"""

import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.absolute()
TRAIN_SCRIPT = BASE_DIR / "src" / "train_enhanced_equiformer_v2.py"
RESULTS_DIR = BASE_DIR / "result" / "eq_lite_ablation" / "cross_lmax_radius_fixed"

# Use direct Python path
PYTHON_EXE = "C:/ProgramData/miniconda3/envs/matgen/python.exe"

# Quick test: 1 config, 5 epochs
lmax = int(sys.argv[1]) if len(sys.argv) > 1 else 3
radius = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 5

exp_name = f"quick_lmax{lmax}_r{radius}_e{epochs}"
print(f"Quick test: lmax={lmax}, radius={radius}, epochs={epochs}")

cmd = [
    str(PYTHON_EXE),
    str(TRAIN_SCRIPT),
    "--exp_name", exp_name,
    "--num_layers", "3",
    "--sphere_channels", "128",
    "--num_heads", "4",
    "--attn_hidden_channels", "32",
    "--attn_alpha_channels", "16",
    "--attn_value_channels", "8",
    "--ffn_hidden_channels", "128",
    "--lmax_list", str(lmax),
    "--max_radius", str(radius),
    "--cutoff", str(radius),
    "--max_neighbors", "20",
    "--edge_channels", "128",
    "--batch_size", "16",
    "--lr", "0.0002",
    "--num_epochs", str(epochs),
    "--seed", "42",
]

start = time.time()
result = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, timeout=3600)
elapsed = time.time() - start

print(f"Return code: {result.returncode}")
print(f"Elapsed: {elapsed:.1f}s")
if result.returncode == 0:
    # Try to parse metrics
    output = result.stdout + result.stderr
    for line in output.split('\n'):
        if 'test_mae' in line.lower() or 'MAE' in line:
            print(f"METRIC: {line.strip()}")
else:
    print("STDERR:", result.stderr[-500:])
