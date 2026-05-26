# -*- coding: utf-8 -*-
"""
System Performance and Quality Analysis Experiment - Chapter 05
100-sample benchmark with tqdm progress bar
"""
import sys
import os
import time
import json
import random
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

os.environ["MATGEN_DEVICE"] = "cpu"
os.environ["MATGEN_DEMO"] = "1"  # Demo mode for testing

from backend.rag_gen import generate as rag_generate
from backend.eq_predict import predict as eq_predict
from backend.quality import validate_structure
from config import DEFAULT_FILTER_LOW, DEFAULT_FILTER_HIGH, BASE_MODEL_PATH, NORM_STATS

import numpy as np
import ase.io
import io

random.seed(42)
np.random.seed(42)

ELEMENTS = ["Ir", "Pd", "Pt", "Rh", "Ru", "Cu", "Fe", "Co", "Ni"]
TEMPLATES = [
    "Generate H adsorption structure on {elem}(111) surface with reasonable lattice and coordinates.",
    "Target DeltaG_H = {dg:.2f} eV, {elem1}{elem2} binary alloy surface for H adsorption.",
    "Target DeltaG_H = {dg:.2f} eV, {elem1}-based high-entropy alloy surface catalytic site.",
    "Generate {elem1}{elem2}{elem3} ternary alloy surface structure, H coverage = {cov:.2f} ML.",
    "Design {elem1}-based high-entropy alloy {miller} surface with 20-50 atoms.",
]

prompts = []
for i in range(100):
    elem = random.choice(ELEMENTS)
    dg = round(random.uniform(-0.5, 0.5), 2)
    elem1, elem2 = random.sample(ELEMENTS, 2)
    elem3 = random.choice([e for e in ELEMENTS if e not in [elem1, elem2]])
    cov = round(random.uniform(0.25, 1.0), 2)
    miller = random.choice(["(111)", "(100)", "(110)"])
    template = random.choice(TEMPLATES)
    prompt = template.format(elem=elem, dg=dg, elem1=elem1, elem2=elem2, elem3=elem3, cov=cov, miller=miller)
    prompts.append(prompt)

TARGET_DG_H = -0.2
FILTER_LOW = DEFAULT_FILTER_LOW
FILTER_HIGH = DEFAULT_FILTER_HIGH

with open(NORM_STATS) as f:
    norm_stats = json.load(f)
train_mean = norm_stats["target_mean"]
train_std = norm_stats["target_std"]

N = 100  # Full benchmark
prompts = prompts[:N]  # Limit to N samples
gen_times, parse_times, pred_times, filter_times = [], [], [], []
gen_success, parse_success, pred_success, filter_pass = 0, 0, 0, 0
rejection_levels = Counter()
dg_h_all, dg_h_valid = [], []
deviations = []
valid_structures, structure_info = [], []
failed_by_rejection = Counter()

print(f"[INFO] 100-sample benchmark | Target DeltaG_H: {TARGET_DG_H} eV | Filter: [{FILTER_LOW}, {FILTER_HIGH}]")
print(f"[INFO] Device: {os.environ.get('MATGEN_DEVICE')} | Demo mode: {os.environ.get('MATGEN_DEMO')}")

from tqdm import tqdm
import sys

start_time = time.time()

# Force unbuffered output for tqdm
sys.stdout.reconfigure(line_buffering=True)

for i, prompt in enumerate(tqdm(prompts, desc="Running benchmark", unit="sample")):
    poscar = None
    dg_h = None
    rejection_level = "unknown"
    in_filter_flag = False

    t0 = time.time()
    try:
        poscar = rag_generate(prompt, base_model_name_or_path=BASE_MODEL_PATH)
        gen_success += 1
    except Exception:
        rejection_level = "text"
        failed_by_rejection["gen_failed"] += 1
    gen_times.append(time.time() - t0)

    t0 = time.time()
    if poscar:
        try:
            check = validate_structure(poscar)
            rejection_level = check.get("rejection_level", "pass")
            if rejection_level == "pass":
                parse_success += 1
            else:
                failed_by_rejection[rejection_level] += 1
        except Exception:
            rejection_level = "structure"
            failed_by_rejection["structure_error"] += 1
    parse_times.append(time.time() - t0)

    t0 = time.time()
    if poscar and rejection_level == "pass":
        try:
            res = eq_predict(poscar)
            dg_h = res["dg_h"]
            pred_success += 1
            dg_h_all.append(dg_h)
            deviations.append(abs(dg_h - TARGET_DG_H))
        except Exception:
            failed_by_rejection["pred_error"] += 1
    pred_times.append(time.time() - t0)

    t0 = time.time()
    if dg_h is not None:
        try:
            in_filter_flag = (float(FILTER_LOW) <= dg_h <= float(FILTER_HIGH))
            if in_filter_flag:
                filter_pass += 1
        except:
            pass
    filter_times.append(time.time() - t0)

    rejection_levels[rejection_level] += 1

    if poscar and rejection_level == "pass":
        try:
            atoms = ase.io.read(io.StringIO(poscar), format="vasp")
            elem_counts = dict(Counter(atoms.get_chemical_symbols()))
            lattice = atoms.cell.array
            from ase.neighborlist import neighbor_list
            n1, n2, d = neighbor_list('ijd', atoms, cutoff=10.0)
            min_dist = float(d.min()) if len(d) > 0 else float('inf')
            a, b, c = np.linalg.norm(lattice[0]), np.linalg.norm(lattice[1]), np.linalg.norm(lattice[2])

            valid_structures.append({
                "index": i, "dg_h": dg_h,
                "deviation": abs(dg_h - TARGET_DG_H) if dg_h else None,
                "in_filter": in_filter_flag,
            })
            structure_info.append({
                "num_atoms": len(atoms), "elements": elem_counts,
                "a": round(a, 3), "b": round(b, 3), "c": round(c, 3),
                "min_distance": round(min_dist, 3),
            })
            if dg_h is not None:
                dg_h_valid.append(dg_h)
        except Exception:
            pass

elapsed = time.time() - start_time

print("\n" + "="*60)
print("EXPERIMENT RESULTS SUMMARY")
print("="*60)

print(f"\n[1] SUCCESS RATES")
print(f"  Generation: {gen_success}/{N} ({gen_success/N*100:.1f}%)")
parse_rate = parse_success/gen_success*100 if gen_success > 0 else 0
print(f"  Parsing:    {parse_success}/{gen_success} ({parse_rate:.1f}%)")
pred_rate = pred_success/parse_success*100 if parse_success > 0 else 0
print(f"  Prediction: {pred_success}/{parse_success} ({pred_rate:.1f}%)")
filter_rate = filter_pass/pred_success*100 if pred_success > 0 else 0
print(f"  Filtered:  {filter_pass}/{pred_success} ({filter_rate:.1f}%)")

print(f"\n[2] REJECTION LEVEL DISTRIBUTION")
for level in ["pass", "text", "structure", "physics", "unknown"]:
    count = rejection_levels.get(level, 0)
    pct = count/N*100
    bar = "#" * int(pct/2)
    print(f"  {level:12s}: {count:3d} ({pct:5.1f}%) {bar}")

print(f"\n[3] DELTA_G_H DISTRIBUTION")
if dg_h_all:
    dg_h_arr = np.array(dg_h_all)
    print(f"  Count:  {len(dg_h_all)}")
    print(f"  Mean:   {np.mean(dg_h_arr):.4f} eV")
    print(f"  Std:    {np.std(dg_h_arr):.4f} eV")
    print(f"  Min:    {np.min(dg_h_arr):.4f} eV | Max: {np.max(dg_h_arr):.4f} eV")
    print(f"  Median: {np.median(dg_h_arr):.4f} eV")
    in_target = np.sum((dg_h_arr >= FILTER_LOW) & (dg_h_arr <= FILTER_HIGH))
    print(f"  In target [{FILTER_LOW}, {FILTER_HIGH}]: {in_target}/{len(dg_h_all)} ({in_target/len(dg_h_all)*100:.1f}%)")
else:
    print("  No valid predictions")

print(f"\n[4] CANDIDATE QUALITY (target = {TARGET_DG_H:.2f} eV)")
if deviations:
    dev_arr = np.array(deviations)
    print(f"  Mean |deviation|: {np.mean(dev_arr):.4f} eV")
    print(f"  Min:  {np.min(dev_arr):.4f} eV | Max: {np.max(dev_arr):.4f} eV")
    sorted_indices = np.argsort(dev_arr)[:5]
    print(f"\n  Top 5 closest to target:")
    for rank, idx in enumerate(sorted_indices, 1):
        if idx < len(valid_structures):
            vs = valid_structures[idx]
            print(f"    #{rank} Sample{vs['index']}: DeltaG_H={vs['dg_h']:.4f} eV, |dev|={vs['deviation']:.4f} eV")

print(f"\n[5] STRUCTURE VALIDITY")
if structure_info:
    num_atoms_list = [s["num_atoms"] for s in structure_info]
    min_dists = [s["min_distance"] for s in structure_info]
    print(f"  Valid structures: {len(structure_info)}")
    print(f"  Atoms: mean={np.mean(num_atoms_list):.1f}, range=[{min(num_atoms_list)}, {max(num_atoms_list)}]")
    print(f"  Min distance: mean={np.mean(min_dists):.3f} A")
    all_elements = Counter()
    for s in structure_info:
        all_elements.update(s["elements"])
    print(f"  Elements: {dict(all_elements.most_common(5))}")
else:
    print("  No valid structure data")

print(f"\n[6] PERFORMANCE")
print(f"  Total time: {elapsed:.1f} s")
print(f"  Throughput: {N/elapsed:.2f} samples/s")
if gen_times:
    print(f"  Gen avg: {np.mean(gen_times)*1000:.1f} ms/sample")
if pred_times:
    print(f"  Pred avg: {np.mean(pred_times)*1000:.1f} ms/sample")

results = {
    "n_samples": N, "elapsed_s": round(elapsed, 2),
    "success_rates": {
        "gen_success": gen_success, "parse_success": parse_success,
        "pred_success": pred_success, "filter_pass": filter_pass,
        "gen_rate": round(gen_success/N, 4),
        "parse_rate": round(parse_success/gen_success, 4) if gen_success > 0 else 0,
        "pred_rate": round(pred_success/parse_success, 4) if parse_success > 0 else 0,
        "filter_rate": round(filter_pass/pred_success, 4) if pred_success > 0 else 0,
    },
    "rejection_level_distribution": dict(rejection_levels),
    "dg_h_stats": {
        "count": len(dg_h_all),
        "mean": round(float(np.mean(dg_h_arr)), 6) if dg_h_all else None,
        "std": round(float(np.std(dg_h_arr)), 6) if dg_h_all else None,
        "min": round(float(np.min(dg_h_arr)), 6) if dg_h_all else None,
        "max": round(float(np.max(dg_h_arr)), 6) if dg_h_all else None,
        "median": round(float(np.median(dg_h_arr)), 6) if dg_h_all else None,
        "all_values": [round(v, 6) for v in dg_h_all],
    },
    "deviation_stats": {
        "mean": round(float(np.mean(dev_arr)), 6) if deviations else None,
        "min": round(float(np.min(dev_arr)), 6) if deviations else None,
        "max": round(float(np.max(dev_arr)), 6) if deviations else None,
        "median": round(float(np.median(dev_arr)), 6) if deviations else None,
    },
    "structure_stats": {
        "valid_count": len(structure_info),
        "num_atoms": {
            "mean": round(np.mean(num_atoms_list), 2) if num_atoms_list else None,
            "min": min(num_atoms_list) if num_atoms_list else None,
            "max": max(num_atoms_list) if num_atoms_list else None,
        },
        "min_distance": {
            "mean": round(np.mean(min_dists), 3) if min_dists else None,
            "min": round(min(min_dists), 3) if min_dists else None,
        },
        "element_distribution": dict(all_elements) if structure_info else {},
    },
    "performance": {
        "gen_avg_ms": round(np.mean(gen_times)*1000, 1) if gen_times else None,
        "pred_avg_ms": round(np.mean(pred_times)*1000, 1) if pred_times else None,
        "throughput_sps": round(N/elapsed, 2),
    },
    "target_dg_h": TARGET_DG_H,
    "filter_range": [FILTER_LOW, FILTER_HIGH],
    "training_set_stats": {"mean": train_mean, "std": train_std},
}

out_path = SCRIPT_DIR / "benchmark_100_results.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n[INFO] Results saved to {out_path}")
print("="*60)
print("EXPERIMENT COMPLETE!")
print("="*60)
