# -*- coding: utf-8 -*-
"""
System Performance and Quality Analysis Experiment
- 100-sample benchmark: generation/parsing/filtering rates, rejection_level distribution, ΔG_H distribution
- Candidate quality analysis: deviation from target -0.2 eV, structure validity, comparison with training set
"""
import sys
import os
import time
import json
import random
from pathlib import Path
from collections import Counter

# Configure paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

# Force DirectML (AMD acceleration)
os.environ["MATGEN_DEVICE"] = "directml"
# Use DEMO mode to bypass LLM generation (for testing pipeline)
os.environ["MATGEN_DEMO"] = "1"

from backend.rag_gen import generate as rag_generate
from backend.eq_predict import predict as eq_predict
from backend.quality import validate_structure
from config import DEFAULT_FILTER_LOW, DEFAULT_FILTER_HIGH, BASE_MODEL_PATH, NORM_STATS

import numpy as np
import ase.io
import io

# Random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Test prompts - English only to avoid encoding issues
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

print(f"[INFO] Generated {len(prompts)} prompts")
print(f"[INFO] Sample prompts:")
for i in range(3):
    print(f"  {i+1}. {prompts[i]}")

# Target parameters
TARGET_DG_H = -0.2
FILTER_LOW = DEFAULT_FILTER_LOW
FILTER_HIGH = DEFAULT_FILTER_HIGH

# Load normalization stats
with open(NORM_STATS) as f:
    norm_stats = json.load(f)
train_mean = norm_stats["target_mean"]
train_std = norm_stats["target_std"]
train_min = norm_stats.get("target_min", None)
train_max = norm_stats.get("target_max", None)
print(f"[INFO] Training set: mean={train_mean:.4f}, std={train_std:.4f}")

# Main experiment loop
N = len(prompts)
gen_times, parse_times, pred_times, filter_times = [], [], [], []

# Counters
gen_success = 0
parse_success = 0
pred_success = 0
filter_pass = 0

# rejection_level distribution
rejection_levels = Counter()

# DeltaG_H lists
dg_h_all = []
dg_h_valid = []

# Deviation stats
deviations = []

# Structure info
valid_structures = []
structure_info = []

# Failure counters
failed_by_rejection = Counter()

print(f"\n[INFO] Starting {N}-sample benchmark...")
print(f"[INFO] Target DeltaG_H: {TARGET_DG_H} eV, Filter: [{FILTER_LOW}, {FILTER_HIGH}] eV")
print(f"[INFO] Device: {os.environ.get('MATGEN_DEVICE', 'auto')}")

start_time = time.time()

for i, prompt in enumerate(prompts):
    if i % 10 == 0:
        print(f"  Progress: {i}/{N}")

    poscar = None
    dg_h = None
    rejection_level = "unknown"
    in_filter_flag = False

    # 1. Generation
    t0 = time.time()
    try:
        poscar = rag_generate(prompt, base_model_name_or_path=BASE_MODEL_PATH)
        gen_success += 1
    except Exception as e:
        rejection_level = "text"
        failed_by_rejection["gen_failed"] += 1
    gen_times.append(time.time() - t0)

    # 2. Parsing & validation
    t0 = time.time()
    if poscar:
        try:
            check = validate_structure(poscar)
            rejection_level = check.get("rejection_level", "pass")
            if rejection_level == "pass":
                parse_success += 1
            else:
                failed_by_rejection[rejection_level] += 1
        except Exception as e:
            rejection_level = "structure"
            failed_by_rejection["structure_error"] += 1
    parse_times.append(time.time() - t0)

    # 3. Prediction
    t0 = time.time()
    if poscar and rejection_level == "pass":
        try:
            res = eq_predict(poscar)
            dg_h = res["dg_h"]
            pred_success += 1
            dg_h_all.append(dg_h)
            deviations.append(abs(dg_h - TARGET_DG_H))
        except Exception as e:
            failed_by_rejection["pred_error"] += 1
    pred_times.append(time.time() - t0)

    # 4. Filtering
    t0 = time.time()
    if dg_h is not None:
        try:
            in_filter_flag = (float(FILTER_LOW) <= dg_h <= float(FILTER_HIGH))
            if in_filter_flag:
                filter_pass += 1
        except:
            pass
    filter_times.append(time.time() - t0)

    # 5. Record rejection_level
    rejection_levels[rejection_level] += 1

    # 6. Extract structure details
    if poscar and rejection_level == "pass":
        try:
            atoms = ase.io.read(io.StringIO(poscar), format="vasp")
            elem_counts = dict(Counter(atoms.get_chemical_symbols()))
            lattice = atoms.cell.array
            from ase.neighborlist import neighbor_list
            n1, n2, d = neighbor_list('ijd', atoms, cutoff=10.0)
            min_dist = float(d.min()) if len(d) > 0 else float('inf')
            a, b, c = np.linalg.norm(lattice[0]), np.linalg.norm(lattice[1]), np.linalg.norm(lattice[2])
            alpha, beta, gamma = atoms.cell.angles(reduced=False)

            valid_structures.append({
                "index": i,
                "dg_h": dg_h,
                "deviation": abs(dg_h - TARGET_DG_H) if dg_h else None,
                "in_filter": in_filter_flag,
            })
            structure_info.append({
                "num_atoms": len(atoms),
                "elements": elem_counts,
                "a": round(a, 3),
                "b": round(b, 3),
                "c": round(c, 3),
                "min_distance": round(min_dist, 3),
            })
            if dg_h is not None:
                dg_h_valid.append(dg_h)
        except Exception as e:
            pass

elapsed = time.time() - start_time

# Statistics
print("\n" + "="*60)
print("EXPERIMENT RESULTS SUMMARY")
print("="*60)

print(f"\n[1] SUCCESS RATES")
print(f"  Generation success:   {gen_success}/{N} ({gen_success/N*100:.1f}%)")
parse_rate = parse_success/gen_success*100 if gen_success > 0 else 0
print(f"  Parsing success:      {parse_success}/{gen_success} ({parse_rate:.1f}% if gen>0)")
pred_rate = pred_success/parse_success*100 if parse_success > 0 else 0
print(f"  Prediction success:   {pred_success}/{parse_success} ({pred_rate:.1f}% if parse>0)")
filter_rate = filter_pass/pred_success*100 if pred_success > 0 else 0
print(f"  Filtered in:          {filter_pass}/{pred_success} ({filter_rate:.1f}% if pred>0)")

print(f"\n[2] REJECTION LEVEL DISTRIBUTION")
total_valid = sum(v for k, v in rejection_levels.items() if k in ["pass", "text", "structure", "physics"])
for level in ["pass", "text", "structure", "physics", "unknown"]:
    count = rejection_levels.get(level, 0)
    pct = count/N*100
    bar = "#" * int(pct/2)
    print(f"  {level:12s}: {count:3d} ({pct:5.1f}%) {bar}")

print(f"\n[3] DELTA_G_H DISTRIBUTION")
if dg_h_all:
    dg_h_arr = np.array(dg_h_all)
    print(f"  Count:            {len(dg_h_all)}")
    print(f"  Mean:             {np.mean(dg_h_arr):.4f} eV")
    print(f"  Std:              {np.std(dg_h_arr):.4f} eV")
    print(f"  Min:              {np.min(dg_h_arr):.4f} eV")
    print(f"  Max:              {np.max(dg_h_arr):.4f} eV")
    print(f"  Median:           {np.median(dg_h_arr):.4f} eV")

    bins = [-2.0, -1.0, -0.5, -0.3, -0.2, -0.1, 0.0, 0.3, 0.5, 1.0, 2.0]
    print(f"  Range distribution:")
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        count = np.sum((dg_h_arr >= lo) & (dg_h_arr < hi))
        bar = "#" * int(count/max(len(dg_h_all),1)*50)
        print(f"    [{lo:5.2f}, {hi:5.2f}): {count:3d} {bar}")

    in_target = np.sum((dg_h_arr >= FILTER_LOW) & (dg_h_arr <= FILTER_HIGH))
    print(f"  In target range [{FILTER_LOW}, {FILTER_HIGH}]: {in_target}/{len(dg_h_all)} ({in_target/len(dg_h_all)*100:.1f}%)")
else:
    print(f"  No valid predictions")

print(f"\n[4] CANDIDATE QUALITY ANALYSIS (target DeltaG_H = {TARGET_DG_H:.2f} eV)")
if deviations:
    dev_arr = np.array(deviations)
    print(f"  Mean |deviation|: {np.mean(dev_arr):.4f} eV")
    print(f"  Min |deviation|:  {np.min(dev_arr):.4f} eV")
    print(f"  Max |deviation|:  {np.max(dev_arr):.4f} eV")
    print(f"  Median |deviation|: {np.median(dev_arr):.4f} eV")

    sorted_indices = np.argsort(dev_arr)[:5]
    print(f"\n  Top 5 best candidates (closest to target):")
    for rank, idx in enumerate(sorted_indices, 1):
        if idx < len(valid_structures):
            vs = valid_structures[idx]
            print(f"    #{rank} Sample{vs['index']}: DeltaG_H={vs['dg_h']:.4f} eV, |deviation|={vs['deviation']:.4f} eV, Filtered:{'Y' if vs['in_filter'] else 'N'}")

print(f"\n[5] STRUCTURE VALIDITY STATISTICS")
if structure_info:
    num_atoms_list = [s["num_atoms"] for s in structure_info]
    min_dists = [s["min_distance"] for s in structure_info]
    print(f"  Valid structures: {len(structure_info)}")
    print(f"  Num atoms: mean={np.mean(num_atoms_list):.1f}, range=[{min(num_atoms_list)}, {max(num_atoms_list)}]")
    print(f"  Min distance: mean={np.mean(min_dists):.3f} A, min={min(min_dists):.3f} A")

    all_elements = Counter()
    for s in structure_info:
        all_elements.update(s["elements"])
    print(f"  Element frequency:")
    for elem, count in all_elements.most_common():
        print(f"    {elem}: {count} times")

    a_vals = [s["a"] for s in structure_info]
    print(f"  Lattice constant a: mean={np.mean(a_vals):.3f} A, range=[{min(a_vals):.3f}, {max(a_vals):.3f}] A")
else:
    print(f"  No valid structure data")

print(f"\n[6] PERFORMANCE STATISTICS")
print(f"  Total time:       {elapsed:.1f} s")
if gen_times:
    print(f"  Gen avg:          {np.mean(gen_times)*1000:.1f} ms/sample")
if parse_times:
    print(f"  Parse avg:         {np.mean(parse_times)*1000:.1f} ms/sample")
if pred_times:
    print(f"  Pred avg:         {np.mean(pred_times)*1000:.1f} ms/sample")
print(f"  Throughput:        {N/elapsed:.2f} samples/s")

# Save results
results = {
    "n_samples": N,
    "elapsed_s": round(elapsed, 2),
    "success_rates": {
        "gen_success": gen_success,
        "parse_success": parse_success,
        "pred_success": pred_success,
        "filter_pass": filter_pass,
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
        "lattice_constant_a": {
            "mean": round(np.mean(a_vals), 3) if a_vals else None,
            "min": round(min(a_vals), 3) if a_vals else None,
            "max": round(max(a_vals), 3) if a_vals else None,
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
    "training_set_stats": {
        "mean": train_mean,
        "std": train_std,
        "min": train_min,
        "max": train_max,
    }
}

out_path = SCRIPT_DIR / "benchmark_100_results.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n[INFO] Results saved to {out_path}")

# Save best candidates
if valid_structures:
    sorted_by_dev = sorted(valid_structures, key=lambda x: x["deviation"] or 999)[:10]
    best_candidates = []
    for vs in sorted_by_dev:
        si = structure_info[vs["index"]] if vs["index"] < len(structure_info) else {}
        best_candidates.append({
            "rank": len(best_candidates) + 1,
            "sample_index": vs["index"],
            "dg_h": vs["dg_h"],
            "deviation": vs["deviation"],
            "in_filter": vs["in_filter"],
            "num_atoms": si.get("num_atoms"),
            "elements": si.get("elements"),
            "min_distance": si.get("min_distance"),
        })
    best_path = SCRIPT_DIR / "benchmark_best_candidates.json"
    with open(best_path, 'w', encoding='utf-8') as f:
        json.dump(best_candidates, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Top 10 candidates saved to {best_path}")

print("\n" + "="*60)
print("EXPERIMENT COMPLETE!")
print("="*60)
