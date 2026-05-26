# Chapter 3 Evidence And Parameter Reallocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn existing Chapter 3 experiments into thesis-ready evidence tables and add one focused parameter-reallocation ablation to justify why Eqv2-Lite allocates capacity to scalar FFN rather than attention, edge, or radial modules.

**Architecture:** First, generate paper-facing tables and text from existing result files without rerunning experiments. Second, add a small standalone ablation runner that varies capacity allocation under the same Eqv2-Lite data/model pipeline and writes new results to a separate directory. Existing historical results, checkpoints, and LMDB files are read-only inputs.

**Tech Stack:** Python 3.10, pandas, PyTorch, existing `StandaloneEquiformerV2`, existing LMDB dataset, Markdown, CSV.

---

## Non-Negotiable Constraints

- Do not edit historical CSV/JSON/PNG result files.
- Do not overwrite `experiments/`, `result/lab01/`, `result/eq_lite_ablation/`, or `result/speed_benchmark/`.
- Treat `Rc=6.0 Å` as a thesis typo. The project evidence uses `max_radius=12.0 Å`.
- Do not change existing model definitions while preparing tables.
- New experiments must write only to `result/eq_lite_ablation/parameter_reallocation/`.
- The new parameter-reallocation experiment must use the same data split as existing Eqv2-Lite experiments: `datasets/custom_hydrogen/train.lmdb` and `datasets/custom_hydrogen/val.lmdb`.

## Files To Create Or Modify

- Create: `docs/chapter3_revision_evidence.md`
  - Thesis-ready evidence summary for reviewer comments.
- Create: `docs/tables/ch3_param_perf_speed.csv`
  - Parameter count, performance, and inference speed comparison table.
- Create: `docs/tables/ch3_architecture_ablation.csv`
  - Lmax, depth, channel, radial, and module ablation table assembled from existing outputs.
- Create: `docs/tables/ch3_local_environment_modeling.md`
  - Local chemical environment modeling explanation and code/result anchors.
- Create: `experiments/ablation/grids/parameter_reallocation.csv`
  - New experiment grid for parameter allocation.
- Create: `scripts/run_parameter_reallocation_ablation.py`
  - New runner based on existing Eqv2-Lite ablation style.
- Create: `docs/parameter_reallocation_experiment.md`
  - Experiment protocol, command, expected output, and interpretation rules.
- Modify: `docs/results_manifest.md`
  - Add references to generated tables and the planned new ablation output.
- Modify: `docs/chapter3-code-inventory.md`
  - Add the new evidence files and parameter-reallocation experiment.

## Existing Evidence To Use

Use these files directly:

- Main paper summary: `/home/leo494/mylab/实验结果/实验结果摘要.csv`
- Eqv2-Lite evaluation: `experiments/2025-07-09_run1/evaluation_metrics.csv`
- Eqv2-Lite training history: `result/lab01/training_history.json`
- Enhanced baseline logs: `experiments/2025-09-10_run1/logs/enhanced_equiformer_v2_test_results.json`
- Enhanced training history: `experiments/2025-09-10_run1/logs/enhanced_equiformer_v2_training_history.json`
- Speed comparison: `result/speed_benchmark/infer_speed_results.json`
- Speed configs: `result/speed_benchmark/eq_lite.json`, `result/speed_benchmark/eqv2_best.json`
- Depth ablation: `result/eq_lite_ablation/depth/summary.csv`
- Channel ablation: `result/eq_lite_ablation/channels/summary.csv`
- Lmax ablation: `result/eq_lite_ablation/lmax/summary.csv`
- Radial basis ablation: `result/eq_lite_ablation/radial/summary.csv`
- Module ablation: `experiments/ablation/results_modules_corrected.csv`
- Dataset summary: `datasets/custom_hydrogen/data_summary.json`
- Local environment code anchors:
  - `datasets/custom_data_processor_simplified.py`
  - `src/standalone_equiformer_v2.py`
  - `src/equiformer_v2/nets/equiformer_v2/input_block.py`
  - `src/equiformer_v2/nets/equiformer_v2/transformer_block.py`

## Task 1: Create Parameter-Performance-Speed Table From Existing Results

**Files:**
- Create: `docs/tables/ch3_param_perf_speed.csv`
- Modify: `docs/chapter3_revision_evidence.md`

- [ ] **Step 1: Create `docs/tables/`**

Run:

```bash
mkdir -p docs/tables
```

Expected:

```text
docs/tables exists
```

- [ ] **Step 2: Write `docs/tables/ch3_param_perf_speed.csv`**

Create this exact table:

```csv
model,role,params_m,param_reduction_vs_original,infer_samples_per_s,infer_speedup_vs_original,r2,mae,rmse,source
Enhanced EquiformerV2,full baseline,38.82,1.00,3.13,1.00,0.8814,0.1320,0.2325,"/home/leo494/mylab/实验结果/实验结果摘要.csv; result/speed_benchmark/infer_speed_results.json"
Eqv2-Lite,lightweight model,4.69,8.29,86.99,27.78,0.9334,0.0868,0.1798,"/home/leo494/mylab/实验结果/实验结果摘要.csv; result/speed_benchmark/infer_speed_results.json"
```

- [ ] **Step 3: Add thesis-ready interpretation to `docs/chapter3_revision_evidence.md`**

Add:

```markdown
# Chapter 3 Revision Evidence

## 1. Lightweight Design Necessity

The existing results show that the full Enhanced EquiformerV2 baseline is not the best choice for this medium-sized hydrogen adsorption dataset. Compared with Enhanced EquiformerV2, Eqv2-Lite reduces parameter count from 38.82M to 4.69M, improves inference throughput from 3.13 to 86.99 samples/s, and improves test performance from MAE 0.1320 eV / RMSE 0.2325 eV / R2 0.8814 to MAE 0.0868 eV / RMSE 0.1798 eV / R2 0.9334.

This supports the revision argument that the lightweight architecture is not only an efficiency optimization, but also a better capacity match for the current high-entropy-alloy hydrogen adsorption dataset.

Table source: `docs/tables/ch3_param_perf_speed.csv`.
```

- [ ] **Step 4: Verify the table contains the required fields**

Run:

```bash
python3 - <<'PY'
import pandas as pd
df = pd.read_csv('docs/tables/ch3_param_perf_speed.csv')
required = {'model','params_m','infer_samples_per_s','r2','mae','rmse'}
missing = required - set(df.columns)
assert not missing, missing
assert len(df) == 2
print('OK param-performance-speed table')
PY
```

Expected:

```text
OK param-performance-speed table
```

## Task 2: Create Architecture Ablation Table From Existing Results

**Files:**
- Create: `docs/tables/ch3_architecture_ablation.csv`
- Modify: `docs/chapter3_revision_evidence.md`

- [ ] **Step 1: Create consolidated ablation CSV**

Create `docs/tables/ch3_architecture_ablation.csv` with these rows:

```csv
ablation_type,setting,mae,rmse,r2,loss,num_parameters,source,interpretation
depth,num_layers=1,0.1035,0.2142,0.9055,0.1018,3.57M,result/eq_lite_ablation/depth/summary.csv,"under-capacity"
depth,num_layers=2,0.1025,0.2002,0.9174,0.0889,6.75M,result/eq_lite_ablation/depth/summary.csv,"improved"
depth,num_layers=3,0.0943,0.1936,0.9228,0.0831,9.94M,result/eq_lite_ablation/depth/summary.csv,"best depth in this ablation"
depth,num_layers=4,0.0981,0.2088,0.9101,0.0967,13.12M,result/eq_lite_ablation/depth/summary.csv,"more parameters but worse than 3 layers"
lmax,lmax=3,0.1041,0.2105,0.9087,0.0982,12.99M,result/eq_lite_ablation/lmax/summary.csv,"under-expressive"
lmax,lmax=4,0.0996,0.1870,0.9280,0.0774,19.49M,result/eq_lite_ablation/lmax/summary.csv,"best angular order"
lmax,lmax=5,0.1076,0.2068,0.9119,0.0947,27.46M,result/eq_lite_ablation/lmax/summary.csv,"higher-order features do not improve"
lmax,lmax=6,0.1138,0.2085,0.9104,0.0963,36.91M,result/eq_lite_ablation/lmax/summary.csv,"largest but worse"
radial,num_gaussians=32,0.1023,0.1936,0.9228,0.0831,19.40M,result/eq_lite_ablation/radial/summary.csv,"coarse radial basis"
radial,num_gaussians=64,0.1132,0.2235,0.8970,0.1108,19.43M,result/eq_lite_ablation/radial/summary.csv,"worse"
radial,num_gaussians=128,0.0971,0.1996,0.9179,0.0884,19.49M,result/eq_lite_ablation/radial/summary.csv,"good"
radial,num_gaussians=256,0.0954,0.1882,0.9270,0.0785,19.61M,result/eq_lite_ablation/radial/summary.csv,"best radial setting"
module,all_on,0.1115,0.2105,,0.0984,,experiments/ablation/results_modules_corrected.csv,"reference"
module,no_attn_renorm,0.1147,0.2182,,0.1057,,experiments/ablation/results_modules_corrected.csv,"attention renormalization helps"
module,no_sep_s2,0.1144,0.2180,,0.1055,,experiments/ablation/results_modules_corrected.csv,"separable S2 helps"
module,no_sep_ln,0.1088,0.2054,,0.0936,,experiments/ablation/results_modules_corrected.csv,"separate LN is not necessary here"
module,only_escn,0.1166,0.2222,,0.1096,,experiments/ablation/results_modules_corrected.csv,"removing all auxiliary modules hurts"
```

- [ ] **Step 2: Add thesis-ready interpretation**

Append to `docs/chapter3_revision_evidence.md`:

```markdown
## 2. Architecture Pruning Rationale

The existing ablations support the pruning choices:

- Angular order: Lmax=4 gives the best balance. Lmax=5 and Lmax=6 increase parameters to 27.46M and 36.91M but degrade MAE/RMSE.
- Depth: 3 layers outperform 1, 2, and 4 layers in the Eqv2-Lite depth ablation, showing that simply deepening the model is not beneficial.
- Radial basis: 256 Gaussian bases gives the best radial result in the existing sweep.
- Module switches: attention renormalization and separable S2 operations improve the tested baseline, while separate layer normalization is not necessary in this dataset.

Table source: `docs/tables/ch3_architecture_ablation.csv`.
```

- [ ] **Step 3: Verify no empty required metrics in non-module rows**

Run:

```bash
python3 - <<'PY'
import pandas as pd
df = pd.read_csv('docs/tables/ch3_architecture_ablation.csv')
non_module = df[df['ablation_type'] != 'module']
assert non_module[['mae','rmse','r2','loss','num_parameters']].notna().all().all()
print('OK architecture ablation table')
PY
```

Expected:

```text
OK architecture ablation table
```

## Task 3: Add Local Chemical Environment Modeling Explanation

**Files:**
- Create: `docs/tables/ch3_local_environment_modeling.md`
- Modify: `docs/chapter3_revision_evidence.md`

- [ ] **Step 1: Create local environment explanation**

Create `docs/tables/ch3_local_environment_modeling.md`:

```markdown
# Local Chemical Environment Modeling Evidence

The project uses `max_radius=12.0 Å`, not `Rc=6.0 Å`. The `Rc=6.0 Å` statement in the thesis draft should be corrected.

## Graph Construction

- `datasets/custom_data_processor_simplified.py` builds a distance graph from atomic coordinates.
- Each atom connects to neighbors within `max_radius=12.0 Å`.
- Neighbor count is capped by `max_neighbors`.
- Prepared dataset summary records `max_radius=12.0` and `max_neighbors=50`.
- Model training and benchmark configurations use `max_radius=12.0` and `max_neighbors=20`.

## Chemical Identity Encoding

- Each structure stores `atomic_numbers`.
- `src/standalone_equiformer_v2.py` maps atomic numbers through `sphere_embedding`.
- EquiformerV2 edge blocks support source/target atomic embeddings through `use_atom_edge_embedding=True`.
- These embeddings let the model distinguish multi-element local coordination environments.

## Local Geometry Encoding

- Edge vectors and edge distances are computed from atomic positions.
- Distances are expanded by Gaussian radial basis functions.
- SO3/SO2 blocks process equivariant angular information from local neighborhoods.

## Tabular Local Descriptors

The prepared dataset also contains local descriptor columns such as:

- `CN`
- `L_bond`
- `R0`, `R1`, `R`
- element-shell descriptors including `Nd`, `Np`, `Ns`, `Out_e`, `First_IE`

These descriptors are used in the structure-plus-tabular fusion experiments and can be cited as auxiliary evidence that the workflow explicitly represents high-entropy-alloy local coordination and chemical heterogeneity.
```

- [ ] **Step 2: Add interpretation to revision evidence**

Append to `docs/chapter3_revision_evidence.md`:

```markdown
## 3. Adaptation To High-Entropy-Alloy HER Surface Configurations

The model is adapted to the target system through graph-based local environment encoding. Atomic coordinates define local neighborhoods, atomic numbers encode multi-element chemical identity, and distance/edge features encode local coordination geometry. The dataset also includes local coordination descriptors such as CN, bond length, and atomic-radius-related features, which are used by the fusion branch as auxiliary descriptors.

The thesis should correct `Rc=6.0 Å` to `max_radius=12.0 Å` for the current project evidence, and explain that the graph uses a radius cutoff plus neighbor cap to cover local coordination shells around adsorption structures.

Detailed source anchors: `docs/tables/ch3_local_environment_modeling.md`.
```

- [ ] **Step 3: Verify the typo correction is explicit**

Run:

```bash
rg -n "Rc=6.0|max_radius=12.0|12.0 Å" docs/tables/ch3_local_environment_modeling.md docs/chapter3_revision_evidence.md
```

Expected output includes:

```text
Rc=6.0 Å
max_radius=12.0 Å
12.0 Å
```

## Task 4: Add Parameter-Reallocation Experiment Grid

**Files:**
- Create: `experiments/ablation/grids/parameter_reallocation.csv`
- Create: `docs/parameter_reallocation_experiment.md`

- [ ] **Step 1: Create experiment grid**

Create `experiments/ablation/grids/parameter_reallocation.csv`:

```csv
name,num_layers,sphere_channels,num_heads,attn_hidden_channels,attn_alpha_channels,attn_value_channels,ffn_hidden_channels,edge_channels,num_gaussians,lmax,max_radius,max_neighbors,seed
ffn_heavy,3,128,4,32,16,8,192,128,256,4,12.0,20,42
ffn_light,3,128,4,32,16,8,64,128,256,4,12.0,20,42
attention_heavy,3,128,4,64,32,16,128,128,256,4,12.0,20,42
edge_heavy,3,128,4,32,16,8,128,192,256,4,12.0,20,42
radial_heavy,3,128,4,32,16,8,128,128,512,4,12.0,20,42
balanced,3,128,4,48,24,12,128,160,256,4,12.0,20,42
```

Then duplicate the same six rows for seeds `0` and `1` if time allows. The first pass with seed `42` is the pilot; the three-seed run is the thesis-quality run.

- [ ] **Step 2: Create protocol document**

Create `docs/parameter_reallocation_experiment.md`:

```markdown
# Parameter Reallocation Ablation

## Purpose

This experiment answers the reviewer concern that the choice to allocate model capacity toward scalar FFN, rather than attention, edge, or radial modules, needs direct evidence.

## Controlled Baseline

All variants keep:

- `num_layers=3`
- `sphere_channels=128`
- `num_heads=4`
- `lmax=4`
- `max_radius=12.0`
- `max_neighbors=20`
- `use_atom_edge_embedding=True`
- same train/validation LMDB splits

## Compared Variants

| Variant | Main change | Hypothesis |
| --- | --- | --- |
| `ffn_heavy` | Increase `ffn_hidden_channels` to 192 | Tests whether scalar nonlinear capacity improves adsorption-energy regression. |
| `ffn_light` | Reduce `ffn_hidden_channels` to 64 | Tests whether removing scalar FFN capacity hurts. |
| `attention_heavy` | Increase attention hidden/alpha/value channels | Tests whether attention capacity is a better target. |
| `edge_heavy` | Increase `edge_channels` | Tests whether edge embedding capacity is a better target. |
| `radial_heavy` | Increase `num_gaussians` to 512 | Tests whether radial basis capacity is a better target. |
| `balanced` | Moderate increases across attention and edge | Tests whether distributed capacity beats FFN-focused capacity. |

## Output

The runner writes:

```text
result/eq_lite_ablation/parameter_reallocation/
  summary.csv
  <variant>_seed<seed>/metrics.json
```

## Decision Rule

The thesis should claim FFN reallocation is supported only if `ffn_heavy` has lower MAE/RMSE than `ffn_light` and is competitive with or better than `attention_heavy`, `edge_heavy`, `radial_heavy`, and `balanced` under comparable parameter counts.
```

## Task 5: Implement Minimal Parameter-Reallocation Runner

**Files:**
- Create: `scripts/run_parameter_reallocation_ablation.py`

- [ ] **Step 1: Copy the existing Eqv2-Lite ablation training pattern**

Use `scripts/run_radial_ablation.py` and `scripts/run_eq_lite_ablation_depth.py` as references.

The new script must:

- read `experiments/ablation/grids/parameter_reallocation.csv`,
- instantiate `StandaloneEquiformerV2`,
- override Gaussian basis count when `num_gaussians` differs,
- train each variant on `datasets/custom_hydrogen/train.lmdb`,
- evaluate on `datasets/custom_hydrogen/val.lmdb`,
- write `metrics.json` per variant,
- write `summary.csv`.

- [ ] **Step 2: Add a dry-run mode**

The script must support:

```bash
python scripts/run_parameter_reallocation_ablation.py --dry-run
```

Expected output:

```text
variant, seed, num_parameters
```

No training happens in dry-run mode.

- [ ] **Step 3: Add a pilot mode**

The script must support:

```bash
python scripts/run_parameter_reallocation_ablation.py --grid experiments/ablation/grids/parameter_reallocation.csv --max-runs 6
```

Expected:

```text
result/eq_lite_ablation/parameter_reallocation/summary.csv
```

- [ ] **Step 4: Add a full mode**

After pilot review, duplicate seeds `0`, `1`, and `42` in the CSV and run:

```bash
python scripts/run_parameter_reallocation_ablation.py --grid experiments/ablation/grids/parameter_reallocation.csv
```

Expected:

```text
result/eq_lite_ablation/parameter_reallocation/summary.csv
```

## Task 6: Add New Results To Thesis Evidence After Experiment Completes

**Files:**
- Modify: `docs/chapter3_revision_evidence.md`
- Modify: `docs/results_manifest.md`
- Modify: `docs/tables/ch3_architecture_ablation.csv`

- [ ] **Step 1: Add parameter-reallocation summary**

After `result/eq_lite_ablation/parameter_reallocation/summary.csv` exists, add:

```markdown
## 4. Parameter Reallocation Evidence

The parameter-reallocation ablation directly compares capacity allocated to FFN, attention, edge embeddings, radial basis, and balanced configurations under the same Eqv2-Lite data split. The result should be interpreted by MAE/RMSE first and parameter count second.

Table source: `result/eq_lite_ablation/parameter_reallocation/summary.csv`.
```

- [ ] **Step 2: Update result manifest**

Add this row to `docs/results_manifest.md` under ablations:

```markdown
| Parameter reallocation | `result/eq_lite_ablation/parameter_reallocation/summary.csv` | `scripts/run_parameter_reallocation_ablation.py` |
```

- [ ] **Step 3: Verify manifest references existing paths**

Run:

```bash
test -f docs/chapter3_revision_evidence.md
test -f docs/tables/ch3_param_perf_speed.csv
test -f docs/tables/ch3_architecture_ablation.csv
test -f docs/tables/ch3_local_environment_modeling.md
test -f experiments/ablation/grids/parameter_reallocation.csv
test -f scripts/run_parameter_reallocation_ablation.py
```

Expected: command exits with status 0.

## Self-Review

- Spec coverage: the plan covers the parameter-performance-speed table, architecture ablation table, local chemical environment explanation, and the new FFN-vs-other-modules parameter reallocation experiment.
- Known correction: `Rc=6.0 Å` is treated as a thesis typo and replaced with `max_radius=12.0 Å`.
- Existing evidence is separated from new experiment evidence.
- No historical result file is overwritten.
- The new experiment has a pilot mode and a dry-run parameter-count mode before any full training.
