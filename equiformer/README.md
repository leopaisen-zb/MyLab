# Eqv2-Lite: Chapter 3 EquiformerV2 Project

This directory contains the Chapter 3 code, documents, and preserved experiment outputs for hydrogen adsorption free-energy prediction.

The main research line is:

- `Eqv2-Lite` / `Equiformer-Light`: lightweight EquiformerV2-style model and main contribution.
- `Enhanced EquiformerV2`: full EquiformerV2-style baseline used for comparison.
- `Eqv2 + tabular fusion`: auxiliary comparison with handcrafted/tabular descriptors.
- Inference-speed benchmark: efficiency evidence for Eqv2-Lite.

## Project Map

Start from these files:

- `docs/documentation_index.md`: which document to open first.
- `docs/project-organization.md`: current code/result/document classification.
- `docs/code_index.md`: concise code navigation index.
- `docs/cleanup_status.md`: current cleanup progress and untouched areas.
- `docs/chapter3-code-inventory.md`: detailed keep/archive/delete recommendation list.
- `docs/results_manifest.md`: mapping from paper results to existing result folders.
- `docs/artifacts.md`: data, checkpoint, and result preservation policy.
- `docs/archive_candidates.md`: files that can be reviewed for future archival.
- `docs/README.md`: original upstream EquiformerV2 documentation.

Directory-level guides:

- `src/README.md`: source-code guide.
- `scripts/README.md`: script guide.
- `datasets/README.md`: dataset guide.
- `result/README.md`: result guide.
- `experiments/README.md`: experiment-output guide.

## Useful Code

Core model and training code:

- `src/standalone_equiformer_v2.py`: Eqv2-Lite / Equiformer-Light model.
- `src/train_equiformer.py`: Eqv2-Lite training entry point.
- `src/evaluate_equiformer_v2.py`: Eqv2-Lite evaluation entry point.
- `src/enhanced_equiformer_v2.py`: full Enhanced EquiformerV2 baseline model.
- `src/train_enhanced_equiformer_v2.py`: Enhanced baseline training entry point.
- `src/predict_enhanced_equiformer_v2.py`: Enhanced baseline prediction entry point.
- `src/equiformer_v2/nets/equiformer_v2/`: EquiformerV2 core network modules.

Data preparation and auxiliary experiments:

- `datasets/custom_data_processor_simplified.py`: VASP/Excel to LMDB preprocessing.
- `scripts/train_test_tabular_fusion.py`: Eqv2 prediction plus tabular fusion.
- `scripts/run_eq_lite_ablation_depth.py`: Eqv2-Lite depth ablation.
- `scripts/run_channels.py`: channel-width ablation.
- `scripts/run_radial_ablation.py`: radial-basis ablation.
- `src/run_lmax_ablation.py`: lmax ablation.
- `result/speed_benchmark/infer_speed_test.py`: inference-speed benchmark.

## Preserved Results

Existing result directories are preserved in place. Do not rename or overwrite them when reorganizing the project.

Primary Chapter 3 result summary:

- `/home/leo494/mylab/实验结果/实验结果摘要.csv`

Important result directories:

- `experiments/2025-07-09_run1/`: Eqv2-Lite main result.
- `experiments/2025-09-10_run1/`: Enhanced EquiformerV2 baseline and ablation outputs.
- `experiments/20251021_160614_tabfusion_run_real_equiformer_with_loss/`: concat fusion result.
- `experiments/20251021_165820_tabfusion_run_gate_fusion_100epochs/`: gate fusion result.
- `result/eq_lite_ablation/`: Eqv2-Lite ablation summaries.
- `result/speed_benchmark/`: inference-speed benchmark files.

See `docs/results_manifest.md` before citing or moving any result.

## Data And Checkpoints

Large artifacts remain local:

- `datasets/custom_hydrogen/`
- `checkpionts/`
- `data/raw/`
- `result/`
- `experiments/`

See `docs/artifacts.md` for the preservation rules.

## Current Cleanup Rule

This repository is being organized for readability, not for rerunning experiments. The first cleanup pass only adds documentation and lightweight index files, and keeps all historical experiment outputs unchanged.
