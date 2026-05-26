# Archive Candidates

This document lists files and directories that can be archived later. It is not an instruction to delete them now.

## Archived

These files were moved without editing their contents.

| Path | Reason |
| --- | --- |
| `archive/demos/example_tabular_fusion.py` | Demo-style fusion script. |
| `archive/public_packaging_helpers/organize_experiment_results.py` | One-off result organization helper. |
| `archive/public_packaging_helpers/prepare_public_results.py` | Public-package helper. |
| `archive/report_helpers/detailed_analysis_report.py` | Report helper. |
| `archive/report_helpers/visualize_branch_architecture.py` | Figure-generation helper. |
| `archive/model_comparison_helpers/run_tree_baselines.py` | Baseline comparison helper. |
| `archive/old_ablation_runners/test_grid_lmax.py` | Test/exploration script under results directory. |
| `archive/old_ablation_runners/test_module_ablation_visual.py` | Test/exploration script under results directory. |
| `archive/old_ablation_runners/run_exp.py` | Historical ablation runner. |
| `archive/old_ablation_runners/run_lmax_experiment.py` | Historical lmax runner. |
| `archive/old_ablation_runners/run_module_ablation_fixed.py` | Historical module-ablation runner. |

## Preserve In Place

Do not archive these in the current cleanup pass.

| Path | Reason |
| --- | --- |
| `src/standalone_equiformer_v2.py` | Main Eqv2-Lite implementation. |
| `src/enhanced_equiformer_v2.py` | Main Enhanced baseline implementation. |
| `src/equiformer_v2/nets/equiformer_v2/` | Shared model internals. |
| `src/train_equiformer.py` | Main Eqv2-Lite training entry. |
| `src/train_enhanced_equiformer_v2.py` | Main baseline training entry. |
| `datasets/custom_data_processor_simplified.py` | Main data preprocessing code. |
| `datasets/custom_hydrogen/` | Main prepared dataset. |
| `checkpionts/` | Historical checkpoints. |
| `experiments/2025-07-09_run1/` | Eqv2-Lite main result. |
| `experiments/2025-09-10_run1/` | Enhanced baseline and ablation result tree. |
| `experiments/20251021_160614_tabfusion_run_real_equiformer_with_loss/` | Concat fusion result. |
| `experiments/20251021_165820_tabfusion_run_gate_fusion_100epochs/` | Gate fusion result. |
| `result/eq_lite_ablation/` | Ablation summaries. |
| `result/speed_benchmark/` | Inference-speed benchmark artifacts. |

## Safe Cleanup Candidates

These generated or local metadata files were deleted.

| Pattern | Reason |
| --- | --- |
| `__pycache__/` | Python bytecode cache. |
| `.DS_Store` | macOS metadata. |
| `*:Zone.Identifier` | Windows download metadata. |
| `micromamba.tar.bz2` | Installer/archive artifact. |

## Suggested Future Archive Layout

The current archive layout is:

```text
archive/
  demos/
  old_ablation_runners/
  report_helpers/
  public_packaging_helpers/
  model_comparison_helpers/
  legacy_oc20_src/
  later_chapter_application/
```

Each subdirectory has a short README explaining original paths.
