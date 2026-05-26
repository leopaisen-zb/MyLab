# Experiments Directory Index

This directory preserves historical experiment outputs for Chapter 3. The cleanup process documents these outputs but does not move or overwrite them.

## Paper-Facing Result Directories

| Path | Role |
| --- | --- |
| `2025-07-09_run1/` | Eqv2-Lite / Standalone EquiformerV2 main result. |
| `2025-09-10_run1/` | Enhanced EquiformerV2 baseline, logs, and ablation outputs. |
| `20251021_160614_tabfusion_run_real_equiformer_with_loss/` | Concat fusion result used in the summary table. |
| `20251021_165820_tabfusion_run_gate_fusion_100epochs/` | Gate fusion result used in the summary table. |

## Supporting Or Intermediate Outputs

| Path | Role |
| --- | --- |
| `struct_preds/` | Intermediate structural predictions. |
| `struct_preds_real/` | Intermediate real-data structural predictions. |
| `model_comparison/` | Model-comparison tables and figures. |
| `architecture_visualization/` | Architecture/fusion mechanism figures. |
| `ablation/` | Ablation runners, collected tables, and plots. |
| `now_result/` | Current/assembled result figures. |

## Needs Review Before Citation

| Path | Reason |
| --- | --- |
| `20251021_135737_tabfusion_run_real_data/` | Earlier fusion run; check source/data alignment before citation. |
| `20251021_135230_tabfusion_run_demo/` | Demo run. |
| `20251021_135335_tabfusion_run_demo/` | Demo run. |
| `20251021_160530_tabfusion_run_real_equiformer_with_loss/` | Earlier fusion run. |
| `20251021_160558_tabfusion_run_real_equiformer_with_loss/` | Earlier fusion run. |
| `20251021_165725_tabfusion_run_gate_fusion/` | Earlier gate-fusion run. |

## Archived Out Of Experiments

Some historical runners were moved out of result folders so result directories contain outputs rather than executable clutter:

- `archive/old_ablation_runners/`
- `archive/model_comparison_helpers/`

## Preservation Rule

Do not rename, delete, or overwrite experiment directories during cleanup. Use `docs/results_manifest.md` for paper-facing mappings and `docs/archive_candidates.md` for future archive review.
