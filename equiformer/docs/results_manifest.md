# Chapter 3 Results Manifest

This manifest maps thesis-facing Chapter 3 results to existing local files. It is an index only; it does not replace or regenerate any result.

Naming note: historical code/result paths use `EnhancedEquiformerV2` and `enhanced_equiformer_v2`, but this code path wraps the original `EquiformerV2_OC20` implementation. Thesis-facing text should call it `Original Eqv2 / EquiformerV2`; historical paths remain unchanged.

## Official Summary

| Item | Path |
| --- | --- |
| Chapter 3 result summary | `/home/leo494/mylab/实验结果/实验结果摘要.csv` |
| Thesis-facing consolidated result table | `docs/thesis_main_results.md` |

## Paper-Ready Result Groups

| Result group | Model / experiment | Metrics from current summary | Main directory |
| --- | --- | --- | --- |
| Main method | Eqv2-Lite | R2 0.9334, MAE 0.0868, RMSE 0.1798 | `experiments/2025-07-09_run1/` |
| Baseline | Original Eqv2 / EquiformerV2 | R2 0.8814, MAE 0.1320, RMSE 0.2325 | `experiments/2025-09-10_run1/enhanced_equiformerv2_baseline/` |
| Tuned original Eqv2 | Original Eqv2 / EquiformerV2, adapted config | R2 0.8951, MAE 0.1108, RMSE 0.2186 | `experiments/2025-09-10_run1/ablation/num_layers=2__sphere_channels=64__num_heads=4__grid_resolution=16__edge_channels=64__REAL_FUSION_BASE/seed=42/` |
| Fusion comparison | Eqv2 + tabular branch, concat | R2 0.9083, MAE 0.1049, RMSE 0.1867 | `experiments/20251021_160614_tabfusion_run_real_equiformer_with_loss/` |
| Fusion comparison | Eqv2 + tabular branch, gate | R2 0.9093, MAE 0.1024, RMSE 0.1857 | `experiments/20251021_165820_tabfusion_run_gate_fusion_100epochs/` |

## Ablation Result Groups

| Ablation | Summary path | Related script |
| --- | --- | --- |
| Depth / layers | `result/eq_lite_ablation/depth/summary.csv` | `scripts/run_eq_lite_ablation_depth.py` |
| Channels | `result/eq_lite_ablation/channels/summary.csv` | `scripts/run_channels.py` |
| Radial basis | `result/eq_lite_ablation/radial/summary.csv` | `scripts/run_radial_ablation.py` |
| lmax | `result/eq_lite_ablation/lmax/summary.csv` | `src/run_lmax_ablation.py` |
| Parameter reallocation | `result/eq_lite_ablation/parameter_reallocation/summary.csv` | `scripts/run_parameter_reallocation_ablation.py` |
| Collected ablation tables | `experiments/ablation/` | `scripts/collect_ablation_results.py` |

## Revision Evidence Tables

| Evidence | Path |
| --- | --- |
| Parameter count, performance, and inference speed | `docs/tables/ch3_param_perf_speed.csv` |
| Architecture ablation summary | `docs/tables/ch3_architecture_ablation.csv` |
| Local chemical environment modeling notes | `docs/tables/ch3_local_environment_modeling.md` |
| Reviewer-response evidence text | `docs/chapter3_revision_evidence.md` |

## Speed Result Group

Chapter 3 maintains inference-speed comparison only.

| Result | Path | Related script |
| --- | --- | --- |
| Inference-speed JSON | `result/speed_benchmark/infer_speed_results.json` | `result/speed_benchmark/infer_speed_test.py` |
| Inference-speed summary CSV | `result/speed_benchmark/speed_compare.csv` | `result/speed_benchmark/generate_comparison.py` |

## Needs Review Before Citation

| Path | Reason |
| --- | --- |
| `experiments/20251021_135737_tabfusion_run_real_data/` | Earlier fusion run; source/data alignment should be checked before using it in the thesis. |
| `experiments/model_comparison/` | Useful comparison tables, but should be matched to the final thesis wording before citation. |
| `experiments/struct_preds/`, `experiments/struct_preds_real/` | Intermediate prediction artifacts, not final metrics by themselves. |

## Preservation Rule

All result folders listed here are historical artifacts. Keep them in place unless a separate backup has been made and all references in this manifest are updated.
