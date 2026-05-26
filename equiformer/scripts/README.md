# Scripts Index

This directory contains Chapter 3 experiment helpers, result-processing scripts, and presentation utilities.

## Maintained Chapter 3 Scripts

| File | Role |
| --- | --- |
| `train_test_tabular_fusion.py` | Main structure-plus-tabular fusion experiment. |
| `run_eq_lite_ablation_depth.py` | Eqv2-Lite depth/layer ablation. |
| `run_channels.py` | Channel-width ablation. |
| `run_radial_ablation.py` | Radial-basis ablation. |
| `run_parameter_reallocation_ablation.py` | FFN/attention/edge/radial parameter-reallocation ablation. |
| `collect_ablation_results.py` | Collects ablation result tables. |
| `generate_ablation_plots.py` | Generates ablation figures. |

## Fusion And Data Helpers

| File | Role |
| --- | --- |
| `extract_equiformer_predictions.py` | Extracts structural model predictions for fusion workflows. |
| `process_real_data.py` | Prepares real tabular/fusion inputs. |
| `run_real_fusion_experiment.py` | Helper for real-data fusion runs. |

## Documentation Notes

| File | Role |
| --- | --- |
| `README_ablation_depth.md` | Notes for depth ablation. |

## Archived Out Of `scripts/`

These files were moved without editing their contents:

| File | Reason |
| --- | --- |
| `archive/demos/example_tabular_fusion.py` | Demo-style fusion script. |
| `archive/report_helpers/detailed_analysis_report.py` | Report-generation helper. |
| `archive/report_helpers/visualize_branch_architecture.py` | Architecture/fusion visualization helper. |
| `archive/public_packaging_helpers/organize_experiment_results.py` | One-off organization helper. |
| `archive/public_packaging_helpers/prepare_public_results.py` | Public-package preparation helper. |

See `archive/README.md` for the archive layout.
