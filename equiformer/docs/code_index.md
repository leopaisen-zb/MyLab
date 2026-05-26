# Chapter 3 Code Index

This is a short code index for day-to-day navigation. For detailed keep/archive reasoning, see `docs/chapter3-code-inventory.md`.

## A. Main Chapter 3 Code

These files form the maintained Chapter 3 model path.

| Path | Category | Keep status |
| --- | --- | --- |
| `src/standalone_equiformer_v2.py` | Eqv2-Lite model | Keep |
| `src/train_equiformer.py` | Eqv2-Lite training | Keep |
| `src/evaluate_equiformer_v2.py` | Eqv2-Lite evaluation | Keep |
| `src/enhanced_equiformer_v2.py` | Enhanced EquiformerV2 baseline | Keep |
| `src/train_enhanced_equiformer_v2.py` | Enhanced baseline training | Keep |
| `src/predict_enhanced_equiformer_v2.py` | Enhanced baseline prediction | Keep |
| `src/equiformer_v2/nets/equiformer_v2/` | Shared EquiformerV2 modules | Keep |
| `datasets/custom_data_processor_simplified.py` | Data preprocessing | Keep |

## B. Chapter 3 Experiment Helpers

These files support fusion experiments, ablations, plotting, and result collection.

| Path | Category | Keep status |
| --- | --- | --- |
| `scripts/train_test_tabular_fusion.py` | Fusion experiment | Keep |
| `scripts/run_real_fusion_experiment.py` | Fusion run helper | Keep as auxiliary |
| `scripts/extract_equiformer_predictions.py` | Fusion input preparation | Keep as auxiliary |
| `scripts/process_real_data.py` | Fusion/tabular data processing | Keep as auxiliary |
| `scripts/run_eq_lite_ablation_depth.py` | Depth ablation | Keep |
| `scripts/run_channels.py` | Channel ablation | Keep |
| `scripts/run_radial_ablation.py` | Radial ablation | Keep |
| `scripts/run_parameter_reallocation_ablation.py` | Parameter reallocation ablation | Keep |
| `src/run_lmax_ablation.py` | lmax ablation | Keep |
| `src/run_radial_ablation.py` | Radial ablation variant | Keep, but reconcile with `scripts/run_radial_ablation.py` later |
| `scripts/collect_ablation_results.py` | Ablation collection | Keep |
| `scripts/generate_ablation_plots.py` | Ablation plotting | Keep as auxiliary |
| `result/speed_benchmark/infer_speed_test.py` | Inference-speed benchmark | Keep |
| `result/speed_benchmark/generate_comparison.py` | Inference-speed summary | Keep |

## C. Documentation And Presentation Helpers

These are useful for explaining the method, but they are not core training/evaluation code.

| Path | Category | Keep status |
| --- | --- | --- |
| `experiments/architecture_visualization/` | Generated figures | Preserve |
| `scripts/README_ablation_depth.md` | Ablation note | Preserve |
| `archive/report_helpers/detailed_analysis_report.py` | Report helper | Archived |
| `archive/report_helpers/visualize_branch_architecture.py` | Architecture figure helper | Archived |

## D. Archived Legacy Or Upstream-Compatible Code

These files were moved out of `src/` because they are not part of the maintained Chapter 3 path.

| Path | Category | Suggested handling |
| --- | --- | --- |
| `archive/legacy_oc20_src/main_oc20.py` | Upstream-style OC20 entry | Archived |
| `archive/legacy_oc20_src/engine.py` | Upstream training utility | Archived |
| `archive/legacy_oc20_src/logger.py` | Upstream logging utility | Archived |
| `archive/legacy_oc20_src/optim_factory.py` | Upstream optimizer utility | Archived |
| `archive/legacy_oc20_src/utils.py` | Upstream utility | Archived |
| `archive/legacy_oc20_src/train_custom_hydrogen.py` | Old custom training entry | Archived |
| `docs/README.md` | Upstream EquiformerV2 README | Keep as upstream reference |

## E. Archived Later-Chapter Or Application Code

These files are useful, but they are outside the current Chapter 3 cleanup scope and have been moved to `archive/later_chapter_application/`.

| Path | Category | Suggested handling |
| --- | --- | --- |
| `archive/later_chapter_application/screen_materials.py` | Materials screening | Archived |
| `archive/later_chapter_application/add_material_cost.py` | Cost annotation | Archived |
| `archive/later_chapter_application/metal_price_dict.py` | Cost dictionary | Archived |
| `archive/later_chapter_application/demjson.py` | Compatibility helper | Archived |

## F. Removed Generated Or Local-Only Files

These generated or local-only files were deleted during cleanup.

| Path pattern | Category | Suggested handling |
| --- | --- | --- |
| `__pycache__/` | Python cache | Deleted |
| `.DS_Store` | macOS metadata | Deleted |
| `*:Zone.Identifier` | Windows/WSL metadata | Deleted |
| `.vscode/` | Local editor config | Keep local or ignore |
| `micromamba.tar.bz2` | Installer/cache artifact | Deleted |

## Current Rule

Core Chapter 3 files remain in their original paths. Archived files were moved without editing their contents.
