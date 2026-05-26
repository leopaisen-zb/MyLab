# Chapter 3 Project Organization

This document is the working index for the Chapter 3 EquiformerV2 / Eqv2-Lite project. It classifies the current files by role so the project can be read without guessing which scripts matter.

## 1. Core Chapter 3 Code

Keep these files as the maintained implementation surface.

For a shorter source-code navigation table, see `docs/code_index.md`.

| Area | Path | Role |
| --- | --- | --- |
| Eqv2-Lite model | `src/standalone_equiformer_v2.py` | Main lightweight EquiformerV2 / Equiformer-Light implementation. |
| Eqv2-Lite training | `src/train_equiformer.py` | Main training entry point for Eqv2-Lite. |
| Eqv2-Lite evaluation | `src/evaluate_equiformer_v2.py` | Evaluation entry point for Eqv2-Lite outputs. |
| Enhanced baseline model | `src/enhanced_equiformer_v2.py` | Full EquiformerV2-style baseline. |
| Enhanced baseline training | `src/train_enhanced_equiformer_v2.py` | Training entry point for the baseline. |
| Enhanced baseline prediction | `src/predict_enhanced_equiformer_v2.py` | Prediction/evaluation helper for the baseline. |
| EquiformerV2 modules | `src/equiformer_v2/nets/equiformer_v2/` | Core network components used by both variants. |
| Data processing | `datasets/custom_data_processor_simplified.py` | Converts raw VASP/Excel data into local LMDB splits. |

## 2. Chapter 3 Auxiliary Code

Keep these for paper analysis, ablation, or result explanation.

| Area | Path | Role |
| --- | --- | --- |
| Tabular fusion | `scripts/train_test_tabular_fusion.py` | Structural prediction plus tabular descriptor fusion. |
| Depth ablation | `scripts/run_eq_lite_ablation_depth.py` | Eqv2-Lite layer-count comparison. |
| Channel ablation | `scripts/run_channels.py` | Hidden-channel comparison. |
| Radial ablation | `scripts/run_radial_ablation.py` | Radial-basis comparison. |
| lmax ablation | `src/run_lmax_ablation.py` | Spherical-harmonic order comparison. |
| Ablation collection | `scripts/collect_ablation_results.py` | Aggregates ablation result tables. |
| Speed benchmark | `result/speed_benchmark/infer_speed_test.py` | Inference-speed benchmark only. |

## 3. Preserved Chapter 3 Results

Do not move, rename, or overwrite these directories in the current cleanup pass.

| Use | Path |
| --- | --- |
| Official summary table | `/home/leo494/mylab/实验结果/实验结果摘要.csv` |
| Eqv2-Lite main result | `experiments/2025-07-09_run1/` |
| Enhanced EquiformerV2 baseline | `experiments/2025-09-10_run1/enhanced_equiformerv2_baseline/` |
| Eqv2-Lite ablations | `experiments/2025-09-10_run1/ablation/`, `result/eq_lite_ablation/` |
| Concat fusion | `experiments/20251021_160614_tabfusion_run_real_equiformer_with_loss/` |
| Gate fusion | `experiments/20251021_165820_tabfusion_run_gate_fusion_100epochs/` |
| Inference speed | `result/speed_benchmark/infer_speed_results.json` |

## 4. Historical Or Secondary Material

Keep these for traceability, but do not cite them as primary Chapter 3 results unless manually reviewed.

For future archival candidates, see `docs/archive_candidates.md`.

| Path | Suggested status |
| --- | --- |
| `experiments/20251021_135737_tabfusion_run_real_data/` | Needs source/data alignment review before citation. |
| `experiments/struct_preds/`, `experiments/struct_preds_real/` | Intermediate prediction tables for fusion experiments. |
| `experiments/model_comparison/` | Useful comparison material; cite only after matching to thesis text. |
| `archive/demos/example_tabular_fusion.py` | Demo/reference script. |
| `archive/report_helpers/detailed_analysis_report.py` | Report-generation helper. |
| `archive/report_helpers/visualize_branch_architecture.py` | Figure/helper script. |

## 5. Other-Chapter Or Non-Chapter-3 Material

These are useful, but they are not part of the current Chapter 3 cleanup scope.

| Path | Reason |
| --- | --- |
| `archive/later_chapter_application/screen_materials.py` | Screening workflow, closer to later application chapters. |
| `archive/later_chapter_application/add_material_cost.py` | Cost annotation/application helper. |
| `Jiang/` | External/reference material. |
| Catalysis-Hub downloader notes | Related data collection material, not the Chapter 3 model core. |

## 6. Cleanup Boundaries

Current cleanup keeps core results stable:

- Do not rerun training.
- Do not rerun evaluation.
- Do not regenerate LMDB files.
- Do not modify existing result CSV/JSON files.
- Do not rename old result directories.
- Do not delete checkpoints or raw data.
- Archived helper scripts are preserved under `archive/`.

## 7. Documentation Map

| Document | Purpose |
| --- | --- |
| `README.md` | Top-level Chapter 3 overview. |
| `docs/documentation_index.md` | Reading order for all project documents. |
| `docs/cleanup_status.md` | Current cleanup progress and untouched areas. |
| `docs/code_index.md` | Concise source-code index. |
| `docs/results_manifest.md` | Result-to-file mapping. |
| `docs/artifacts.md` | Data/checkpoint/result preservation policy. |
| `docs/archive_candidates.md` | Future archive candidates. |
| `docs/chapter3-code-inventory.md` | Detailed review and recommendation document. |
| `src/README.md` | Source-code directory guide. |
| `scripts/README.md` | Script directory guide. |
| `datasets/README.md` | Dataset directory guide. |
| `result/README.md` | Result directory guide. |
| `experiments/README.md` | Experiment-output directory guide. |
