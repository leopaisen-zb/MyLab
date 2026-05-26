# Source Code Index

This directory contains model code and older compatibility utilities. For the Chapter 3 thesis workflow, start with the files listed under "Main Chapter 3 path".

## Main Chapter 3 Path

| File | Role |
| --- | --- |
| `standalone_equiformer_v2.py` | Eqv2-Lite / Equiformer-Light model and dataset helper definitions. |
| `train_equiformer.py` | Eqv2-Lite training entry point. |
| `evaluate_equiformer_v2.py` | Eqv2-Lite evaluation entry point. |
| `enhanced_equiformer_v2.py` | Full Enhanced EquiformerV2 baseline model. |
| `train_enhanced_equiformer_v2.py` | Enhanced baseline training entry point. |
| `predict_enhanced_equiformer_v2.py` | Enhanced baseline prediction/evaluation helper. |
| `equiformer_v2/nets/equiformer_v2/` | Shared EquiformerV2 network modules. |

## Ablation And Auxiliary Chapter 3 Code

| File | Role |
| --- | --- |
| `run_lmax_ablation.py` | lmax ablation. |
| `run_radial_ablation.py` | Radial-basis ablation variant. Compare with `scripts/run_radial_ablation.py` before editing. |

## Archived Out Of `src/`

Old OC20-style files were moved to `archive/legacy_oc20_src/`.

Later-chapter screening and cost-analysis files were moved to `archive/later_chapter_application/`.

## Notes

- Do not edit training defaults just for cleanup; that can change old experiment behavior.
