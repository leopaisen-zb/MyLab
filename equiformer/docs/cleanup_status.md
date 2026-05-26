# Cleanup Status

This document records what has been organized so far. It is a status index, not an experiment log.

## Completed

- Replaced the root `README.md` with a Chapter 3 Eqv2-Lite project overview.
- Added document navigation in `docs/documentation_index.md`.
- Added source-code classification in `docs/code_index.md`.
- Added project-level organization notes in `docs/project-organization.md`.
- Added result-to-folder mapping in `docs/results_manifest.md`.
- Added artifact preservation rules in `docs/artifacts.md`.
- Added future archive candidates in `docs/archive_candidates.md`.
- Updated `docs/chapter3-code-inventory.md` to match the current organization.
- Limited `result/speed_benchmark/` documentation and helper scripts to inference-speed comparison.
- Fixed `datasets/__init__.py` so it points to the maintained simplified data processor instead of missing legacy modules.
- Updated `requirements.txt` so it no longer looks like a Catalysis-Hub-only project dependency file.
- Added directory-level README files for `src/`, `scripts/`, `datasets/`, `result/`, and `experiments/`.
- Moved demo, report, public-packaging, old ablation, model-comparison helper, legacy OC20, and later-chapter application scripts into `archive/`.
- Deleted generated cache and local metadata files: `__pycache__/`, `.DS_Store`, `*:Zone.Identifier`, and `micromamba.tar.bz2`.
- Added Chapter 3 reviewer-response evidence files under `docs/chapter3_revision_evidence.md` and `docs/tables/`.
- Added the planned parameter-reallocation ablation grid, protocol, and runner without running the experiment.

## Not Changed

- No training was rerun.
- No evaluation was rerun.
- The new parameter-reallocation experiment was not run.
- No LMDB file was regenerated.
- No checkpoint was moved or renamed.
- No historical result CSV/JSON/PNG file was edited.
- No paper-facing experiment result directory was moved.

## Still Worth Doing Later

- Add a local ignore/cleanup policy for cache files such as `__pycache__/`, `.DS_Store`, and `*:Zone.Identifier`.
- Reconcile duplicate radial-ablation scripts in `scripts/run_radial_ablation.py` and `src/run_radial_ablation.py`.
- Decide whether old OC20-style legacy files should stay in `src/` or move under an archive directory with compatibility notes.
- Fix hard-coded local paths in training scripts only if you plan to rerun them.
- Create a split manifest only if you need formal traceability from LMDB indices back to raw structures.
