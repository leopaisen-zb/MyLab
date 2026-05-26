# Result Directory Index

This directory contains preserved Chapter 3 outputs and benchmark artifacts. It is not a scratch directory.

## Main Result Groups

| Path | Role |
| --- | --- |
| `lab01/` | Eqv2-Lite main training output: model file, config, predictions, training history, and figures. |
| `eq_lite_ablation/` | Eqv2-Lite ablation summaries. |
| `speed_benchmark/` | Inference-speed benchmark configuration, preserved JSON output, and summary helper scripts. |

## Preservation Rule

- Do not overwrite result files unless a new experiment run is intentional and documented.
- Do not rename historical result directories during cleanup.
- Use `docs/results_manifest.md` to map thesis-facing results back to files.
