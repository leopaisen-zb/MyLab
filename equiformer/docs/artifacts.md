# Data And Artifact Policy

This project keeps historical Chapter 3 artifacts local and stable. The cleanup goal is readability, not result regeneration.

## Keep In Place

Do not move, rename, or overwrite these paths during the current cleanup pass:

- `datasets/custom_hydrogen/`
- `datasets/custom_hydrogen_ocp/`
- `checkpionts/`
- `data/raw/`
- `experiments/`
- `result/`
- `/home/leo494/mylab/实验结果/`

## Large Artifact Types

| Type | Examples | Policy |
| --- | --- | --- |
| LMDB splits | `datasets/custom_hydrogen/*.lmdb` | Preserve as generated experiment inputs. |
| Checkpoints | `checkpionts/`, model `.pth` files | Preserve for reproducibility; do not rename casually. |
| Result tables | `experiments/**/*.csv`, `result/**/*.csv` | Treat as historical outputs. |
| Result JSON | `experiments/**/*.json`, `result/**/*.json` | Treat as historical outputs. |
| Raw source data | `data/raw/` | Preserve as input evidence. |

## Naming Notes

- The existing directory name `checkpionts/` appears to be misspelled, but it should not be renamed in this cleanup pass because scripts or notes may already refer to it.
- Windows-style paths in older scripts should be documented before being changed.
- Existing result directories encode dates and should remain stable.

## Cleanup Rule

When organizing the project, add index documents first. Only move or archive files after the maintained code paths and result references are documented.
