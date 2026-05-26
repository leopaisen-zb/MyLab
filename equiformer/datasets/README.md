# Dataset Directory Index

This directory contains data-processing code and prepared local datasets for Chapter 3.

## Maintained Entry

| Path | Role |
| --- | --- |
| `custom_data_processor_simplified.py` | Converts raw structures and feature tables into LMDB splits. |
| `custom_hydrogen/` | Main prepared LMDB dataset used by Chapter 3 experiments. |
| `custom_hydrogen_ocp/` | OCP-style or alternate prepared dataset material; preserve until confirmed unused. |

## Package Entry

`__init__.py` exposes the maintained simplified processor lazily:

- `SimpleData`
- `VASPDataProcessor`

The lazy import keeps `import datasets` usable even when the full training/preprocessing environment is not active.

## Preservation Rule

Do not regenerate, rename, or overwrite LMDB files during documentation cleanup. Treat prepared datasets as historical experiment inputs.
