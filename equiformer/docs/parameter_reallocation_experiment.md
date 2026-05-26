# Parameter Reallocation Ablation

## Purpose

This experiment answers the reviewer concern that the choice to allocate model capacity toward scalar FFN, rather than attention, edge, or radial modules, needs direct evidence.

## Controlled Baseline

All variants keep:

- `num_layers=3`
- `sphere_channels=128`
- `num_heads=4`
- `lmax=4`
- `max_radius=12.0`
- `max_neighbors=20`
- `use_atom_edge_embedding=True`
- same train/validation LMDB splits

## Compared Variants

| Variant | Main change | Hypothesis |
| --- | --- | --- |
| `ffn_heavy` | Increase `ffn_hidden_channels` to 192 | Tests whether scalar nonlinear capacity improves adsorption-energy regression. |
| `ffn_light` | Reduce `ffn_hidden_channels` to 64 | Tests whether removing scalar FFN capacity hurts. |
| `attention_heavy` | Increase attention hidden/alpha/value channels | Tests whether attention capacity is a better target. |
| `edge_heavy` | Increase `edge_channels` | Tests whether edge embedding capacity is a better target. |
| `radial_heavy` | Increase `num_gaussians` to 512 | Tests whether radial basis capacity is a better target. |
| `balanced` | Moderate increases across attention and edge | Tests whether distributed capacity beats FFN-focused capacity. |

## Output

The runner writes:

```text
result/eq_lite_ablation/parameter_reallocation/
  summary.csv
  <variant>_seed<seed>/metrics.json
```

## Commands

Dry-run parameter count only:

```bash
python scripts/run_parameter_reallocation_ablation.py --dry-run
```

Pilot run:

```bash
python scripts/run_parameter_reallocation_ablation.py --grid experiments/ablation/grids/parameter_reallocation.csv --max-runs 6
```

Thesis-quality run after pilot review:

```bash
python scripts/run_parameter_reallocation_ablation.py --grid experiments/ablation/grids/parameter_reallocation.csv
```

## Decision Rule

The thesis should claim FFN reallocation is supported only if `ffn_heavy` has lower MAE/RMSE than `ffn_light` and is competitive with or better than `attention_heavy`, `edge_heavy`, `radial_heavy`, and `balanced` under comparable parameter counts.
