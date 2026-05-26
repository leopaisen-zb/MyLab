#!/usr/bin/env python3
"""Run Eqv2-Lite parameter-reallocation ablations.

This script compares where additional model capacity is allocated: scalar FFN,
attention, edge embeddings, radial basis, or a balanced allocation. It writes
new outputs only under `result/eq_lite_ablation/parameter_reallocation/`.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from standalone_equiformer_v2 import (  # noqa: E402
    HydrogenDataset,
    SimpleGaussianSmearing,
    StandaloneEquiformerV2,
    custom_collate_fn,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Eqv2-Lite parameter-reallocation ablations."
    )
    parser.add_argument(
        "--grid",
        default="experiments/ablation/grids/parameter_reallocation.csv",
        help="CSV grid with parameter allocation variants.",
    )
    parser.add_argument(
        "--output-root",
        default="result/eq_lite_ablation/parameter_reallocation",
        help="Output directory for new ablation results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only instantiate variants and report parameter counts.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Limit number of variants for a pilot run.",
    )
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args()


def read_grid(path):
    grid_path = PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
    with open(grid_path, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows found in grid: {grid_path}")
    return rows


def as_int(row, key):
    return int(float(row[key]))


def as_float(row, key):
    return float(row[key])


def rebuild_radial_basis(model, num_gaussians, block_config):
    """Replace the radial basis and dependent modules after model init."""
    from equiformer_v2.nets.equiformer_v2.input_block import EdgeDegreeEmbedding
    from equiformer_v2.nets.equiformer_v2.transformer_block import TransBlockV2

    model.distance_expansion = SimpleGaussianSmearing(
        0.0, model.max_radius, num_gaussians, 2.0
    )
    model.edge_channels_list = [model.distance_expansion.num_output] + [
        model.edge_channels
    ] * 2

    if model.share_atom_edge_embedding and model.use_atom_edge_embedding:
        model.edge_channels_list[0] += 2 * model.edge_channels_list[-1]

    model.edge_degree_embedding = EdgeDegreeEmbedding(
        model.sphere_channels,
        model.lmax_list,
        model.mmax_list,
        model.SO3_rotation,
        model.mappingReduced,
        model.max_num_elements,
        model.edge_channels_list,
        model.block_use_atom_edge_embedding,
        rescale_factor=model._avg_degree,
    )

    model.blocks = torch.nn.ModuleList()
    for _ in range(model.num_layers):
        model.blocks.append(
            TransBlockV2(
                model.sphere_channels,
                block_config["attn_hidden_channels"],
                block_config["num_heads"],
                block_config["attn_alpha_channels"],
                block_config["attn_value_channels"],
                block_config["ffn_hidden_channels"],
                model.sphere_channels,
                model.lmax_list,
                model.mmax_list,
                model.SO3_rotation,
                model.mappingReduced,
                model.SO3_grid,
                model.max_num_elements,
                model.edge_channels_list,
                model.block_use_atom_edge_embedding,
                use_m_share_rad=False,
                attn_activation="scaled_silu",
                use_s2_act_attn=False,
                use_attn_renorm=True,
                ffn_activation="scaled_silu",
                use_gate_act=False,
                use_grid_mlp=False,
                use_sep_s2_act=True,
                norm_type="layer_norm_sh",
                alpha_drop=0.1,
                drop_path_rate=0.05,
                proj_drop=0.0,
            )
        )


def build_model(row):
    block_config = {
        "attn_hidden_channels": as_int(row, "attn_hidden_channels"),
        "num_heads": as_int(row, "num_heads"),
        "attn_alpha_channels": as_int(row, "attn_alpha_channels"),
        "attn_value_channels": as_int(row, "attn_value_channels"),
        "ffn_hidden_channels": as_int(row, "ffn_hidden_channels"),
    }

    model = StandaloneEquiformerV2(
        max_radius=as_float(row, "max_radius"),
        max_neighbors=as_int(row, "max_neighbors"),
        max_num_elements=90,
        num_layers=as_int(row, "num_layers"),
        sphere_channels=as_int(row, "sphere_channels"),
        attn_hidden_channels=block_config["attn_hidden_channels"],
        num_heads=block_config["num_heads"],
        attn_alpha_channels=block_config["attn_alpha_channels"],
        attn_value_channels=block_config["attn_value_channels"],
        ffn_hidden_channels=block_config["ffn_hidden_channels"],
        lmax_list=[as_int(row, "lmax")],
        mmax_list=[2],
        grid_resolution=18,
        edge_channels=as_int(row, "edge_channels"),
        use_atom_edge_embedding=True,
        share_atom_edge_embedding=False,
        alpha_drop=0.1,
        drop_path_rate=0.05,
        proj_drop=0.0,
    )

    num_gaussians = as_int(row, "num_gaussians")
    if num_gaussians != model.distance_expansion.num_output:
        rebuild_radial_basis(model, num_gaussians, block_config)

    return model


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def make_loaders(batch_size):
    train_path = PROJECT_ROOT / "datasets" / "custom_hydrogen" / "train.lmdb"
    val_path = PROJECT_ROOT / "datasets" / "custom_hydrogen" / "val.lmdb"

    train_dataset = HydrogenDataset(str(train_path))
    val_dataset = HydrogenDataset(str(val_path))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=custom_collate_fn,
    )
    return train_loader, val_loader


def row_to_config(row, args):
    return {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "target_mean": -0.2042156457901001,
        "target_std": 0.6712170839309692,
        "variant": row["name"],
        "seed": as_int(row, "seed"),
    }


def serializable_metrics(metrics):
    clean = {}
    for key, value in metrics.items():
        if key in {"predictions", "targets"}:
            continue
        if isinstance(value, np.generic):
            clean[key] = value.item()
        else:
            clean[key] = value
    return clean


def run_training(rows, args):
    from train_equiformer import EquiformerV2Trainer

    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = make_loaders(args.batch_size)
    summary_rows = []

    for row in rows:
        seed = as_int(row, "seed")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = build_model(row)
        num_parameters = count_parameters(model)
        run_name = f"{row['name']}_seed{seed}"
        run_dir = output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        config = row_to_config(row, args)
        trainer = EquiformerV2Trainer(model, train_loader, val_loader, config)
        metrics = trainer.train(args.num_epochs, output_dir=str(run_dir))
        if metrics is None:
            raise RuntimeError(f"Training produced no metrics for {run_name}")

        clean_metrics = serializable_metrics(metrics)
        result = {
            "name": row["name"],
            "seed": seed,
            "num_parameters": num_parameters,
            **clean_metrics,
            "config": dict(row),
        }

        with open(run_dir / "metrics.json", "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)

        summary_rows.append(
            {
                "name": row["name"],
                "seed": seed,
                "num_parameters": num_parameters,
                "r2": clean_metrics.get("r2"),
                "mae": clean_metrics.get("mae"),
                "rmse": clean_metrics.get("rmse"),
                "loss": clean_metrics.get("loss"),
            }
        )

    summary_path = output_root / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "name",
                "seed",
                "num_parameters",
                "r2",
                "mae",
                "rmse",
                "loss",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {summary_path}")


def run_dry_run(rows):
    print("variant,seed,num_parameters")
    for row in rows:
        model = build_model(row)
        print(f"{row['name']},{row['seed']},{count_parameters(model)}")


def main():
    args = parse_args()
    rows = read_grid(args.grid)
    if args.max_runs is not None:
        rows = rows[: args.max_runs]

    if args.dry_run:
        run_dry_run(rows)
    else:
        run_training(rows, args)


if __name__ == "__main__":
    main()
