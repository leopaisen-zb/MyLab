#!/usr/bin/env python3
"""
Eqv2-Lite lmax × max_radius 交叉消融实验
回应答辩意见中"剪枝参数只做单因素实验"的问题
"""

import json
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
import warnings
from pathlib import Path
import time
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
plt.switch_backend('Agg')

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from standalone_equiformer_v2 import (
    StandaloneEquiformerV2,
    HydrogenDataset,
    custom_collate_fn,
)

# MPS 加速
DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE} (MPS has compatibility issues, using CPU)")


class EquiformerV2Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = DEVICE
        self.use_amp = False
        self.scaler = None
        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 1e-4),
        )
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=10, min_lr=1e-6
        )
        self.target_mean = config.get('target_mean', 0.0)
        self.target_std = config.get('target_std', 1.0)

    def normalize_target(self, y):
        return (y - self.target_mean) / self.target_std

    def denormalize_target(self, y):
        return y * self.target_std + self.target_mean

    def _move_to_device(self, batch):
        for field in ['pos', 'atomic_numbers', 'edge_index', 'edge_distance', 'batch', 'natoms', 'y']:
            if hasattr(batch, field):
                val = getattr(batch, field)
                if val is not None and isinstance(val, torch.Tensor):
                    setattr(batch, field, val.to(self.device, non_blocking=True))
        return batch

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(self.train_loader, desc="  Training", leave=False, unit="batch")
        for batch in pbar:
            batch = self._move_to_device(batch)
            target = self.normalize_target(batch.y)
            self.optimizer.zero_grad()
            pred = self.model(batch)
            if pred.dim() > 1:
                pred = pred.squeeze(-1)
            if target.dim() > 1:
                target = target.squeeze(-1)
            loss = self.criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix_str(f"loss={loss.item():.4f}")
        return total_loss / max(n_batches, 1)

    def evaluate(self):
        self.model.eval()
        preds, targs = [], []
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_to_device(batch)
                target = self.normalize_target(batch.y)
                pred = self.model(batch)
                if pred.dim() > 1:
                    pred = pred.squeeze(-1)
                if target.dim() > 1:
                    target = target.squeeze(-1)
                preds.extend(self.denormalize_target(pred).cpu().numpy().tolist())
                targs.extend(batch.y.cpu().numpy().tolist())
        preds, targs = np.array(preds), np.array(targs)
        mae = mean_absolute_error(targs, preds)
        rmse = np.sqrt(mean_squared_error(targs, preds))
        r2 = r2_score(targs, preds)
        return {'mae': mae, 'rmse': rmse, 'r2': r2}

    def train(self, num_epochs: int, experiment_desc: str = ""):
        best_r2 = -float('inf')
        best_state = None
        epochs_no_improve = 0

        pbar = tqdm(range(num_epochs), desc=f"Epochs {experiment_desc}", unit="epoch")
        for epoch in pbar:
            epoch_start = time.time()
            train_loss = self.train_epoch()
            val_metrics = self.evaluate()
            elapsed = time.time() - epoch_start

            self.scheduler.step(val_metrics['mae'])
            if val_metrics['r2'] > best_r2:
                best_r2 = val_metrics['r2']
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            pbar.set_postfix_str(
                f"Train={train_loss:.4f} ValMAE={val_metrics['mae']:.4f} "
                f"ValR2={val_metrics['r2']:.4f} ({elapsed:.1f}s/ep)"
            )

            if epochs_no_improve >= 15:
                pbar.write(f"  [EARLY STOP] epoch {epoch+1}")
                break

        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        return self.evaluate()


def run_single(lmax, max_radius, train_loader, val_loader, config, output_root, combo_idx: int, total_combos: int):
    print(f"\n--- [{combo_idx}/{total_combos}] lmax={lmax}, max_radius={max_radius} ---")
    model = StandaloneEquiformerV2(
        max_radius=max_radius,
        max_neighbors=20,
        max_num_elements=90,
        num_layers=6,
        sphere_channels=128,
        attn_hidden_channels=64,
        num_heads=8,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=256,
        lmax_list=[lmax],
        mmax_list=[2],
        grid_resolution=18,
        edge_channels=128,
        use_atom_edge_embedding=True,
        share_atom_edge_embedding=False,
        alpha_drop=0.1,
        drop_path_rate=0.05,
        proj_drop=0.0,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {num_params:,}")

    trainer = EquiformerV2Trainer(model, train_loader, val_loader, config)
    exp_desc = f"lmax={lmax}, r={max_radius}"
    metrics = trainer.train(config['num_epochs'], experiment_desc=exp_desc)

    exp_dir = output_root / f"lmax{lmax}_r{max_radius}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    result = {
        'lmax': lmax,
        'max_radius': max_radius,
        'mae': float(metrics['mae']),
        'rmse': float(metrics['rmse']),
        'r2': float(metrics['r2']),
        'num_parameters': num_params,
    }
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  => MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f}")
    return result


def plot_heatmap(df, metric, output_path):
    pivot = df.pivot(index='max_radius', columns='lmax', values=metric)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': metric})
    ax.set_title(f'{metric} Heatmap (lmax × max_radius)')
    ax.set_xlabel('lmax')
    ax.set_ylabel('max_radius (Å)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Heatmap saved: {output_path}")


def main():
    print("Eqv2-Lite lmax × max_radius 交叉消融实验")
    print("=" * 60)

    output_root = project_root / "result" / "eq_lite_ablation" / "cross_lmax_radius"
    output_root.mkdir(parents=True, exist_ok=True)

    # 2x3 矩阵: lmax in [3,4], max_radius in [8.0, 12.0, 16.0]
    lmax_list = [3, 4]
    radius_list = [8.0, 12.0, 16.0]
    total_combos = len(lmax_list) * len(radius_list)

    config = {
        'lr': 0.0003,
        'weight_decay': 1e-4,
        'num_epochs': 15,
        'batch_size': 2,
        'target_mean': -0.2042156457901001,
        'target_std': 0.6712170839309692,
    }

    max_train = 500
    max_val = 100

    print("\n加载数据集...")
    train_ds = HydrogenDataset(str(project_root / "datasets/custom_hydrogen/train.lmdb"))
    val_ds = HydrogenDataset(str(project_root / "datasets/custom_hydrogen/val.lmdb"))

    if len(train_ds) > max_train:
        train_ds = SubsetDataset(train_ds, max_train)
    if len(val_ds) > max_val:
        val_ds = SubsetDataset(val_ds, max_val)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,
                              num_workers=0, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False,
                            num_workers=0, collate_fn=custom_collate_fn)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    results = []
    combo_idx = 0
    total_start = time.time()

    for lmax in lmax_list:
        for radius in radius_list:
            combo_idx += 1
            combo_start = time.time()
            try:
                res = run_single(lmax, radius, train_loader, val_loader, config, output_root, combo_idx, total_combos)
                results.append(res)
            except Exception as e:
                print(f"  [ERROR] lmax={lmax}, radius={radius}: {e}")
                import traceback; traceback.print_exc()
            combo_elapsed = time.time() - combo_start
            avg_per_combo = (time.time() - total_start) / combo_idx
            remaining = avg_per_combo * (total_combos - combo_idx)
            print(f"  [{combo_idx}/{total_combos}] 此配置耗时: {combo_elapsed:.1f}s, 预计剩余: {remaining:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\n总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_root / "summary.csv", index=False)
        print("\n" + "=" * 60)
        print("交叉实验汇总:")
        print(df.to_string(index=False))

        plot_heatmap(df, 'mae', output_root / "heatmap_mae.png")
        plot_heatmap(df, 'r2', output_root / "heatmap_r2.png")

        param_pivot = df.pivot(index='max_radius', columns='lmax', values='num_parameters')
        print("\n参数量 (lmax × max_radius):")
        print(param_pivot.to_string())

    print("\n[OK] 交叉消融实验完成!")


class SubsetDataset:
    def __init__(self, dataset, max_size):
        self.dataset = dataset
        self.max_size = max_size
    def __len__(self):
        return min(len(self.dataset), self.max_size)
    def __getitem__(self, idx):
        return self.dataset[idx]


if __name__ == "__main__":
    main()