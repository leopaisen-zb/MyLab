#!/usr/bin/env python3
"""
Eqv2-Lite sphere_channels 消融实验脚本
测试不同球谐通道数 (sphere_channels) 对模型性能的影响
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
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

# 导入必要组件
from standalone_equiformer_v2 import (
    StandaloneEquiformerV2,
    HydrogenDataset,
    custom_collate_fn,
)


class EquiformerV2Trainer:
    """EquiformerV2 训练器（用于 channels 消融）"""

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # 设备与 AMP
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            self.use_amp = hasattr(torch.cuda.amp, "autocast")
            self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        else:
            self.device = torch.device("cpu")
            self.use_amp = False
            self.scaler = None

        self.model.to(self.device)

        # 优化器与调度器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 1e-4),
        )
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.8, patience=10, min_lr=1e-6
        )

        # 归一化参数
        self.target_mean = config.get("target_mean", 0.0)
        self.target_std = config.get("target_std", 1.0)

        # 记录
        self.train_losses = []
        self.val_losses = []
        self.val_r2s = []

    def _move_to_device(self, batch):
        if hasattr(batch, "pos"):
            batch.pos = batch.pos.to(self.device, non_blocking=True)
        if hasattr(batch, "atomic_numbers"):
            batch.atomic_numbers = batch.atomic_numbers.to(self.device, non_blocking=True)
        if hasattr(batch, "edge_index"):
            batch.edge_index = batch.edge_index.to(self.device, non_blocking=True)
        if hasattr(batch, "edge_distance"):
            batch.edge_distance = batch.edge_distance.to(self.device, non_blocking=True)
        if hasattr(batch, "batch"):
            batch.batch = batch.batch.to(self.device, non_blocking=True)
        if hasattr(batch, "natoms"):
            batch.natoms = batch.natoms.to(self.device, non_blocking=True)
        if hasattr(batch, "y"):
            batch.y = batch.y.to(self.device, non_blocking=True)
        return batch

    def normalize_target(self, y):
        return (y - self.target_mean) / self.target_std

    def denormalize_target(self, y):
        return y * self.target_std + self.target_mean

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for batch in tqdm(self.train_loader, desc="训练", leave=False):
            batch = self._move_to_device(batch)
            target = self.normalize_target(batch.y)

            self.optimizer.zero_grad()

            if self.use_amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    pred = self.model(batch)
                    if pred.dim() > 1:
                        pred = pred.squeeze(-1)
                    if target.dim() > 1:
                        target = target.squeeze(-1)
                    loss = self.criterion(pred, target)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
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

        return total_loss / max(n_batches, 1)

    def evaluate(self):
        self.model.eval()
        preds, targs = [], []
        total_loss = 0.0
        valid_batches = 0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="评估", leave=False):
                batch = self._move_to_device(batch)
                target = self.normalize_target(batch.y)

                if self.use_amp and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        pred = self.model(batch)
                else:
                    pred = self.model(batch)

                if pred.dim() > 1:
                    pred = pred.squeeze(-1)
                if target.dim() > 1:
                    target = target.squeeze(-1)

                loss = self.criterion(pred, target)

                pred_denorm = self.denormalize_target(pred).cpu().numpy()
                target_denorm = batch.y.cpu().numpy()

                preds.extend(pred_denorm.tolist())
                targs.extend(target_denorm.tolist())
                total_loss += loss.item()
                valid_batches += 1

        if valid_batches == 0:
            return None

        preds = np.array(preds)
        targs = np.array(targs)

        mae = mean_absolute_error(targs, preds)
        mse = mean_squared_error(targs, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(targs, preds)

        return {
            "loss": total_loss / valid_batches,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

    def train(self, num_epochs: int):
        best_r2 = -float("inf")
        best_state = None

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_metrics = self.evaluate()
            if val_metrics is None:
                continue

            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics["loss"])
            self.val_r2s.append(val_metrics["r2"])

            self.scheduler.step(val_metrics["loss"])

            if val_metrics["r2"] > best_r2:
                best_r2 = val_metrics["r2"]
                best_state = self.model.state_dict().copy()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch+1:3d}/{num_epochs}: "
                    f"Train Loss={train_loss:.4f}, "
                    f"Val Loss={val_metrics['loss']:.4f}, "
                    f"Val R²={val_metrics['r2']:.4f}, "
                    f"Val MAE={val_metrics['mae']:.4f}, "
                    f"LR={lr:.6f}"
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.evaluate()


def run_single_experiment(
    sphere_channels: int,
    output_root: Path,
    config,
    train_loader,
    val_loader,
):
    """运行单个 sphere_channels 实验，并保存 metrics.json"""
    print("\n" + "=" * 60)
    print(f"🔬 开始实验: sphere_channels = {sphere_channels}")
    print("=" * 60)

    model = StandaloneEquiformerV2(
        max_radius=12.0,
        max_neighbors=20,
        max_num_elements=90,
        num_layers=6,  # Lite 默认层数
        sphere_channels=sphere_channels,  # 消融变量
        attn_hidden_channels=64,
        num_heads=8,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=256,
        lmax_list=[4],
        mmax_list=[2],
        grid_resolution=18,
        edge_channels=128,
        use_atom_edge_embedding=True,
        share_atom_edge_embedding=False,
        alpha_drop=0.1,
        drop_path_rate=0.05,
        proj_drop=0.0,
    )

    print(f"✅ 模型已创建，参数数量: {sum(p.numel() for p in model.parameters()):,}")

    trainer = EquiformerV2Trainer(model, train_loader, val_loader, config)
    final_metrics = trainer.train(config["num_epochs"])

    if final_metrics is None:
        print(f"❌ 实验失败: sphere_channels = {sphere_channels}")
        return None

    exp_dir = output_root / f"channels_{sphere_channels}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "sphere_channels": sphere_channels,
        "r2": float(final_metrics["r2"]),
        "mae": float(final_metrics["mae"]),
        "rmse": float(final_metrics["rmse"]),
        "loss": float(final_metrics["loss"]),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "config": config,
    }

    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"✅ 实验完成: sphere_channels = {sphere_channels}")
    print(f"   R² = {final_metrics['r2']:.4f}")
    print(f"   MAE = {final_metrics['mae']:.4f}")
    print(f"   RMSE = {final_metrics['rmse']:.4f}")
    print(f"   结果已保存到: {exp_dir}")

    return metrics


def main():
    print("🚀 Eqv2-Lite sphere_channels 消融实验")
    print("=" * 60)

    # 输出目录（注意是 results）
    output_root = project_root / "results" / "eq_lite_ablation" / "channels"
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"📁 结果将保存到: {output_root}")

    # 实验配置
    sphere_channels_list = [32, 64, 128]
    config = {
        "lr": 0.0003,
        "weight_decay": 1e-4,
        "num_epochs": 50,
        "batch_size": 6,
        "target_mean": -0.2042156457901001,
        "target_std": 0.6712170839309692,
    }

    # 数据集
    print("\n📂 加载数据集...")
    train_lmdb_path = project_root / "datasets" / "custom_hydrogen" / "train.lmdb"
    val_lmdb_path = project_root / "datasets" / "custom_hydrogen" / "val.lmdb"

    print(f"训练集路径: {train_lmdb_path}")
    print(f"验证集路径: {val_lmdb_path}")

    train_dataset = HydrogenDataset(str(train_lmdb_path))
    val_dataset = HydrogenDataset(str(val_lmdb_path))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=custom_collate_fn,
    )

    all_results = []
    for sc in sphere_channels_list:
        try:
            res = run_single_experiment(sc, output_root, config, train_loader, val_loader)
            if res is not None:
                all_results.append(res)
        except Exception as e:
            print(f"❌ 实验失败 (sphere_channels={sc}): {e}")
            import traceback

            traceback.print_exc()

    if all_results:
        summary_df = pd.DataFrame(all_results)[
            ["sphere_channels", "r2", "mae", "rmse", "loss", "num_parameters"]
        ]
        summary_df.to_csv(output_root / "summary.csv", index=False, encoding="utf-8-sig")
        print("\n📊 消融实验汇总:")
        print(summary_df.to_string(index=False))
        print(f"\n✅ 汇总结果已保存到: {output_root / 'summary.csv'}")

    train_dataset.close()
    val_dataset.close()

    print("\n🎉 sphere_channels 消融实验完成！")


if __name__ == "__main__":
    main()

{
  "cells": [],
  "metadata": {
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 2
}