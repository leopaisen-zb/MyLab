#!/usr/bin/env python3
"""
Eqv2-Lite lmax 消融实验脚本
测试不同 lmax 值对模型性能的影响
"""

import json
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd  # type: ignore[import]
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # type: ignore[import]
from tqdm import tqdm
import warnings
from pathlib import Path
import time
import matplotlib.pyplot as plt  # type: ignore[import]
plt.switch_backend('Agg')  # 使用非交互式后端
warnings.filterwarnings('ignore')

# 添加项目路径（脚本在src目录下，所以parent.parent是项目根目录）
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

# 导入必要组件
from standalone_equiformer_v2 import (
    StandaloneEquiformerV2,
    HydrogenDataset,
    custom_collate_fn,
)


class EquiformerV2Trainer:
    """EquiformerV2 训练器（用于 lmax 消融）"""

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        # 早停相关
        self.early_stopping_patience = config.get('early_stopping_patience', 10)

        # 设备与 AMP
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
            self.use_amp = hasattr(torch.cuda.amp, 'autocast')
            self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        else:
            self.device = torch.device('cpu')
            self.use_amp = False
            self.scaler = None

        self.model.to(self.device)

        # 优化器与调度器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 1e-4),
        )
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=10, min_lr=1e-6
        )

        # 归一化参数
        self.target_mean = config.get('target_mean', 0.0)
        self.target_std = config.get('target_std', 1.0)

        # 记录
        self.train_losses = []
        self.val_losses = []
        self.val_r2s = []

    def _move_to_device(self, batch):
        if hasattr(batch, 'pos'):
            batch.pos = batch.pos.to(self.device, non_blocking=True)
        if hasattr(batch, 'atomic_numbers'):
            batch.atomic_numbers = batch.atomic_numbers.to(self.device, non_blocking=True)
        if hasattr(batch, 'edge_index'):
            batch.edge_index = batch.edge_index.to(self.device, non_blocking=True)
        if hasattr(batch, 'edge_distance'):
            batch.edge_distance = batch.edge_distance.to(self.device, non_blocking=True)
        if hasattr(batch, 'batch'):
            batch.batch = batch.batch.to(self.device, non_blocking=True)
        if hasattr(batch, 'natoms'):
            batch.natoms = batch.natoms.to(self.device, non_blocking=True)
        if hasattr(batch, 'y'):
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

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="训练", leave=False)):
            batch_start_time = time.time()
            
            try:
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
                
                batch_time = time.time() - batch_start_time
                # 如果单个batch处理时间超过60秒，跳过这个batch
                if batch_time > 60.0:
                    print(f"  [WARNING] Batch {batch_idx} 处理时间过长({batch_time:.2f}秒)，跳过此batch")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  [WARNING] Batch {batch_idx}: GPU内存不足，跳过此batch")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"  [ERROR] Batch {batch_idx} 错误: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            except Exception as e:
                print(f"  [ERROR] Batch {batch_idx} 未知错误: {e}")
                import traceback
                traceback.print_exc()
                continue

        return total_loss / max(n_batches, 1)

    def evaluate(self):
        self.model.eval()
        preds, targs = [], []
        total_loss = 0.0
        valid_batches = 0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="评估", leave=False)):
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
            'loss': total_loss / valid_batches,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
        }

    def train(self, num_epochs: int):
        best_r2 = -float('inf')
        best_state = None
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            try:
                train_loss = self.train_epoch()
            except Exception as e:
                print(f"  [ERROR] Epoch {epoch+1} 训练失败: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            try:
                val_metrics = self.evaluate()
                if val_metrics is None:
                    print(f"  [WARNING] Epoch {epoch+1} 评估失败，跳过")
                    continue
            except Exception as e:
                print(f"  [ERROR] Epoch {epoch+1} 评估失败: {e}")
                import traceback
                traceback.print_exc()
                continue

            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            self.val_r2s.append(val_metrics['r2'])

            self.scheduler.step(val_metrics['loss'])

            if val_metrics['r2'] > best_r2:
                best_r2 = val_metrics['r2']
                best_state = self.model.state_dict().copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch+1:3d}/{num_epochs}: "
                    f"Train Loss={train_loss:.4f}, "
                    f"Val Loss={val_metrics['loss']:.4f}, "
                    f"Val R²={val_metrics['r2']:.4f}, "
                    f"Val MAE={val_metrics['mae']:.4f}, "
                    f"LR={lr:.6f}"
                )
            
            # 早停判断
            if self.early_stopping_patience is not None and epochs_no_improve >= self.early_stopping_patience:
                print(
                    f"[EARLY STOPPING] 验证集 R² 已经连续 {epochs_no_improve} 个 epoch "
                    f"没有提升，提前停止训练。"
                )
                break
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.evaluate()


def run_single_experiment(lmax: int, output_root: Path, config, train_loader, val_loader):
    """运行单个 lmax 实验，并保存 metrics.json"""
    print("\n" + "=" * 60)
    print(f"开始实验: lmax = {lmax}")
    print("=" * 60)

    # 创建模型，使用指定的 lmax
    print(f"[DEBUG] 开始创建模型 (lmax={lmax})...")
    model = StandaloneEquiformerV2(
        max_radius=12.0,
        max_neighbors=20,
        max_num_elements=90,
        num_layers=6,
        sphere_channels=128,
        attn_hidden_channels=64,
        num_heads=8,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=256,
        lmax_list=[lmax],  # 消融变量
        mmax_list=[2],
        grid_resolution=18,
        edge_channels=128,
        use_atom_edge_embedding=True,
        share_atom_edge_embedding=False,
        alpha_drop=0.1,
        drop_path_rate=0.05,
        proj_drop=0.0,
    )

    print(f"[OK] 模型已创建，参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   lmax: {lmax}")
    
    # 将模型移动到GPU（如果可用）
    if torch.cuda.is_available():
        print("[DEBUG] 移动模型到GPU...")
        model = model.cuda()
        print(f"  - GPU内存: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")
    
    # 测试模型是否能正常前向传播
    print("[DEBUG] 测试模型前向传播...")
    try:
        model.eval()
        test_batch = next(iter(val_loader))
        
        if torch.cuda.is_available():
            test_batch.pos = test_batch.pos.cuda()
            test_batch.atomic_numbers = test_batch.atomic_numbers.cuda()
            test_batch.edge_index = test_batch.edge_index.cuda()
            test_batch.edge_distance = test_batch.edge_distance.cuda()
            test_batch.batch = test_batch.batch.cuda()
            test_batch.natoms = test_batch.natoms.cuda()
        
        with torch.no_grad():
            output = model(test_batch)
        print(f"  - 输出形状: {output.shape}")
        print("[OK] 模型前向传播测试通过")
    except Exception as e:
        print(f"[ERROR] 模型前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    print("[DEBUG] 创建训练器...")
    trainer = EquiformerV2Trainer(model, train_loader, val_loader, config)
    print("[OK] 训练器创建完成")
    
    print("开始训练...")
    print(f"[DEBUG] 训练配置: epochs={config['num_epochs']}, batch_size={config['batch_size']}, lr={config['lr']}")
    final_metrics = trainer.train(config["num_epochs"])

    if final_metrics is None:
        print(f"[ERROR] 实验失败: lmax = {lmax}")
        return None

    exp_dir = output_root / f"lmax_{lmax}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "lmax": lmax,
        "mae": float(final_metrics["mae"]),
        "rmse": float(final_metrics["rmse"]),
        "r2": float(final_metrics["r2"]),
        "loss": float(final_metrics["loss"]),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "config": config,
    }

    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[OK] 实验完成: lmax = {lmax}")
    print(f"   MAE = {final_metrics['mae']:.4f}")
    print(f"   RMSE = {final_metrics['rmse']:.4f}")
    print(f"   R² = {final_metrics['r2']:.4f}")
    print(f"   结果已保存到: {exp_dir}")

    return metrics


def plot_ablation_results(all_results, output_path):
    """绘制消融实验结果图"""
    if not all_results:
        print("[WARNING] 没有结果可绘制")
        return
    
    # 准备数据
    df = pd.DataFrame(all_results)
    df = df.sort_values('lmax')
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Lmax Ablation Study Results', fontsize=16, fontweight='bold')
    
    # 1. MAE vs Lmax
    ax1 = axes[0, 0]
    ax1.plot(df['lmax'], df['mae'], 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Lmax', fontsize=12)
    ax1.set_ylabel('MAE', fontsize=12)
    ax1.set_title('Mean Absolute Error vs Lmax', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(df['lmax'])
    
    # 2. RMSE vs Lmax
    ax2 = axes[0, 1]
    ax2.plot(df['lmax'], df['rmse'], 's-', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Lmax', fontsize=12)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.set_title('Root Mean Squared Error vs Lmax', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(df['lmax'])
    
    # 3. R² vs Lmax
    ax3 = axes[1, 0]
    ax3.plot(df['lmax'], df['r2'], '^-', linewidth=2, markersize=8, color='#F18F01')
    ax3.set_xlabel('Lmax', fontsize=12)
    ax3.set_ylabel('R²', fontsize=12)
    ax3.set_title('R² Score vs Lmax', fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(df['lmax'])
    
    # 4. 参数量 vs Lmax
    ax4 = axes[1, 1]
    ax4.plot(df['lmax'], df['num_parameters'] / 1e6, 'd-', linewidth=2, markersize=8, color='#C73E1D')
    ax4.set_xlabel('Lmax', fontsize=12)
    ax4.set_ylabel('Parameters (M)', fontsize=12)
    ax4.set_title('Model Parameters vs Lmax', fontsize=13)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(df['lmax'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 图表已保存到: {output_path}")
    plt.close()


def main():
    print("Eqv2-Lite lmax 消融实验")
    print("=" * 60)

    # 输出目录
    output_root = project_root / "result" / "eq_lite_ablation" / "lmax"
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"结果将保存到: {output_root}")

    # 实验配置
    lmax_list = [3, 4, 5, 6]
    config = {
        'lr': 0.0003,
        'weight_decay': 1e-4,
        'num_epochs': 50,
        'batch_size': 16,
        'target_mean': -0.2042156457901001,
        'target_std': 0.6712170839309692,
        # 早停机制：如果验证集 R² 连续若干个 epoch 没有提升则停止训练
        'early_stopping_patience': 10,
    }

    # 根据平台设置 DataLoader 的 num_workers
    # 注意：Windows 下 DataLoader 使用多进程会要求 Dataset 可被 pickle，
    # 对部分自定义 Dataset（尤其是带有环境句柄的）可能导致 “cannot pickle 'Environment' object” 等错误，
    # 因此在 Windows 平台上强制使用单进程（num_workers=0）更安全。
    if sys.platform.startswith("win"):
        num_workers = 0
        print("[INFO] 检测到 Windows 平台，DataLoader 将使用 num_workers=0（单进程，避免 pickling 问题）")
    else:
        num_workers = 2
        print(f"[INFO] 非 Windows 平台，DataLoader 使用 num_workers={num_workers}")

    # 数据集
    print("\n加载数据集...")
    train_lmdb_path = project_root / "datasets" / "custom_hydrogen" / "train.lmdb"
    val_lmdb_path = project_root / "datasets" / "custom_hydrogen" / "val.lmdb"

    print(f"训练集路径: {train_lmdb_path}")
    print(f"验证集路径: {val_lmdb_path}")
    print(f"路径是否存在: train={train_lmdb_path.exists()}, val={val_lmdb_path.exists()}")

    print("[DEBUG] 开始创建训练数据集...")
    train_dataset = HydrogenDataset(str(train_lmdb_path))
    print(f"[OK] 训练数据集创建完成，样本数: {len(train_dataset)}")
    
    print("[DEBUG] 开始创建验证数据集...")
    val_dataset = HydrogenDataset(str(val_lmdb_path))
    print(f"[OK] 验证数据集创建完成，样本数: {len(val_dataset)}")

    print("[DEBUG] 开始创建训练数据加载器...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=custom_collate_fn,
    )
    print(f"[OK] 训练数据加载器创建完成，batch数: {len(train_loader)}")
    
    print("[DEBUG] 开始创建验证数据加载器...")
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=custom_collate_fn,
    )
    print(f"[OK] 验证数据加载器创建完成，batch数: {len(val_loader)}")
    
    # 测试数据加载器是否能正常工作
    print("[DEBUG] 测试数据加载器...")
    try:
        test_train_batch = next(iter(train_loader))
        print(f"  [OK] 成功获取训练batch，batch大小: {len(test_train_batch.y) if hasattr(test_train_batch, 'y') else 'N/A'}")
        
        test_val_batch = next(iter(val_loader))
        print(f"  [OK] 成功获取验证batch，batch大小: {len(test_val_batch.y) if hasattr(test_val_batch, 'y') else 'N/A'}")
    except Exception as e:
        print(f"  [ERROR] 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return

    all_results = []
    print(f"\n[DEBUG] 准备运行 {len(lmax_list)} 个实验: {lmax_list}")
    
    for idx, lmax in enumerate(lmax_list):
        print(f"\n{'='*60}")
        print(f"实验进度: {idx+1}/{len(lmax_list)}")
        print(f"{'='*60}")
        try:
            print(f"[DEBUG] 开始实验 {idx+1}: lmax={lmax}")
            res = run_single_experiment(lmax, output_root, config, train_loader, val_loader)
            if res is not None:
                all_results.append(res)
                print(f"[OK] 实验 {idx+1} 完成")
            else:
                print(f"[WARNING] 实验 {idx+1} 返回None")
        except Exception as e:
            print(f"[ERROR] 实验失败 (lmax={lmax}): {e}")
            import traceback
            traceback.print_exc()
            print(f"[WARNING] 继续下一个实验...")
            continue

    if all_results:
        summary_df = pd.DataFrame(all_results)[
            ['lmax', 'mae', 'rmse', 'r2', 'loss', 'num_parameters']
        ]
        summary_df.to_csv(output_root / "summary.csv", index=False, encoding="utf-8-sig")
        print("\n消融实验汇总:")
        print(summary_df.to_string(index=False))
        print(f"\n[OK] 汇总结果已保存到: {output_root / 'summary.csv'}")
        
        # 绘制结果图
        plot_path = output_root / "ablation_lmax.png"
        plot_ablation_results(all_results, plot_path)

    train_dataset.close()
    val_dataset.close()

    print("\n[OK] lmax 消融实验完成！")


if __name__ == "__main__":
    main()

