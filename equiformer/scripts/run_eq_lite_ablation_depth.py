#!/usr/bin/env python3
"""
Eqv2-Lite Depth 消融实验脚本
测试不同层数 (num_layers) 对模型性能的影响
"""

import json
import os
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
import matplotlib.pyplot as plt
from pathlib import Path
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

# 导入必要的组件
from standalone_equiformer_v2 import (
    StandaloneEquiformerV2,
    HydrogenDataset,
    custom_collate_fn
)

class EquiformerV2Trainer:
    """EquiformerV2训练器"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # GPU设备检测
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
            self.use_amp = torch.cuda.is_available() and hasattr(torch.cuda.amp, 'autocast')
            if self.use_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            else:
                self.scaler = None
        else:
            self.device = torch.device('cpu')
            self.use_amp = False
            self.scaler = None
        
        self.model.to(self.device)
        
        # 优化器和损失函数
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['lr'], 
            weight_decay=config.get('weight_decay', 1e-4)
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
    
    def _move_to_device(self, batch_data):
        """将数据移动到GPU设备"""
        if hasattr(batch_data, 'pos'):
            batch_data.pos = batch_data.pos.to(self.device, non_blocking=True)
        if hasattr(batch_data, 'atomic_numbers'):
            batch_data.atomic_numbers = batch_data.atomic_numbers.to(self.device, non_blocking=True)
        if hasattr(batch_data, 'edge_index'):
            batch_data.edge_index = batch_data.edge_index.to(self.device, non_blocking=True)
        if hasattr(batch_data, 'edge_distance'):
            batch_data.edge_distance = batch_data.edge_distance.to(self.device, non_blocking=True)
        if hasattr(batch_data, 'batch'):
            batch_data.batch = batch_data.batch.to(self.device, non_blocking=True)
        if hasattr(batch_data, 'natoms'):
            batch_data.natoms = batch_data.natoms.to(self.device, non_blocking=True)
        if hasattr(batch_data, 'y'):
            batch_data.y = batch_data.y.to(self.device, non_blocking=True)
        return batch_data
    
    def normalize_target(self, target):
        """归一化目标值"""
        return (target - self.target_mean) / self.target_std
    
    def denormalize_target(self, target):
        """反归一化目标值"""
        return target * self.target_std + self.target_mean
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for batch_data in tqdm(self.train_loader, desc="训练", leave=False):
            try:
                batch_data = self._move_to_device(batch_data)
                target = self.normalize_target(batch_data.y)
                
                self.optimizer.zero_grad()
                
                if self.use_amp and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        pred_energy = self.model(batch_data)
                        if pred_energy.dim() > 1:
                            pred_energy = pred_energy.squeeze(-1)
                        if target.dim() > 1:
                            target = target.squeeze(-1)
                        loss = self.criterion(pred_energy, target)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    pred_energy = self.model(batch_data)
                    if pred_energy.dim() > 1:
                        pred_energy = pred_energy.squeeze(-1)
                    if target.dim() > 1:
                        target = target.squeeze(-1)
                    loss = self.criterion(pred_energy, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️ GPU内存不足，跳过此批次")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"训练批次错误: {e}")
                    continue
        
        return total_loss / max(num_batches, 1)
    
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        predictions = []
        targets = []
        total_loss = 0.0
        valid_batches = 0
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="评估", leave=False):
                try:
                    batch_data = self._move_to_device(batch_data)
                    target = self.normalize_target(batch_data.y)
                    
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            pred_energy = self.model(batch_data)
                    else:
                        pred_energy = self.model(batch_data)
                    
                    if pred_energy.dim() > 1:
                        pred_energy = pred_energy.squeeze(-1)
                    if target.dim() > 1:
                        target = target.squeeze(-1)
                    
                    loss = self.criterion(pred_energy, target)
                    
                    pred_denorm = self.denormalize_target(pred_energy).cpu().numpy()
                    target_denorm = batch_data.y.cpu().numpy()
                    
                    predictions.extend(pred_denorm.tolist())
                    targets.extend(target_denorm.tolist())
                    total_loss += loss.item()
                    valid_batches += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        continue
        
        if valid_batches == 0:
            return None
            
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)
        
        return {
            'loss': total_loss / valid_batches,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions,
            'targets': targets
        }
    
    def train(self, num_epochs):
        """训练模型"""
        best_r2 = -float('inf')
        best_model_state = None
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_metrics = self.evaluate()
            
            if val_metrics is None:
                continue
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            self.val_r2s.append(val_metrics['r2'])
            
            self.scheduler.step(val_metrics['loss'])
            
            if val_metrics['r2'] > best_r2:
                best_r2 = val_metrics['r2']
                best_model_state = self.model.state_dict().copy()
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                      f"Train Loss={train_loss:.4f}, "
                      f"Val Loss={val_metrics['loss']:.4f}, "
                      f"Val R²={val_metrics['r2']:.4f}, "
                      f"Val MAE={val_metrics['mae']:.4f}, "
                      f"LR={current_lr:.6f}")
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        final_metrics = self.evaluate()
        return final_metrics

def plot_loss_curve(trainer, output_dir, num_layers):
    """绘制损失曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(trainer.train_losses) + 1)
    
    # 损失曲线
    axes[0].plot(epochs, trainer.train_losses, 'b-', label='训练损失', linewidth=2)
    axes[0].plot(epochs, trainer.val_losses, 'r-', label='验证损失', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'训练和验证损失曲线 (Layers={num_layers})', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # R²曲线
    axes[1].plot(epochs, trainer.val_r2s, 'g-', label='验证 R²', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('R² Score', fontsize=12)
    axes[1].set_title(f'验证集 R² 分数曲线 (Layers={num_layers})', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def run_single_experiment(num_layers, output_dir, config, train_loader, val_loader):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"🔬 开始实验: num_layers = {num_layers}")
    print(f"{'='*60}")
    
    # 创建模型
    model = StandaloneEquiformerV2(
        max_radius=12.0,
        max_neighbors=20,
        max_num_elements=90,
        num_layers=num_layers,  # 消融变量
        sphere_channels=128,
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
    
    # 创建训练器
    trainer = EquiformerV2Trainer(model, train_loader, val_loader, config)
    
    # 训练
    final_metrics = trainer.train(config['num_epochs'])
    
    if final_metrics is None:
        print(f"❌ 实验失败: num_layers = {num_layers}")
        return None
    
    # 保存结果
    exp_output_dir = output_dir / f"layers_{num_layers}"
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存metrics.json
    metrics = {
        'num_layers': num_layers,
        'r2': float(final_metrics['r2']),
        'mae': float(final_metrics['mae']),
        'rmse': float(final_metrics['rmse']),
        'loss': float(final_metrics['loss']),
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'config': config
    }
    
    with open(exp_output_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # 保存predictions.csv
    predictions_df = pd.DataFrame({
        'targets': final_metrics['targets'].flatten(),
        'predictions': final_metrics['predictions'].flatten()
    })
    predictions_df['residuals'] = predictions_df['targets'] - predictions_df['predictions']
    predictions_df.to_csv(exp_output_dir / 'predictions.csv', index=False)
    
    # 绘制并保存损失曲线
    plot_loss_curve(trainer, exp_output_dir, num_layers)
    
    print(f"✅ 实验完成: num_layers = {num_layers}")
    print(f"   R² = {final_metrics['r2']:.4f}")
    print(f"   MAE = {final_metrics['mae']:.4f}")
    print(f"   RMSE = {final_metrics['rmse']:.4f}")
    print(f"   结果已保存到: {exp_output_dir}")
    
    return metrics

def main():
    """主函数"""
    print("🚀 Eqv2-Lite Depth 消融实验")
    print("=" * 60)
    
    # 设置输出目录
    output_dir = project_root / "result" / "eq_lite_ablation" / "depth"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 结果将保存到: {output_dir}")
    
    # 实验配置
    num_layers_list = [1, 2, 3, 4]
    config = {
        'lr': 0.0003,
        'weight_decay': 1e-4,
        'num_epochs': 50,  # 每个模型训练50个epoch
        'batch_size': 6,
        'target_mean': -0.2042156457901001,
        'target_std': 0.6712170839309692,
    }
    
    # 数据加载
    print("\n📂 加载数据集...")
    train_lmdb_path = project_root / "datasets" / "custom_hydrogen" / "train.lmdb"
    val_lmdb_path = project_root / "datasets" / "custom_hydrogen" / "val.lmdb"
    
    print(f"训练集路径: {train_lmdb_path}")
    print(f"验证集路径: {val_lmdb_path}")
    
    train_dataset = HydrogenDataset(str(train_lmdb_path))
    val_dataset = HydrogenDataset(str(val_lmdb_path))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=custom_collate_fn
    )
    
    # 运行消融实验
    all_results = []
    
    for num_layers in num_layers_list:
        try:
            result = run_single_experiment(
                num_layers, 
                output_dir, 
                config, 
                train_loader, 
                val_loader
            )
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"❌ 实验失败 (num_layers={num_layers}): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存汇总结果
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df = summary_df[['num_layers', 'r2', 'mae', 'rmse', 'loss', 'num_parameters']]
        summary_df.to_csv(output_dir / 'summary.csv', index=False)
        
        print(f"\n{'='*60}")
        print("📊 消融实验汇总:")
        print(f"{'='*60}")
        print(summary_df.to_string(index=False))
        print(f"\n✅ 所有结果已保存到: {output_dir}")
    
    # 关闭数据集
    train_dataset.close()
    val_dataset.close()
    
    print("\n🎉 消融实验完成！")

if __name__ == "__main__":
    main()

