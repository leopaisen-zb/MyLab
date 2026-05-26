#!/usr/bin/env python3
"""
EquiformerV2训练脚本
使用独立的EquiformerV2模型进行氢吸附能量预测训练
"""

import json
import os
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

# 从独立的EquiformerV2模块导入必要组件
from standalone_equiformer_v2 import (
    StandaloneEquiformerV2,
    HydrogenDataset,
    custom_collate_fn
)

class EquiformerV2Trainer:
    """EquiformerV2训练器（GPU优化版本）"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # GPU设备检测和优化
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            # 启用GPU优化
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # 检查是否支持混合精度
            self.use_amp = torch.cuda.is_available() and hasattr(torch.cuda.amp, 'autocast')
            if self.use_amp:
                self.scaler = torch.cuda.amp.GradScaler()
                print("🚀 使用混合精度训练 (AMP) 来加速GPU训练")
            else:
                self.scaler = None
            
            # 显示GPU信息
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🎯 GPU设备: {gpu_name}")
            print(f"🎯 GPU显存: {gpu_memory:.1f} GB")
        else:
            self.device = torch.device('cpu')
            self.use_amp = False
            self.scaler = None
            print("⚠️  警告: 未检测到CUDA，将使用CPU训练（速度较慢）")
        
        # 将模型移动到设备
        self.model.to(self.device)
        
        # 暂时使用单GPU模式，避免EquiformerV2的DataParallel兼容性问题
        if torch.cuda.device_count() > 1:
            print(f"🔍 检测到 {torch.cuda.device_count()} 个GPU，但EquiformerV2与DataParallel存在兼容性问题")
            print("🚀 使用单GPU模式确保稳定训练（建议未来升级到DDP）")
        else:
            print("🔍 使用单GPU模式")
        
        # 优化器和损失函数
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['lr'], 
            weight_decay=config.get('weight_decay', 1e-4)
        )
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=15, min_lr=1e-6
        )
        
        # 归一化参数
        self.target_mean = config.get('target_mean', 0.0)
        self.target_std = config.get('target_std', 1.0)
        
        # 记录
        self.train_losses = []
        self.val_losses = []
        self.val_r2s = []
        
        print(f"🎯 训练设备: {self.device}")
        print(f"🎯 模型参数: {sum(p.numel() for p in model.parameters()):,}")
        
        # GPU内存监控
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self._log_gpu_memory("初始化后")
    
    def _log_gpu_memory(self, stage):
        """记录GPU内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"🔍 GPU内存 ({stage}): 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
    
    def _move_to_device(self, batch_data):
        """高效地将数据移动到GPU设备"""
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
        """训练一个epoch（GPU优化版本）"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # 训练前清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for batch_data in tqdm(self.train_loader, desc="🚀 GPU训练", leave=False):
            try:
                # 高效移动数据到GPU
                batch_data = self._move_to_device(batch_data)
                target = self.normalize_target(batch_data.y)
                
                self.optimizer.zero_grad()
                
                # 使用混合精度训练
                if self.use_amp and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        pred_energy = self.model(batch_data)
                        
                        # 确保预测和目标的形状匹配
                        if pred_energy.dim() > 1:
                            pred_energy = pred_energy.squeeze(-1)
                        if target.dim() > 1:
                            target = target.squeeze(-1)
                        
                        loss = self.criterion(pred_energy, target)
                    
                    # 反向传播（混合精度）
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                
                else:
                    # 标准精度训练
                    pred_energy = self.model(batch_data)
                    
                    # 确保预测和目标的形状匹配
                    if pred_energy.dim() > 1:
                        pred_energy = pred_energy.squeeze(-1)
                    if target.dim() > 1:
                        target = target.squeeze(-1)
                    
                    loss = self.criterion(pred_energy, target)
                    
                    # 反向传播
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # 定期清理GPU缓存
                if num_batches % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️ GPU内存不足，跳过此批次: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"训练批次错误: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            except Exception as e:
                print(f"训练批次错误: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return total_loss / max(num_batches, 1)
    
    def evaluate(self):
        """评估模型（GPU优化版本）"""
        self.model.eval()
        predictions = []
        targets = []
        total_loss = 0.0
        valid_batches = 0
        
        # 评估前清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="🎯 GPU评估", leave=False):
                try:
                    # 高效移动数据到GPU
                    batch_data = self._move_to_device(batch_data)
                    target = self.normalize_target(batch_data.y)
                    
                    # 使用混合精度评估
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            pred_energy = self.model(batch_data)
                    else:
                        pred_energy = self.model(batch_data)
                    
                    # 确保预测和目标的形状匹配
                    if pred_energy.dim() > 1:
                        pred_energy = pred_energy.squeeze(-1)
                    if target.dim() > 1:
                        target = target.squeeze(-1)
                    
                    loss = self.criterion(pred_energy, target)
                    
                    # 反归一化进行评估
                    pred_denorm = self.denormalize_target(pred_energy).cpu().numpy()
                    target_denorm = batch_data.y.cpu().numpy()
                    
                    predictions.extend(pred_denorm.tolist())
                    targets.extend(target_denorm.tolist())
                    total_loss += loss.item()
                    valid_batches += 1
                    
                    # 定期清理GPU缓存
                    if valid_batches % 20 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"⚠️ GPU内存不足，跳过此评估批次: {e}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        print(f"评估批次错误: {e}")
                        continue
                except Exception as e:
                    print(f"评估批次错误: {e}")
                    continue
        
        if valid_batches == 0:
            return None
            
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # 计算指标
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
    
    def train(self, num_epochs, output_dir=None):
        """训练模型"""
        print(f"🚀 Starting Standalone EquiformerV2 training for {num_epochs} epochs...")
        
        best_r2 = -float('inf')
        best_model_state = None
        patience_counter = 0
        max_patience = 50  # 增加patience，确保能训练100个epoch
        
        for epoch in range(num_epochs):
            # 训练
            train_loss = self.train_epoch()
            
            # 评估
            val_metrics = self.evaluate()
            
            if val_metrics is None:
                print(f"Epoch {epoch+1}: Evaluation failed, skipping...")
                continue
            
            # 记录
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            self.val_r2s.append(val_metrics['r2'])
            
            # 学习率调度
            self.scheduler.step(val_metrics['loss'])
            
            # 保存最佳模型
            if val_metrics['r2'] > best_r2:
                best_r2 = val_metrics['r2']
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 早停（但patience足够大，确保能训练完100个epoch）
            if patience_counter >= max_patience and epoch >= 50:  # 至少训练50个epoch
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # 打印进度
            if (epoch + 1) % 1 == 0 or epoch == 0:  # 每个epoch都打印
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                      f"Train Loss={train_loss:.4f}, "
                      f"Val Loss={val_metrics['loss']:.4f}, "
                      f"Val R²={val_metrics['r2']:.4f}, "
                      f"Val MAE={val_metrics['mae']:.4f}, "
                      f"LR={current_lr:.6f}")
        
        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # 最终评估
        final_metrics = self.evaluate()
        
        if final_metrics is not None:
            print(f"\n🎯 Standalone EquiformerV2 Training completed!")
            print(f"Best R²: {best_r2:.4f}")
            print(f"Final metrics:")
            print(f"  R²: {final_metrics['r2']:.4f}")
            print(f"  MAE: {final_metrics['mae']:.4f}")
            print(f"  RMSE: {final_metrics['rmse']:.4f}")
            
            if final_metrics['r2'] >= 0.93:
                print("🎉 Congratulations! R² >= 0.93 achieved with EquiformerV2!")
            else:
                print(f"R² = {final_metrics['r2']:.4f} < 0.93")
                print("EquiformerV2 optimization suggestions:")
                print("1. Increase model complexity (more layers/channels)")
                print("2. Adjust learning rate and warmup")
                print("3. Increase training epochs")
                print("4. Tune attention and FFN parameters")
        else:
            print("Training failed - no valid evaluation results")
            final_metrics = {'r2': 0.0, 'mae': float('inf'), 'rmse': float('inf')}
        
        return final_metrics

def plot_training_curves(trainer, output_dir):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(trainer.train_losses) + 1)
    
    # 损失曲线
    axes[0].plot(epochs, trainer.train_losses, 'b-', label='训练损失', linewidth=2)
    axes[0].plot(epochs, trainer.val_losses, 'r-', label='验证损失', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('训练和验证损失曲线', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # R²曲线
    axes[1].plot(epochs, trainer.val_r2s, 'g-', label='验证 R²', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('R² Score', fontsize=12)
    axes[1].set_title('验证集 R² 分数曲线', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.93, color='r', linestyle='--', label='目标 R²=0.93', alpha=0.7)
    axes[1].legend(fontsize=11)
    
    # 预测vs真实值散点图
    if hasattr(trainer, 'final_predictions') and hasattr(trainer, 'final_targets'):
        axes[2].scatter(trainer.final_targets, trainer.final_predictions, alpha=0.5, s=20)
        min_val = min(trainer.final_targets.min(), trainer.final_predictions.min())
        max_val = max(trainer.final_targets.max(), trainer.final_predictions.max())
        axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测线')
        axes[2].set_xlabel('真实值', fontsize=12)
        axes[2].set_ylabel('预测值', fontsize=12)
        axes[2].set_title('预测值 vs 真实值', fontsize=14, fontweight='bold')
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: {os.path.join(output_dir, 'training_curves.png')}")
    plt.close()

def plot_prediction_scatter(final_metrics, output_dir):
    """绘制预测散点图"""
    if 'predictions' not in final_metrics or 'targets' not in final_metrics:
        return
    
    predictions = final_metrics['predictions']
    targets = final_metrics['targets']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 预测vs真实值
    axes[0].scatter(targets, predictions, alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测线')
    axes[0].set_xlabel('真实值 (ΔGH)', fontsize=12)
    axes[0].set_ylabel('预测值 (ΔGH)', fontsize=12)
    axes[0].set_title(f'预测值 vs 真实值 (R² = {final_metrics["r2"]:.4f})', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 残差图
    residuals = targets - predictions
    axes[1].scatter(predictions, residuals, alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('预测值 (ΔGH)', fontsize=12)
    axes[1].set_ylabel('残差 (真实值 - 预测值)', fontsize=12)
    axes[1].set_title(f'残差图 (MAE = {final_metrics["mae"]:.4f})', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"预测分析图已保存到: {os.path.join(output_dir, 'prediction_analysis.png')}")
    plt.close()

def main():
    """主函数"""
    print("🚀 Starting Standalone EquiformerV2-based hydrogen adsorption model training...")
    
    # 设置输出目录
    output_dir = Path("D:/mylab/equiformer/result/lab01")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 结果将保存到: {output_dir}")
    
    # 配置 - 单GPU优化训练
    config = {
        'lr': 0.0003,  # 单GPU适中的学习率
        'weight_decay': 1e-4,
        'num_epochs': 100,  # 训练100个epoch
        'batch_size': 6,  # 单GPU时适中的批次大小
        'target_mean': -0.2042156457901001,
        'target_std': 0.6712170839309692,
    }
    
    # 数据加载
    print("📂 加载数据集...")
    # 获取项目根目录（假设脚本在 src/ 目录下）
    script_dir = Path(__file__).parent
    project_root = script_dir.parent if script_dir.name == 'src' else Path.cwd()
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
        num_workers=0,  # Windows上LMDB环境对象无法pickle，使用单进程
        pin_memory=True if torch.cuda.is_available() else False,  # 启用内存固定以加速GPU传输
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=0,  # Windows上LMDB环境对象无法pickle，使用单进程
        pin_memory=True if torch.cuda.is_available() else False,  # 启用内存固定
        collate_fn=custom_collate_fn
    )
    
    # 创建独立的EquiformerV2模型
    print("🔧 创建 Standalone EquiformerV2 模型...")
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
    
    print(f"✅ Standalone EquiformerV2 模型已创建，参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练器
    trainer = EquiformerV2Trainer(model, train_loader, val_loader, config)
    
    # 训练
    final_metrics = trainer.train(config['num_epochs'], output_dir=str(output_dir))
    
    # 保存预测结果
    if 'predictions' in final_metrics and 'targets' in final_metrics:
        predictions_df = pd.DataFrame({
            'targets': final_metrics['targets'].flatten(),
            'predictions': final_metrics['predictions'].flatten()
        })
        predictions_df['residuals'] = predictions_df['targets'] - predictions_df['predictions']
        predictions_path = output_dir / 'predictions.csv'
        predictions_df.to_csv(predictions_path, index=False)
        print(f"✅ 预测结果已保存到: {predictions_path}")
        
        # 保存给可视化函数使用
        trainer.final_predictions = final_metrics['predictions']
        trainer.final_targets = final_metrics['targets']
    
    # 保存模型
    model_path = output_dir / 'best_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"✅ 模型已保存到: {model_path}")
    
    # 保存训练历史
    training_history = {
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'val_r2s': trainer.val_r2s,
        'config': config,
        'final_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                         for k, v in final_metrics.items() if k not in ['predictions', 'targets']}
    }
    
    history_path = output_dir / 'training_history.json'
    with open(history_path, "w", encoding='utf-8') as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)
    print(f"✅ 训练历史已保存到: {history_path}")
    
    # 绘制训练曲线
    try:
        plot_training_curves(trainer, str(output_dir))
        plot_prediction_scatter(final_metrics, str(output_dir))
    except Exception as e:
        print(f"⚠️ 绘制图表时出错: {e}")
    
    # 保存配置
    config_path = output_dir / 'config.json'
    with open(config_path, "w", encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✅ 配置已保存到: {config_path}")
    
    # 关闭数据集
    train_dataset.close()
    val_dataset.close()
    
    print(f"\n🎉 Standalone EquiformerV2 训练完成！所有结果已保存到: {output_dir}")

if __name__ == "__main__":
    main() 