#!/usr/bin/env python3
"""
EquiformerV2模型评估脚本
- 加载预训练的最佳模型
- 在验证集上评估性能
- 生成详细的评估报告和可视化
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

# 导入自定义模块
from standalone_equiformer_v2 import (
    StandaloneEquiformerV2,
    HydrogenDataset,
    custom_collate_fn,
    SimpleData  # 添加SimpleData导入以解决pickle问题
)

# 确保pickle可以找到SimpleData类
import sys
sys.modules['__main__'].SimpleData = SimpleData



def set_style():
    """设置matplotlib样式"""
    try:
        plt.style.use('seaborn')
    except OSError:
        # 新版本matplotlib使用seaborn-v0_8
        try:
            plt.style.use('seaborn-v0_8')
        except:
            # 如果都不可用，使用默认样式
            plt.style.use('default')
    
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

def create_model():
    """创建模型实例"""
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
    return model

def load_best_model(model, model_path):
    """加载最佳模型权重"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    print(f"Successfully loaded model from {model_path}")
    return model

def evaluate_model(model, val_loader, device, target_mean, target_std):
    """评估模型性能"""
    model.eval()
    predictions = []
    targets = []
    
    print("Evaluating model on validation set...")
    with torch.no_grad():
        for batch_data in val_loader:
            # 移动数据到设备
            batch_data.pos = batch_data.pos.to(device)
            batch_data.atomic_numbers = batch_data.atomic_numbers.to(device)
            batch_data.edge_index = batch_data.edge_index.to(device)
            batch_data.edge_distance = batch_data.edge_distance.to(device)
            batch_data.batch = batch_data.batch.to(device)
            batch_data.natoms = batch_data.natoms.to(device)
            
            # 前向传播
            pred = model(batch_data)
            
            # 反归一化
            pred = pred * target_std + target_mean
            
            # 收集预测和目标
            predictions.extend(pred.cpu().numpy())
            targets.extend(batch_data.y.numpy())
    
    # 转换为numpy数组
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # 计算指标
    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    
    return {
        'predictions': predictions,
        'targets': targets,
        'r2': r2,
        'mae': mae,
        'rmse': rmse
    }

def plot_results(results, save_dir):
    """可视化评估结果"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置绘图样式
    set_style()
    
    # 1. 散点图：预测值 vs 真实值
    plt.figure(figsize=(10, 8))
    plt.scatter(results['targets'], results['predictions'], alpha=0.5)
    plt.plot([min(results['targets']), max(results['targets'])],
             [min(results['targets']), max(results['targets'])],
             'r--', label='Perfect Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs True Values (R² = {results["r2"]:.4f})')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'predictions_vs_true.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 残差图
    residuals = results['predictions'] - results['targets']
    plt.figure(figsize=(10, 8))
    plt.scatter(results['predictions'], residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    plt.grid(True)
    plt.savefig(save_dir / 'residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 残差分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Count')
    plt.title('Distribution of Residuals')
    plt.grid(True)
    plt.savefig(save_dir / 'residuals_dist.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 保存评估指标
    metrics_df = pd.DataFrame({
        'Metric': ['R²', 'MAE', 'RMSE'],
        'Value': [results['r2'], results['mae'], results['rmse']]
    })
    metrics_df.to_csv(save_dir / 'evaluation_metrics.csv', index=False)
    
    # 5. 生成评估报告
    with open(save_dir / 'evaluation_report.txt', 'w') as f:
        f.write("EquiformerV2 Model Evaluation Report\n")
        f.write("================================\n\n")
        f.write(f"R² Score: {results['r2']:.4f}\n")
        f.write(f"Mean Absolute Error: {results['mae']:.4f}\n")
        f.write(f"Root Mean Square Error: {results['rmse']:.4f}\n")
        f.write("\nData Statistics:\n")
        f.write(f"Number of samples: {len(results['targets'])}\n")
        f.write(f"Target range: [{min(results['targets']):.4f}, {max(results['targets']):.4f}]\n")
        f.write(f"Prediction range: [{min(results['predictions']):.4f}, {max(results['predictions']):.4f}]\n")
    
    print(f"Evaluation results saved to {save_dir}")

def main():
    """主函数"""
    # 配置
    config = {
        'model_path': 'best_standalone_equiformer_v2_model.pt',
        'batch_size': 32,  # 评估时可以用更大的批次
        'target_mean': -0.2042156457901001,
        'target_std': 0.6712170839309692,
    }
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建并加载模型
    model = create_model()
    model = load_best_model(model, config['model_path'])
    model = model.to(device)
    
    # 加载验证数据集
    val_dataset = HydrogenDataset("datasets/custom_hydrogen/val.lmdb")
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn
    )
    
    # 评估模型
    results = evaluate_model(
        model,
        val_loader,
        device,
        config['target_mean'],
        config['target_std']
    )
    
    # 打印主要指标
    print("\nEvaluation Results:")
    print(f"R² Score: {results['r2']:.4f}")
    print(f"Mean Absolute Error: {results['mae']:.4f}")
    print(f"Root Mean Square Error: {results['rmse']:.4f}")
    
    # 可视化结果
    plot_results(results, 'evaluation_results')
    
    # 关闭数据集
    val_dataset.close()

if __name__ == "__main__":
    main() 