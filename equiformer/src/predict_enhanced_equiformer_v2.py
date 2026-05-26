#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用训练好的增强版EquiformerV2模型进行预测，并导出丰富评估指标与可视化
"""

import os
import sys
import json
import time
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
try:
    from scipy import stats as _scipy_stats
except Exception:
    _scipy_stats = None

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 导入模型和数据集
from src.enhanced_equiformer_v2 import EnhancedEquiformerV2, HydrogenDataset, custom_collate_fn, SimpleData

# 确保 SimpleData 在全局命名空间中，以便 pickle 可以找到它
import sys
sys.modules['__main__'].SimpleData = SimpleData

def load_normalization_stats(stats_file):
    """加载归一化统计信息"""
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    return stats

def normalize_targets(targets, stats):
    """归一化目标值"""
    return (targets - stats['target_mean']) / stats['target_std']

def denormalize_targets(normalized_targets, stats):
    """反归一化目标值"""
    return normalized_targets * stats['target_std'] + stats['target_mean']

def _nan_safe_div(numer: np.ndarray, denom: np.ndarray, eps: float) -> np.ndarray:
    denom_safe = np.where(np.abs(denom) < eps, eps, denom)
    return numer / denom_safe

def predict(model, dataloader, device, stats, latency_warmup: int = 10, latency_iters: int = 100):
    """使用模型进行预测，并返回丰富指标与延迟统计"""
    model.eval()

    true_values = []
    predictions = []
    lat_ms = []
    seen = 0

    with torch.no_grad():
        for batch in dataloader:
            # 将数据移至设备
            batch.pos = batch.pos.to(device)
            batch.atomic_numbers = batch.atomic_numbers.to(device)
            batch.edge_index = batch.edge_index.to(device)
            batch.edge_distance = batch.edge_distance.to(device)
            batch.y = batch.y.to(device)
            batch.batch = batch.batch.to(device)
            batch.natoms = batch.natoms.to(device)

            # 前向传播与延迟
            if torch.cuda.is_available() and device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model(batch)
            if torch.cuda.is_available() and device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if seen >= latency_warmup and len(lat_ms) < latency_iters:
                lat_ms.append((t1 - t0) * 1000.0)
            seen += 1

            # 反归一化预测值
            denorm_outputs = denormalize_targets(outputs, stats)

            # 保存预测值和真实值
            predictions.append(denorm_outputs.detach().cpu().numpy())
            true_values.append(batch.y.detach().cpu().numpy())

    # 合并预测值和真实值
    predictions = np.concatenate(predictions).astype(np.float64)
    true_values = np.concatenate(true_values).astype(np.float64)

    # 误差与基础指标
    err = predictions - true_values
    abs_err = np.abs(err)
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_err))
    median_ae = float(np.median(abs_err))
    me = float(np.mean(err))
    std_error = float(np.std(err))
    abs_error_p10 = float(np.percentile(abs_err, 10))
    abs_error_p50 = float(np.percentile(abs_err, 50))
    abs_error_p90 = float(np.percentile(abs_err, 90))

    # 比例类（加 eps 防护）
    eps = 1e-6
    mape = float(np.mean(np.abs(_nan_safe_div(err, np.abs(true_values), eps))))
    smape = float(np.mean(2.0 * abs_err / (np.abs(true_values) + np.abs(predictions) + eps)))
    mdape = float(np.median(np.abs(_nan_safe_div(err, np.abs(true_values), eps))))
    mpe = float(np.mean(_nan_safe_div(err, np.abs(true_values), eps)))

    # 相关性与拟合
    try:
        pearson_r = float(np.corrcoef(true_values, predictions)[0, 1])
    except Exception:
        pearson_r = float('nan')
    ss_res = float(np.sum((true_values - predictions) ** 2))
    ss_tot = float(np.sum((true_values - np.mean(true_values)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    n = len(true_values)
    p = 1
    r2_adj = float(1 - (1 - r2) * (n - 1) / (n - p - 1)) if n > p + 1 else float('nan')
    try:
        slope, intercept = np.polyfit(true_values, predictions, 1)
        slope = float(slope); intercept = float(intercept)
    except Exception:
        slope = float('nan'); intercept = float('nan')

    # 参数量、延迟、显存
    params_million = float(sum(p_.numel() for p_ in model.parameters()) / 1e6)
    lat_ms = np.array(lat_ms, dtype=np.float64) if lat_ms else np.array([], dtype=np.float64)
    if lat_ms.size:
        latency_ms_p50 = float(np.percentile(lat_ms, 50))
        latency_ms_p90 = float(np.percentile(lat_ms, 90))
        latency_ms_p99 = float(np.percentile(lat_ms, 99))
        throughput_sps = float(1000.0 / latency_ms_p50) if latency_ms_p50 > 0 else float('inf')
    else:
        latency_ms_p50 = latency_ms_p90 = latency_ms_p99 = throughput_sps = float('nan')
    if torch.cuda.is_available() and device.type == 'cuda':
        try:
            peak_mem_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        except Exception:
            peak_mem_mb = float('nan')
    else:
        peak_mem_mb = float('nan')

    metrics = {
        'mse': mse, 'rmse': rmse, 'mae': mae, 'median_ae': median_ae,
        'me': me, 'std_error': std_error,
        'abs_error_p10': abs_error_p10, 'abs_error_p50': abs_error_p50, 'abs_error_p90': abs_error_p90,
        'mape': mape, 'smape': smape, 'mdape': mdape, 'mpe': mpe,
        'r2': r2, 'r2_adj': r2_adj, 'pearson_r': pearson_r,
        'calib_slope': slope, 'calib_intercept': intercept,
        'params_million': params_million,
        'latency_ms_p50': latency_ms_p50, 'latency_ms_p90': latency_ms_p90, 'latency_ms_p99': latency_ms_p99,
        'throughput_sps': throughput_sps, 'peak_mem_mb': peak_mem_mb,
    }

    return predictions, true_values, metrics, abs_err

def plot_results(predictions, true_values, save_dir):
    """绘制预测结果"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 预测值与真实值对比图
    plt.figure(figsize=(10, 10))
    plt.scatter(true_values, predictions, alpha=0.5)
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')
    plt.xlabel('True Values (eV)')
    plt.ylabel('Predictions (eV)')
    plt.title('Predictions vs True Values')
    plt.savefig(os.path.join(save_dir, "predictions_vs_true.png"))
    plt.close()
    
    # 残差图
    plt.figure(figsize=(10, 10))
    residuals = predictions - true_values
    plt.scatter(true_values, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('True Values (eV)')
    plt.ylabel('Residuals (eV)')
    plt.title('Residuals')
    plt.savefig(os.path.join(save_dir, "residuals.png"))
    plt.close()
    
    # 残差分布图
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Residuals (eV)')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    plt.savefig(os.path.join(save_dir, "residuals_dist.png"))
    plt.close()

def save_predictions(predictions, true_values, save_path):
    """保存预测结果到CSV文件"""
    df = pd.DataFrame({
        'true': true_values,
        'predicted': predictions,
        'residual': predictions - true_values
    })
    df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据路径
    data_dir = project_root / "datasets" / "custom_hydrogen"
    test_path = data_dir / "test.lmdb"
    stats_path = data_dir / "normalization_stats.json"
    
    # 模型路径
    model_path = project_root / "checkpionts" / "best_enhanced_equiformer_v2.pt"
    
    # 输出目录 - 使用与原始equiformer_v2相同的目录结构
    experiment_date = "2025-09-10"
    output_dir = project_root / "experiments" / f"{experiment_date}_run1"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建日志目录
    logs_dir = output_dir / "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # 加载归一化统计信息
    stats = load_normalization_stats(stats_path)
    print(f"Loaded normalization stats: mean={stats['target_mean']}, std={stats['target_std']}")
    
    # 加载测试数据集
    test_dataset = HydrogenDataset(test_path)
    print(f"Test dataset size: {len(test_dataset)}")
    
    # 创建数据加载器
    batch_size = 32
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=custom_collate_fn,
        num_workers=0
    )
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    
    # 从checkpoint中读取模型配置，如果没有则使用默认值
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"Loading model config from checkpoint: {config}")
        # 使用checkpoint中的配置
        model = EnhancedEquiformerV2(
            max_radius=config.get('max_radius', 12.0),
            max_neighbors=config.get('max_neighbors', 20),
            max_num_elements=config.get('max_num_elements', 90),
            num_layers=config.get('num_layers', 3),
            sphere_channels=config.get('sphere_channels', 64),
            attn_hidden_channels=config.get('attn_hidden_channels', 32),
            num_heads=config.get('num_heads', 4),
            attn_alpha_channels=config.get('attn_alpha_channels', 32),
            attn_value_channels=config.get('attn_value_channels', 8),
            ffn_hidden_channels=config.get('ffn_hidden_channels', 64),
            lmax_list=config.get('lmax_list', [4]),
            mmax_list=config.get('mmax_list', [2]),
            grid_resolution=config.get('grid_resolution', 12),
            edge_channels=config.get('edge_channels', 64),
            use_atom_edge_embedding=config.get('use_atom_edge_embedding', True),
            share_atom_edge_embedding=config.get('share_atom_edge_embedding', False),
            alpha_drop=config.get('alpha_drop', 0.1),
            drop_path_rate=config.get('drop_path_rate', 0.05),
            proj_drop=config.get('proj_drop', 0.0),
            norm_type=config.get('norm_type', 'layer_norm_sh')
        ).to(device)
    else:
        print("No config found in checkpoint, using default values")
        # 使用默认配置
        model = EnhancedEquiformerV2(
            max_radius=12.0,
            max_neighbors=20,
            max_num_elements=90,
            num_layers=3,
            sphere_channels=64,
            attn_hidden_channels=32,
            num_heads=4,
            attn_alpha_channels=32,
            attn_value_channels=8,
            ffn_hidden_channels=64,
            lmax_list=[4],
            mmax_list=[2],
            grid_resolution=12,
            edge_channels=64,
            use_atom_edge_embedding=True,
            share_atom_edge_embedding=False,
            alpha_drop=0.1,
            drop_path_rate=0.05,
            proj_drop=0.0,
            norm_type='layer_norm_sh'
        ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {model_path}")
    print(f"Model validation MAE: {checkpoint.get('val_mae', 'N/A')} eV")
    
    # 进行预测
    print("Running predictions...")
    predictions, true_values, mae, rmse = predict(model, test_loader, device, stats)
    print(f"Test MAE: {mae:.6f} eV")
    print(f"Test RMSE: {rmse:.6f} eV")
    
    # 绘制结果
    plot_results(predictions, true_values, output_dir)
    
    # 保存预测结果
    save_predictions(predictions, true_values, os.path.join(output_dir, "enhanced_equiformer_v2_predictions.csv"))
    
    # 保存评估指标
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
    }
    with open(os.path.join(logs_dir, "enhanced_equiformer_v2_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # 关闭数据集
    test_dataset.close()

if __name__ == "__main__":
    main()
