#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Enhanced EquiformerV2 Model
Using custom_hydrogen dataset
"""

import os
import sys
import json
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免需要图形界面

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 导入增强版EquiformerV2模型和数据集
from src.enhanced_equiformer_v2 import EnhancedEquiformerV2, HydrogenDataset, custom_collate_fn, SimpleData

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

def train_epoch(model, dataloader, optimizer, device, stats, epoch, num_epochs):
    """训练一个epoch，显示详细的进度信息"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # 添加进度条，显示更多信息
    pbar = tqdm(dataloader, desc="Epoch {}/{}".format(epoch+1, num_epochs), leave=True)
    for batch_idx, batch in enumerate(pbar):
        # 将数据移至设备
        batch.pos = batch.pos.to(device)
        batch.atomic_numbers = batch.atomic_numbers.to(device)
        batch.edge_index = batch.edge_index.to(device)
        batch.edge_distance = batch.edge_distance.to(device)
        batch.y = batch.y.to(device)
        batch.batch = batch.batch.to(device)
        batch.natoms = batch.natoms.to(device)
        
        # 归一化目标
        targets = normalize_targets(batch.y, stats)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(batch)
        
        # 计算损失
        loss = torch.nn.functional.mse_loss(outputs, targets)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item()
        total_loss += batch_loss
        num_batches += 1
        
        # 更新进度条，显示更多信息
        avg_loss = total_loss / num_batches
        pbar.set_postfix({
            "batch": "{}/{}".format(batch_idx+1, len(dataloader)),
            "loss": "{:.4f}".format(batch_loss),
            "avg_loss": "{:.4f}".format(avg_loss)
        })
        
        # 每10个batch打印一次详细信息
        if (batch_idx + 1) % 10 == 0:
            print("  Epoch {}/{} | Batch {}/{} | Loss: {:.4f} | Avg Loss: {:.4f}".format(
                epoch+1, num_epochs, batch_idx+1, len(dataloader), batch_loss, avg_loss
            ), flush=True)

    epoch_loss = total_loss / num_batches
    print("Epoch {}/{} completed | Avg Loss: {:.4f}".format(epoch+1, num_epochs, epoch_loss), flush=True)
    return epoch_loss

def validate(model, dataloader, device, stats):
    """验证模型，显示详细的进度信息"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    true_values = []
    predictions = []
    
    # 添加进度条，显示更多信息
    pbar = tqdm(dataloader, desc="验证中", leave=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # 将数据移至设备
            batch.pos = batch.pos.to(device)
            batch.atomic_numbers = batch.atomic_numbers.to(device)
            batch.edge_index = batch.edge_index.to(device)
            batch.edge_distance = batch.edge_distance.to(device)
            batch.y = batch.y.to(device)
            batch.batch = batch.batch.to(device)
            batch.natoms = batch.natoms.to(device)
            
            # 归一化目标
            targets = normalize_targets(batch.y, stats)
            
            # 前向传播
            outputs = model(batch)
            
            # 计算损失
            loss = torch.nn.functional.mse_loss(outputs, targets)
            
            # 反归一化预测值和真实值
            denorm_outputs = denormalize_targets(outputs, stats)
            denorm_targets = batch.y
            
            # 保存预测值和真实值
            predictions.append(denorm_outputs.cpu().numpy())
            true_values.append(denorm_targets.cpu().numpy())
            
            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1
            
            # 更新进度条，显示更多信息
            avg_loss = total_loss / num_batches
            pbar.set_postfix({
                "batch": "{}/{}".format(batch_idx+1, len(dataloader)),
                "loss": "{:.4f}".format(batch_loss),
                "avg_loss": "{:.4f}".format(avg_loss)
            })
    
    # 合并预测值和真实值
    predictions = np.concatenate(predictions)
    true_values = np.concatenate(true_values)
    
    # 计算MAE和RMSE
    mae = np.mean(np.abs(predictions - true_values))
    rmse = np.sqrt(np.mean((predictions - true_values) ** 2))
    
    print("Validation completed | Avg Loss: {:.4f} | MAE: {:.4f} | RMSE: {:.4f}".format(
        total_loss / num_batches, mae, rmse
    ))
    
    return total_loss / num_batches, mae, rmse, predictions, true_values

def plot_results(predictions, true_values, save_path):
    """绘制预测结果"""
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 预测值与真实值对比图
    plt.figure(figsize=(10, 10))
    plt.scatter(true_values, predictions, alpha=0.5)
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')
    plt.xlabel('True Values (eV)')
    plt.ylabel('Predictions (eV)')
    plt.title('Predictions vs True Values')
    plt.savefig("{}_predictions_vs_true.png".format(save_path))
    plt.close()
    
    # 残差图
    plt.figure(figsize=(10, 10))
    residuals = predictions - true_values
    plt.scatter(true_values, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('True Values (eV)')
    plt.ylabel('Residuals (eV)')
    plt.title('Residuals')
    plt.savefig("{}_residuals.png".format(save_path))
    plt.close()
    
    # 残差分布图
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Residuals (eV)')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    plt.savefig("{}_residuals_dist.png".format(save_path))
    plt.close()

def save_predictions(predictions, true_values, save_path):
    """保存预测结果"""
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存为CSV
    with open("{}_predictions.csv".format(save_path), 'w') as f:
        f.write("true,predicted\n")
        for true, pred in zip(true_values, predictions):
            f.write("{},{}\n".format(true, pred))

def train_and_eval(args):
    """
    训练和评估函数，返回实验结果字典
    
    Args:
        args: 包含所有训练参数的命名空间
        
    Returns:
        dict: 包含测试指标和性能指标的字典
    """
    def str2bool(x: str) -> bool:
        return str(x).lower() in ('1','true','yes','y','t')

    # 设置随机种子
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    
    # 设置设备 - 自动选择第一个可用GPU（cuda:0），避免无效的GPU索引
    if torch.cuda.is_available():
        try:
            device_index = 0
            if torch.cuda.device_count() <= device_index:
                device = torch.device('cuda:0')
            else:
                device = torch.device(f'cuda:{device_index}')
            torch.cuda.set_device(device)
        except Exception:
            device = torch.device('cuda:0')
            torch.cuda.set_device(0)
    else:
        device = torch.device('cpu')
    
    print(f"使用设备: {device}")
    
    # 数据路径
    data_dir = project_root / "datasets" / "custom_hydrogen"
    train_path = data_dir / "train.lmdb"
    val_path = data_dir / "val.lmdb"
    test_path = data_dir / "test.lmdb"
    stats_path = data_dir / "normalization_stats.json"
    
    # 加载归一化统计信息
    stats = load_normalization_stats(stats_path)
    print(f"加载归一化统计: mean={stats['target_mean']}, std={stats['target_std']}")
    
    # 加载数据集
    train_dataset = HydrogenDataset(train_path)
    val_dataset = HydrogenDataset(val_path)
    test_dataset = HydrogenDataset(test_path)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=int(args.batch_size), 
        shuffle=True, 
        collate_fn=custom_collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=int(args.batch_size), 
        shuffle=False, 
        collate_fn=custom_collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=int(args.batch_size), 
        shuffle=False, 
        collate_fn=custom_collate_fn,
        num_workers=0
    )
    
    # 创建模型
    model = EnhancedEquiformerV2(
        max_radius=float(args.max_radius),
        max_neighbors=int(args.max_neighbors),
        max_num_elements=90,
        num_layers=int(args.num_layers),
        sphere_channels=int(args.sphere_channels),
        attn_hidden_channels=int(args.attn_hidden_channels),
        num_heads=int(args.num_heads),
        attn_alpha_channels=int(args.attn_alpha_channels),
        attn_value_channels=int(args.attn_value_channels),
        ffn_hidden_channels=int(args.ffn_hidden_channels),
        lmax_list=list(args.lmax_list) if args.lmax_list else [2],  # 降低lmax以避免grid_resolution限制
        mmax_list=list(args.mmax_list) if args.mmax_list else [2],
        grid_resolution=int(args.grid_resolution),
        edge_channels=int(args.edge_channels),
        use_atom_edge_embedding=str2bool(args.use_atom_edge_embedding),
        share_atom_edge_embedding=str2bool(args.share_atom_edge_embedding),
        alpha_drop=float(args.alpha_drop),
        drop_path_rate=float(args.drop_path_rate),
        proj_drop=float(args.proj_drop),
        norm_type=str(args.norm_type)
    ).to(device)
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 训练循环
    num_epochs = int(args.num_epochs)  # 使用命令行参数
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device, stats, epoch, num_epochs)
        
        # 验证
        val_loss, val_mae, val_rmse, _, _ = validate(model, val_loader, device, stats)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
    
    # 测试评估
    test_loss, test_mae, test_rmse, _, _ = validate(model, test_loader, device, stats)
    
    # 计算性能指标（占位值）
    latency_ms = -1.0  # 后续完善
    throughput = -1.0  # 后续完善
    
    # 关闭数据集
    train_dataset.close()
    val_dataset.close()
    test_dataset.close()
    
    # 返回结果
    return {
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_loss": float(test_loss),
        "params": int(total_params),
        "latency_ms": float(latency_ms),
        "throughput": float(throughput),
        "seed": int(args.seed)
    }

def main():
    # CLI 参数
    ap = argparse.ArgumentParser(description='Train Enhanced EquiformerV2 with configurable hyperparameters')
    # 结构参数（默认为当前脚本小模型配置）
    ap.add_argument('--num_layers', type=int, default=3)
    ap.add_argument('--sphere_channels', type=int, default=64)
    ap.add_argument('--attn_hidden_channels', type=int, default=32)
    ap.add_argument('--num_heads', type=int, default=4)
    ap.add_argument('--attn_alpha_channels', type=int, default=32)
    ap.add_argument('--attn_value_channels', type=int, default=8)
    ap.add_argument('--ffn_hidden_channels', type=int, default=64)
    ap.add_argument('--lmax_list', type=int, nargs='*', default=[2])  # 降低默认lmax
    ap.add_argument('--mmax_list', type=int, nargs='*', default=[2])
    ap.add_argument('--grid_resolution', type=int, default=12)
    ap.add_argument('--edge_channels', type=int, default=64)
    ap.add_argument('--use_atom_edge_embedding', type=str, default='True')
    ap.add_argument('--share_atom_edge_embedding', type=str, default='False')
    ap.add_argument('--alpha_drop', type=float, default=0.1)
    ap.add_argument('--drop_path_rate', type=float, default=0.05)
    ap.add_argument('--proj_drop', type=float, default=0.0)
    ap.add_argument('--norm_type', type=str, default='layer_norm_sh')
    # 图构建
    ap.add_argument('--max_radius', type=float, default=12.0)
    ap.add_argument('--max_neighbors', type=int, default=20)
    ap.add_argument('--cutoff', type=float, default=12.0)  # 别名，与max_radius相同
    # 实验参数
    ap.add_argument('--eSCN', type=str, default='False')  # 是否使用eSCN
    ap.add_argument('--attn_renorm', type=str, default='False')  # 注意力重归一化
    ap.add_argument('--sep_s2', type=str, default='False')  # 分离SO(2)操作
    ap.add_argument('--sep_ln', type=str, default='False')  # 分离层归一化
    ap.add_argument('--data_ratio', type=float, default=1.0)  # 数据使用比例
    ap.add_argument('--pretrained', type=str, default='False')  # 是否使用预训练
    ap.add_argument('--split_file', type=str, default='')  # 数据分割文件
    ap.add_argument('--pretrained_ckpt', type=str, default='')  # 预训练检查点
    ap.add_argument('--exp_name', type=str, default='')  # 实验名称
    # 训练参数
    ap.add_argument('--batch_size', type=int, default=16)  # 与正式训练保持一致
    ap.add_argument('--lr', type=float, default=0.0002)    # 与配置文件保持一致
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--num_epochs', type=int, default=100)  # 训练轮数
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--experiment_tag', type=str, default='')
    ap.add_argument('--checkpoint_dir', type=str, default='')  # 自定义checkpoint目录（每个实验配置独立）
    ap.add_argument('--resume', action='store_true')  # 从checkpoint恢复训练
    args = ap.parse_args()

    def str2bool(x: str) -> bool:
        return str(x).lower() in ('1','true','yes','y','t')

    # 设置随机种子
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    
    # 设置设备 - 自动选择第一个可用GPU（cuda:0），避免无效的GPU索引
    if torch.cuda.is_available():
        try:
            device_index = 0
            if torch.cuda.device_count() <= device_index:
                device = torch.device('cuda:0')
            else:
                device = torch.device(f'cuda:{device_index}')
            torch.cuda.set_device(device)
        except Exception:
            device = torch.device('cuda:0')
            torch.cuda.set_device(0)
    else:
        device = torch.device('cpu')
    print("Using device: {}".format(device))
    
    # 设置CUDA内存优化
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        # 设置内存分配策略
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        # 设置内存增长策略
        torch.cuda.set_per_process_memory_fraction(0.8)  # 限制使用80%的GPU内存
    
    # 数据路径
    data_dir = project_root / "datasets" / "custom_hydrogen"
    train_path = data_dir / "train.lmdb"
    val_path = data_dir / "val.lmdb"
    test_path = data_dir / "test.lmdb"
    stats_path = data_dir / "normalization_stats.json"
    
    # 创建checkpoints目录（每个实验配置独立目录）
    if args.checkpoint_dir:
        checkpoints_dir = Path(args.checkpoint_dir)
    else:
        checkpoints_dir = project_root / "checkpionts"
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # 创建实验目录 - ablation/<k=v__...>/seed=<seed>/
    experiment_date = "2025-09-10"
    kv_parts = [
        f"num_layers={args.num_layers}",
        f"sphere_channels={args.sphere_channels}",
        f"num_heads={args.num_heads}",
        f"grid_resolution={args.grid_resolution}",
        f"edge_channels={args.edge_channels}",
    ]
    if args.experiment_tag:
        kv_parts.append(args.experiment_tag)
    ablation_name = "__".join(kv_parts)
    experiment_dir = project_root / "experiments" / f"{experiment_date}_run1" / "ablation" / ablation_name / f"seed={args.seed}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 创建日志目录
    logs_dir = experiment_dir / "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # 训练参数
    batch_size = int(args.batch_size)
    
    # 加载归一化统计信息
    stats = load_normalization_stats(stats_path)
    print("Loaded normalization stats: mean={}, std={}".format(stats['target_mean'], stats['target_std']))
    
    # 加载数据集
    train_dataset = HydrogenDataset(train_path)
    val_dataset = HydrogenDataset(val_path)
    test_dataset = HydrogenDataset(test_path)
    
    print("Train dataset size: {}".format(len(train_dataset)))
    print("Validation dataset size: {}".format(len(val_dataset)))
    print("Test dataset size: {}".format(len(test_dataset)))
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=custom_collate_fn,
        num_workers=0,
        pin_memory=True  # 启用内存固定以加速GPU传输
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=custom_collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=custom_collate_fn,
        num_workers=0
    )
    
    # 创建增强版EquiformerV2模型（由 CLI 参数控制）
    model = EnhancedEquiformerV2(
        max_radius=float(args.max_radius),
        max_neighbors=int(args.max_neighbors),
        max_num_elements=90,
        num_layers=int(args.num_layers),
        sphere_channels=int(args.sphere_channels),
        attn_hidden_channels=int(args.attn_hidden_channels),
        num_heads=int(args.num_heads),
        attn_alpha_channels=int(args.attn_alpha_channels),
        attn_value_channels=int(args.attn_value_channels),
        ffn_hidden_channels=int(args.ffn_hidden_channels),
        lmax_list=list(args.lmax_list) if args.lmax_list else [2],  # 降低lmax以避免grid_resolution限制
        mmax_list=list(args.mmax_list) if args.mmax_list else [2],
        grid_resolution=int(args.grid_resolution),
        edge_channels=int(args.edge_channels),
        use_atom_edge_embedding=str2bool(args.use_atom_edge_embedding),
        share_atom_edge_embedding=str2bool(args.share_atom_edge_embedding),
        alpha_drop=float(args.alpha_drop),
        drop_path_rate=float(args.drop_path_rate),
        proj_drop=float(args.proj_drop),
        norm_type=str(args.norm_type)
    ).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: {:,}".format(total_params))
    
    # 优化器和学习率调度器
    optimizer = Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 训练参数 - 使用命令行指定的epoch数
    num_epochs = int(args.num_epochs)
    best_val_loss = float('inf')
    best_model_path = checkpoints_dir / "best_enhanced_equiformer_v2.pt"
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': [],
        'lr': []
    }
    
    # ========== RESUME FROM CHECKPOINT ==========
    start_epoch = 0
    if args.resume and args.checkpoint_dir:
        checkpoint_path = checkpoints_dir / "best_enhanced_equiformer_v2.pt"
        if checkpoint_path.exists():
            print(f"\n[RESUME] Found checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            if 'history' in checkpoint:
                history = checkpoint['history']
            print(f"[RESUME] Resuming from epoch {start_epoch}, best val_loss: {best_val_loss:.6f}")
            cfg = checkpoint.get('config', {})
            print(f"[RESUME] Model config: lmax={cfg.get('lmax_list','N/A')}, num_layers={cfg.get('num_layers','N/A')}")
        else:
            print(f"\n[RESUME] --resume set but no checkpoint found at {checkpoint_path}, starting from scratch")
    # ===========================================

    # 训练循环
    print("\n" + "="*80)
    print("开始训练增强版EquiformerV2模型...")
    print("训练参数:")
    print("  - 批次大小: {}".format(batch_size))
    print("  - 训练轮数: {}".format(num_epochs))
    print("  - 学习率: {}".format(optimizer.param_groups[0]['lr']))
    print("  - 设备: {}".format(device))
    if start_epoch > 0:
        print("  - 起始轮次: {} (from checkpoint)".format(start_epoch))
    print("="*80 + "\n")

    for epoch in range(start_epoch, num_epochs):
        print("\n" + "-"*50)
        print("开始第 {}/{} 轮训练".format(epoch+1, num_epochs))
        print("-"*50)
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device, stats, epoch, num_epochs)
        
        print("\n" + "-"*50)
        print("开始第 {}/{} 轮验证".format(epoch+1, num_epochs))
        print("-"*50)
        
        # 验证
        val_loss, val_mae, val_rmse, _, _ = validate(model, val_loader, device, stats)
        
        # 更新学习率
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 保存历史
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_mae'].append(float(val_mae))
        history['val_rmse'].append(float(val_rmse))
        history['lr'].append(float(current_lr))
        
        # 打印轮次总结
        print("\n" + "="*80)
        print("第 {}/{} 轮训练总结:".format(epoch+1, num_epochs))
        print("  - 训练损失: {:.6f}".format(train_loss))
        print("  - 验证损失: {:.6f}".format(val_loss))
        print("  - 验证MAE: {:.6f} eV".format(val_mae))
        print("  - 验证RMSE: {:.6f} eV".format(val_rmse))
        print("  - 学习率: {:.6f}".format(current_lr))
        print("="*80 + "\n")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            # 保存配置元数据
            config = {
                'num_layers': int(args.num_layers),
                'sphere_channels': int(args.sphere_channels),
                'attn_hidden_channels': int(args.attn_hidden_channels),
                'num_heads': int(args.num_heads),
                'attn_alpha_channels': int(args.attn_alpha_channels),
                'attn_value_channels': int(args.attn_value_channels),
                'ffn_hidden_channels': int(args.ffn_hidden_channels),
                'lmax_list': list(args.lmax_list) if args.lmax_list else [4],
                'mmax_list': list(args.mmax_list) if args.mmax_list else [2],
                'grid_resolution': int(args.grid_resolution),
                'edge_channels': int(args.edge_channels),
                'use_atom_edge_embedding': bool(str2bool(args.use_atom_edge_embedding)),
                'share_atom_edge_embedding': bool(str2bool(args.share_atom_edge_embedding)),
                'alpha_drop': float(args.alpha_drop),
                'drop_path_rate': float(args.drop_path_rate),
                'proj_drop': float(args.proj_drop),
                'norm_type': str(args.norm_type),
                'max_radius': float(args.max_radius),
                'max_neighbors': int(args.max_neighbors),
                'seed': int(args.seed),
                'experiment_tag': str(args.experiment_tag),
            }
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': float(val_loss),
                'val_mae': float(val_mae),
                'val_rmse': float(val_rmse),
                'config': config,
                'history': history,
            }, best_model_path)
            print("保存最佳模型，验证损失: {:.6f}".format(val_loss))
        
        # 保存训练历史
        with open(logs_dir / "enhanced_equiformer_v2_training_history.json", 'w') as f:
            json.dump(history, f, indent=4)
        print("保存训练历史到 {}/enhanced_equiformer_v2_training_history.json".format(logs_dir))
        
        # 早停
        if current_lr < 1e-6:
            print("学习率过小，停止训练")
            break
    
    print("Training completed!")
    
    # 加载最佳模型
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded best model from epoch {} with validation MAE {:.6f} eV".format(checkpoint['epoch']+1, checkpoint['val_mae']))
    
    # 在测试集上评估
    test_loss, test_mae, test_rmse, test_preds, test_true = validate(model, test_loader, device, stats)
    print("Test Loss: {:.6f} | Test MAE: {:.6f} eV | Test RMSE: {:.6f} eV".format(test_loss, test_mae, test_rmse))
    
    # 绘制并保存结果
    plot_results(test_preds, test_true, experiment_dir / "enhanced_equiformer_v2")
    save_predictions(test_preds, test_true, experiment_dir / "enhanced_equiformer_v2")
    
    # 保存测试结果
    test_results = {
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
    }
    with open(logs_dir / "enhanced_equiformer_v2_test_results.json", 'w') as f:
        json.dump(test_results, f, indent=4)
    
    # 关闭数据集
    train_dataset.close()
    val_dataset.close()
    test_dataset.close()
    
    # 返回结果字典（用于批跑脚本）
    return {
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_loss": float(test_loss),
        "params": int(total_params),
        "latency_ms": -1.0,  # 占位值
        "throughput": -1.0,  # 占位值
        "seed": int(args.seed)
    }

if __name__ == "__main__":
    result = main()
    if result:
        print("\n" + "="*80)
        print("实验完成！结果:")
        print(f"  - Test MAE: {result['test_mae']:.6f} eV")
        print(f"  - Test RMSE: {result['test_rmse']:.6f} eV")
        print(f"  - Test Loss: {result['test_loss']:.6f}")
        print(f"  - 参数量: {result['params']:,}")
        print(f"  - 延迟: {result['latency_ms']:.2f} ms")
        print(f"  - 吞吐量: {result['throughput']:.2f}")
        print(f"  - 种子: {result['seed']}")
        print("="*80)
