#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘图工具模块
提供各种消融实验的可视化功能

主要功能:
- draw_lmax_curve: 绘制Lmax曲线
- draw_module_ablation: 绘制模块消融柱状图
- draw_capacity_bars: 绘制容量对比柱状图
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def draw_lmax_curve(df: pd.DataFrame, save_path: str, 
                   lmax_col: str = 'lmax_list', 
                   mae_col: str = 'test_mae',
                   rmse_col: str = 'test_rmse') -> None:
    """
    绘制Lmax曲线图
    
    Args:
        df: 包含实验结果的DataFrame
        save_path: 保存路径
        lmax_col: Lmax列名
        mae_col: MAE列名
        rmse_col: RMSE列名
    """
    # 过滤有效数据
    valid_df = df[(df[mae_col] > 0) & (df[rmse_col] > 0)].copy()
    
    if valid_df.empty:
        print("警告: 没有有效的Lmax数据")
        return
    
    # 处理lmax_list列（可能是列表格式）
    if valid_df[lmax_col].dtype == 'object':
        # 尝试解析列表格式
        try:
            valid_df['lmax_value'] = valid_df[lmax_col].apply(
                lambda x: eval(x)[0] if isinstance(x, str) and x.startswith('[') else x
            )
        except:
            valid_df['lmax_value'] = valid_df[lmax_col]
    else:
        valid_df['lmax_value'] = valid_df[lmax_col]
    
    # 按lmax分组计算均值
    grouped = valid_df.groupby('lmax_value').agg({
        mae_col: ['mean', 'std'],
        rmse_col: ['mean', 'std']
    }).reset_index()
    
    # 展平列名
    grouped.columns = ['lmax', 'mae_mean', 'mae_std', 'rmse_mean', 'rmse_std']
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # MAE曲线
    ax1.errorbar(grouped['lmax'], grouped['mae_mean'], 
                yerr=grouped['mae_std'], marker='o', capsize=5, capthick=2)
    ax1.set_xlabel('Lmax')
    ax1.set_ylabel('Test MAE (eV)')
    ax1.set_title('Lmax vs Test MAE')
    ax1.grid(True, alpha=0.3)
    
    # RMSE曲线
    ax2.errorbar(grouped['lmax'], grouped['rmse_mean'], 
                yerr=grouped['rmse_std'], marker='s', capsize=5, capthick=2, color='orange')
    ax2.set_xlabel('Lmax')
    ax2.set_ylabel('Test RMSE (eV)')
    ax2.set_title('Lmax vs Test RMSE')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Lmax曲线图已保存到: {save_path}")

def draw_module_ablation(df: pd.DataFrame, save_path: str,
                         module_cols: List[str] = None,
                         mae_col: str = 'test_mae',
                         rmse_col: str = 'test_rmse') -> None:
    """
    绘制模块消融柱状图
    
    Args:
        df: 包含实验结果的DataFrame
        save_path: 保存路径
        module_cols: 模块列名列表
        mae_col: MAE列名
        rmse_col: RMSE列名
    """
    if module_cols is None:
        # 自动检测布尔类型的模块列
        module_cols = [col for col in df.columns if df[col].dtype == bool]
    
    if not module_cols:
        print("警告: 没有找到模块列")
        return
    
    # 过滤有效数据
    valid_df = df[(df[mae_col] > 0) & (df[rmse_col] > 0)].copy()
    
    if valid_df.empty:
        print("警告: 没有有效的模块数据")
        return
    
    # 创建模块组合标签
    valid_df['module_config'] = valid_df[module_cols].apply(
        lambda row: '+'.join([col for col in module_cols if row[col]]), axis=1
    )
    
    # 按模块配置分组
    grouped = valid_df.groupby('module_config').agg({
        mae_col: ['mean', 'std'],
        rmse_col: ['mean', 'std']
    }).reset_index()
    
    # 展平列名
    grouped.columns = ['config', 'mae_mean', 'mae_std', 'rmse_mean', 'rmse_std']
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # MAE柱状图
    bars1 = ax1.bar(range(len(grouped)), grouped['mae_mean'], 
                   yerr=grouped['mae_std'], capsize=5, alpha=0.7)
    ax1.set_xlabel('Module Configuration')
    ax1.set_ylabel('Test MAE (eV)')
    ax1.set_title('Module Ablation - MAE')
    ax1.set_xticks(range(len(grouped)))
    ax1.set_xticklabels(grouped['config'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # RMSE柱状图
    bars2 = ax2.bar(range(len(grouped)), grouped['rmse_mean'], 
                   yerr=grouped['rmse_std'], capsize=5, alpha=0.7, color='orange')
    ax2.set_xlabel('Module Configuration')
    ax2.set_ylabel('Test RMSE (eV)')
    ax2.set_title('Module Ablation - RMSE')
    ax2.set_xticks(range(len(grouped)))
    ax2.set_xticklabels(grouped['config'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"模块消融图已保存到: {save_path}")

def draw_capacity_bars(df: pd.DataFrame, save_path: str,
                      capacity_cols: List[str] = None,
                      mae_col: str = 'test_mae',
                      rmse_col: str = 'test_rmse') -> None:
    """
    绘制容量对比柱状图
    
    Args:
        df: 包含实验结果的DataFrame
        save_path: 保存路径
        capacity_cols: 容量相关列名列表
        mae_col: MAE列名
        rmse_col: RMSE列名
    """
    if capacity_cols is None:
        # 自动检测数值类型的容量列
        capacity_cols = ['num_layers', 'sphere_channels', 'num_heads', 
                        'grid_resolution', 'edge_channels']
        capacity_cols = [col for col in capacity_cols if col in df.columns]
    
    if not capacity_cols:
        print("警告: 没有找到容量列")
        return
    
    # 过滤有效数据
    valid_df = df[(df[mae_col] > 0) & (df[rmse_col] > 0)].copy()
    
    if valid_df.empty:
        print("警告: 没有有效的容量数据")
        return
    
    # 创建容量标签
    valid_df['capacity_config'] = valid_df[capacity_cols].apply(
        lambda row: '_'.join([f"{col}={row[col]}" for col in capacity_cols]), axis=1
    )
    
    # 按容量配置分组
    grouped = valid_df.groupby('capacity_config').agg({
        mae_col: ['mean', 'std'],
        rmse_col: ['mean', 'std']
    }).reset_index()
    
    # 展平列名
    grouped.columns = ['config', 'mae_mean', 'mae_std', 'rmse_mean', 'rmse_std']
    
    # 按MAE排序
    grouped = grouped.sort_values('mae_mean')
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # MAE柱状图
    bars1 = ax1.bar(range(len(grouped)), grouped['mae_mean'], 
                   yerr=grouped['mae_std'], capsize=5, alpha=0.7)
    ax1.set_xlabel('Capacity Configuration')
    ax1.set_ylabel('Test MAE (eV)')
    ax1.set_title('Capacity Comparison - MAE')
    ax1.set_xticks(range(len(grouped)))
    ax1.set_xticklabels(grouped['config'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # RMSE柱状图
    bars2 = ax2.bar(range(len(grouped)), grouped['rmse_mean'], 
                   yerr=grouped['rmse_std'], capsize=5, alpha=0.7, color='orange')
    ax2.set_xlabel('Capacity Configuration')
    ax2.set_ylabel('Test RMSE (eV)')
    ax2.set_title('Capacity Comparison - RMSE')
    ax2.set_xticks(range(len(grouped)))
    ax2.set_xticklabels(grouped['config'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"容量对比图已保存到: {save_path}")

def draw_parameter_sweep(df: pd.DataFrame, save_path: str,
                        param_col: str,
                        mae_col: str = 'test_mae',
                        rmse_col: str = 'test_rmse') -> None:
    """
    绘制参数扫描图
    
    Args:
        df: 包含实验结果的DataFrame
        save_path: 保存路径
        param_col: 参数列名
        mae_col: MAE列名
        rmse_col: RMSE列名
    """
    # 过滤有效数据
    valid_df = df[(df[mae_col] > 0) & (df[rmse_col] > 0)].copy()
    
    if valid_df.empty:
        print(f"警告: 没有有效的{param_col}数据")
        return
    
    # 按参数分组
    grouped = valid_df.groupby(param_col).agg({
        mae_col: ['mean', 'std'],
        rmse_col: ['mean', 'std']
    }).reset_index()
    
    # 展平列名
    grouped.columns = [param_col, 'mae_mean', 'mae_std', 'rmse_mean', 'rmse_std']
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # MAE曲线
    ax1.errorbar(grouped[param_col], grouped['mae_mean'], 
                yerr=grouped['mae_std'], marker='o', capsize=5, capthick=2)
    ax1.set_xlabel(param_col)
    ax1.set_ylabel('Test MAE (eV)')
    ax1.set_title(f'{param_col} vs Test MAE')
    ax1.grid(True, alpha=0.3)
    
    # RMSE曲线
    ax2.errorbar(grouped[param_col], grouped['rmse_mean'], 
                yerr=grouped['rmse_std'], marker='s', capsize=5, capthick=2, color='orange')
    ax2.set_xlabel(param_col)
    ax2.set_ylabel('Test RMSE (eV)')
    ax2.set_title(f'{param_col} vs Test RMSE')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"参数扫描图已保存到: {save_path}")

def create_summary_plot(df: pd.DataFrame, save_path: str) -> None:
    """
    创建综合摘要图
    
    Args:
        df: 包含实验结果的DataFrame
        save_path: 保存路径
    """
    # 过滤有效数据
    valid_df = df[(df['test_mae'] > 0) & (df['test_rmse'] > 0)].copy()
    
    if valid_df.empty:
        print("警告: 没有有效数据")
        return
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. MAE vs RMSE 散点图
    axes[0, 0].scatter(valid_df['test_mae'], valid_df['test_rmse'], alpha=0.6)
    axes[0, 0].set_xlabel('Test MAE (eV)')
    axes[0, 0].set_ylabel('Test RMSE (eV)')
    axes[0, 0].set_title('MAE vs RMSE')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 参数量 vs MAE
    if 'params' in valid_df.columns and valid_df['params'].max() > 0:
        axes[0, 1].scatter(valid_df['params'], valid_df['test_mae'], alpha=0.6)
        axes[0, 1].set_xlabel('Parameters')
        axes[0, 1].set_ylabel('Test MAE (eV)')
        axes[0, 1].set_title('Parameters vs MAE')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 种子 vs MAE
    if 'seed' in valid_df.columns:
        seed_stats = valid_df.groupby('seed')['test_mae'].agg(['mean', 'std']).reset_index()
        axes[1, 0].errorbar(seed_stats['seed'], seed_stats['mean'], 
                           yerr=seed_stats['std'], marker='o', capsize=5)
        axes[1, 0].set_xlabel('Seed')
        axes[1, 0].set_ylabel('Test MAE (eV)')
        axes[1, 0].set_title('Seed vs MAE')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 最佳配置柱状图
    best_configs = valid_df.nsmallest(5, 'test_mae')
    axes[1, 1].bar(range(len(best_configs)), best_configs['test_mae'], alpha=0.7)
    axes[1, 1].set_xlabel('Configuration Rank')
    axes[1, 1].set_ylabel('Test MAE (eV)')
    axes[1, 1].set_title('Top 5 Configurations')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"综合摘要图已保存到: {save_path}")

if __name__ == "__main__":
    # 测试代码
    print("绘图工具模块已加载")
