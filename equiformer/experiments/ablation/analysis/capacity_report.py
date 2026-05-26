#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
容量分析报告
分析不同模型容量对性能的影响
"""

import pandas as pd
import argparse
from pathlib import Path
import sys

# 添加analysis目录到路径
analysis_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(analysis_dir))

from plot_utils import draw_capacity_bars, draw_parameter_sweep

def analyze_capacity(results_csv: str, output_dir: str):
    """分析模型容量对性能的影响"""
    df = pd.read_csv(results_csv)
    
    # 过滤有效数据
    valid_df = df[(df['test_mae'] > 0) & (df['test_rmse'] > 0)]
    
    if valid_df.empty:
        print("没有有效的容量数据")
        return
    
    print(f"分析容量数据: {len(valid_df)} 个有效实验")
    
    # 容量相关参数
    capacity_params = ['num_layers', 'sphere_channels', 'num_heads', 
                      'grid_resolution', 'edge_channels', 'attn_hidden_channels']
    
    available_params = [col for col in capacity_params if col in valid_df.columns]
    print(f"可用容量参数: {available_params}")
    
    # 绘制容量对比图
    output_path = Path(output_dir) / "capacity_comparison.png"
    draw_capacity_bars(valid_df, str(output_path), available_params)
    
    # 绘制各参数的扫描图
    for param in available_params:
        param_path = Path(output_dir) / f"{param}_sweep.png"
        draw_parameter_sweep(valid_df, str(param_path), param)
    
    # 生成容量统计报告
    capacity_stats = []
    for param in available_params:
        stats = valid_df.groupby(param).agg({
            'test_mae': ['mean', 'std'],
            'test_rmse': ['mean', 'std'],
            'params': ['mean', 'std']
        })
        stats.columns = ['mae_mean', 'mae_std', 'rmse_mean', 'rmse_std', 'params_mean', 'params_std']
        stats['parameter'] = param
        capacity_stats.append(stats.reset_index())
    
    if capacity_stats:
        combined_stats = pd.concat(capacity_stats, ignore_index=True)
        print("\n容量统计报告:")
        print(combined_stats)
        
        # 保存统计报告
        stats_path = Path(output_dir) / "capacity_stats.csv"
        combined_stats.to_csv(stats_path, index=False)
        print(f"统计报告已保存到: {stats_path}")
    
    # 分析参数量与性能的关系
    if 'params' in valid_df.columns and valid_df['params'].max() > 0:
        print("\n参数量与性能关系分析:")
        correlation_mae = valid_df['params'].corr(valid_df['test_mae'])
        correlation_rmse = valid_df['params'].corr(valid_df['test_rmse'])
        print(f"参数量与MAE的相关性: {correlation_mae:.4f}")
        print(f"参数量与RMSE的相关性: {correlation_rmse:.4f}")

def main():
    parser = argparse.ArgumentParser(description='容量分析报告')
    parser.add_argument('--results_csv', type=str, required=True, help='实验结果CSV文件')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    
    args = parser.parse_args()
    
    analyze_capacity(args.results_csv, args.output_dir)

if __name__ == "__main__":
    main()
