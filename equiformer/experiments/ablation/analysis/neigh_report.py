#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
邻居数分析报告
分析不同max_neighbors对模型性能的影响
"""

import pandas as pd
import argparse
from pathlib import Path
import sys

# 添加analysis目录到路径
analysis_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(analysis_dir))

from plot_utils import draw_parameter_sweep

def analyze_neighbors(results_csv: str, output_dir: str):
    """分析max_neighbors对性能的影响"""
    df = pd.read_csv(results_csv)
    
    # 过滤有效数据
    valid_df = df[(df['test_mae'] > 0) & (df['test_rmse'] > 0)]
    
    if valid_df.empty:
        print("没有有效的邻居数数据")
        return
    
    print(f"分析邻居数数据: {len(valid_df)} 个有效实验")
    
    # 检查max_neighbors列
    if 'max_neighbors' not in valid_df.columns:
        print("没有找到max_neighbors列")
        return
    
    # 绘制邻居数扫描图
    output_path = Path(output_dir) / "neighbors_sweep.png"
    draw_parameter_sweep(valid_df, str(output_path), 'max_neighbors')
    
    # 生成邻居数统计报告
    neighbors_stats = valid_df.groupby('max_neighbors').agg({
        'test_mae': ['mean', 'std', 'min', 'max'],
        'test_rmse': ['mean', 'std', 'min', 'max'],
        'params': ['mean', 'std']
    }).round(6)
    
    print("\n邻居数统计报告:")
    print(neighbors_stats)
    
    # 保存统计报告
    stats_path = Path(output_dir) / "neighbors_stats.csv"
    neighbors_stats.to_csv(stats_path)
    print(f"统计报告已保存到: {stats_path}")
    
    # 分析最优邻居数
    best_neighbors = valid_df.loc[valid_df['test_mae'].idxmin(), 'max_neighbors']
    best_mae = valid_df['test_mae'].min()
    
    print(f"\n最优邻居数分析:")
    print(f"最佳邻居数: {best_neighbors}")
    print(f"最佳MAE: {best_mae:.6f} eV")
    
    # 分析邻居数与性能的关系
    correlation_mae = valid_df['max_neighbors'].corr(valid_df['test_mae'])
    correlation_rmse = valid_df['max_neighbors'].corr(valid_df['test_rmse'])
    print(f"邻居数与MAE的相关性: {correlation_mae:.4f}")
    print(f"邻居数与RMSE的相关性: {correlation_rmse:.4f}")

def main():
    parser = argparse.ArgumentParser(description='邻居数分析报告')
    parser.add_argument('--results_csv', type=str, required=True, help='实验结果CSV文件')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    
    args = parser.parse_args()
    
    analyze_neighbors(args.results_csv, args.output_dir)

if __name__ == "__main__":
    main()
