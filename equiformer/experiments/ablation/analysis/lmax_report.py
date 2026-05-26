#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lmax分析报告
分析不同Lmax值对模型性能的影响
"""

import pandas as pd
import argparse
from pathlib import Path
import sys

# 添加analysis目录到路径
analysis_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(analysis_dir))

from plot_utils import draw_lmax_curve, draw_parameter_sweep

def analyze_lmax(results_csv: str, output_dir: str):
    """分析Lmax对性能的影响"""
    df = pd.read_csv(results_csv)
    
    # 过滤有效数据
    valid_df = df[(df['test_mae'] > 0) & (df['test_rmse'] > 0)]
    
    if valid_df.empty:
        print("没有有效的Lmax数据")
        return
    
    print(f"分析Lmax数据: {len(valid_df)} 个有效实验")
    
    # 使用grid_resolution作为Lmax值
    lmax_col = 'grid_resolution'
    
    # 按grid_resolution分组计算均值±std
    lmax_stats = valid_df.groupby(lmax_col).agg({
        'test_mae': ['mean', 'std'],
        'test_rmse': ['mean', 'std'],
        'params': ['mean', 'std']
    }).round(6)
    
    # 展平列名
    lmax_stats.columns = ['mae_mean', 'mae_std', 'rmse_mean', 'rmse_std', 'params_mean', 'params_std']
    lmax_stats = lmax_stats.reset_index()
    
    print("\nLmax统计报告 (按grid_resolution聚合):")
    print(lmax_stats)
    
    # 保存统计报告
    stats_path = Path(output_dir) / "lmax_stats.csv"
    lmax_stats.to_csv(stats_path, index=False)
    print(f"统计报告已保存到: {stats_path}")
    
    # 绘制Lmax曲线
    output_path = Path(output_dir) / "lmax_curve.png"
    draw_lmax_curve(valid_df, str(output_path), lmax_col)
    
    # 分析最优Lmax
    best_lmax = lmax_stats.loc[lmax_stats['mae_mean'].idxmin(), lmax_col]
    best_mae = lmax_stats['mae_mean'].min()
    
    print(f"\n最优Lmax分析:")
    print(f"最佳grid_resolution: {best_lmax}")
    print(f"最佳MAE: {best_mae:.6f} eV")
    
    # 分析Lmax与性能的关系
    correlation_mae = valid_df[lmax_col].corr(valid_df['test_mae'])
    correlation_rmse = valid_df[lmax_col].corr(valid_df['test_rmse'])
    print(f"grid_resolution与MAE的相关性: {correlation_mae:.4f}")
    print(f"grid_resolution与RMSE的相关性: {correlation_rmse:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Lmax分析报告')
    parser.add_argument('--input', type=str, default='experiments/ablation/results_lmax.csv', help='实验结果CSV文件')
    parser.add_argument('--output_dir', type=str, default='experiments/ablation/plots', help='输出目录')
    
    args = parser.parse_args()
    
    analyze_lmax(args.input, args.output_dir)

if __name__ == "__main__":
    main()
