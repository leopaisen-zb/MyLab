#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据量分析报告
分析不同data_ratio对模型性能的影响
"""

import pandas as pd
import argparse
from pathlib import Path
import sys

# 添加analysis目录到路径
analysis_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(analysis_dir))

from plot_utils import draw_parameter_sweep

def analyze_datasize(results_csv: str, output_dir: str):
    """分析data_ratio对性能的影响"""
    df = pd.read_csv(results_csv)
    
    # 过滤有效数据
    valid_df = df[(df['test_mae'] > 0) & (df['test_rmse'] > 0)]
    
    if valid_df.empty:
        print("没有有效的数据量数据")
        return
    
    print(f"分析数据量数据: {len(valid_df)} 个有效实验")
    
    # 检查data_ratio列
    if 'data_ratio' not in valid_df.columns:
        print("没有找到data_ratio列")
        return
    
    # 绘制数据量扫描图
    output_path = Path(output_dir) / "datasize_sweep.png"
    draw_parameter_sweep(valid_df, str(output_path), 'data_ratio')
    
    # 生成数据量统计报告
    datasize_stats = valid_df.groupby('data_ratio').agg({
        'test_mae': ['mean', 'std', 'min', 'max'],
        'test_rmse': ['mean', 'std', 'min', 'max'],
        'params': ['mean', 'std']
    }).round(6)
    
    print("\n数据量统计报告:")
    print(datasize_stats)
    
    # 保存统计报告
    stats_path = Path(output_dir) / "datasize_stats.csv"
    datasize_stats.to_csv(stats_path)
    print(f"统计报告已保存到: {stats_path}")
    
    # 分析数据量的影响
    print(f"\n数据量影响分析:")
    print(f"数据量范围: {valid_df['data_ratio'].min():.2f} - {valid_df['data_ratio'].max():.2f}")
    
    # 分析数据量与性能的关系
    correlation_mae = valid_df['data_ratio'].corr(valid_df['test_mae'])
    correlation_rmse = valid_df['data_ratio'].corr(valid_df['test_rmse'])
    print(f"数据量与MAE的相关性: {correlation_mae:.4f}")
    print(f"数据量与RMSE的相关性: {correlation_rmse:.4f}")
    
    if correlation_mae < 0:
        print("更多数据有助于提升性能")
    else:
        print("数据量对性能的影响不明显")

def main():
    parser = argparse.ArgumentParser(description='数据量分析报告')
    parser.add_argument('--results_csv', type=str, required=True, help='实验结果CSV文件')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    
    args = parser.parse_args()
    
    analyze_datasize(args.results_csv, args.output_dir)

if __name__ == "__main__":
    main()
