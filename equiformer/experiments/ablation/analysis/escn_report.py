#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eSCN分析报告
分析eSCN模块对模型性能的影响
"""

import pandas as pd
import argparse
from pathlib import Path
import sys

# 添加analysis目录到路径
analysis_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(analysis_dir))

from plot_utils import draw_module_ablation

def analyze_escn(results_csv: str, output_dir: str):
    """分析eSCN对性能的影响"""
    df = pd.read_csv(results_csv)
    
    # 过滤有效数据
    valid_df = df[(df['test_mae'] > 0) & (df['test_rmse'] > 0)]
    
    if valid_df.empty:
        print("没有有效的eSCN数据")
        return
    
    print(f"分析eSCN数据: {len(valid_df)} 个有效实验")
    
    # 检查eSCN列
    if 'eSCN' not in valid_df.columns:
        print("没有找到eSCN列")
        return
    
    # 绘制eSCN消融图
    output_path = Path(output_dir) / "escn_ablation.png"
    draw_module_ablation(valid_df, str(output_path), ['eSCN'])
    
    # 生成eSCN统计报告
    escn_stats = valid_df.groupby('eSCN').agg({
        'test_mae': ['mean', 'std', 'min', 'max'],
        'test_rmse': ['mean', 'std', 'min', 'max'],
        'params': ['mean', 'std']
    }).round(6)
    
    print("\neSCN统计报告:")
    print(escn_stats)
    
    # 保存统计报告
    stats_path = Path(output_dir) / "escn_stats.csv"
    escn_stats.to_csv(stats_path)
    print(f"统计报告已保存到: {stats_path}")
    
    # 分析eSCN的效果
    escn_on = valid_df[valid_df['eSCN'] == True]['test_mae'].mean()
    escn_off = valid_df[valid_df['eSCN'] == False]['test_mae'].mean()
    
    print(f"\neSCN效果分析:")
    print(f"eSCN开启时平均MAE: {escn_on:.6f} eV")
    print(f"eSCN关闭时平均MAE: {escn_off:.6f} eV")
    print(f"eSCN带来的MAE变化: {escn_on - escn_off:.6f} eV")
    
    if escn_on < escn_off:
        print("eSCN有助于提升性能")
    else:
        print("eSCN可能对性能有负面影响")

def main():
    parser = argparse.ArgumentParser(description='eSCN分析报告')
    parser.add_argument('--results_csv', type=str, required=True, help='实验结果CSV文件')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    
    args = parser.parse_args()
    
    analyze_escn(args.results_csv, args.output_dir)

if __name__ == "__main__":
    main()
