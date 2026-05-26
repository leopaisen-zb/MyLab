#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预训练分析报告
分析预训练对模型性能的影响
"""

import pandas as pd
import argparse
from pathlib import Path
import sys

# 添加analysis目录到路径
analysis_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(analysis_dir))

from plot_utils import draw_module_ablation

def analyze_pretrain(results_csv: str, output_dir: str):
    """分析预训练对性能的影响"""
    df = pd.read_csv(results_csv)
    
    # 过滤有效数据
    valid_df = df[(df['test_mae'] > 0) & (df['test_rmse'] > 0)]
    
    if valid_df.empty:
        print("没有有效的预训练数据")
        return
    
    print(f"分析预训练数据: {len(valid_df)} 个有效实验")
    
    # 检查pretrained列
    if 'pretrained' not in valid_df.columns:
        print("没有找到pretrained列")
        return
    
    # 绘制预训练消融图
    output_path = Path(output_dir) / "pretrain_ablation.png"
    draw_module_ablation(valid_df, str(output_path), ['pretrained'])
    
    # 生成预训练统计报告
    pretrain_stats = valid_df.groupby('pretrained').agg({
        'test_mae': ['mean', 'std', 'min', 'max'],
        'test_rmse': ['mean', 'std', 'min', 'max'],
        'params': ['mean', 'std']
    }).round(6)
    
    print("\n预训练统计报告:")
    print(pretrain_stats)
    
    # 保存统计报告
    stats_path = Path(output_dir) / "pretrain_stats.csv"
    pretrain_stats.to_csv(stats_path)
    print(f"统计报告已保存到: {stats_path}")
    
    # 分析预训练的效果
    pretrain_on = valid_df[valid_df['pretrained'] == True]['test_mae'].mean()
    pretrain_off = valid_df[valid_df['pretrained'] == False]['test_mae'].mean()
    
    print(f"\n预训练效果分析:")
    print(f"预训练开启时平均MAE: {pretrain_on:.6f} eV")
    print(f"预训练关闭时平均MAE: {pretrain_off:.6f} eV")
    print(f"预训练带来的MAE变化: {pretrain_on - pretrain_off:.6f} eV")
    
    if pretrain_on < pretrain_off:
        print("预训练有助于提升性能")
    else:
        print("预训练可能对性能有负面影响")
    
    # 分析预训练检查点的影响
    if 'pretrained_ckpt' in valid_df.columns:
        ckpt_stats = valid_df.groupby('pretrained_ckpt').agg({
            'test_mae': ['mean', 'std'],
            'test_rmse': ['mean', 'std']
        })
        print(f"\n预训练检查点统计:")
        print(ckpt_stats)

def main():
    parser = argparse.ArgumentParser(description='预训练分析报告')
    parser.add_argument('--results_csv', type=str, required=True, help='实验结果CSV文件')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    
    args = parser.parse_args()
    
    analyze_pretrain(args.results_csv, args.output_dir)

if __name__ == "__main__":
    main()
