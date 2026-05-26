#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lmax曲线实验一键运行脚本
"""

import subprocess
import sys
from pathlib import Path

def run_lmax_experiment():
    """运行Lmax曲线实验"""
    print("🚀 开始运行Lmax曲线实验")
    print("="*60)
    
    # 检查实验网格文件
    grid_csv = Path("experiments/ablation/grids/lmax.csv")
    if not grid_csv.exists():
        print("❌ 错误: lmax.csv文件不存在")
        return False
    
    print(f"✅ 找到实验网格文件: {grid_csv}")
    
    # 创建输出目录
    plots_dir = Path("experiments/ablation/plots")
    plots_dir.mkdir(exist_ok=True)
    
    # 1. 运行批量实验
    print("\n📊 运行批量实验...")
    cmd1 = [
        sys.executable,
        "experiments/ablation/run_exp.py",
        "--grid_csv", str(grid_csv),
        "--repeat_seeds", "3",
        "--output_csv", "experiments/ablation/results_lmax.csv",
        "--exp_tag", "LMAX"
    ]
    
    print(f"执行命令: {' '.join(cmd1)}")
    
    try:
        result1 = subprocess.run(cmd1, check=True, capture_output=True, text=True)
        print("批量实验完成!")
        print("输出:", result1.stdout[-500:])  # 显示最后500字符
    except subprocess.CalledProcessError as e:
        print(f"❌ 批量实验失败: {e}")
        print("错误输出:", e.stderr)
        return False
    
    # 2. 生成Lmax分析报告
    print("\n📈 生成Lmax分析报告...")
    cmd2 = [
        sys.executable,
        "experiments/ablation/analysis/lmax_report.py",
        "--input", "experiments/ablation/results_lmax.csv"
    ]
    
    print(f"执行命令: {' '.join(cmd2)}")
    
    try:
        result2 = subprocess.run(cmd2, check=True, capture_output=True, text=True)
        print("Lmax分析报告完成!")
        print("输出:", result2.stdout[-500:])  # 显示最后500字符
    except subprocess.CalledProcessError as e:
        print(f"❌ Lmax分析报告失败: {e}")
        print("错误输出:", e.stderr)
        return False
    
    print("\nLmax曲线实验完成!")
    print("="*60)
    print("生成的文件:")
    print(f"  - 实验结果: experiments/ablation/results_lmax.csv")
    print(f"  - Lmax曲线图: experiments/ablation/plots/lmax_curve.png")
    print(f"  - 统计报告: experiments/ablation/plots/lmax_stats.csv")
    
    return True

def main():
    """主函数"""
    print("EquiformerV2 Lmax曲线实验")
    print("="*60)
    
    if run_lmax_experiment():
        print("\n实验完成!")
        print("查看结果:")
        print("  - 实验结果: experiments/ablation/results_lmax.csv")
        print("  - Lmax曲线图: experiments/ablation/plots/lmax_curve.png")
    else:
        print("\n❌ 实验失败，请检查错误信息")

if __name__ == "__main__":
    main()
