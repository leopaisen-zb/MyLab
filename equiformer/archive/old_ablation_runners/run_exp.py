#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一批跑入口脚本
读取实验网格CSV，批量运行训练实验，收集结果

使用示例:
    python experiments/ablation/run_exp.py \
        --grid_csv experiments/ablation/grids/demo.csv \
        --repeat_seeds 2 \
        --output_csv experiments/ablation/results_demo.csv \
        --exp_tag DEMO
"""

import os
import sys
import csv
import json
import argparse
import subprocess
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import traceback

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

def parse_grid_csv(grid_csv_path: str) -> List[Dict[str, Any]]:
    """解析实验网格CSV文件"""
    configs = []
    with open(grid_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 转换字符串值为适当类型
            config = {}
            for key, value in row.items():
                if value.lower() in ('true', 'false'):
                    config[key] = value.lower() == 'true'
                elif value.isdigit():
                    config[key] = int(value)
                elif value.replace('.', '').isdigit():
                    config[key] = float(value)
                else:
                    config[key] = value
            configs.append(config)
    return configs

def build_command(config: Dict[str, Any], seed: int, exp_tag: str, 
                 split_file: str = '', pretrained_ckpt: str = '') -> List[str]:
    """构建训练命令"""
    cmd = [
        sys.executable,  # python
        str(project_root / "src" / "train_enhanced_equiformer_v2.py")
    ]
    
    # 添加所有配置参数
    for key, value in config.items():
        if isinstance(value, bool):
            cmd.extend([f"--{key}", str(value).lower()])
        elif isinstance(value, (int, float)):
            cmd.extend([f"--{key}", str(value)])
        elif isinstance(value, str):
            cmd.extend([f"--{key}", value])
        elif isinstance(value, list):
            cmd.extend([f"--{key}"] + [str(v) for v in value])
    
    # 添加种子和实验名称
    cmd.extend(["--seed", str(seed)])
    
    # 构建实验名称 - 包含所有重要参数
    config_summary = "_".join([f"{k}={v}" for k, v in config.items() if k in 
                              ['num_layers', 'sphere_channels', 'num_heads', 'grid_resolution', 'edge_channels',
                               'eSCN', 'attn_renorm', 'sep_s2', 'sep_ln']])
    exp_name = f"{exp_tag}__{config_summary}__seed{seed}"
    cmd.extend(["--exp_name", exp_name])
    
    # 添加可选参数
    if split_file:
        cmd.extend(["--split_file", split_file])
    if pretrained_ckpt:
        cmd.extend(["--pretrained_ckpt", pretrained_ckpt])
    
    return cmd

def run_single_experiment(cmd: List[str], config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """运行单个实验"""
    print(f"\n{'='*80}")
    print(f"运行实验: seed={seed}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    try:
        # 运行训练脚本 - 实时显示输出
        result = subprocess.run(cmd, cwd=str(project_root))
        
        if result.returncode == 0:
            print("实验成功完成!")
            
            # 从JSON文件读取结果
            try:
                # 构建实验名称来找到结果文件 - 使用与训练脚本相同的格式
                kv_parts = [
                    f"num_layers={config['num_layers']}",
                    f"sphere_channels={config['sphere_channels']}",
                    f"num_heads={config['num_heads']}",
                    f"grid_resolution={config['grid_resolution']}",
                    f"edge_channels={config['edge_channels']}",
                ]
                if exp_tag:
                    kv_parts.append(exp_tag)
                ablation_name = "__".join(kv_parts)
                
                # 查找实验结果目录
                experiments_dir = project_root / "experiments" / f"{experiment_date}_run1" / "ablation"
                exp_dirs = list(experiments_dir.glob(f"{ablation_name}"))
                
                if exp_dirs:
                    exp_dir = exp_dirs[0] / f"seed={seed}"
                    logs_dir = exp_dir / "logs"
                    
                    # 读取测试结果
                    test_results_file = logs_dir / "enhanced_equiformer_v2_test_results.json"
                    if test_results_file.exists():
                        with open(test_results_file, 'r') as f:
                            test_results = json.load(f)
                        
                        # 读取训练历史获取参数量
                        history_file = logs_dir / "enhanced_equiformer_v2_training_history.json"
                        params = 0
                        if history_file.exists():
                            with open(history_file, 'r') as f:
                                history = json.load(f)
                                params = history.get('total_params', 0)
                        
                        return {
                            "test_mae": float(test_results.get("test_mae", 0.0)),
                            "test_rmse": float(test_results.get("test_rmse", 0.0)),
                            "test_loss": float(test_results.get("test_loss", 0.0)),
                            "params": int(params),
                            "latency_ms": -1.0,  # 占位值
                            "throughput": -1.0,  # 占位值
                            "seed": seed,
                            "error": ""
                        }
                    else:
                        print(f"警告: 未找到测试结果文件 {test_results_file}")
                        return {
                            "test_mae": -1.0,
                            "test_rmse": -1.0,
                            "test_loss": -1.0,
                            "params": -1,
                            "latency_ms": -1.0,
                            "throughput": -1.0,
                            "seed": seed,
                            "error": "未找到测试结果文件"
                        }
                else:
                    print(f"警告: 未找到实验目录 {exp_name}")
                    return {
                        "test_mae": -1.0,
                        "test_rmse": -1.0,
                        "test_loss": -1.0,
                        "params": -1,
                        "latency_ms": -1.0,
                        "throughput": -1.0,
                        "seed": seed,
                        "error": "未找到实验目录"
                    }
            except Exception as e:
                print(f"读取结果文件时出错: {e}")
                return {
                    "test_mae": -1.0,
                    "test_rmse": -1.0,
                    "test_loss": -1.0,
                    "params": -1,
                    "latency_ms": -1.0,
                    "throughput": -1.0,
                    "seed": seed,
                    "error": f"读取结果失败: {str(e)}"
                }
        else:
            print(f"实验失败! 返回码: {result.returncode}")
            error_msg = result.stderr if result.stderr else "未知错误"
            print("STDERR:", error_msg)
            return {
                "test_mae": -1.0,
                "test_rmse": -1.0,
                "test_loss": -1.0,
                "params": -1,
                "latency_ms": -1.0,
                "throughput": -1.0,
                "seed": seed,
                "error": f"训练失败: {error_msg[:200]}"
            }
    except Exception as e:
        print(f"实验异常: {e}")
        traceback.print_exc()
        return {
            "test_mae": -1.0,
            "test_rmse": -1.0,
            "test_loss": -1.0,
            "params": -1,
            "latency_ms": -1.0,
            "throughput": -1.0,
            "seed": seed,
            "error": f"异常: {str(e)[:200]}"
        }

def main():
    parser = argparse.ArgumentParser(description='批量运行消融实验')
    parser.add_argument('--grid_csv', type=str, required=True, help='实验网格CSV文件路径')
    parser.add_argument('--repeat_seeds', type=int, default=1, help='重复种子数量')
    parser.add_argument('--output_csv', type=str, required=True, help='输出结果CSV文件路径')
    parser.add_argument('--exp_tag', type=str, required=True, help='实验标签')
    parser.add_argument('--split_file', type=str, default='', help='数据分割文件')
    parser.add_argument('--pretrained_ckpt', type=str, default='', help='预训练检查点')
    
    args = parser.parse_args()
    
    # 解析实验网格
    print(f"读取实验网格: {args.grid_csv}")
    configs = parse_grid_csv(args.grid_csv)
    print(f"找到 {len(configs)} 个配置")
    
    # 准备输出文件
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已有输出文件
    file_exists = output_path.exists()
    
    # 收集所有结果
    all_results = []
    
    for config_idx, config in enumerate(configs):
        print(f"\n处理配置 {config_idx + 1}/{len(configs)}: {config}")
        
        for seed in range(args.repeat_seeds):
            # 构建命令
            cmd = build_command(config, seed, args.exp_tag, args.split_file, args.pretrained_ckpt)
            
            # 运行实验
            result = run_single_experiment(cmd, config, seed)
            
            # 合并配置和结果
            combined_result = {**config, **result}
            all_results.append(combined_result)
            
            # 立即写入CSV（增量保存）
            df = pd.DataFrame(all_results)
            df.to_csv(output_path, index=False)
            print(f"结果已保存到: {output_path}")
    
    print(f"\n{'='*80}")
    print(f"批量实验完成!")
    print(f"总共运行了 {len(all_results)} 个实验")
    print(f"结果保存在: {output_path}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
