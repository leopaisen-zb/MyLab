#!/usr/bin/env python3
"""
收集Lmax消融实验结果
"""
import os
import json
import pandas as pd
from pathlib import Path

def collect_lmax_results():
    """收集所有Lmax消融实验的结果"""
    results = []
    ablation_dir = Path("experiments/2025-09-10_run1/ablation")
    
    # 查找所有grid_resolution相关的实验
    for exp_dir in ablation_dir.iterdir():
        if exp_dir.is_dir() and "grid_resolution" in exp_dir.name:
            # 解析配置
            config_parts = exp_dir.name.split("__")
            config = {}
            for part in config_parts:
                if "=" in part:
                    key, value = part.split("=", 1)
                    try:
                        config[key] = float(value) if "." in value else int(value)
                    except ValueError:
                        config[key] = value
            
            # 查找seed=0的结果
            seed_dir = exp_dir / "seed=0"
            if seed_dir.exists():
                results_file = seed_dir / "logs" / "enhanced_equiformer_v2_test_results.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        test_results = json.load(f)
                    
                    # 合并配置和结果
                    result = {**config, **test_results}
                    results.append(result)
    
    # 转换为DataFrame并排序
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('grid_resolution')
        
        # 保存结果
        output_file = "experiments/ablation/lmax_results_collected.csv"
        df.to_csv(output_file, index=False)
        
        print("Lmax消融实验结果收集完成!")
        print(f"结果保存到: {output_file}")
        print("\n结果摘要:")
        print(df[['grid_resolution', 'test_mae', 'test_rmse', 'test_loss']].to_string(index=False))
        
        return df
    else:
        print("未找到任何Lmax消融实验结果")
        return None

if __name__ == "__main__":
    collect_lmax_results()
