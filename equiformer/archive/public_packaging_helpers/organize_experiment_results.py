#!/usr/bin/env python3
"""
整理实验结果脚本
梳理Eqv2、Eqv2微调、Eqv2+分支等实验，计算R²系数，并汇总到总JSON文件
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime

def calculate_r2_from_csv(csv_path, true_col='targets', pred_col='predictions'):
    """从CSV文件计算R²系数和其他指标"""
    try:
        df = pd.read_csv(csv_path)
        
        # 尝试不同的列名
        possible_true_cols = ['targets', 'true', 'y_true', 'true_value']
        possible_pred_cols = ['predictions', 'predicted', 'y_hat', 'pred', 'y_pred']
        
        true_col_found = None
        pred_col_found = None
        
        for col in possible_true_cols:
            if col in df.columns:
                true_col_found = col
                break
        
        for col in possible_pred_cols:
            if col in df.columns:
                pred_col_found = col
                break
        
        if true_col_found is None or pred_col_found is None:
            print(f"⚠️ 无法找到合适的列: {csv_path}")
            print(f"   可用列: {list(df.columns)}")
            return None
        
        y_true = df[true_col_found].values
        y_pred = df[pred_col_found].values
        
        # 计算指标
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        return {
            'r2': float(r2),
            'mae': float(mae),
            'rmse': float(rmse),
            'n_samples': len(y_true)
        }
    except Exception as e:
        print(f"❌ 读取CSV文件失败 {csv_path}: {e}")
        return None

def load_json_metrics(json_path):
    """加载JSON格式的指标文件"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ 读取JSON文件失败 {json_path}: {e}")
        return None

def organize_experiments():
    """整理所有实验结果"""
    project_root = Path(__file__).resolve().parent.parent
    experiments_dir = project_root / "experiments"
    output_dir = Path("D:/mylab/实验结果")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "整理时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "实验类型": {
            "Eqv2_Baseline": [],
            "Eqv2_微调": [],
            "Eqv2_分支融合": []
        }
    }
    
    print("🔍 开始整理实验结果...")
    print("=" * 60)
    
    # 1. Eqv2 Baseline 实验
    print("\n📊 1. Eqv2 Baseline 实验")
    print("-" * 60)
    
    # 1.1 Standalone EquiformerV2 (2025-07-09_run1)
    eqv2_standalone_dir = experiments_dir / "2025-07-09_run1"
    if eqv2_standalone_dir.exists():
        print(f"  ✓ 找到: Standalone EquiformerV2")
        exp_info = {
            "实验名称": "Standalone EquiformerV2",
            "实验目录": str(eqv2_standalone_dir.relative_to(project_root)),
            "实验日期": "2025-07-09",
            "模型类型": "EquiformerV2 Baseline"
        }
        
        # 读取训练历史
        history_path = eqv2_standalone_dir / "logs" / "standalone_equiformer_v2_training_history.json"
        if history_path.exists():
            history = load_json_metrics(history_path)
            if history and 'final_metrics' in history:
                exp_info.update(history['final_metrics'])
                exp_info['训练配置'] = history.get('config', {})
        
        # 从预测结果计算R²
        pred_path = eqv2_standalone_dir / "standalone_equiformer_v2_predictions.csv"
        if pred_path.exists():
            metrics = calculate_r2_from_csv(pred_path)
            if metrics:
                exp_info['预测结果指标'] = metrics
                if 'r2' not in exp_info:
                    exp_info['r2'] = metrics['r2']
        
        results["实验类型"]["Eqv2_Baseline"].append(exp_info)
        print(f"    R² = {exp_info.get('r2', 'N/A'):.4f}")
    
    # 1.2 Enhanced EquiformerV2 Baseline (2025-09-10_run1)
    eqv2_enhanced_dir = experiments_dir / "2025-09-10_run1" / "enhanced_equiformerv2_baseline"
    if eqv2_enhanced_dir.exists():
        print(f"  ✓ 找到: Enhanced EquiformerV2 Baseline")
        exp_info = {
            "实验名称": "Enhanced EquiformerV2 Baseline",
            "实验目录": str(eqv2_enhanced_dir.relative_to(project_root)),
            "实验日期": "2025-09-10",
            "模型类型": "EquiformerV2 Baseline (Enhanced)"
        }
        
        # 读取测试结果
        test_results_path = experiments_dir / "2025-09-10_run1" / "logs" / "enhanced_equiformer_v2_test_results.json"
        if test_results_path.exists():
            test_results = load_json_metrics(test_results_path)
            if test_results:
                exp_info.update({
                    'test_loss': test_results.get('test_loss'),
                    'test_mae': test_results.get('test_mae'),
                    'test_rmse': test_results.get('test_rmse')
                })
        
        # 从预测结果计算R²
        pred_path = eqv2_enhanced_dir / "enhanced_equiformer_v2_predictions.csv"
        if pred_path.exists():
            metrics = calculate_r2_from_csv(pred_path, true_col='true', pred_col='predicted')
            if metrics:
                exp_info['预测结果指标'] = metrics
                exp_info['r2'] = metrics['r2']
        
        results["实验类型"]["Eqv2_Baseline"].append(exp_info)
        print(f"    R² = {exp_info.get('r2', 'N/A'):.4f}")
    
    # 2. Eqv2 微调实验 (消融实验)
    print("\n📊 2. Eqv2 微调实验 (消融实验)")
    print("-" * 60)
    
    ablation_dir = experiments_dir / "2025-09-10_run1" / "ablation"
    if ablation_dir.exists():
        # 查找最佳配置的实验
        best_configs = [
            "num_layers=2__sphere_channels=64__num_heads=4__grid_resolution=16__edge_channels=64__REAL_FUSION_BASE"
        ]
        
        for config_name in best_configs:
            config_dir = ablation_dir / config_name
            if config_dir.exists():
                # 查找seed目录
                for seed_dir in config_dir.iterdir():
                    if seed_dir.is_dir() and seed_dir.name.startswith('seed='):
                        print(f"  ✓ 找到: {config_name} ({seed_dir.name})")
                        exp_info = {
                            "实验名称": f"Eqv2微调 - {config_name}",
                            "实验目录": str(seed_dir.relative_to(project_root)),
                            "实验日期": "2025-09-10",
                            "模型类型": "EquiformerV2 微调",
                            "配置": config_name,
                            "随机种子": seed_dir.name
                        }
                        
                        # 读取测试结果
                        test_results_path = seed_dir / "logs" / "enhanced_equiformer_v2_test_results.json"
                        if test_results_path.exists():
                            test_results = load_json_metrics(test_results_path)
                            if test_results:
                                exp_info.update({
                                    'test_loss': test_results.get('test_loss'),
                                    'test_mae': test_results.get('test_mae'),
                                    'test_rmse': test_results.get('test_rmse')
                                })
                        
                        # 从预测结果计算R²
                        pred_path = seed_dir / "enhanced_equiformer_v2_predictions.csv"
                        if pred_path.exists():
                            metrics = calculate_r2_from_csv(pred_path, true_col='true', pred_col='predicted')
                            if metrics:
                                exp_info['预测结果指标'] = metrics
                                exp_info['r2'] = metrics['r2']
                        
                        results["实验类型"]["Eqv2_微调"].append(exp_info)
                        print(f"    R² = {exp_info.get('r2', 'N/A'):.4f}")
    
    # 3. Eqv2 + 分支融合实验
    print("\n📊 3. Eqv2 + 分支融合实验")
    print("-" * 60)
    
    fusion_experiments = [
        {
            "name": "Eqv2 + 表格分支 (Concat融合)",
            "dir": "20251021_160614_tabfusion_run_real_equiformer_with_loss",
            "fusion_type": "concat"
        },
        {
            "name": "Eqv2 + 表格分支 (Gate融合)",
            "dir": "20251021_165820_tabfusion_run_gate_fusion_100epochs",
            "fusion_type": "gate"
        }
    ]
    
    for exp in fusion_experiments:
        fusion_dir = experiments_dir / exp["dir"]
        if fusion_dir.exists():
            print(f"  ✓ 找到: {exp['name']}")
            exp_info = {
                "实验名称": exp["name"],
                "实验目录": str(fusion_dir.relative_to(project_root)),
                "实验日期": exp["dir"][:8],
                "模型类型": "EquiformerV2 + 表格分支融合",
                "融合方式": exp["fusion_type"]
            }
            
            # 读取metrics.json
            metrics_path = fusion_dir / "metrics.json"
            if metrics_path.exists():
                metrics = load_json_metrics(metrics_path)
                if metrics:
                    exp_info.update({
                        'test_loss': metrics.get('test_loss'),
                        'test_mae': metrics.get('test_mae'),
                        'test_rmse': metrics.get('test_rmse'),
                        '融合配置': {
                            'align_mode': metrics.get('align_mode'),
                            'fusion': metrics.get('fusion'),
                            'mc_T': metrics.get('mc_T')
                        }
                    })
            
            # 从预测结果计算R²
            pred_path = fusion_dir / "predictions_test.csv"
            if pred_path.exists():
                metrics = calculate_r2_from_csv(pred_path, true_col='y_true', pred_col='y_hat')
                if metrics:
                    exp_info['预测结果指标'] = metrics
                    exp_info['r2'] = metrics['r2']
            
            results["实验类型"]["Eqv2_分支融合"].append(exp_info)
            print(f"    R² = {exp_info.get('r2', 'N/A'):.4f}")
    
    # 保存结果
    output_file = output_dir / "实验结果汇总.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print(f"✅ 实验结果整理完成！")
    print(f"📁 结果已保存到: {output_file}")
    
    # 生成摘要报告
    print("\n📊 实验结果摘要:")
    print("-" * 60)
    print(f"Eqv2 Baseline 实验数量: {len(results['实验类型']['Eqv2_Baseline'])}")
    for exp in results['实验类型']['Eqv2_Baseline']:
        print(f"  - {exp['实验名称']}: R² = {exp.get('r2', 'N/A'):.4f}")
    
    print(f"\nEqv2 微调实验数量: {len(results['实验类型']['Eqv2_微调'])}")
    for exp in results['实验类型']['Eqv2_微调']:
        print(f"  - {exp['实验名称']}: R² = {exp.get('r2', 'N/A'):.4f}")
    
    print(f"\nEqv2 + 分支融合实验数量: {len(results['实验类型']['Eqv2_分支融合'])}")
    for exp in results['实验类型']['Eqv2_分支融合']:
        print(f"  - {exp['实验名称']}: R² = {exp.get('r2', 'N/A'):.4f}")
    
    # 生成CSV摘要
    summary_data = []
    for exp_type, exps in results['实验类型'].items():
        for exp in exps:
            summary_data.append({
                '实验类型': exp_type,
                '实验名称': exp.get('实验名称', ''),
                'R²': exp.get('r2', ''),
                'MAE': exp.get('test_mae', exp.get('mae', '')),
                'RMSE': exp.get('test_rmse', exp.get('rmse', '')),
                '实验目录': exp.get('实验目录', '')
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = output_dir / "实验结果摘要.csv"
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    print(f"\n📊 摘要CSV已保存到: {summary_csv}")
    
    return results

if __name__ == "__main__":
    organize_experiments()

