#!/usr/bin/env python3
"""
收集模块消融实验的结果
修复目录查找问题
"""
import json
import pandas as pd
from pathlib import Path

def collect_module_ablation_results():
    """收集模块消融实验结果"""
    print("🔍 收集模块消融实验结果")
    print("=" * 60)
    
    ablation_dir = Path("experiments/2025-09-10_run1/ablation")
    
    # 模块消融实验配置
    module_configs = [
        {
            'name': 'All_Modules_ON',
            'eSCN': 1, 'attn_renorm': 1, 'sep_s2': 1, 'sep_ln': 1,
            'exp_tag': 'MODULE_VIS_All_Modules_ON'
        },
        {
            'name': 'No_Attn_Renorm',
            'eSCN': 1, 'attn_renorm': 0, 'sep_s2': 1, 'sep_ln': 1,
            'exp_tag': 'MODULE_VIS_No_Attn_Renorm'
        },
        {
            'name': 'No_Sep_S2',
            'eSCN': 1, 'attn_renorm': 1, 'sep_s2': 0, 'sep_ln': 1,
            'exp_tag': 'MODULE_VIS_No_Sep_S2'
        },
        {
            'name': 'No_Sep_LN',
            'eSCN': 1, 'attn_renorm': 1, 'sep_s2': 1, 'sep_ln': 0,
            'exp_tag': 'MODULE_VIS_No_Sep_LN'
        },
        {
            'name': 'Only_eSCN',
            'eSCN': 1, 'attn_renorm': 0, 'sep_s2': 0, 'sep_ln': 0,
            'exp_tag': 'MODULE_VIS_Only_eSCN'
        }
    ]
    
    results = []
    
    for config in module_configs:
        print(f"\n📁 处理配置: {config['name']}")
        
        # 构建实验目录名
        exp_dir_name = f"num_layers=2__sphere_channels=64__num_heads=4__grid_resolution=16__edge_channels=64__{config['exp_tag']}"
        exp_dir = ablation_dir / exp_dir_name
        
        print(f"   查找目录: {exp_dir}")
        
        if not exp_dir.exists():
            print(f"   ❌ 目录不存在")
            continue
        
        print(f"   ✅ 目录存在")
        
        # 查找种子目录
        seed_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith('seed=')]
        print(f"   找到 {len(seed_dirs)} 个种子目录")
        
        for seed_dir in seed_dirs:
            seed = seed_dir.name.replace('seed=', '')
            logs_dir = seed_dir / 'logs'
            test_results_file = logs_dir / 'enhanced_equiformer_v2_test_results.json'
            
            print(f"   🌱 种子 {seed}:")
            
            if test_results_file.exists():
                try:
                    with open(test_results_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    result = {
                        'num_layers': 2,
                        'sphere_channels': 64,
                        'num_heads': 4,
                        'grid_resolution': 16,
                        'edge_channels': 64,
                        'eSCN': config['eSCN'],
                        'attn_renorm': config['attn_renorm'],
                        'sep_s2': config['sep_s2'],
                        'sep_ln': config['sep_ln'],
                        'test_mae': data.get('test_mae', -1.0),
                        'test_rmse': data.get('test_rmse', -1.0),
                        'test_loss': data.get('test_loss', -1.0),
                        'params': -1,  # 暂时设为-1
                        'latency_ms': -1.0,
                        'throughput': -1.0,
                        'seed': int(seed),
                        'error': ''
                    }
                    
                    results.append(result)
                    print(f"     ✅ MAE: {result['test_mae']:.6f}, RMSE: {result['test_rmse']:.6f}")
                    
                except Exception as e:
                    print(f"     ❌ 读取失败: {e}")
            else:
                print(f"     ❌ 测试结果文件不存在")
    
    # 保存结果
    if results:
        results_df = pd.DataFrame(results)
        output_file = Path("experiments/ablation/results_modules_corrected.csv")
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 结果已保存到: {output_file}")
        print(f"📊 总共收集到 {len(results)} 个结果")
        
        # 显示结果摘要
        print(f"\n📋 结果摘要:")
        for config in module_configs:
            config_results = results_df[
                (results_df['eSCN'] == config['eSCN']) &
                (results_df['attn_renorm'] == config['attn_renorm']) &
                (results_df['sep_s2'] == config['sep_s2']) &
                (results_df['sep_ln'] == config['sep_ln'])
            ]
            
            if not config_results.empty:
                mae_mean = config_results['test_mae'].mean()
                rmse_mean = config_results['test_rmse'].mean()
                print(f"   {config['name']:>15}: MAE={mae_mean:.6f}, RMSE={rmse_mean:.6f} ({len(config_results)} 个种子)")
            else:
                print(f"   {config['name']:>15}: 无结果")
        
        return results_df
    else:
        print("❌ 没有收集到任何结果")
        return None

def main():
    """主函数"""
    results_df = collect_module_ablation_results()
    
    if results_df is not None:
        print(f"\n🎉 模块消融实验结果收集完成!")
        print(f"📁 结果文件: experiments/ablation/results_modules_corrected.csv")
    else:
        print(f"\n❌ 结果收集失败")

if __name__ == "__main__":
    main()
