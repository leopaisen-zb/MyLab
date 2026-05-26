#!/usr/bin/env python3
"""
EquiformerV2模块开关消融实验分析
分析attn_renorm、sep_s2、sep_ln模块对性能的影响
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def load_results(input_file):
    """加载实验结果"""
    df = pd.read_csv(input_file)
    print(f"加载实验结果: {len(df)} 行")
    print(f"列名: {list(df.columns)}")
    return df

def aggregate_by_modules(df):
    """按模块配置聚合结果"""
    # 按模块开关分组
    module_cols = ['attn_renorm', 'sep_s2', 'sep_ln']
    grouped = df.groupby(module_cols).agg({
        'test_mae': ['mean', 'std', 'count'],
        'test_rmse': ['mean', 'std'],
        'test_loss': ['mean', 'std']
    }).round(6)
    
    # 展平列名
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns]
    grouped = grouped.reset_index()
    
    return grouped

def create_module_ablation_plot(df, output_dir):
    """创建模块消融可视化"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('EquiformerV2 Module Ablation Study', fontsize=16, fontweight='bold')
    
    # 创建模块标签
    df['module_label'] = df.apply(lambda x: f"AR={x['attn_renorm']}, S2={x['sep_s2']}, LN={x['sep_ln']}", axis=1)
    
    # 1. MAE对比
    axes[0, 0].bar(range(len(df)), df['test_mae_mean'], 
                   yerr=df['test_mae_std'], capsize=5, alpha=0.7, color='#2E86AB')
    axes[0, 0].set_xlabel('Module Configuration', fontsize=12)
    axes[0, 0].set_ylabel('Test MAE (eV)', fontsize=12)
    axes[0, 0].set_title('Test MAE by Module Configuration', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(range(len(df)))
    axes[0, 0].set_xticklabels(df['module_label'], rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. RMSE对比
    axes[0, 1].bar(range(len(df)), df['test_rmse_mean'], 
                   yerr=df['test_rmse_std'], capsize=5, alpha=0.7, color='#A23B72')
    axes[0, 1].set_xlabel('Module Configuration', fontsize=12)
    axes[0, 1].set_ylabel('Test RMSE (eV)', fontsize=12)
    axes[0, 1].set_title('Test RMSE by Module Configuration', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(range(len(df)))
    axes[0, 1].set_xticklabels(df['module_label'], rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Loss对比
    axes[1, 0].bar(range(len(df)), df['test_loss_mean'], 
                   yerr=df['test_loss_std'], capsize=5, alpha=0.7, color='#F18F01')
    axes[1, 0].set_xlabel('Module Configuration', fontsize=12)
    axes[1, 0].set_ylabel('Test Loss', fontsize=12)
    axes[1, 0].set_title('Test Loss by Module Configuration', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(range(len(df)))
    axes[1, 0].set_xticklabels(df['module_label'], rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 热力图 - 模块影响
    module_matrix = df[['attn_renorm', 'sep_s2', 'sep_ln', 'test_mae_mean']].copy()
    module_matrix = module_matrix.set_index(['attn_renorm', 'sep_s2', 'sep_ln'])
    
    # 创建热力图数据
    heatmap_data = np.zeros((2, 2))
    heatmap_labels = []
    
    # 简化热力图：只显示sep_s2和sep_ln的组合
    for i, s2 in enumerate([0, 1]):
        for j, ln in enumerate([0, 1]):
            subset = df[(df['sep_s2'] == s2) & (df['sep_ln'] == ln)]
            if len(subset) > 0:
                heatmap_data[i, j] = subset['test_mae_mean'].mean()
            heatmap_labels.append(f'S2={s2}, LN={ln}')
    
    im = axes[1, 1].imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_xticklabels(['LN=0', 'LN=1'])
    axes[1, 1].set_yticklabels(['S2=0', 'S2=1'])
    axes[1, 1].set_title('Module Impact Heatmap (MAE)', fontsize=14, fontweight='bold')
    
    # 添加数值标注
    for i in range(2):
        for j in range(2):
            text = axes[1, 1].text(j, i, f'{heatmap_data[i, j]:.4f}',
                                 ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'module_ablation.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"模块消融图表已保存到: {output_file}")
    
    return output_file

def print_summary(df):
    """打印结果摘要"""
    print("\n=== 模块消融实验结果摘要 ===")
    print("配置说明: AR=attn_renorm, S2=sep_s2, LN=sep_ln")
    print()
    
    # 按MAE排序
    df_sorted = df.sort_values('test_mae_mean')
    
    print("性能排名 (按Test MAE):")
    print("-" * 80)
    print(f"{'排名':<4} {'AR':<2} {'S2':<2} {'LN':<2} {'MAE (eV)':<12} {'RMSE (eV)':<12} {'Loss':<12}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"{i:<4} {row['attn_renorm']:<2} {row['sep_s2']:<2} {row['sep_ln']:<2} "
              f"{row['test_mae_mean']:<12.6f} {row['test_rmse_mean']:<12.6f} {row['test_loss_mean']:<12.6f}")
    
    print("-" * 80)
    
    # 最佳配置
    best_config = df_sorted.iloc[0]
    print(f"\n最佳配置:")
    print(f"  attn_renorm={best_config['attn_renorm']}")
    print(f"  sep_s2={best_config['sep_s2']}")
    print(f"  sep_ln={best_config['sep_ln']}")
    print(f"  Test MAE: {best_config['test_mae_mean']:.6f} ± {best_config['test_mae_std']:.6f} eV")
    print(f"  Test RMSE: {best_config['test_rmse_mean']:.6f} ± {best_config['test_rmse_std']:.6f} eV")
    print(f"  Test Loss: {best_config['test_loss_mean']:.6f} ± {best_config['test_loss_std']:.6f}")
    
    # 模块影响分析
    print(f"\n模块影响分析:")
    
    # attn_renorm影响
    ar_on = df[df['attn_renorm'] == 1]['test_mae_mean'].mean()
    ar_off = df[df['attn_renorm'] == 0]['test_mae_mean'].mean()
    print(f"  attn_renorm: ON={ar_on:.6f} vs OFF={ar_off:.6f} (差异: {ar_on-ar_off:+.6f})")
    
    # sep_s2影响
    s2_on = df[df['sep_s2'] == 1]['test_mae_mean'].mean()
    s2_off = df[df['sep_s2'] == 0]['test_mae_mean'].mean()
    print(f"  sep_s2: ON={s2_on:.6f} vs OFF={s2_off:.6f} (差异: {s2_on-s2_off:+.6f})")
    
    # sep_ln影响
    ln_on = df[df['sep_ln'] == 1]['test_mae_mean'].mean()
    ln_off = df[df['sep_ln'] == 0]['test_mae_mean'].mean()
    print(f"  sep_ln: ON={ln_on:.6f} vs OFF={ln_off:.6f} (差异: {ln_on-ln_off:+.6f})")

def main():
    parser = argparse.ArgumentParser(description='EquiformerV2模块消融实验分析')
    parser.add_argument('--input', type=str, default='experiments/ablation/results_modules.csv',
                       help='输入结果文件路径')
    parser.add_argument('--output_dir', type=str, default='experiments/ablation/plots',
                       help='输出图片目录')
    
    args = parser.parse_args()
    
    # 加载结果
    df = load_results(args.input)
    
    # 聚合结果
    aggregated = aggregate_by_modules(df)
    
    # 打印摘要
    print_summary(aggregated)
    
    # 创建可视化
    plot_file = create_module_ablation_plot(aggregated, args.output_dir)
    
    print(f"\n分析完成！")
    print(f"结果文件: {args.input}")
    print(f"可视化图表: {plot_file}")

if __name__ == "__main__":
    main()