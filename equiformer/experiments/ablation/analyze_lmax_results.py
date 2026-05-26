#!/usr/bin/env python3
"""
分析Lmax消融实验结果并创建可视化
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def analyze_lmax_results():
    """分析Lmax消融实验结果"""
    # 读取结果
    df = pd.read_csv("experiments/ablation/lmax_results_collected.csv")
    
    # 过滤出2层模型的Lmax实验结果
    lmax_df = df[(df['num_layers'] == 2) & (df['grid_resolution'] >= 10)].copy()
    
    print("=== Lmax消融实验结果分析 ===")
    print(f"实验配置: 2层Transformer, 64球面通道, 4注意力头, 64边通道")
    print(f"训练轮数: 100 epochs")
    print(f"批次大小: 16")
    print()
    
    # 基本统计
    print("性能指标统计:")
    print(f"Test MAE范围: {lmax_df['test_mae'].min():.6f} - {lmax_df['test_mae'].max():.6f}")
    print(f"Test RMSE范围: {lmax_df['test_rmse'].min():.6f} - {lmax_df['test_rmse'].max():.6f}")
    print(f"Test Loss范围: {lmax_df['test_loss'].min():.6f} - {lmax_df['test_loss'].max():.6f}")
    print()
    
    # 详细结果
    print("详细结果:")
    result_cols = ['grid_resolution', 'test_mae', 'test_rmse', 'test_loss']
    print(lmax_df[result_cols].to_string(index=False, float_format='%.6f'))
    print()
    
    # 性能变化分析
    print("性能变化分析:")
    mae_diff = lmax_df['test_mae'].max() - lmax_df['test_mae'].min()
    rmse_diff = lmax_df['test_rmse'].max() - lmax_df['test_rmse'].min()
    loss_diff = lmax_df['test_loss'].max() - lmax_df['test_loss'].min()
    
    print(f"MAE变化范围: {mae_diff:.6f} eV")
    print(f"RMSE变化范围: {rmse_diff:.6f} eV") 
    print(f"Loss变化范围: {loss_diff:.6f}")
    print()
    
    # 最佳配置
    best_idx = lmax_df['test_mae'].idxmin()
    best_config = lmax_df.loc[best_idx]
    print(f"最佳配置 (最低MAE):")
    print(f"  Grid Resolution: {best_config['grid_resolution']}")
    print(f"  Test MAE: {best_config['test_mae']:.6f} eV")
    print(f"  Test RMSE: {best_config['test_rmse']:.6f} eV")
    print(f"  Test Loss: {best_config['test_loss']:.6f}")
    print()
    
    # 创建可视化
    create_visualizations(lmax_df)
    
    return lmax_df

def create_visualizations(df):
    """创建可视化图表"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('EquiformerV2 Lmax消融实验结果分析', fontsize=16, fontweight='bold')
    
    # 1. MAE vs Grid Resolution
    axes[0, 0].plot(df['grid_resolution'], df['test_mae'], 'o-', linewidth=2, markersize=8, color='#2E86AB')
    axes[0, 0].set_xlabel('Grid Resolution (Lmax)', fontsize=12)
    axes[0, 0].set_ylabel('Test MAE (eV)', fontsize=12)
    axes[0, 0].set_title('Test MAE vs Grid Resolution', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='both', which='major', labelsize=10)
    
    # 2. RMSE vs Grid Resolution
    axes[0, 1].plot(df['grid_resolution'], df['test_rmse'], 'o-', linewidth=2, markersize=8, color='#A23B72')
    axes[0, 1].set_xlabel('Grid Resolution (Lmax)', fontsize=12)
    axes[0, 1].set_ylabel('Test RMSE (eV)', fontsize=12)
    axes[0, 1].set_title('Test RMSE vs Grid Resolution', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=10)
    
    # 3. Loss vs Grid Resolution
    axes[1, 0].plot(df['grid_resolution'], df['test_loss'], 'o-', linewidth=2, markersize=8, color='#F18F01')
    axes[1, 0].set_xlabel('Grid Resolution (Lmax)', fontsize=12)
    axes[1, 0].set_ylabel('Test Loss', fontsize=12)
    axes[1, 0].set_title('Test Loss vs Grid Resolution', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='both', which='major', labelsize=10)
    
    # 4. 综合性能对比
    # 归一化到0-1范围
    mae_norm = (df['test_mae'] - df['test_mae'].min()) / (df['test_mae'].max() - df['test_mae'].min())
    rmse_norm = (df['test_rmse'] - df['test_rmse'].min()) / (df['test_rmse'].max() - df['test_rmse'].min())
    loss_norm = (df['test_loss'] - df['test_loss'].min()) / (df['test_loss'].max() - df['test_loss'].min())
    
    x = df['grid_resolution']
    axes[1, 1].plot(x, mae_norm, 'o-', label='MAE (归一化)', linewidth=2, markersize=6)
    axes[1, 1].plot(x, rmse_norm, 's-', label='RMSE (归一化)', linewidth=2, markersize=6)
    axes[1, 1].plot(x, loss_norm, '^-', label='Loss (归一化)', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Grid Resolution (Lmax)', fontsize=12)
    axes[1, 1].set_ylabel('归一化性能指标', fontsize=12)
    axes[1, 1].set_title('综合性能对比 (归一化)', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    plt.savefig('experiments/ablation/lmax_ablation_analysis.png', dpi=300, bbox_inches='tight')
    print("可视化图表已保存到: experiments/ablation/lmax_ablation_analysis.png")
    
    # 创建性能变化表格
    create_performance_table(df)

def create_performance_table(df):
    """创建性能变化表格"""
    print("\n=== 性能变化详细分析 ===")
    
    # 计算相对于最小值的改善
    mae_min = df['test_mae'].min()
    rmse_min = df['test_rmse'].min()
    loss_min = df['test_loss'].min()
    
    print("相对于最佳性能的改善:")
    print("Grid Resolution | MAE改善(%) | RMSE改善(%) | Loss改善(%)")
    print("-" * 60)
    
    for _, row in df.iterrows():
        mae_improve = (row['test_mae'] - mae_min) / mae_min * 100
        rmse_improve = (row['test_rmse'] - rmse_min) / rmse_min * 100
        loss_improve = (row['test_loss'] - loss_min) / loss_min * 100
        
        print(f"{row['grid_resolution']:>14} | {mae_improve:>10.4f} | {rmse_improve:>11.4f} | {loss_improve:>10.4f}")

if __name__ == "__main__":
    analyze_lmax_results()
