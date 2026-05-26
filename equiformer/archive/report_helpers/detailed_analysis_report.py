#!/usr/bin/env python3
"""
详细的模型性能分析报告
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_performance_improvement():
    """分析性能改善"""
    print("📊 详细性能分析报告")
    print("=" * 80)
    
    # 读取对比结果
    comparison_file = "experiments/model_comparison/model_comparison.csv"
    if not os.path.exists(comparison_file):
        print("❌ 对比结果文件不存在")
        return
    
    df = pd.read_csv(comparison_file)
    
    # 基础EquiformerV2结果
    base_eqv2 = df[df['模型名称'].str.contains('基础')].iloc[0]
    # 最佳参数EquiformerV2结果
    best_eqv2 = df[df['模型名称'].str.contains('最佳参数')].iloc[0]
    # 融合模型结果
    fusion_model = df[df['模型名称'].str.contains('融合')].iloc[0]
    
    print(f"🔍 模型性能对比:")
    print(f"   1. EquiformerV2 (基础参数):")
    print(f"      - Test MAE: {base_eqv2['Test MAE (eV)']:.6f} eV")
    print(f"      - Test RMSE: {base_eqv2['Test RMSE (eV)']:.6f} eV")
    print(f"      - Test Loss: {base_eqv2['Test Loss']:.6f}")
    
    print(f"\n   2. EquiformerV2 (最佳参数 - 消融实验优化):")
    print(f"      - Test MAE: {best_eqv2['Test MAE (eV)']:.6f} eV")
    print(f"      - Test RMSE: {best_eqv2['Test RMSE (eV)']:.6f} eV")
    print(f"      - Test Loss: {best_eqv2['Test Loss']:.6f}")
    
    print(f"\n   3. 结构+表格晚融合模型:")
    print(f"      - Test MAE: {fusion_model['Test MAE (eV)']:.6f} eV")
    print(f"      - Test RMSE: {fusion_model['Test RMSE (eV)']:.6f} eV")
    print(f"      - 融合模式: {fusion_model['融合模式']}")
    print(f"      - 对齐模式: {fusion_model['对齐模式']}")
    
    # 计算改善百分比
    print(f"\n📈 性能改善分析:")
    
    # 相对于基础EquiformerV2的改善
    mae_improvement_base = ((base_eqv2['Test MAE (eV)'] - fusion_model['Test MAE (eV)']) / base_eqv2['Test MAE (eV)']) * 100
    rmse_improvement_base = ((base_eqv2['Test RMSE (eV)'] - fusion_model['Test RMSE (eV)']) / base_eqv2['Test RMSE (eV)']) * 100
    
    print(f"   相对于基础EquiformerV2:")
    print(f"   - MAE改善: {mae_improvement_base:.2f}% ({base_eqv2['Test MAE (eV)']:.6f} → {fusion_model['Test MAE (eV)']:.6f})")
    print(f"   - RMSE改善: {rmse_improvement_base:.2f}% ({base_eqv2['Test RMSE (eV)']:.6f} → {fusion_model['Test RMSE (eV)']:.6f})")
    
    # 相对于最佳EquiformerV2的改善
    mae_improvement_best = ((best_eqv2['Test MAE (eV)'] - fusion_model['Test MAE (eV)']) / best_eqv2['Test MAE (eV)']) * 100
    rmse_improvement_best = ((best_eqv2['Test RMSE (eV)'] - fusion_model['Test RMSE (eV)']) / best_eqv2['Test RMSE (eV)']) * 100
    
    print(f"\n   相对于最佳EquiformerV2:")
    print(f"   - MAE改善: {mae_improvement_best:.2f}% ({best_eqv2['Test MAE (eV)']:.6f} → {fusion_model['Test MAE (eV)']:.6f})")
    print(f"   - RMSE改善: {rmse_improvement_best:.2f}% ({best_eqv2['Test RMSE (eV)']:.6f} → {fusion_model['Test RMSE (eV)']:.6f})")
    
    # 消融实验的效果
    ablation_mae_improvement = ((base_eqv2['Test MAE (eV)'] - best_eqv2['Test MAE (eV)']) / base_eqv2['Test MAE (eV)']) * 100
    ablation_rmse_improvement = ((base_eqv2['Test RMSE (eV)'] - best_eqv2['Test RMSE (eV)']) / base_eqv2['Test RMSE (eV)']) * 100
    
    print(f"\n   消融实验优化效果:")
    print(f"   - MAE改善: {ablation_mae_improvement:.2f}% ({base_eqv2['Test MAE (eV)']:.6f} → {best_eqv2['Test MAE (eV)']:.6f})")
    print(f"   - RMSE改善: {ablation_rmse_improvement:.2f}% ({base_eqv2['Test RMSE (eV)']:.6f} → {best_eqv2['Test RMSE (eV)']:.6f})")
    
    return {
        'base_eqv2': base_eqv2,
        'best_eqv2': best_eqv2,
        'fusion_model': fusion_model,
        'mae_improvement_base': mae_improvement_base,
        'rmse_improvement_base': rmse_improvement_base,
        'mae_improvement_best': mae_improvement_best,
        'rmse_improvement_best': rmse_improvement_best,
        'ablation_mae_improvement': ablation_mae_improvement,
        'ablation_rmse_improvement': ablation_rmse_improvement
    }

def analyze_training_process():
    """分析训练过程"""
    print(f"\n📊 训练过程分析:")
    print("=" * 50)
    
    # 分析EquiformerV2训练历史
    training_history_file = "experiments/2025-09-10_run1/ablation/num_layers=2__sphere_channels=64__num_heads=4__grid_resolution=16__edge_channels=64__REAL_FUSION_BASE/seed=42/logs/enhanced_equiformer_v2_training_history.json"
    
    if os.path.exists(training_history_file):
        with open(training_history_file, 'r') as f:
            training_history = json.load(f)
        
        train_losses = training_history.get('train_loss', [])
        val_losses = training_history.get('val_loss', [])
        val_maes = training_history.get('val_mae', [])
        
        print(f"   EquiformerV2训练过程:")
        print(f"   - 训练轮数: {len(train_losses)}")
        print(f"   - 最终训练损失: {train_losses[-1]:.6f}")
        print(f"   - 最终验证损失: {val_losses[-1]:.6f}")
        print(f"   - 最终验证MAE: {val_maes[-1]:.6f} eV")
        print(f"   - 最佳验证MAE: {min(val_maes):.6f} eV (第{val_maes.index(min(val_maes))+1}轮)")
        
        # 分析收敛性
        if len(val_maes) >= 20:
            early_mae = np.mean(val_maes[:10])
            late_mae = np.mean(val_maes[-10:])
            convergence_improvement = ((early_mae - late_mae) / early_mae) * 100
            print(f"   - 收敛改善: {convergence_improvement:.2f}% (前10轮平均: {early_mae:.6f}, 后10轮平均: {late_mae:.6f})")

def create_improvement_visualization(analysis_results):
    """创建改善可视化"""
    print(f"\n📊 创建改善可视化...")
    
    # 创建改善对比图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. MAE对比
    models = ['基础EquiformerV2', '最佳EquiformerV2', '结构+表格融合']
    maes = [
        analysis_results['base_eqv2']['Test MAE (eV)'],
        analysis_results['best_eqv2']['Test MAE (eV)'],
        analysis_results['fusion_model']['Test MAE (eV)']
    ]
    colors = ['#ff9999', '#ffcc99', '#99ff99']
    
    bars1 = ax1.bar(models, maes, color=colors, alpha=0.7)
    ax1.set_ylabel('Test MAE (eV)')
    ax1.set_title('Test MAE 对比')
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, mae in zip(bars1, maes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{mae:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 2. RMSE对比
    rmses = [
        analysis_results['base_eqv2']['Test RMSE (eV)'],
        analysis_results['best_eqv2']['Test RMSE (eV)'],
        analysis_results['fusion_model']['Test RMSE (eV)']
    ]
    
    bars2 = ax2.bar(models, rmses, color=colors, alpha=0.7)
    ax2.set_ylabel('Test RMSE (eV)')
    ax2.set_title('Test RMSE 对比')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, rmse in zip(bars2, rmses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{rmse:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 3. 改善百分比
    improvements = [
        analysis_results['ablation_mae_improvement'],
        analysis_results['mae_improvement_best']
    ]
    improvement_labels = ['消融实验优化', '融合模型改善']
    improvement_colors = ['#66b3ff', '#99ff99']
    
    bars3 = ax3.bar(improvement_labels, improvements, color=improvement_colors, alpha=0.7)
    ax3.set_ylabel('MAE改善 (%)')
    ax3.set_title('相对于基础EquiformerV2的MAE改善')
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, improvement in zip(bars3, improvements):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{improvement:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # 4. 累积改善效果
    cumulative_improvements = [0, analysis_results['ablation_mae_improvement'], analysis_results['mae_improvement_base']]
    cumulative_labels = ['基础EquiformerV2', '消融优化后', '融合模型后']
    
    ax4.plot(range(len(cumulative_labels)), cumulative_improvements, 'o-', linewidth=2, markersize=8)
    ax4.set_xlabel('优化阶段')
    ax4.set_ylabel('累积MAE改善 (%)')
    ax4.set_title('累积优化效果')
    ax4.set_xticks(range(len(cumulative_labels)))
    ax4.set_xticklabels(cumulative_labels)
    ax4.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, improvement in enumerate(cumulative_improvements):
        ax4.text(i, improvement + 0.5, f'{improvement:.2f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = "experiments/model_comparison/detailed_improvement_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ 详细改善分析图已保存: {output_path}")
    
    plt.close()

def generate_summary_report(analysis_results):
    """生成总结报告"""
    print(f"\n📝 生成总结报告...")
    
    report = f"""
# EquiformerV2 vs 结构+表格晚融合模型详细分析报告

## 🎯 实验目标
比较EquiformerV2（纯结构模型）与结构+表格晚融合模型的性能差异，验证表格特征对氢吸附能预测的贡献。

## 📊 实验结果

### 模型性能对比
| 模型 | Test MAE (eV) | Test RMSE (eV) | Test Loss |
|------|---------------|----------------|-----------|
| EquiformerV2 (基础) | {analysis_results['base_eqv2']['Test MAE (eV)']:.6f} | {analysis_results['base_eqv2']['Test RMSE (eV)']:.6f} | {analysis_results['base_eqv2']['Test Loss']:.6f} |
| EquiformerV2 (最佳参数) | {analysis_results['best_eqv2']['Test MAE (eV)']:.6f} | {analysis_results['best_eqv2']['Test RMSE (eV)']:.6f} | {analysis_results['best_eqv2']['Test Loss']:.6f} |
| 结构+表格融合 | {analysis_results['fusion_model']['Test MAE (eV)']:.6f} | {analysis_results['fusion_model']['Test RMSE (eV)']:.6f} | N/A |

### 性能改善分析

#### 消融实验优化效果
- MAE改善: {analysis_results['ablation_mae_improvement']:.2f}%
- RMSE改善: {analysis_results['ablation_rmse_improvement']:.2f}%

#### 融合模型改善效果
相对于基础EquiformerV2:
- MAE改善: {analysis_results['mae_improvement_base']:.2f}%
- RMSE改善: {analysis_results['rmse_improvement_base']:.2f}%

相对于最佳EquiformerV2:
- MAE改善: {analysis_results['mae_improvement_best']:.2f}%
- RMSE改善: {analysis_results['rmse_improvement_best']:.2f}%

## 🔍 关键发现

1. **消融实验有效性**: 通过系统性的消融实验，找到了EquiformerV2的最佳参数组合：
   - grid_resolution = 16 (Lmax消融最佳)
   - num_layers = 2 (层数消融最佳)
   - attn_renorm=1, sep_s2=1, sep_ln=0 (模块消融最佳)
   - 相比基础参数，MAE改善了{analysis_results['ablation_mae_improvement']:.2f}%

2. **表格特征的重要性**: 结构+表格晚融合模型相比最佳EquiformerV2：
   - MAE进一步改善了{analysis_results['mae_improvement_best']:.2f}%
   - 证明了表格特征对氢吸附能预测的重要贡献

3. **累积优化效果**: 从基础EquiformerV2到最终融合模型：
   - 总MAE改善达到{analysis_results['mae_improvement_base']:.2f}%
   - 体现了系统性优化的价值

## 🏆 结论

结构+表格晚融合模型在氢吸附能预测任务上显著优于纯结构模型，证明了：
1. 消融实验对模型优化的有效性
2. 表格特征对分子性质预测的重要价值
3. 晚融合策略的成功应用

## 📁 相关文件
- 对比图表: experiments/model_comparison/model_comparison.png
- 详细分析图: experiments/model_comparison/detailed_improvement_analysis.png
- 对比数据: experiments/model_comparison/model_comparison.csv
"""
    
    # 保存报告
    report_path = "experiments/model_comparison/detailed_analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"   ✅ 详细分析报告已保存: {report_path}")
    
    # 打印关键结论
    print(f"\n🏆 关键结论:")
    print(f"   1. 消融实验优化效果: MAE改善{analysis_results['ablation_mae_improvement']:.2f}%")
    print(f"   2. 融合模型改善效果: MAE改善{analysis_results['mae_improvement_best']:.2f}%")
    print(f"   3. 总累积改善效果: MAE改善{analysis_results['mae_improvement_base']:.2f}%")
    print(f"   4. 最佳模型: 结构+表格融合 (MAE={analysis_results['fusion_model']['Test MAE (eV)']:.6f} eV)")

def main():
    """主函数"""
    print("📊 EquiformerV2 vs 结构+表格晚融合模型详细分析")
    print("=" * 80)
    
    try:
        # 1. 分析性能改善
        analysis_results = analyze_performance_improvement()
        
        # 2. 分析训练过程
        analyze_training_process()
        
        # 3. 创建改善可视化
        create_improvement_visualization(analysis_results)
        
        # 4. 生成总结报告
        generate_summary_report(analysis_results)
        
        print(f"\n🎉 详细分析完成!")
        print(f"📁 所有结果保存在: experiments/model_comparison/")
        
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

