#!/usr/bin/env python3
"""
可视化分支架构脚本
生成优美的网络结构图，展示结构+表格晚融合的分支架构
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_branch_architecture_diagram():
    """创建分支架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 定义颜色方案
    colors = {
        'input': '#E8F4FD',
        'structure': '#FFE6E6', 
        'tabular': '#E6F7E6',
        'fusion': '#FFF2E6',
        'output': '#F0E6FF',
        'arrow': '#666666',
        'text': '#333333'
    }
    
    # 1. 输入层
    input_box = FancyBboxPatch((1, 10), 8, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['input'], 
                              edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 10.75, '输入数据', ha='center', va='center', fontsize=16, fontweight='bold')
    
    # 输入子项
    ax.text(2.5, 10.3, '• 分子结构 (原子坐标)', ha='center', va='center', fontsize=12)
    ax.text(5, 10.3, '• 表格特征 (12维)', ha='center', va='center', fontsize=12)
    ax.text(7.5, 10.3, '• 真实标签', ha='center', va='center', fontsize=12)
    
    # 2. 分支分离
    # 结构分支
    struct_box = FancyBboxPatch((0.5, 7.5), 3.5, 2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['structure'], 
                               edgecolor='red', linewidth=2)
    ax.add_patch(struct_box)
    ax.text(2.25, 8.5, '结构分支', ha='center', va='center', fontsize=14, fontweight='bold', color='red')
    
    # EquiformerV2详细结构
    ax.text(2.25, 8.1, 'EquiformerV2', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(2.25, 7.8, '• Spherical Harmonics', ha='center', va='center', fontsize=10)
    ax.text(2.25, 7.6, '• SO(3) Equivariant Attention', ha='center', va='center', fontsize=10)
    
    # 表格分支
    tabular_box = FancyBboxPatch((6, 7.5), 3.5, 2, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['tabular'], 
                                edgecolor='green', linewidth=2)
    ax.add_patch(tabular_box)
    ax.text(7.75, 8.5, '表格分支', ha='center', va='center', fontsize=14, fontweight='bold', color='green')
    
    # TabularBranch详细结构
    ax.text(7.75, 8.1, 'TabularBranch', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(7.75, 7.8, '• TabularStandardizer', ha='center', va='center', fontsize=10)
    ax.text(7.75, 7.6, '• TabularMLP (64→64)', ha='center', va='center', fontsize=10)
    
    # 3. 中间表示
    # 结构分支输出
    struct_output_box = FancyBboxPatch((1, 5.5), 2.5, 1, 
                                      boxstyle="round,pad=0.1", 
                                      facecolor=colors['structure'], 
                                      edgecolor='red', linewidth=1)
    ax.add_patch(struct_output_box)
    ax.text(2.25, 6, 'y_struct (B,1)', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 表格分支输出
    tabular_output_box = FancyBboxPatch((6.5, 5.5), 2.5, 1, 
                                        boxstyle="round,pad=0.1", 
                                        facecolor=colors['tabular'], 
                                        edgecolor='green', linewidth=1)
    ax.add_patch(tabular_output_box)
    ax.text(7.75, 6, 'h_feat (B,64)', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 4. 融合层
    fusion_box = FancyBboxPatch((3, 3.5), 4, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['fusion'], 
                               edgecolor='orange', linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(5, 4.25, 'FusionHead', ha='center', va='center', fontsize=14, fontweight='bold', color='orange')
    
    # 融合模式
    ax.text(5, 3.9, 'Concat模式: [s; h_feat] → MLP(128→64→1)', ha='center', va='center', fontsize=11)
    ax.text(5, 3.7, 'Gate模式: s * σ(g) + h_feat → MLP(64→1)', ha='center', va='center', fontsize=11)
    
    # 5. 输出层
    output_box = FancyBboxPatch((4, 1.5), 2, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['output'], 
                               edgecolor='purple', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 2, 'y_hat (B,1)', ha='center', va='center', fontsize=14, fontweight='bold', color='purple')
    
    # 6. 添加箭头连接
    # 输入到分支
    arrow1 = ConnectionPatch((5, 10), (2.25, 9.5), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc=colors['arrow'], ec=colors['arrow'])
    ax.add_patch(arrow1)
    
    arrow2 = ConnectionPatch((5, 10), (7.75, 9.5), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc=colors['arrow'], ec=colors['arrow'])
    ax.add_patch(arrow2)
    
    # 分支到中间表示
    arrow3 = ConnectionPatch((2.25, 7.5), (2.25, 6.5), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc=colors['arrow'], ec=colors['arrow'])
    ax.add_patch(arrow3)
    
    arrow4 = ConnectionPatch((7.75, 7.5), (7.75, 6.5), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc=colors['arrow'], ec=colors['arrow'])
    ax.add_patch(arrow4)
    
    # 中间表示到融合
    arrow5 = ConnectionPatch((2.25, 5.5), (3, 4.25), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc=colors['arrow'], ec=colors['arrow'])
    ax.add_patch(arrow5)
    
    arrow6 = ConnectionPatch((7.75, 5.5), (7, 4.25), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc=colors['arrow'], ec=colors['arrow'])
    ax.add_patch(arrow6)
    
    # 融合到输出
    arrow7 = ConnectionPatch((5, 3.5), (5, 2.5), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc=colors['arrow'], ec=colors['arrow'])
    ax.add_patch(arrow7)
    
    # 7. 添加标题和说明
    ax.text(5, 11.5, '结构+表格晚融合分支架构', ha='center', va='center', 
            fontsize=20, fontweight='bold', color=colors['text'])
    
    # 添加性能指标
    ax.text(0.5, 0.5, '性能提升:', ha='left', va='center', fontsize=12, fontweight='bold')
    ax.text(0.5, 0.2, '• MAE改善: 74.30% (0.132 → 0.034 eV)', ha='left', va='center', fontsize=11)
    ax.text(0.5, 0.0, '• RMSE改善: 75.24% (0.232 → 0.058 eV)', ha='left', va='center', fontsize=11)
    
    plt.tight_layout()
    return fig

def create_detailed_fusion_diagram():
    """创建详细的融合机制图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    colors = {
        'concat': '#E6F3FF',
        'gate': '#FFF0E6', 
        'arrow': '#666666',
        'text': '#333333'
    }
    
    # 标题
    ax.text(5, 7.5, 'FusionHead 融合机制详解', ha='center', va='center', 
            fontsize=18, fontweight='bold')
    
    # Concat模式
    ax.text(2.5, 6.5, 'Concat模式', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='blue')
    
    concat_box = FancyBboxPatch((0.5, 4.5), 4, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['concat'], 
                               edgecolor='blue', linewidth=2)
    ax.add_patch(concat_box)
    
    ax.text(2.5, 5.5, 'y_struct (B,1) → Linear(1,64) → s (B,64)', ha='center', va='center', fontsize=11)
    ax.text(2.5, 5.1, 'h_feat (B,64)', ha='center', va='center', fontsize=11)
    ax.text(2.5, 4.7, '[s; h_feat] → (B,128) → MLP(128→64→1) → y_hat', ha='center', va='center', fontsize=11)
    
    # Gate模式
    ax.text(7.5, 6.5, 'Gate模式', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='orange')
    
    gate_box = FancyBboxPatch((5.5, 4.5), 4, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['gate'], 
                             edgecolor='orange', linewidth=2)
    ax.add_patch(gate_box)
    
    ax.text(7.5, 5.5, 'y_struct (B,1) → Linear(1,64) → s (B,64)', ha='center', va='center', fontsize=11)
    ax.text(7.5, 5.1, 'h_feat (B,64) → Linear(64,1) → g (B,1)', ha='center', va='center', fontsize=11)
    ax.text(7.5, 4.7, 'h = s * σ(g) + h_feat → MLP(64→1) → y_hat', ha='center', va='center', fontsize=11)
    
    # 添加箭头
    arrow1 = ConnectionPatch((2.5, 4.5), (2.5, 3.5), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc=colors['arrow'], ec=colors['arrow'])
    ax.add_patch(arrow1)
    
    arrow2 = ConnectionPatch((7.5, 4.5), (7.5, 3.5), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc=colors['arrow'], ec=colors['arrow'])
    ax.add_patch(arrow2)
    
    # 输出
    output_box = FancyBboxPatch((4, 2.5), 2, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#F0E6FF', 
                               edgecolor='purple', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 2.9, 'y_hat (B,1)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 特点说明
    ax.text(5, 1.5, '特点:', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(5, 1.1, '• Concat: 简单拼接，信息完整保留', ha='center', va='center', fontsize=10)
    ax.text(5, 0.8, '• Gate: 自适应权重，选择性融合', ha='center', va='center', fontsize=10)
    ax.text(5, 0.5, '• 动态维度适配，支持不同结构表示', ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig

def create_training_flow_diagram():
    """创建训练流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    colors = {
        'phase1': '#E6F7E6',
        'phase2': '#FFF2E6', 
        'phase3': '#E6F3FF',
        'arrow': '#666666',
        'text': '#333333'
    }
    
    # 标题
    ax.text(6, 9.5, '训练流程: 结构+表格晚融合', ha='center', va='center', 
            fontsize=18, fontweight='bold')
    
    # 阶段1: 结构分支训练
    phase1_box = FancyBboxPatch((0.5, 7), 3, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['phase1'], 
                               edgecolor='green', linewidth=2)
    ax.add_patch(phase1_box)
    ax.text(2, 7.75, '阶段1: 结构分支训练', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(2, 7.4, 'EquiformerV2', ha='center', va='center', fontsize=11)
    ax.text(2, 7.1, '输入: 分子结构', ha='center', va='center', fontsize=10)
    ax.text(2, 6.8, '输出: y_struct', ha='center', va='center', fontsize=10)
    
    # 阶段2: 表格分支训练
    phase2_box = FancyBboxPatch((4.5, 7), 3, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['phase2'], 
                               edgecolor='orange', linewidth=2)
    ax.add_patch(phase2_box)
    ax.text(6, 7.75, '阶段2: 表格分支训练', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(6, 7.4, 'TabularBranch', ha='center', va='center', fontsize=11)
    ax.text(6, 7.1, '输入: 表格特征', ha='center', va='center', fontsize=10)
    ax.text(6, 6.8, '输出: h_feat', ha='center', va='center', fontsize=10)
    
    # 阶段3: 融合训练
    phase3_box = FancyBboxPatch((8.5, 7), 3, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['phase3'], 
                               edgecolor='blue', linewidth=2)
    ax.add_patch(phase3_box)
    ax.text(10, 7.75, '阶段3: 融合训练', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(10, 7.4, 'FusionHead', ha='center', va='center', fontsize=11)
    ax.text(10, 7.1, '输入: y_struct + h_feat', ha='center', va='center', fontsize=10)
    ax.text(10, 6.8, '输出: y_hat', ha='center', va='center', fontsize=10)
    
    # 添加箭头
    arrow1 = ConnectionPatch((3.5, 7.75), (4.5, 7.75), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc=colors['arrow'], ec=colors['arrow'])
    ax.add_patch(arrow1)
    
    arrow2 = ConnectionPatch((7.5, 7.75), (8.5, 7.75), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc=colors['arrow'], ec=colors['arrow'])
    ax.add_patch(arrow2)
    
    # 详细训练步骤
    ax.text(6, 5.5, '详细训练步骤:', ha='center', va='center', fontsize=14, fontweight='bold')
    
    steps = [
        '1. 使用最佳参数训练EquiformerV2 (grid_resolution=16, num_layers=2)',
        '2. 提取结构预测结果 y_struct (train/val/test)',
        '3. 标准化表格特征 (TabularStandardizer)',
        '4. 训练TabularMLP编码器 (in_dim→64→64)',
        '5. 训练FusionHead融合器 (concat/gate模式)',
        '6. 使用MAE损失和AdamW优化器',
        '7. 早停机制 (patience=20)',
        '8. 在测试集上评估最终性能'
    ]
    
    for i, step in enumerate(steps):
        ax.text(0.5, 4.5-i*0.4, step, ha='left', va='center', fontsize=11)
    
    plt.tight_layout()
    return fig

def main():
    """主函数"""
    print("🎨 生成分支架构可视化图表...")
    
    # 创建输出目录
    output_dir = "experiments/architecture_visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. 生成主架构图
        print("📊 生成主架构图...")
        fig1 = create_branch_architecture_diagram()
        fig1.savefig(os.path.join(output_dir, "branch_architecture.png"), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✅ 主架构图已保存: {output_dir}/branch_architecture.png")
        
        # 2. 生成融合机制图
        print("🔗 生成融合机制图...")
        fig2 = create_detailed_fusion_diagram()
        fig2.savefig(os.path.join(output_dir, "fusion_mechanism.png"), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✅ 融合机制图已保存: {output_dir}/fusion_mechanism.png")
        
        # 3. 生成训练流程图
        print("🔄 生成训练流程图...")
        fig3 = create_training_flow_diagram()
        fig3.savefig(os.path.join(output_dir, "training_flow.png"), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✅ 训练流程图已保存: {output_dir}/training_flow.png")
        
        print(f"\n🎉 所有架构图生成完成!")
        print(f"📁 输出目录: {output_dir}")
        print(f"📊 包含文件:")
        print(f"   - branch_architecture.png (主架构图)")
        print(f"   - fusion_mechanism.png (融合机制图)")
        print(f"   - training_flow.png (训练流程图)")
        
        # 显示图表
        plt.show()
        
    except Exception as e:
        print(f"❌ 生成图表失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

