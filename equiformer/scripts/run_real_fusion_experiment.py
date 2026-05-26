#!/usr/bin/env python3
"""
使用真实的EquiformerV2模型和数据集进行结构+表格晚融合实验
"""
import os
import sys
import pandas as pd
import numpy as np
import subprocess
import json
from pathlib import Path
import torch

def prepare_real_data():
    """准备真实的数据集"""
    print("🔧 准备真实数据集...")
    
    # 1. 处理特征表格数据
    print("   处理10features_for_ML.xlsx...")
    df = pd.read_excel('data/raw/10features_for_ML.xlsx')
    
    # 选择特征列
    exclude_cols = {'ΔGH', 'Equation', 'reactants', 'products', 'structures', 'Structure'}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # 创建处理后的数据
    processed_data = df[feature_cols + ['ΔGH']].copy()
    processed_data['idx'] = range(len(processed_data))
    
    # 保存处理后的数据
    os.makedirs("data/processed", exist_ok=True)
    processed_data.to_csv("data/processed/cleaned_data.csv", index=False)
    print(f"   ✅ 特征数据已保存: data/processed/cleaned_data.csv ({len(processed_data)} 样本, {len(feature_cols)} 特征)")
    
    return processed_data, feature_cols

def train_equiformer_v2():
    """训练真实的EquiformerV2模型（带可视化）"""
    print("\n🚀 训练真实的EquiformerV2模型...")
    print("📊 使用消融实验最佳参数组合:")
    print("   - grid_resolution = 16 (Lmax消融最佳)")
    print("   - num_layers = 2 (层数消融最佳)")
    print("   - attn_renorm=1, sep_s2=1, sep_ln=0 (模块消融最佳)")
    
    # 构建训练命令
    cmd = [
        "python", "src/train_enhanced_equiformer_v2.py",
        "--num_layers", "2",
        "--sphere_channels", "64", 
        "--num_heads", "4",
        "--grid_resolution", "16",  # Lmax消融最佳
        "--edge_channels", "64",
        "--eSCN", "1",
        "--attn_renorm", "1",  # 模块消融最佳
        "--sep_s2", "1",       # 模块消融最佳
        "--sep_ln", "0",       # 模块消融最佳
        "--batch_size", "16",
        "--lr", "0.0002",
        "--num_epochs", "100",
        "--seed", "42",
        "--experiment_tag", "REAL_FUSION_BASE"
    ]
    
    print(f"📋 训练命令:")
    print(f"   {' '.join(cmd)}")
    
    # 运行训练（实时显示输出）
    print("\n⏳ 开始训练（实时显示进度）...")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              text=True, bufsize=1, universal_newlines=True)
    
    # 实时显示训练输出
    epoch_count = 0
    batch_count = 0
    training_losses = []
    validation_maes = []
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            
            # 解析训练进度
            if "Epoch" in output and "/" in output and "completed" in output:
                epoch_count += 1
                print(f"✅ Epoch {epoch_count} 完成")
            
            # 解析验证结果
            elif "验证MAE:" in output:
                try:
                    mae_value = float(output.split("验证MAE:")[1].split("eV")[0].strip())
                    validation_maes.append(mae_value)
                    print(f"   📈 验证MAE: {mae_value:.6f} eV")
                except:
                    pass
            
            # 解析训练损失
            elif "训练损失:" in output:
                try:
                    train_loss = float(output.split("训练损失:")[1].strip())
                    training_losses.append(train_loss)
                except:
                    pass
    
    return_code = process.poll()
    
    if return_code == 0:
        print("✅ EquiformerV2训练成功完成!")
        
        # 创建训练过程可视化
        if training_losses and validation_maes:
            create_training_visualization(training_losses, validation_maes)
        
        return True
    else:
        print("❌ EquiformerV2训练失败!")
        return False

def create_training_visualization(train_losses, val_maes):
    """创建训练过程可视化"""
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 训练损失曲线
        if train_losses:
            epochs = range(1, len(train_losses) + 1)
            ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('EquiformerV2 Training Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 验证MAE曲线
        if val_maes:
            epochs = range(1, len(val_maes) + 1)
            ax2.plot(epochs, val_maes, 'r-', linewidth=2, label='Validation MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE (eV)')
            ax2.set_title('EquiformerV2 Validation MAE')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        output_path = "experiments/equiformer_v2_training_progress.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 训练可视化已保存: {output_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"⚠️  创建训练可视化失败: {e}")

def extract_predictions():
    """提取EquiformerV2的预测结果"""
    print("\n📊 提取EquiformerV2预测结果...")
    
    # 查找最新的实验目录
    experiments_dir = Path("experiments")
    equiformer_dirs = [d for d in experiments_dir.iterdir() if d.is_dir() and "REAL_FUSION_BASE" in d.name]
    
    if not equiformer_dirs:
        print("❌ 未找到EquiformerV2实验目录")
        return False
    
    # 选择最新的目录
    latest_dir = max(equiformer_dirs, key=lambda x: x.stat().st_mtime)
    print(f"   使用实验目录: {latest_dir}")
    
    # 查找预测文件
    ablation_dir = latest_dir / "ablation"
    if not ablation_dir.exists():
        print("❌ 未找到ablation目录")
        return False
    
    # 查找具体的实验子目录
    exp_subdirs = [d for d in ablation_dir.iterdir() if d.is_dir() and "REAL_FUSION_BASE" in d.name]
    if not exp_subdirs:
        print("❌ 未找到具体的实验子目录")
        return False
    
    exp_dir = exp_subdirs[0] / "seed=0"
    logs_dir = exp_dir / "logs"
    
    print(f"   实验子目录: {exp_dir}")
    
    # 检查关键文件
    test_results_file = logs_dir / "enhanced_equiformer_v2_test_results.json"
    predictions_file = exp_dir / "enhanced_equiformer_v2_predictions.csv"
    
    if not test_results_file.exists():
        print(f"❌ 测试结果文件不存在: {test_results_file}")
        return False
    
    if not predictions_file.exists():
        print(f"❌ 预测文件不存在: {predictions_file}")
        return False
    
    # 读取预测结果
    predictions_df = pd.read_csv(predictions_file)
    print(f"   ✅ 预测结果已读取: {len(predictions_df)} 样本")
    print(f"   列名: {predictions_df.columns.tolist()}")
    
    # 创建结构预测数据目录
    struct_preds_dir = Path("experiments/struct_preds_real")
    struct_preds_dir.mkdir(parents=True, exist_ok=True)
    
    # 分割数据（假设预测文件包含所有数据，需要按原始数据集分割）
    # 这里我们需要根据custom_hydrogen数据集的分割来创建train/val/test
    n_total = len(predictions_df)
    train_size = int(0.7 * n_total)
    val_size = int(0.15 * n_total)
    
    # 创建分割
    train_df = predictions_df.iloc[:train_size].copy()
    val_df = predictions_df.iloc[train_size:train_size+val_size].copy()
    test_df = predictions_df.iloc[train_size+val_size:].copy()
    
    # 添加必要的列
    for df, split_name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        df['idx'] = range(len(df))
        df['y_true'] = df['y_true']  # 假设已经有这个列
        df['y_pred'] = df['y_pred']  # 假设已经有这个列
        
        output_file = struct_preds_dir / f"{split_name}_preds.csv"
        df[['idx', 'y_true', 'y_pred']].to_csv(output_file, index=False)
        print(f"   ✅ {split_name}预测已保存: {output_file} ({len(df)} 样本)")
    
    return True

def run_fusion_experiment():
    """运行融合实验（带可视化）"""
    print("\n🔗 运行结构+表格晚融合实验...")
    print("📊 融合配置:")
    print("   - 结构分支: EquiformerV2 (最佳参数)")
    print("   - 表格分支: 12维特征 → TabularMLP")
    print("   - 融合模式: concat")
    
    # 构建融合命令
    cmd = [
        "python", "scripts/train_test_tabular_fusion.py",
        "--mode", "train_test",
        "--struct_train_cmd", "",  # 跳过结构训练
        "--struct_pred_train", "experiments/struct_preds_real/train_preds.csv",
        "--struct_pred_val", "experiments/struct_preds_real/val_preds.csv", 
        "--struct_pred_test", "experiments/struct_preds_real/test_preds.csv",
        "--tab_csv", "data/processed/cleaned_data.csv",
        "--tab_cols", "",  # 自动选择特征列
        "--tab_id_col", "",  # 使用idx对齐
        "--fusion", "concat",
        "--epochs", "100",
        "--batch_size", "256",
        "--seed", "42",
        "--extra_note", "real_equiformer"
    ]
    
    print(f"📋 融合命令:")
    print(f"   {' '.join(cmd)}")
    
    # 运行融合实验
    print("\n⏳ 开始融合训练...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ 融合实验成功完成!")
        print("\n📊 输出结果:")
        print(result.stdout)
        
        # 解析融合训练进度
        fusion_maes = []
        for line in result.stdout.split('\n'):
            if "[Fusion]" in line and "val MAE=" in line:
                try:
                    mae_value = float(line.split("val MAE=")[1].split(" best=")[0].strip())
                    fusion_maes.append(mae_value)
                except:
                    pass
        
        # 提取结果
        output_dir = None
        for line in result.stdout.split('\n'):
            if 'save_dir=' in line:
                output_dir = line.split('save_dir=')[-1].strip()
                break
        
        if output_dir and os.path.exists(output_dir):
            print(f"\n📁 输出目录: {output_dir}")
            
            # 读取并显示结果
            metrics_file = os.path.join(output_dir, "metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                print(f"\n📈 最终结果:")
                print(f"   Test MAE: {metrics['test_mae']:.6f} eV")
                print(f"   Test RMSE: {metrics['test_rmse']:.6f} eV")
                print(f"   融合模式: {metrics['fusion']}")
                print(f"   对齐模式: {metrics['align_mode']}")
                
                # 创建融合过程可视化
                if fusion_maes:
                    create_fusion_visualization(fusion_maes, output_dir)
                
                return output_dir, metrics
    else:
        print("❌ 融合实验失败!")
        return None, None

def create_fusion_visualization(fusion_maes, output_dir):
    """创建融合训练过程可视化"""
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # 融合验证MAE曲线
        if fusion_maes:
            epochs = range(1, len(fusion_maes) + 1)
            ax.plot(epochs, fusion_maes, 'g-', linewidth=2, label='Fusion Validation MAE')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MAE (eV)')
            ax.set_title('Structure + Tabular Late Fusion Training Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 标记最佳点
            best_mae = min(fusion_maes)
            best_epoch = fusion_maes.index(best_mae) + 1
            ax.plot(best_epoch, best_mae, 'ro', markersize=8, label=f'Best MAE: {best_mae:.6f} eV')
            ax.legend()
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(output_dir, "fusion_training_progress.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 融合训练可视化已保存: {output_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"⚠️  创建融合可视化失败: {e}")

def compare_results():
    """比较不同方法的结果"""
    print("\n📊 比较不同方法的结果...")
    
    # 这里可以添加与纯EquiformerV2结果的比较
    # 以及与纯表格特征结果的比较
    
    print("💡 建议:")
    print("   1. 比较融合后的MAE与纯EquiformerV2的MAE")
    print("   2. 比较融合后的MAE与纯表格特征的MAE") 
    print("   3. 分析哪些样本的融合效果最好")
    print("   4. 尝试不同的融合模式（concat vs gate）")

def main():
    """主函数"""
    print("🎯 真实EquiformerV2 + 表格特征晚融合实验")
    print("=" * 60)
    
    try:
        # 1. 准备数据
        processed_data, feature_cols = prepare_real_data()
        
        # 2. 训练EquiformerV2
        if not train_equiformer_v2():
            print("❌ EquiformerV2训练失败，退出")
            return
        
        # 3. 提取预测结果
        if not extract_predictions():
            print("❌ 预测结果提取失败，退出")
            return
        
        # 4. 运行融合实验
        output_dir, metrics = run_fusion_experiment()
        
        if output_dir and metrics:
            print(f"\n🎉 实验完成!")
            print(f"📁 结果目录: {output_dir}")
            print(f"📈 融合后MAE: {metrics['test_mae']:.6f} eV")
            
            # 5. 比较结果
            compare_results()
        else:
            print("❌ 融合实验失败")
        
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
