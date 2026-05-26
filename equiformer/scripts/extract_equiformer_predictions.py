#!/usr/bin/env python3
"""
提取EquiformerV2预测结果，创建融合实验所需的结构预测文件
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

def extract_equiformer_predictions():
    """提取EquiformerV2的预测结果"""
    print("📊 提取EquiformerV2预测结果...")
    
    # EquiformerV2预测文件路径
    pred_file = "experiments/2025-09-10_run1/ablation/num_layers=2__sphere_channels=64__num_heads=4__grid_resolution=16__edge_channels=64__REAL_FUSION_BASE/seed=42/enhanced_equiformer_v2_predictions.csv"
    
    if not os.path.exists(pred_file):
        print(f"❌ EquiformerV2预测文件不存在: {pred_file}")
        return False
    
    # 读取预测结果
    df = pd.read_csv(pred_file)
    print(f"   ✅ 读取预测结果: {len(df)} 样本")
    
    # 重命名列
    df = df.rename(columns={'true': 'y_true', 'predicted': 'y_pred'})
    
    # 添加idx列
    df['idx'] = np.arange(len(df))
    
    # 重新排列列顺序
    df = df[['idx', 'y_true', 'y_pred']]
    
    # 创建输出目录
    output_dir = "experiments/struct_preds_real"
    os.makedirs(output_dir, exist_ok=True)
    
    # 由于我们没有train/val/test分割信息，我们假设：
    # - 前80%为训练集
    # - 中间10%为验证集  
    # - 后10%为测试集
    
    n_total = len(df)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    train_df = df[:n_train].copy()
    val_df = df[n_train:n_train+n_val].copy()
    test_df = df[n_train+n_val:].copy()
    
    # 重新分配idx
    train_df['idx'] = np.arange(len(train_df))
    val_df['idx'] = np.arange(len(val_df))
    test_df['idx'] = np.arange(len(test_df))
    
    # 保存文件
    train_file = os.path.join(output_dir, "train_preds.csv")
    val_file = os.path.join(output_dir, "val_preds.csv")
    test_file = os.path.join(output_dir, "test_preds.csv")
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"   ✅ 训练集预测: {train_file} ({len(train_df)} 样本)")
    print(f"   ✅ 验证集预测: {val_file} ({len(val_df)} 样本)")
    print(f"   ✅ 测试集预测: {test_file} ({len(test_df)} 样本)")
    
    return True

def main():
    """主函数"""
    print("🔧 提取EquiformerV2预测结果用于融合实验")
    print("=" * 50)
    
    if extract_equiformer_predictions():
        print("\n🎉 预测结果提取完成!")
        print("📁 输出目录: experiments/struct_preds_real/")
    else:
        print("\n❌ 预测结果提取失败!")

if __name__ == "__main__":
    main()
