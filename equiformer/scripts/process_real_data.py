#!/usr/bin/env python3
"""
处理真实的10features_for_ML.xlsx数据，生成结构预测数据用于融合实验
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def process_real_data():
    """处理真实的特征表格数据"""
    print("🔧 处理真实的10features_for_ML.xlsx数据...")
    
    # 读取Excel文件
    df = pd.read_excel('data/raw/10features_for_ML.xlsx')
    print(f"   原始数据形状: {df.shape}")
    print(f"   列名: {df.columns.tolist()}")
    
    # 选择特征列（排除目标列和无关列）
    exclude_cols = {'ΔGH', 'Equation', 'reactants', 'products', 'structures', 'Structure'}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"   选择的特征列: {feature_cols}")
    print(f"   特征列数量: {len(feature_cols)}")
    
    # 创建处理后的数据
    processed_data = df[feature_cols + ['ΔGH']].copy()
    
    # 添加idx列用于对齐
    processed_data['idx'] = range(len(processed_data))
    
    # 保存处理后的数据
    os.makedirs("data/processed", exist_ok=True)
    processed_data.to_csv("data/processed/cleaned_data.csv", index=False)
    print("   ✅ 处理后的数据已保存: data/processed/cleaned_data.csv")
    
    return processed_data, feature_cols

def create_mock_structure_predictions(data, feature_cols):
    """创建模拟的结构预测数据"""
    print("\n🎭 创建模拟的结构预测数据...")
    
    n_samples = len(data)
    
    # 模拟结构模型的预测（比真实值稍差一些）
    np.random.seed(42)
    noise_factor = 0.3  # 添加30%的噪声
    
    # 基于真实值生成模拟预测
    true_values = data['ΔGH'].values
    mock_predictions = true_values + np.random.normal(0, noise_factor, n_samples)
    
    # 创建训练/验证/测试分割
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    train_indices = np.random.choice(n_samples, train_size, replace=False)
    remaining_indices = np.setdiff1d(range(n_samples), train_indices)
    val_indices = np.random.choice(remaining_indices, val_size, replace=False)
    test_indices = np.setdiff1d(remaining_indices, val_indices)
    
    # 创建预测数据
    experiments_dir = Path("experiments/struct_preds")
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    splits = [
        ("train", train_indices),
        ("val", val_indices), 
        ("test", test_indices)
    ]
    
    for split_name, indices in splits:
        split_data = data.iloc[indices].copy()
        split_data['y_pred'] = mock_predictions[indices]
        split_data['y_true'] = split_data['ΔGH']
        
        # 只保留必要的列
        output_data = split_data[['idx', 'y_true', 'y_pred']].copy()
        
        output_file = experiments_dir / f"{split_name}_preds.csv"
        output_data.to_csv(output_file, index=False)
        print(f"   ✅ {split_name}预测数据已保存: {output_file} ({len(output_data)} 样本)")
    
    return experiments_dir

def run_real_fusion_experiment():
    """运行真实的融合实验"""
    print("\n🚀 运行真实的融合实验...")
    
    # 构建命令
    cmd = [
        "python", "scripts/train_test_tabular_fusion.py",
        "--mode", "train_test",
        "--struct_train_cmd", "",  # 跳过结构训练
        "--struct_pred_train", "experiments/struct_preds/train_preds.csv",
        "--struct_pred_val", "experiments/struct_preds/val_preds.csv", 
        "--struct_pred_test", "experiments/struct_preds/test_preds.csv",
        "--tab_csv", "data/processed/cleaned_data.csv",
        "--tab_cols", "",  # 自动选择特征列
        "--tab_id_col", "",  # 使用idx对齐
        "--fusion", "concat",
        "--epochs", "100",
        "--batch_size", "256",
        "--seed", "42",
        "--extra_note", "real_data"
    ]
    
    print(f"📋 运行命令:")
    print(f"   {' '.join(cmd)}")
    
    # 运行命令
    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ 真实数据融合训练成功完成!")
        print("\n📊 输出结果:")
        print(result.stdout)
        
        # 检查输出文件
        output_dir = None
        for line in result.stdout.split('\n'):
            if 'save_dir=' in line:
                output_dir = line.split('save_dir=')[-1].strip()
                break
        
        if output_dir and os.path.exists(output_dir):
            print(f"\n📁 输出目录: {output_dir}")
            files = os.listdir(output_dir)
            print("📄 生成的文件:")
            for f in files:
                print(f"   - {f}")
                
            # 读取并显示结果
            metrics_file = os.path.join(output_dir, "metrics.json")
            if os.path.exists(metrics_file):
                import json
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                print(f"\n📈 最终结果:")
                print(f"   Test MAE: {metrics['test_mae']:.6f} eV")
                print(f"   Test RMSE: {metrics['test_rmse']:.6f} eV")
                print(f"   融合模式: {metrics['fusion']}")
                print(f"   对齐模式: {metrics['align_mode']}")
    else:
        print("❌ 真实数据融合训练失败!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

def show_real_data_usage():
    """显示真实数据的使用示例"""
    print("\n📖 真实数据使用示例:")
    print("=" * 60)
    
    print("\n🎯 使用10features_for_ML.xlsx进行结构+表格晚融合:")
    print("""
# 1. 处理真实数据
python scripts/process_real_data.py

# 2. 运行融合实验
python scripts/train_test_tabular_fusion.py \\
  --mode train_test \\
  --struct_pred_train experiments/struct_preds/train_preds.csv \\
  --struct_pred_val   experiments/struct_preds/val_preds.csv \\
  --struct_pred_test  experiments/struct_preds/test_preds.csv \\
  --tab_csv data/processed/cleaned_data.csv \\
  --tab_cols "" \\
  --fusion concat \\
  --epochs 100 --batch_size 256 --seed 42

# 3. 比较不同融合模式
python scripts/train_test_tabular_fusion.py \\
  --mode train_test \\
  --struct_pred_train experiments/struct_preds/train_preds.csv \\
  --struct_pred_val   experiments/struct_preds/val_preds.csv \\
  --struct_pred_test  experiments/struct_preds/test_preds.csv \\
  --tab_csv data/processed/cleaned_data.csv \\
  --fusion gate \\
  --epochs 100 --batch_size 256 --seed 42 \\
  --extra_note gate_fusion
""")

def main():
    """主函数"""
    print("🎯 处理真实10features_for_ML.xlsx数据")
    print("=" * 60)
    
    try:
        # 1. 处理真实数据
        processed_data, feature_cols = process_real_data()
        
        # 2. 创建模拟结构预测
        experiments_dir = create_mock_structure_predictions(processed_data, feature_cols)
        
        # 3. 运行融合实验
        run_real_fusion_experiment()
        
        # 4. 显示使用示例
        show_real_data_usage()
        
        print("\n🎉 真实数据处理和融合实验完成!")
        print("💡 提示: 现在您可以使用真实的EquiformerV2预测结果替换模拟数据")
        
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
