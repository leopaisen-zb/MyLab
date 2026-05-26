#!/usr/bin/env python3
"""
示例：如何使用 train_test_tabular_fusion.py 进行结构+表格晚融合
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_data():
    """创建示例数据用于测试"""
    print("🔧 创建示例数据...")
    
    # 创建示例表格数据（不包含idx列，让脚本自动处理）
    np.random.seed(42)
    n_samples = 1000
    
    tabular_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(5, 2, n_samples),
        'feature3': np.random.uniform(0, 10, n_samples),
        'feature4': np.random.exponential(1, n_samples),
        'feature5': np.random.beta(2, 5, n_samples),
        'target': np.random.normal(0, 0.5, n_samples)  # 这个会被排除
    })
    
    # 保存表格数据
    os.makedirs("data/processed", exist_ok=True)
    tabular_data.to_csv("data/processed/cleaned_data.csv", index=False)
    print("   ✅ 表格数据已保存: data/processed/cleaned_data.csv")
    
    # 创建示例结构预测数据
    experiments_dir = Path("experiments/struct_preds")
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练集预测
    train_preds = pd.DataFrame({
        'idx': range(700),
        'y_true': np.random.normal(0, 0.5, 700),
        'y_pred': np.random.normal(0, 0.4, 700)  # 结构模型预测
    })
    train_preds.to_csv(experiments_dir / "train_preds.csv", index=False)
    
    # 验证集预测
    val_preds = pd.DataFrame({
        'idx': range(700, 850),
        'y_true': np.random.normal(0, 0.5, 150),
        'y_pred': np.random.normal(0, 0.4, 150)
    })
    val_preds.to_csv(experiments_dir / "val_preds.csv", index=False)
    
    # 测试集预测
    test_preds = pd.DataFrame({
        'idx': range(850, 1000),
        'y_true': np.random.normal(0, 0.5, 150),
        'y_pred': np.random.normal(0, 0.4, 150)
    })
    test_preds.to_csv(experiments_dir / "test_preds.csv", index=False)
    
    print("   ✅ 结构预测数据已保存:")
    print(f"      - {experiments_dir / 'train_preds.csv'}")
    print(f"      - {experiments_dir / 'val_preds.csv'}")
    print(f"      - {experiments_dir / 'test_preds.csv'}")

def run_fusion_example():
    """运行融合示例"""
    print("\n🚀 运行结构+表格晚融合示例...")
    
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
        "--epochs", "50",  # 减少epoch用于快速测试
        "--batch_size", "128",
        "--seed", "42",
        "--extra_note", "demo"
    ]
    
    print(f"📋 运行命令:")
    print(f"   {' '.join(cmd)}")
    
    # 运行命令
    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ 融合训练成功完成!")
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
    else:
        print("❌ 融合训练失败!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

def show_usage_examples():
    """显示使用示例"""
    print("\n📖 使用示例:")
    print("=" * 60)
    
    print("\n1️⃣ 训练+测试融合模型（晚融合）:")
    print("""
python scripts/train_test_tabular_fusion.py \\
  --mode train_test \\
  --struct_train_cmd "" \\
  --struct_pred_train experiments/struct_preds/train_preds.csv \\
  --struct_pred_val   experiments/struct_preds/val_preds.csv \\
  --struct_pred_test  experiments/struct_preds/test_preds.csv \\
  --tab_csv data/processed/cleaned_data.csv \\
  --tab_cols "" \\
  --tab_id_col "" \\
  --fusion concat \\
  --epochs 100 --batch_size 256 --seed 0
""")
    
    print("\n2️⃣ 仅测试（已训练好 fusion_ckpt.pt）:")
    print("""
python scripts/train_test_tabular_fusion.py \\
  --mode test \\
  --struct_pred_train experiments/struct_preds/train_preds.csv \\
  --struct_pred_val   experiments/struct_preds/val_preds.csv \\
  --struct_pred_test  experiments/struct_preds/test_preds.csv \\
  --tab_csv data/processed/cleaned_data.csv \\
  --fusion concat \\
  --mc_dropout_T 20 \\
  --save_dir experiments/2025-10-21_tabfusion_run
""")
    
    print("\n3️⃣ 使用门控融合:")
    print("""
python scripts/train_test_tabular_fusion.py \\
  --mode train_test \\
  --struct_pred_train experiments/struct_preds/train_preds.csv \\
  --struct_pred_val   experiments/struct_preds/val_preds.csv \\
  --struct_pred_test  experiments/struct_preds/test_preds.csv \\
  --tab_csv data/processed/cleaned_data.csv \\
  --fusion gate \\
  --epochs 100 --batch_size 256
""")

def main():
    """主函数"""
    print("🎯 结构+表格晚融合示例")
    print("=" * 60)
    
    try:
        # 1. 创建示例数据
        create_sample_data()
        
        # 2. 运行融合示例
        run_fusion_example()
        
        # 3. 显示使用示例
        show_usage_examples()
        
        print("\n🎉 示例完成!")
        print("💡 提示: 这个脚本完全保留原始训练脚本，通过晚融合方式实现结构+表格特征结合")
        
    except Exception as e:
        print(f"\n❌ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
