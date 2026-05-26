
"""
自定义氢吸附能量预测模型训练脚本
"""

import os
import sys
import torch
import yaml
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
import datasets.custom_dataset
import nets
import oc20.trainer

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import setup_logging, build_config
from main_oc20 import Runner


class CustomHydrogenTrainer:
    """自定义氢吸附训练器"""
    
    def __init__(self, config_path: str, run_dir: str = "./runs"):
        """
        Args:
            config_path: 配置文件路径
            run_dir: 运行目录
        """
        self.config_path = Path(config_path)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 设置基本参数
        self.config.update({
            "mode": "train",
            "identifier": "custom_hydrogen",
            "run_dir": str(self.run_dir),
            "is_debug": False,
            "distributed": False,
            "local_rank": 0,
            "seed": 42,
            "amp": False,
            "cpu": False,
            "noddp": True
        })
        
        print("Configuration loaded:")
        print(yaml.dump(self.config, default_flow_style=False))

    def setup_trainer(self):
        """初始化训练器"""
        setup_logging()
        
        # 创建训练器
        self.trainer = registry.get_trainer_class(
            self.config.get("trainer", "energy_v2")
        )(
            task=self.config["task"],
            model=self.config["model"],
            dataset=self.config["dataset"],
            optimizer=self.config["optim"],
            identifier=self.config["identifier"],
            run_dir=self.config["run_dir"],
            is_debug=self.config["is_debug"],
            seed=self.config["seed"],
            logger=self.config.get("logger", "tensorboard"),
            local_rank=self.config["local_rank"],
            amp=self.config["amp"],
            cpu=self.config["cpu"],
            noddp=self.config["noddp"],
        )
        
        print(f"Trainer setup complete: {self.trainer.__class__.__name__}")
        
        # 设置任务
        self.task = registry.get_task_class(self.config["mode"])(self.config)
        self.task.setup(self.trainer)

    def train(self):
        """执行训练"""
        print("=" * 50)
        print("Starting training...")
        print("=" * 50)
        
        try:
            self.task.run()
            print("Training completed successfully!")
        except Exception as e:
            print(f"Training failed with error: {e}")
            raise

    def evaluate_model(self, checkpoint_path: str = None):
        """评估模型性能"""
        if checkpoint_path:
            self.trainer.load_checkpoint(checkpoint_path)
        
        print("=" * 50)
        print("Evaluating model...")
        print("=" * 50)
        
        # 在验证集上评估
        if hasattr(self.trainer, 'val_loader') and self.trainer.val_loader:
            predictions = self.trainer.predict(
                self.trainer.val_loader,
                results_file=None,
                disable_tqdm=False
            )
            
            # 计算指标
            y_true = predictions['targets']
            y_pred = predictions['predictions']
            
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            print(f"Validation Results:")
            print(f"MAE: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R²: {r2:.4f}")
            
            if r2 >= 0.93:
                print("🎉 Congratulations! R² >= 0.93 achieved!")
            else:
                print(f"R² = {r2:.4f} < 0.93. Consider:")
                print("1. Increasing model complexity")
                print("2. Training for more epochs")
                print("3. Adjusting learning rate")
                print("4. Adding regularization")
                print("5. Feature engineering")
            
            return {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred,
                'targets': y_true
            }

    def save_predictions(self, results: dict, output_path: str):
        """保存预测结果"""
        import pandas as pd
        
        df = pd.DataFrame({
            'targets': results['targets'],
            'predictions': results['predictions'],
            'residuals': results['targets'] - results['predictions']
        })
        
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    def run_full_pipeline(self, evaluate: bool = True):
        """运行完整训练流程"""
        # 1. 设置训练器
        self.setup_trainer()
        
        # 2. 训练模型
        self.train()
        
        # 3. 评估模型（可选）
        if evaluate:
            results = self.evaluate_model()
            
            # 保存预测结果
            if results:
                predictions_path = self.run_dir / "predictions.csv"
                self.save_predictions(results, str(predictions_path))
            
            return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Train custom hydrogen adsorption model")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to configuration file")
    parser.add_argument("--run-dir", type=str, default="./runs",
                       help="Directory to save runs")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate model after training")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint for evaluation")
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = CustomHydrogenTrainer(args.config, args.run_dir)
    
    if args.checkpoint:
        # 只评估
        trainer.setup_trainer()
        results = trainer.evaluate_model(args.checkpoint)
    else:
        # 训练+评估
        results = trainer.run_full_pipeline(args.evaluate)
    
    print("Training/Evaluation pipeline completed!")


if __name__ == "__main__":
    main() 