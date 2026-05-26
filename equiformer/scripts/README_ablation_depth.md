# Eqv2-Lite Depth 消融实验

## 脚本说明

`run_eq_lite_ablation_depth.py` - 运行 StandaloneEquiformerV2 的层数（depth）消融实验

## 功能

- 测试不同层数 `num_layers = [1, 2, 3, 4]` 对模型性能的影响
- 每个模型训练 50 个 epochs
- 自动保存实验结果到指定目录

## 使用方法

```bash
cd D:\mylab\equiformer
python scripts/run_eq_lite_ablation_depth.py
```

## 输出结构

```
result/eq_lite_ablation/depth/
├── layers_1/
│   ├── metrics.json          # 评估指标
│   ├── predictions.csv       # 预测结果
│   └── loss_curve.png       # 损失曲线图
├── layers_2/
│   ├── metrics.json
│   ├── predictions.csv
│   └── loss_curve.png
├── layers_3/
│   ├── metrics.json
│   ├── predictions.csv
│   └── loss_curve.png
├── layers_4/
│   ├── metrics.json
│   ├── predictions.csv
│   └── loss_curve.png
└── summary.csv              # 所有实验的汇总结果
```

## 输出文件说明

### metrics.json
包含以下指标：
- `num_layers`: 模型层数
- `r2`: R² 分数
- `mae`: 平均绝对误差
- `rmse`: 均方根误差
- `loss`: 验证损失
- `num_parameters`: 模型参数量
- `config`: 训练配置

### predictions.csv
包含三列：
- `targets`: 真实值
- `predictions`: 预测值
- `residuals`: 残差（真实值 - 预测值）

### loss_curve.png
包含两个子图：
- 训练和验证损失曲线
- 验证集 R² 分数曲线

### summary.csv
所有实验的汇总表格，便于对比分析

## 实验配置

- **数据集**: `datasets/custom_hydrogen`
- **训练轮数**: 50 epochs
- **批次大小**: 6
- **学习率**: 0.0003
- **权重衰减**: 1e-4
- **模型**: StandaloneEquiformerV2

## 注意事项

- 确保 GPU 可用（如果使用 CPU 训练会很慢）
- 每个实验会依次运行，总时间约为 4 × 50 epochs
- 如果某个实验失败，会跳过并继续下一个

