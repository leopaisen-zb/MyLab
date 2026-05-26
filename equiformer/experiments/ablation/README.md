# EquiformerV2 最小可复现实验脚手架

这是一个用于EquiformerV2模型消融实验的完整脚手架，支持批量运行、结果收集和可视化分析。

## 📁 目录结构

```
experiments/ablation/
├── grids/                    # 实验网格CSV文件
│   └── demo.csv             # 示例实验配置
├── plots/                   # 生成的图表
├── results_demo.csv         # demo结果（运行后生成）
├── run_exp.py              # 统一批跑入口
└── analysis/               # 分析工具
    ├── plot_utils.py       # 绘图工具
    ├── lmax_report.py      # Lmax分析报告
    ├── module_report.py    # 模块消融分析
    ├── escn_report.py      # eSCN分析
    ├── capacity_report.py  # 容量分析
    ├── neigh_report.py     # 邻居数分析
    ├── datasize_report.py  # 数据量分析
    ├── pretrain_report.py  # 预训练分析
    └── summary.py          # 综合分析
```

## 🚀 快速开始

### 1. 准备实验网格

创建CSV文件定义实验配置，例如 `grids/demo.csv`:

```csv
num_layers,sphere_channels,num_heads,grid_resolution,edge_channels,eSCN,attn_renorm,sep_s2,sep_ln,data_ratio,pretrained
2,64,4,8,64,False,False,False,False,1.0,False
2,64,4,12,64,False,False,False,False,1.0,False
3,64,4,8,64,False,False,False,False,1.0,False
3,64,4,12,64,False,False,False,False,1.0,False
```

### 2. 运行批量实验

```bash
python experiments/ablation/run_exp.py \
    --grid_csv experiments/ablation/grids/demo.csv \
    --repeat_seeds 2 \
    --output_csv experiments/ablation/results_demo.csv \
    --exp_tag DEMO
```

### 3. 生成分析报告

```bash
# 综合分析
python experiments/ablation/analysis/summary.py \
    --results_csv experiments/ablation/results_demo.csv \
    --output_dir experiments/ablation/plots/

# 特定分析
python experiments/ablation/analysis/capacity_report.py \
    --results_csv experiments/ablation/results_demo.csv \
    --output_dir experiments/ablation/plots/
```

## 📊 支持的参数

### 模型结构参数
- `num_layers`: Transformer层数
- `sphere_channels`: 球面通道数
- `num_heads`: 注意力头数
- `grid_resolution`: 网格分辨率
- `edge_channels`: 边通道数
- `attn_hidden_channels`: 注意力隐藏通道数
- `attn_alpha_channels`: 注意力alpha通道数
- `attn_value_channels`: 注意力value通道数
- `ffn_hidden_channels`: 前馈网络隐藏通道数

### 实验参数
- `eSCN`: 是否使用eSCN模块
- `attn_renorm`: 注意力重归一化
- `sep_s2`: 分离SO(2)操作
- `sep_ln`: 分离层归一化
- `data_ratio`: 数据使用比例
- `pretrained`: 是否使用预训练
- `split_file`: 数据分割文件
- `pretrained_ckpt`: 预训练检查点

### 训练参数
- `batch_size`: 批次大小
- `lr`: 学习率
- `weight_decay`: 权重衰减
- `seed`: 随机种子

## 📈 输出结果

### 实验指标
- `test_mae`: 测试MAE (eV)
- `test_rmse`: 测试RMSE (eV)
- `test_loss`: 测试损失
- `params`: 模型参数量
- `latency_ms`: 推理延迟 (ms)
- `throughput`: 吞吐量
- `seed`: 随机种子

### 生成文件
- `results_*.csv`: 实验结果汇总
- `*_analysis.png`: 各种分析图表
- `*_stats.csv`: 统计报告
- `summary_report.html`: HTML格式综合报告

## 🔧 高级用法

### 自定义实验网格

```python
import pandas as pd

# 创建参数网格
configs = []
for num_layers in [2, 3, 4]:
    for sphere_channels in [64, 128]:
        for grid_resolution in [8, 12, 16]:
            configs.append({
                'num_layers': num_layers,
                'sphere_channels': sphere_channels,
                'grid_resolution': grid_resolution,
                'num_heads': 4,
                'edge_channels': 64,
                'eSCN': False,
                'attn_renorm': False,
                'sep_s2': False,
                'sep_ln': False,
                'data_ratio': 1.0,
                'pretrained': False
            })

# 保存为CSV
df = pd.DataFrame(configs)
df.to_csv('experiments/ablation/grids/custom_grid.csv', index=False)
```

### 并行运行

```bash
# 使用GNU parallel (Linux/Mac)
parallel -j 4 python experiments/ablation/run_exp.py \
    --grid_csv experiments/ablation/grids/demo.csv \
    --repeat_seeds 1 \
    --output_csv experiments/ablation/results_{}.csv \
    --exp_tag DEMO_{} ::: {1..4}
```

## 🐛 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少 `batch_size`
   - 减少 `num_layers` 或 `sphere_channels`

2. **实验失败**
   - 检查 `error` 列查看具体错误
   - 确保数据路径正确
   - 检查依赖包版本

3. **结果为空**
   - 确认实验配置正确
   - 检查训练脚本是否正常返回结果

### 调试模式

```bash
# 运行单个实验进行调试
python src/train_enhanced_equiformer_v2.py \
    --num_layers 2 \
    --sphere_channels 64 \
    --num_heads 4 \
    --grid_resolution 12 \
    --edge_channels 64 \
    --seed 0 \
    --exp_name DEBUG_TEST
```

## 📝 扩展功能

### 添加新的分析报告

1. 在 `analysis/` 目录创建新的报告脚本
2. 使用 `plot_utils.py` 中的绘图函数
3. 参考现有报告脚本的格式

### 自定义绘图

```python
from analysis.plot_utils import draw_parameter_sweep

# 绘制自定义参数扫描图
draw_parameter_sweep(df, 'custom_plot.png', 'your_parameter')
```

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个脚手架！

## 📄 许可证

本项目遵循与EquiformerV2相同的许可证。
