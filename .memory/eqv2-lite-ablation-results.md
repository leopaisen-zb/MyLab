---
name: eqv2-lite-ablation-results
description: Eqv2-Lite轻量化模型消融实验结果，含单因素实验和交叉验证实验
metadata:
  node_type: memory
  type: project
  date: 2026-05-26
  originSessionId: f8223fbb-6c00-44dd-bce1-1075a23262bb
---

# Eqv2-Lite 消融实验结果

## 1. 单因素消融实验（已完成）

存储位置：`equiformer/result/eq_lite_ablation/`

### lmax（球谐展开阶数）
| lmax | R² | MAE | RMSE | 参数量 |
|------|-----|------|------|--------|
| 3 | 0.909 | 0.104 | 0.211 | 13.0M |
| **4** | **0.928** | **0.100** | **0.187** | 19.5M |
| 5 | 0.912 | 0.108 | 0.207 | 27.5M |
| 6 | 0.910 | 0.114 | 0.209 | 36.9M |

**结论**：lmax=4 最优

### num_layers（网络深度）
| layers | R² | MAE | RMSE | 参数量 |
|--------|-----|------|------|--------|
| 1 | 0.905 | 0.104 | 0.214 | 3.6M |
| 2 | 0.917 | 0.102 | 0.200 | 6.8M |
| **3** | **0.923** | **0.094** | **0.194** | 9.9M |
| 4 | 0.910 | 0.098 | 0.209 | 13.1M |

**结论**：layers=3 最优

### sphere_channels（通道维度）
| channels | R² | MAE | RMSE | 参数量 |
|-----------|-----|------|------|--------|
| 32 | 0.912 | 0.102 | 0.207 | 8.1M |
| 64 | 0.921 | 0.098 | 0.196 | 11.9M |
| **128** | **0.935** | **0.089** | **0.178** | 19.5M |

**结论**：128 channels 最优

### 参数重分配策略
| 策略 | R² | MAE | RMSE | 参数量 |
|------|-----|------|------|--------|
| ffn_heavy | 0.920 | 0.099 | 0.197 | 5.2M |
| ffn_light | 0.926 | 0.104 | 0.189 | 4.5M |
| attention_heavy | 0.918 | 0.102 | 0.199 | 7.8M |
| **edge_heavy** | **0.932** | **0.096** | **0.181** | 5.8M |
| radial_heavy | 0.896 | 0.113 | 0.225 | 5.0M |
| balanced | 0.917 | 0.105 | 0.201 | 6.8M |

**结论**：edge_heavy（边特征增强）最优

### 径向基函数数量
| gaussians | R² | MAE | RMSE |
|-----------|-----|------|------|
| 32 | 0.923 | 0.102 | 0.194 |
| 64 | 0.897 | 0.113 | 0.224 |
| 128 | 0.918 | 0.097 | 0.200 |
| **256** | **0.927** | **0.095** | **0.188** |

**结论**：256 gaussians 最优

### 最优单因素配置
- lmax=4, layers=3, sphere_channels=128, edge_heavy, 256 gaussians
- 参数量约 4.69M，MAE=0.0868 eV，R²=0.9334

---

## 2. 交叉验证实验（lmax × radius，第一版有缺陷，2026-05-25完成）

**重要说明**：第一版交叉实验使用了 num_layers=6，与单因素实验的最优配置（num_layers=3）不一致，
导致结果不可直接比较。该版本结果记录于此，但**不能**作为最终结论。

存储位置：`equiformer/result/eq_lite_ablation/cross_lmax_radius/`

| 配置 | lmax | radius | R² | MAE | RMSE | 参数量 |
|------|------|--------|-----|------|------|--------|
| 1 | 3 | 8.0 | 0.572 | 0.274 | 0.431 | 13.0M |
| 2 | 3 | 12.0 | 0.459 | 0.311 | 0.485 | 13.0M |
| **3** | **3** | **16.0** | **0.677** | **0.260** | **0.374** | 13.0M |
| 4 | 4 | 8.0 | 0.583 | 0.267 | 0.425 | 19.5M |
| 5 | 4 | 12.0 | 0.499 | 0.296 | 0.466 | 19.5M |
| 6 | 4 | 16.0 | 0.591 | 0.279 | 0.421 | 19.5M |

**注意**：此版本 R² 显著低于单因素最优（R²=0.933），原因：使用了 num_layers=6 而非最优的 3

---

## 3. 交叉验证实验（lmax × radius，第二版修正，num_layers=3 固定）

**脚本**：`equiformer/scripts/run_lmax_radius_cross_ablation_fixed.py`

**配置**：
- 固定参数：num_layers=3, sphere_channels=128, num_heads=4, ffn_hidden_channels=128,
  edge_channels=128, num_gaussians=256, max_neighbors=20, batch_size=16, lr=0.0002
- 变量：lmax=[3,4], max_radius=[8.0, 12.0, 16.0]
- 总配置数：6
- 数据：LMDB（equiformer/datasets/custom_hydrogen/）

**训练参数（CPU版）**：
- num_epochs: 30（原100，减少以适应CPU训练时间）
- patience: 10（原20）

**状态**：脚本已创建，等待 CPU 训练执行

**预期结果**：验证 lmax 和 max_radius 之间是否存在显著交互效应

---

## 4. MatGen-Eq系统实验数据

存储位置：`matgen_app/results.db`（SQLite）

### 现有数据统计（11条记录）
- 总样本数：11
- 生成成功率：100%
- 通过初筛率：72.7%
- 平均ΔG_H：-0.082 eV
- ΔG_H范围：[-0.452, 0.618] eV

### 已补充图表
- `MS_1/figures/dg_h_distribution.png`（ΔG_H分布直方图）

### 优先级3改动（rejection_level追踪）
- `backend/quality.py`：新增 `rejection_level` 返回值（pass/text/structure/physics）
- `backend/db.py`：新增 `rejection_level` 字段
- `app.py`：Tab3批处理时记录每条样本的拒绝级别

---

## 5. GitHub 同步（2026-05-26）

仓库：https://github.com/leopaisen-zb/MyLab

**已上传**：
- MS_1/：论文 LaTeX 源码
- .memory/：记忆文档（8个）
- .claude/skills/：academic-paper 等 skill
- equiformer/：模型代码 + 消融结果
- RAG/：检索系统代码
- matgen_app/：Web 应用
- mylab/, 实验结果/, 答辩材料/

**已排除**：
- .conda 环境（21GB）
- RAG checkpoints（13GB）
- LMDB .mdb 数据文件（>100MB）
- everything-claude-code-zh/
- Zone.Identifier 流文件
- settings.json（包含本机路径）
