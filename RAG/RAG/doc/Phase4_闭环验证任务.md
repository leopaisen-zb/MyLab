# Phase 4 核心任务：闭环物理验证

## 目标概述

手握两个核心组件：
1. **刚微调好的 Qwen 生成器**：能根据性质描述生成 VASP 结构文本  
2. **第一章自己改的 Eqv2-Lite 评价器**：能读入 `.vasp` 文件并预测 \(\Delta G\)

本阶段的任务是让它们**会师**：写一个**闭环验证脚本**，用“生成 → 预测 → 对比”的方式验证**生成结构的物理合理性**。

---

## 核心脚本：`src/validate_physics.py`

### 1. 大规模生成（Generation）

- **输入**：测试集（与训练集切分时已确定，对 LLM 保密）
- **操作**：
  - 用微调后的 Qwen 模型在测试集上生成 **100 个（或更多）** VASP 结构文本
  - 将每条生成结果解析并保存为**真实的 `.vasp` 文件**（如 `gen_0.vasp`, `gen_1.vasp`, ...）
- **输出**：一个目录下的多份 `.vasp` 文件，与测试集样本一一对应（或按 index 编号）

### 2. 物理打分（Scoring with Eqv2-Lite）

- **输入**：上一步保存的 `.vasp` 文件
- **操作**：
  - 用你**第一章自己改的 Eqv2-Lite** 读取这些生成的 `.vasp` 文件
  - 对每个结构预测 **\(\Delta G_{pred}\)**（或你脚本里对应的输出名称）
- **输出**：每个生成结构对应的 \(\Delta G_{pred}\) 列表/表

### 3. 计算误差（Error Metrics）

- **目标值**：这 100 个材料在 **Excel 表里原本应有的 \(\Delta G_{target}\)**（即测试集对应的那一列，如 `reactionEn` / ΔG_H）
- **操作**：
  - 将 **\(\Delta G_{pred}\)** 与 **\(\Delta G_{target}\)** 一一对应
  - 计算 **MAE（平均绝对误差）**：  
    \(\text{MAE} = \frac{1}{N}\sum_{i=1}^{N} |\Delta G_{pred}^{(i)} - \Delta G_{target}^{(i)}|\)
- **输出**：MAE 数值（以及可选：逐条误差、RMSE 等）

### 4. 画图出结果（Visualization）

- **散点图**：
  - **X 轴**：目标性质 \(\Delta G_{target}\)（来自 Excel/测试集）
  - **Y 轴**：生成结构经 Eqv2-Lite 预测的性质 \(\Delta G_{pred}\)
- **理想情况**：点分布在 **y = x** 附近，表示“生成的结构其预测性质与目标性质一致”
- **图上标注**：在图中或图例中写出 **MAE**（平均绝对误差）
- **输出**：保存为图片文件（如 `doc/figures/validate_physics_scatter.png` 或项目内约定路径）

---

## 实现要点小结

| 步骤 | 输入 | 核心操作 | 输出 |
|------|------|----------|------|
| 1. 大规模生成 | 测试集 JSON + 微调 Qwen | 模型生成 → 解析为 POSCAR → 写 `.vasp` | `gen_0.vasp` … `gen_99.vasp`（或更多） |
| 2. 物理打分 | 生成的 `.vasp` 文件 | Eqv2-Lite 推理 → \(\Delta G_{pred}\) | 预测值列表/表 |
| 3. 计算误差 | \(\Delta G_{pred}\) + Excel/测试集 \(\Delta G_{target}\) | 对齐 id/index → 算 MAE | MAE（及可选指标） |
| 4. 画图 | 上述两列数据 | 散点图 X=target, Y=pred，标注 MAE | 散点图文件 |

---

## 与现有资源对接

- **测试集**：来自 `data/processed/dataset_test_rag.json`（或 `dataset_test.json`），需能拿到每条对应的 **index / id** 以及 **\(\Delta G_{target}\)**（若 JSON 里没有，需从原始 Excel 或 `cleaned_data.csv` 按 index 对齐）
- **微调模型**：`checkpoints/qwen_vasp_lora`，推理方式可参考 `eval_baseline.py`（Unsloth 4-bit 加载 + 生成）
- **Eqv2-Lite**：第一章已改好的脚本与权重路径，需在 `validate_physics.py` 中调用（读 `.vasp` → 输出 \(\Delta G_{pred}\)）
- **Excel/目标值**：确保测试集 100 条的 \(\Delta G_{target}\) 可从表或 CSV 中按同一 index 取到

---

## 成功标准（与 todo 一致）

- 预测值与目标值**高度相关**：散点图接近 **y = x**
- **MAE** 在可接受范围内（具体阈值可根据领域或基线设定）
- 脚本可复现：给定测试集与模型，能稳定跑通“生成 → Eqv2-Lite 预测 → 误差计算 → 出图”

完成本脚本后，即完成 **Phase 4：推理与闭环验证** 的核心实现。
