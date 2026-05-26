---
name: thesis-revision-todo
description: 论文大修执行计划TODO清单，基于答辩意见和大修执行计划
metadata:
  node_type: memory
  type: project
  date: 2026-05-26
  originSessionId: f8223fbb-6c00-44dd-bce1-1075a23262bb
---

# 论文大修 TODO 清单

## 优先级说明
- **P0**：必须修改，否则影响论文主线逻辑
- **P1**：重要修改，有明确评审意见
- **P2**：格式、附录、参考文献问题

---

## P0：论文主线与逻辑问题

- [ ] **数据口径核验**：核验 `.vasp` 结构中真实元素、样本划分(8301/1037/1039)、特征维度、图构建参数、DFT标签来源
  - 输出：`outputs/thesis-revision/data_audit/`
  - 影响：摘要、第1-4章、第6章

- [ ] **第1章重写**：统一术语(Eqv2-Lite/HEA-Gen/MatGen-Eq)，删除重复句，把"发现新材料"改为"候选结构生成与预筛选"

- [ ] **第3章 Eqv2-Lite剪枝实验支撑不足**：
  - [x] 已完成：lmax/depth/channels/参数重分配/gaussians 单因素消融实验
  - [x] 已完成：lmax×radius 交叉验证实验（发现lmax=3+radius=16.0最优，R²=0.677）
  - [ ] 核验第3章现有实验口径与真实数据是否一致
  - [ ] 补充"多因素耦合实验"章节，说明参数交互影响

- [ ] **第4章 HEA-Gen可合成性说明**：
  - [ ] 补充数据量（训练/验证/测试样本数）
  - [ ] 补充训练环境（Qwen2.5-7B + LoRA）
  - [ ] 补充时间复杂度分析
  - [ ] 明确局限性：未建模可合成性、未做VASP复算

- [ ] **第5章系统章节降调**：
  - [x] 已修改代码：rejection_level追踪（quality.py/db.py/app.py）
  - [ ] 删除或降级高风险表述（见下方清单）
  - [ ] 改标题为"候选结构生成与计算筛选原型系统"
  - [ ] 候选材料用编号展示，不公开完整配方

- [ ] **第6章总结重写**：
  - [ ] 明确写出口数据集边界、DFT噪声、可合成性不足、无新候选VASP复算

---

## P1：格式与附录问题

- [ ] **LaTeX编译结构修复**：`main.tex` 中 `\include{03_behindPart/appendix_C}` 引用不存在的文件
- [ ] **附录C处理**：要么删除引用，要么新建并补充内容
- [ ] **附录B简化**：删除过长prompt模板，脱敏候选结构示例
- [ ] **附录A统一**：训练环境、RAG索引、模型参数一致
- [ ] **成果页 `achievement.tex`**：改为"候选结构生成与计算筛选原型系统"
- [ ] **参考文献格式**：DOI字段不能写成URL格式

---

## P2：候选材料展示原则

需修改位置：第5章系统案例、第6章总结展望、附录B

展示内容：
- 候选编号：`HEA-HER-01`、`Candidate-01`、`M-01`
- 预测ΔG_H值
- 与目标值偏差
- 格式合规性
- 结构合理性
- 筛选状态

禁止展示：
- 完整元素配方
- 完整POSCAR
- 可直接复现的候选结构细节

---

## 大修执行顺序（推荐）

1. 备份LaTeX源文件和计划文档
2. 修复 `main.tex` 附录C引用问题（保证编译通过）
3. **数据口径核验**（最重要，影响所有章节）
4. 对照审计结果重写摘要、第1-2章数据口径
5. 核验HEA-Gen任务形态，重写第4章
6. 核验MatGen-Eq代码实现，重写第5章（原型系统，不写未实现功能）
7. 补充Eqv2-Lite交叉实验章节
8. 重写第6章，把局限写在展望前面
9. 修改附录、成果页、参考文献
10. 编译PDF，按答辩意见做最终排版复查

---

## 第5章需删除/降级的高风险表述

| 原表述 | 降级后 |
|--------|--------|
| 端到端自动化材料发现系统 | 候选结构生成与计算筛选原型系统 |
| 自动回流训练数据库并驱动模型增量迭代 | 预留数据回流机制接口 |
| 100个候选结构端到端处理不超过30分钟 | 需补充实验数据 |
| TorchScript编译优化 | 预留优化接口 |
| 显存动态调节 | 预留显存自适应接口 |
| 单张NVIDIA RTX 4080 Super GPU稳定运行完整端到端流程 | 需实测验证 |
| 输出完整POSCAR文本内容 | 候选结构可导出为POSCAR格式 |
| 发现新材料 | 生成候选结构，进行预筛选 |

---

## 论文文件位置

| 文件 | 路径 |
|------|------|
| 论文主文件 | `MS_1/main.tex` |
| 中文摘要 | `MS_1/01_frontPart/cAbstract.tex` |
| 英文摘要 | `MS_1/01_frontPart/eAbstract.tex` |
| 第1-6章 | `MS_1/02_bodyPart/chapter_01~06.tex` |
| 附录 | `MS_1/03_behindPart/` |
| 参考文献 | `MS_1/references/` |

---

## 记忆文档索引

- [Reviewer 2 Comments](reviewer2_comments.md)
- [Writing Pitfalls](writing_pitfalls.md)
- [swjtuThesis V3.0 模板格式规范](swjtuthesis_template_v3.md)
- [Eqv2-Lite 消融实验结果](eqv2-lite-ablation-results.md)
- [MatGen-Eq 系统说明](matgen-system.md)
- [本文档：论文大修TODO](thesis-revision-todo.md)
