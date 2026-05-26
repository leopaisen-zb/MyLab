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

- [x] **第1章重写**：已完成（术语统一、删除重复、修正夸大表述）

- [ ] **第3章 Eqv2-Lite交叉验证实验**：
  - [x] 单因素消融实验（已完成）
  - [x] lmax×radius 交叉验证第一版（有缺陷，num_layers=6）
  - [ ] **lmax×radius 交叉验证修正版（num_layers=3固定）**
    - 脚本已创建：`run_lmax_radius_cross_ablation_fixed.py`
    - 训练参数：30 epochs, batch_size=16, patience=10
    - 6个配置，CPU训练，预计 2-3 小时
    - 待执行
  - [ ] 核验第3章现有实验口径与真实数据是否一致
  - [ ] 补充"多因素耦合实验"章节（基于修正版结果）

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

展示内容：候选编号、预测ΔG_H值、与目标值偏差、格式合规性、结构合理性、筛选状态

禁止展示：完整元素配方、完整POSCAR、可直接复现的候选结构细节

---

## 大修执行顺序（推荐）

1. 备份LaTeX源文件和计划文档
2. 修复 `main.tex` 附录C引用问题（保证编译通过）
3. **数据口径核验**（最重要，影响所有章节）
4. 对照审计结果重写摘要、第1-2章数据口径
5. **运行 Eqv2-Lite CPU 交叉消融实验**（最高优先级）
6. 核验HEA-Gen任务形态，重写第4章
7. 核验MatGen-Eq代码实现，重写第5章（原型系统，不写未实现功能）
8. 补充Eqv2-Lite交叉实验章节（第3章新增一节）
9. 重写第6章，把局限写在展望前面
10. 修改附录、成果页、参考文献
11. 编译PDF，按答辩意见做最终排版复查

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

## GitHub 同步（2026-05-26）

仓库：https://github.com/leopaisen-zb/MyLab

已上传全部论文源码、记忆文档、Claude Code skills、equiformer代码。
训练数据（LMDB）未上传（>100MB），需在新设备上重新生成或从备份恢复。

---

## 记忆文档索引

- [Reviewer 2 Comments](reviewer2_comments.md)
- [Writing Pitfalls](writing_pitfalls.md)
- [swjtuThesis V3.0 模板格式规范](swjtuthesis_template_v3.md)
- [Eqv2-Lite 消融实验结果](eqv2-lite-ablation-results.md)
- [MatGen-Eq 系统说明](matgen-system.md)
- [本文档：论文大修TODO](thesis-revision-todo.md)
