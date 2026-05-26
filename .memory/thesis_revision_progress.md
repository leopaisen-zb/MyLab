---
name: thesis-revision-progress
description: 论文大修进度记录（2026-05-26），包含摘要和第1章修改详情
metadata:
  node_type: memory
  type: project
  date: 2026-05-26
  originSessionId: be2d39b7-049b-4191-a389-669428f215d7
---

# 论文大修进度记录

## 已完成修改

### 摘要修改（2026-05-26）

**文件**：
- `MS_1/01_frontPart/cAbstract.tex`
- `MS_1/01_frontPart/eAbstract.tex`

**修改内容**：

1. **术语统一**：将"氢吸附自由能（ΔGH）"改为"催化性能数据/性能"，与标题"性能预测"对齐
2. **删除括号解释**：删除"（即最优催化性能）"等冗余括号
3. **夸大表述修正**：
   - "完整闭环工作流" → "工作流程"
   - "大幅降低准入门槛" → 删除
   - "实际工程场景" → "实际筛选场景"
4. **关键词格式**：按swjtuThesis V3.0模板，逗号分隔
5. **英文语法修复**：`closed-loop alignment with optimal...` → `closed-loop alignment, with optimal...`
6. **去除"不仅...而且"夸大结构**

---

### 第1章修改（2026-05-26）

**文件**：`MS_1/02_bodyPart/chapter_01.tex`

**修改内容**：

1. **术语统一（8处）**：统一为"性能预测/评估"
2. **删除重复表述**：删除第82-84行重复内容
3. **"发现新材料"类表述修正（4处）**：改为"筛选"/"候选结构生成与预筛选"
4. **系统能力边界夸大表述修正（3处）**：改为"原型验证"/"原型系统"

---

## 当前状态（2026-05-26）

### GitHub 仓库已同步
- 仓库：https://github.com/leopaisen-zb/MyLab
- 包含：论文源码、记忆文档、Claude Code skills、equiformer代码

### Eqv2-Lite 消融实验进展
- ✅ 单因素消融实验全部完成（lmax/depth/channels/参数重分配/gaussians）
- ⚠️ lmax×radius 交叉验证（第一版）有缺陷（num_layers=6 而非最优的3）
- 📝 修正版脚本已创建：`run_lmax_radius_cross_ablation_fixed.py`（num_layers=3固定）
- ⏳ 待执行：CPU 训练（当前设备仅有 CPU，无 GPU）

### 消融实验脚本
- `equiformer/scripts/run_lmax_radius_cross_ablation_fixed.py`
  - 6个配置：lmax=[3,4] × radius=[8.0,12.0,16.0]，num_layers=3固定
  - CPU 训练参数：30 epochs, patience=10
  - 预计每配置 20-40 分钟（CPU），总计约 2-3 小时

---

## P0级别（答辩核心意见）

1. **消融实验** ⚠️ 交叉实验有缺陷，修正版待运行
   - 第一版交叉实验用 num_layers=6（与最优3不一致），结论不可比
   - 修正版脚本已就绪，待 CPU 训练

2. **第4章 HEA-Gen 可合成性说明** — 待补充

3. **第5章 MatGen-Eq 可扩展性** — 待按高风险表述清单降调

4. **数据集局限 + DFT噪声未量化** — 待补充

5. **闭环未与 VASP 深度联动** — 已通过措辞修改规避

6. **第4章实验细节缺失** — 待补充

7. **第5章与第3/4章逻辑关系** — 待明确说明

---

## academic-paper-reviewer 评审结果

**第一章评审得分**：加权总分 70.3（Minor Revision区间65-79）
**Decision**: Minor Revision

---

## 下一步工作

1. **运行 Eqv2-Lite CPU 交叉消融实验**（最高优先级）
2. **第3章补充交叉验证章节**（基于新实验结果）
3. **第2章**：理论基础与相关技术（检查术语统一）
4. **第4章**：HEA-Gen（补充可合成性说明）
5. **第5章**：MatGen-Eq（降调系统能力边界）
6. **第6章**：总结（明确局限）
7. **附录、参考文献、成果页**

---

## 记忆文档索引

- [thesis-revision-todo.md](thesis-revision-todo.md) — 论文大修TODO清单
- [defense_comments.md](defense_comments.md) — 答辩意见汇总
- [reviewer2_comments.md](reviewer2_comments.md) — 评阅人二意见
- [eqv2-lite-ablation-results.md](eqv2-lite-ablation-results.md) — Eqv2-Lite消融实验结果
- [writing_pitfalls.md](writing_pitfalls.md) — 论文写作踩坑总结
- [swjtuthesis_template_v3.md](swjtuthesis_template_v3.md) — 模板格式规范
- [matgen-system.md](matgen-system.md) — MatGen-Eq系统说明
