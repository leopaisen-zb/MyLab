---
name: matgen-system
description: MatGen-Eq系统架构与功能说明文档，供其他agent参考
metadata:
  node_type: memory
  type: project
  date: 2026-05-26
  originSessionId: f8223fbb-6c00-44dd-bce1-1075a23262bb
---

# MatGen-Eq 系统说明

## 系统位置

路径：`/Users/leo/Downloads/AI4S/mylab(1)/matgen_app/`

启动命令：
```bash
cd "/Users/leo/Downloads/AI4S/mylab(1)/matgen_app" && bash run.sh
```

访问地址：http://localhost:8501（默认端口8501）

---

## 系统架构

```
matgen_app/
├── app.py                    # Streamlit 主界面（3个Tab）
├── config.py                 # 全局配置（筛选阈值、模型路径等）
├── run.sh                    # 启动脚本
├── requirements.txt          # Python依赖
├── benchmark_batch.py        # 批量性能测试脚本（待运行）
├── benchmark_100_results.json
└── backend/
    ├── __init__.py
    ├── db.py                 # SQLite 持久化
    ├── rag_gen.py            # HEA-Gen LLM 生成（Qwen2.5-7B + LoRA）
    ├── eq_predict.py         # Eqv2-Lite 预测
    ├── quality.py            # 结构合理性校验（新增rejection_level）
    ├── cluster.py            # 候选结构聚类分析
    ├── compare.py            # 结构横向对比
    ├── rag_visual.py         # RAG 检索可视化
    └── stats.py              # 统计计算
```

---

## 核心数据流

```
用户输入 prompt
    ↓
HEA-Gen (LLM) → POSCAR 文本
    ↓
质量校验 (quality.py) → rejection_level 追踪
    ↓
Eqv2-Lite → ΔG_H 预测
    ↓
区间筛选 (filter_low/high)
    ↓
SQLite 持久化 (results.db)
    ↓
专家审查 (approved/rejected/validated)
    ↓
数据回流 (export_validated)
```

---

## 功能模块（Tab）

| Tab | 功能 | 核心文件 |
|-----|------|----------|
| Tab1 文本生成 POSCAR | 单条/多候选 RAG 生成，RAG 可视化 | `rag_gen.py`, `rag_visual.py` |
| Tab2 上传 POSCAR 预测 | Eqv2-Lite ΔG_H 推理，结构合理性校验 | `eq_predict.py`, `quality.py` |
| Tab3 闭环批处理 | 批量生成→预测→筛选→专家审查→数据回流 | `db.py`, `app.py` |

---

## 关键资源路径

| 资源 | 路径 |
|------|------|
| 结果数据库 | `matgen_app/results.db` |
| ΔG_H 分布图 | `MS_1/figures/dg_h_distribution.png` |
| LLM Checkpoint | `RAG/RAG/checkpoints/qwen_vasp_lora/checkpoint-3114` |
| Eqv2-Lite 模型 | `equiformer/checkpionts/best_standalone_equiformer_v2_model.pt` |

---

## rejection_level 追踪（已实现）

`quality.py` 中 `validate_structure()` 返回：
- `pass`：通过所有校验
- `text`：文本级格式错误
- `structure`：结构级解析失败（pymatgen）
- `physics`：物理级不合理（原子间距过近、晶格常数负值）

`db.py` 中 `batch_samples` 表新增 `rejection_level` 字段
`app.py` 中 Tab3 批处理时记录每条样本的拒绝级别

---

## 待补充实验

1. **100条批量性能基准**（需MPS/GPU环境）
   - 脚本：`benchmark_batch.py`
   - 测量：总耗时、各阶段耗时、吞吐量

2. **三级解析防呆统计**
   - 需新跑一批数据后才能有完整 rejection_level 分布
   - 当前11条数据均为 pass

---

## 与论文第5章的关系

MatGen-Eq 是论文第5章"大语言模型与等变网络协同的智能材料发现系统"的原型实现。

**注意**：论文中的系统能力主张需对照代码验证，未实现的功能不能写成已验证结论。高风险表述见 `thesis-revision-todo.md`。
