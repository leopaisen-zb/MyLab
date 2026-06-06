# MatGen-Eq 数据飞轮建设计划

> 目标：把 matgen_app 从"开环演示"升级为"闭环数据飞轮 + 多模型可切换"系统。
> 定调：**优先答辩可演示**——真实逻辑跑通、故事讲圆，重活（真 DFT、全自动重训）留接口。

## 飞轮闭环

```
①生成 → ②预测 → ③初筛 → ④DFT验证(回填) → ⑤研究员确认 → ⑥导出
  ↑                                                              ↓
  └── ⑧一键重训+注册新版本 ← ⑦数据集回灌 ←─────────────────────┘
       (生成/预测模型可切换)
```

## 范围决策（已与用户确认 2026-06-06）

| 环节 | 决策 | 含义 |
|------|------|------|
| ④ DFT | 留接口+人工回填 | 加 `dft_verified` 状态+字段+回填API/界面，**不接 VASP** |
| ⑧ 重训 | 导出+一键脚本+注册版本 | 包装 `equiformer/src/train_equiformer.py` 与 LoRA 脚本，半自动 |
| 多模型 | 只需切换使用 | 模型注册表+按名切换，**砍掉**自动评估/A-B 基准 |
| 目标 | 答辩可演示 | stub/半自动，架构留扩展点 |

---

## 现状关键事实（Research 结论）

- ①②已接通真实模型（`adapters/` → `backend/rag_gen` + `backend/eq_predict`，2026-06-06 完成）。
- **`backend/db.py` 是死代码**：除测试外无人 import。它有富状态机（8状态+`state_history`+`export_validated`），但没 `task_id`。
- **`core/persistence/state_store.py`（`StateStore`）是 live 路径**：orchestrator 用它，但缺状态历史/导出/DFT/版本字段。
- **`core/state_machine.py`（`StateTransition`）是 live 状态机**，已被 `test_state_machine.py` 覆盖；`backend/db.py` 里的 `VALID_TRANSITIONS` 是重复定义。
- `app.py` 只走 API → orchestrator → StateStore 一条线。

**统一策略**：以 `StateStore` 为唯一存储 + `core/state_machine.py` 为唯一状态机，把 `backend/db.py` 的飞轮能力并入 `StateStore`，废弃 `backend/db.py`。

---

## 任务分期

### P0 — 地基（~3.5 天）

**P0-1 统一 DB + 状态机**（本期实施，TDD）
- `StateStore` 新增：`state_history` 表、`record_state_change()`、`get_state_history()`、`export_validated()`、`get_state_distribution()`
- orchestrator 的 `_transition_state` 每次转移写入 `state_history`（traceability，论文卖点）
- 删除 `backend/db.py` 重复的 `VALID_TRANSITIONS`，统一引用 `core/state_machine.py`
- `backend/db.py` 标记为 legacy（确认无引用后移除）
- 文件：`persistence/state_store.py`、`core/task_orchestrator.py`、`tests/test_state_store.py`

**P0-2 模型注册表 + 切换**
- `config.py` 增加 `GEN_MODELS` / `PRED_MODELS` 注册表（id/name/path/loader）
- `GenerationRequest` 增加 `gen_model_id` / `pred_model_id`
- orchestrator 从注册表按 id 取 adapter（替代写死单例）
- 文件：`config.py`、`api/schemas.py`、`core/task_orchestrator.py`、`adapters/`

### P1 — 闭环（~3 天）

**P1-1 DFT 回填节点**
- `core/state_machine.py` 增加 `DFT_VERIFIED`（位于 `FILTERED_IN` → `VALIDATED` 之间）
- `StateStore` 增加 `dft_dg_h`/`dft_energy`/`dft_status` 字段
- API：`POST /structures/{id}/dft`（单条回填）+ 批量 CSV 导入
- `app.py`：DFT 录入界面

**P1-2 数据集回灌 + 索引重建**
- `export_validated()` → 追加进 `dataset_train_rag.json`
- 重建 `near_neighbor_indices.json` 脚本
- 文件：新增 `backend/dataset_feedback.py`

**P1-3 一键重训 + 版本注册**
- 包装 `train_equiformer.py` / LoRA 训练入口为 `scripts/retrain_*.sh`
- 训完把新 ckpt 注册回模型表（自动 version 号）
- 字段：`model_version` 记录到每条预测

### P2 — 收尾（~0.5 天）
- Streamlit「飞轮全景」监控页
- 一键跑通整圈的演示脚本

---

## 验证策略
- 每个任务 TDD：先写 `tests/test_*.py`（RED）→ 实现（GREEN）→ 全量 `pytest` 回归。
- 端到端：`MATGEN_DEMO=1` 跑通 ①→⑧ 整圈。
- 现有 99 个测试不得回归。
