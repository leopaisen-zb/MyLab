# 开发范式：数据完整性优先（Data-Integrity-First）

> 起因：`elements` 字段是占位假串、`predict` fallback 返回与结构无关常数——
> 这两个 bug 逻辑 review + 单元测试全没拦住，最后靠"跑系统肉眼看输出"才发现。
> 本范式把那条"肉眼核对"固化成可执行规则，使这类错误从源头不再发生。

## 这类 bug 的共性（要警惕的信号）
- **非关键路径**：显示字段、元数据、fallback/降级路径——错了不影响功能跑通，最易被忽视。
- **看着合理**：`Ir1Pd4Pt4` 是合法组成串、`-0.4` 是合理 ΔG_H——扫一眼过得去，只有跨字段对照才露馅。
- **被搬运的旧代码**：重构时原样复制的占位逻辑，diff review 只看"新改动"会放过。
- **被另一个 bug 掩盖**：假绿测试因双重巧合通过（如 `"Ir2Pd2"` 无空格解析失败 == 空组成）。

## 五条规则（每次开发都遵守）

1. **单一真相源（Single Source of Truth）**
   任何展示/派生/存储字段必须**从规范数据源计算得来**（如 elements 从 POSCAR 提取），
   绝不独立编造或重复维护。禁止"装饰性占位值"。

2. **不变式测试跑真实输出（Invariant tests over real output）**
   凡是产生/变换数据记录的功能，必须配一个**跑通真实流水线**并断言**跨字段不变式**的测试
   （不是只测孤立函数逻辑）。见 `tests/test_data_consistency.py`，新增展示字段就在此补一条。

3. **fallback 必须诚实（Honest fallbacks）**
   降级/兜底路径要么仍**从真实输入派生**，要么是**显式哨兵值**；
   绝不静默返回一个"貌似真实却与输入无关"的常数。

4. **审查也要查被搬运的旧代码（Review the copied code）**
   重构/复制代码时，对**移动来的、预先存在的**行同样质疑正确性。
   "外科手术式改动"≠"信任旧代码"。多 agent review 应含一个**数据语义 lens**
   （专门核对输出自洽 + 审查 copied/pre-existing 代码），不止查 diff 逻辑。

5. **交付前跑系统看真实输出（Eyeball before delivery）**
   自动化测试 + diff review 对"输出语义自洽"有结构盲区。
   任何 demo/截图/交付前，**真跑一遍系统、核对一条真实记录端到端**。

## 接入 everything-claude 流程
- **TDD 阶段**：success criteria 里显式包含"数据一致性不变式测试"。
- **Code Review 阶段**：除逻辑/安全/设计 lens 外，固定加一个**数据语义一致性 reviewer**。
- **验证阶段**：交付前必跑 `pytest tests/test_data_consistency.py` + 跑 `scripts/demo_flywheel.py` 看输出。
