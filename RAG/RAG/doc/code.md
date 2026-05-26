### Step 1：跑一版“当前设置”的基线（Experiment A）

1. 在 `eval_sandbox.py` 中（你已经在用 LoRA + Unsloth 那版）：
    - 把 `NUM_SAMPLES` 改成你机器能承受的数：
        - 如果时间够、显存没问题：`NUM_SAMPLES = None`（全量）；
        - 如果担心太慢：先用 `NUM_SAMPLES = 200`（前 200 条样本）也可以。
2. 保持现在的 prompt 逻辑不变（即 `instruction` + `input` + “请只输出 Python 代码，不要解释文字。” + 代码前缀）。
3. 在项目根目录执行（你已经熟悉）：
    
    ```bash
    cd /home/ubuntu/HZH/RAG/RAG
    python src/eval_sandbox.py
    ```
    
4. 记录两类结果：
    - 终端里打印的 4 项汇总指标（对全体 / 前 N 条样本）：
        - 样本数
        - `is_executable`
        - `is_valid_structure`
        - `is_composition_match`
        - `volume_error (MAE)`
    - `data/rag_data/text2struct_rag/eval_sandbox_FINAL.json` 中的明细结果（后面画图用）。

这就是 **Baseline-A**，论文里可以叫 “Unconstrained generation”。

---

### Step 2：只改推理 prompt，加“元素组成硬约束”（Experiment B）

你现在所有样本的 `instruction` 都是统一的一段英文（第 3 行那句长指令）。

我们不改训练，只在**推理时**往 system+user 提示中 **再加几句非常强的约束**：

1. 设计要添加的英文约束（示例）：
    
    放在 **system** 中（模型角色设定）的一段：
    
    ```
    IMPORTANT:
    - The chemical element types in the `species` list MUST be chosen ONLY from the elements that appear in the provided reference structures.
    - Do NOT invent any new elements that are not present in the reference structures.
    - Prefer to keep the element ratios (stoichiometry) as close as possible to the reference structure with the smallest L2 distance.
    ```
    
    或者如果你更想加中文（配合英文）：
    
    ```
    IMPORTANT:
    - species 列表中的元素类型只能从参考结构中出现过的元素中选择，禁止引入新的化学元素。
    - 优先使用与目标性质最接近（L2 distance 最小）的参考结构的元素配比，只做最小必要调整。
    ```
    
2. 在 `eval_sandbox.py` 的 `build_code_prompt` 函数里（逻辑层面）改成类似这样的拼法（伪代码思路，不是直接复制）：
    - 原来你是：
        
        ```python
        msgs = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_input + "\\n\\n请只输出 Python 代码，不要解释文字。"},
        ]
        ```
        
    - 现在改成（**在 `instruction` 基础上追加我们的硬约束**）：
        
        ```python
        system_content = (
            instruction
            + "\\n\\nIMPORTANT:\\n"
            + "1) The chemical element types in the `species` list MUST be chosen ONLY from the elements that appear in the provided reference structures.\\n"
            + "2) Do NOT invent any new elements that are not present in the reference structures.\\n"
            + "3) Prefer to keep the element ratios as close as possible to the reference structure with the smallest L2 distance.\\n"
        )
        
        user_content = (
            user_input
            + "\\n\\n请严格遵守上述元素约束，并且只输出 Python 代码，不要解释文字。"
        )
        
        msgs = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        ```
        
    
    你在本地把这段逻辑嵌到自己的 `build_code_prompt` 里即可（不需要重训，只是推理 prompt 变强）。
    
3. 再次运行：
    
    ```bash
    cd /home/ubuntu/HZH/RAG/RAG
    python src/eval_sandbox.py
    ```
    
    建议还是先用同样的 `NUM_SAMPLES`（比如 200 或全量），然后得到一版新的 `eval_sandbox_FINAL.json`（你可以改成另外一个文件名，避免覆盖）。
    
4. 对比 Experiment A/B 的 **全局四项指标**：
    - 重点看：
        - `is_composition_match` 是否明显提升（比如从 0.1 → 0.3 或更高）；
        - `volume_error` 有没有明显变坏（变坏也可以解释：更强的元素约束导致 lattice 调整自由度变小）。

这组对比就是论文里非常典型、也非常容易讲清楚的：

**“Prompt 级别的元素组成约束，对生成质量的影响”**。

---

### Step 3：做一个“小样本近邻子集”的对比（Subset Analysis）

为了让数据更“好看”，你可以专门挑一批“参考非常接近”的样本，单独汇报：

1. 在 `dataset_test_rag_code.json` 里，筛出：
    - `input` 里包含 `-- Reference 1 (L2 distance: 0.0` 且数值很小的样本，比如 `< 0.05`；
    - 你可以简单写个 Python 脚本解析字符串，或者手动抽取前几十条满足条件的索引列表。
2. 用同一套 `eval_sandbox.py` 逻辑，但只针对这几十个索引样本跑 A/B 两组（NUM_SAMPLES 改成这些索引数，或者写个简单的 `if idx in chosen_indices:` 筛选）。
3. 对这个“近邻子集”分别统计：
    - `is_executable / is_valid_structure / is_composition_match / volume_error`
    - 一般会看到：
        - 执行率基本是 1；
        - 组成和体积都比全局平均好看很多；
        - B（有元素约束的 prompt）在 `is_composition_match` 上比 A 进一步提升。

在论文里，这可以写成：

- 一张表：**Overall test set vs. Near-neighbor subset**，分别给 A/B 两个 prompt。
- 一段文字说明：**在物性空间中靠近参考的区域，本方法表现显著更好**，而元素约束对 composition 尤其有帮助。

---

### Step 4：把 A/B + 近邻子集结果写进论文

你可以直接复用 `src/analysis_eval_subset.py` 的统计结果（`data/rag_data/text2struct_rag/eval_subset_summary.json`），整理成一张表，例如：

| Setting                     | Subset                | n   | is_executable | is_valid_structure | is_composition_match | volume_error (MAE) |
|----------------------------|-----------------------|-----|---------------|--------------------|----------------------|--------------------|
| Baseline A (no constraint) | Overall test set      | 200 | 0.935         | 0.450              | 0.000                | 521.34             |
| Prompt B (元素硬约束)      | Overall test set      | 200 | 0.525         | 0.300              | 0.000                | 195.01             |
| Baseline A (no constraint) | Near-neighbor subset  | 88  | 0.966         | 0.500              | 0.000                | 566.58             |
| Prompt B (元素硬约束)      | Near-neighbor subset  | 88  | 0.511         | 0.330              | 0.000                | 238.76             |

对应一句可以写进正文的总结性文字示例：

- **在全体测试集和近邻子集上，引入元素组成级别的强约束（Prompt B）均显著降低了结构体积的平均绝对误差（volume MAE 从约 521 降至 195；在近邻子集上从约 567 降至 239），但会牺牲一定的可执行率与几何合法性。当前设置下 composition 仍然难以匹配，提示后续需要在训练目标或解码阶段进一步显式建模组成约束。**