# 硕士论文第二章执行计划（To-Do）

课题名称：基于检索增强大语言模型（LLM+RAG）的储氢材料逆向设计  
角色设定：算法研究员 / AI 辅助编程专家  
核心目标：实现“文本到结构（Text-to-Structure）”生成流水线  
输入：物理化学性质（如 ΔG_H 等）  
输出：`.vasp` 原子结构文件，并使用预训练 Eqv2-Lite 模型闭环验证  

## 数据资产与结构
- 输入（性质）：Excel 表或从 `image_9f1143.png` 解析数据
- 特征：reactionEn（ΔG_H 目标值）、Nd1、Np0、First_IE0、CN、L_bond、R0 等
- 输出（结构）：与 Excel 行一一对应的 `.vasp`（POSCAR）文件夹
- 验证器：预训练 Eqv2-Lite 模型（本地 Python 脚本/权重）

## Phase 1：数据预处理与序列化（Preprocessing）
目标：将 Excel+VASP 转为 SFT 所需 JSON 数据集

- [ ] 1.1 数据加载与对齐
  - pandas 读取 Excel
  - 遍历 VASP 文件夹，确保文件名（如 `0.vasp`）与 Excel 行索引一一对应
  - 切分训练集（80%）/测试集（20%），测试集对 
LLM 保密
- [ ] 1.2 VASP 序列化（文本化）
  - 编写 `vasp_to_string(filepath)` 读取 `.vasp`
  - 坐标浮点数保留小数点后 4 位（截断），保留元素行与晶格常数行
- [ ] 1.3 Prompt 构造（JSON 生成）
  - 生成 JSON：
    - instruction：生成结构的指令
    - input：特征描述
    - output：序列化 VASP 字符串
  - 输出为 `dataset_train.json` 与 `dataset_test.json`

## Phase 2：RAG 检索系统搭建（RAG Setup）
目标：检索相似结构作为 In-Context 参考模板

- [ ] 2.1 向量数据库初始化
  - 使用 ChromaDB 或 FAISS
  - 对训练集特征向量化（embedding 或归一化数值向量）
- [ ] 2.2 检索逻辑实现
  - 实现 `retrieve_references(target_features, k=3)`
  - 返回训练集相似度 Top-3 的 VASP 结构
- [ ] 2.3 Prompt 增强（In-Context Learning）
  - 将检索结构加入 JSON `input` 字段作为提示

## Phase 3：模型微调（SFT）
目标：微调 Qwen2.5 学习“性质描述 → VASP 结构”映射

- [ ] 3.1 训练环境配置
  - 技术栈：PyTorch、Unsloth（推荐）或 PEFT/Transformers
  - 基座模型：`Qwen/Qwen2.5-7B-Instruct`（显存允许可上 7B）
- [ ] 3.2 LoRA 参数配置与训练
  - LoRA：rank=16、alpha=32
  - target_modules：`["q_proj","k_proj","v_proj","o_proj"]`
  - 损失：标准交叉熵（Next Token Prediction）
  - 使用 `dataset_train.json` 启动训练

## Phase 4：推理与闭环验证（Validation）
目标：对测试集生成结构并使用 Eqv2-Lite 进行物理验证

- [ ] 4.1 生成流水线
  - 遍历测试集：读取目标性质
  - 从训练集检索 3 个参考结构
  - 构造 Prompt → 输入微调 LLM
  - 解析输出保存为 `gen_{id}.vasp`
- [ ] 4.2 结构健全性检查（Sanity Check）
  - `pymatgen.Structure.from_str(gen_string, fmt="poscar")`
  - 检查解析错误与原子重叠（距离 < 0.8 Å）
- [ ] 4.3 物理验证（Eqv2-Lite）
  - 预测 `pred_delta_G`
  - 计算 MAE（pred vs target）
  - 成功标准：预测值与目标值高度相关（散点图接近 y=x）

## 技术栈要求
- Python：`pandas`, `pymatgen`, `transformers`, `peft`, `chromadb`, `unsloth`（可选）, `torch`

