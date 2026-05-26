"""
Phase 1: 数据预处理与序列化
将 Excel + VASP 文件 → SFT 所需的 JSON 数据集

输出:
  - data/processed/dataset_train.json
  - data/processed/dataset_test.json
"""

import os
import json
import math
import random
import pandas as pd
from pathlib import Path

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # /home/ajifang/HZH/RAG
EXCEL_PATH = PROJECT_ROOT / "data" / "raw" / "10features_for_ML.xlsx"
VASP_DIR = PROJECT_ROOT / "data" / "raw" / "the_atomic_structure_for_ML_model"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# 特征列（用于构造 input prompt）
FEATURE_COLS = ["φ", "Nd1", "Np0", "Out_e0", "Out_e1", "ψ1",
                "First_IE0", "CN", "L_bond", "R0"]
TARGET_COL = "ΔGH"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


# ============================================================
# 1.2  VASP 序列化：将 .vasp 文件转为规范化字符串
# ============================================================
def truncate_float(value_str: str, decimals: int = 4) -> str:
    """将浮点数字符串截断到指定小数位（非四舍五入）。"""
    try:
        val = float(value_str)
    except ValueError:
        return value_str
    # 截断而非四舍五入
    factor = 10 ** decimals
    truncated = math.trunc(val * factor) / factor
    return f"{truncated:.{decimals}f}"


def vasp_to_string(filepath: str) -> str:
    """
    读取 .vasp (POSCAR) 文件并转为规范化文本字符串。
    - 保留注释行（元素名）
    - 保留缩放因子行
    - 晶格常数行：截断到 4 位小数
    - 保留元素符号行和原子数目行
    - 坐标类型行（Cartesian / Direct）
    - 坐标行：截断到 4 位小数
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    result = []

    # 第 1 行：注释行（元素名）
    result.append(lines[0].strip())

    # 第 2 行：缩放因子
    result.append(lines[1].strip())

    # 第 3-5 行：晶格向量（3 行，每行 3 个浮点数）
    for i in range(2, 5):
        tokens = lines[i].split()
        truncated = [truncate_float(t) for t in tokens]
        result.append("  ".join(truncated))

    # 第 6 行：元素符号
    result.append(lines[5].strip())

    # 第 7 行：各元素原子数
    result.append(lines[6].strip())

    # 第 8 行：坐标类型 (Cartesian / Direct)
    result.append(lines[7].strip())

    # 第 9 行起：原子坐标
    for i in range(8, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        tokens = line.split()
        # 坐标部分（前 3 列是浮点数）
        coord_tokens = [truncate_float(t) for t in tokens[:3]]
        # 后面可能有额外标记（T/F 等），保留原样
        extra = tokens[3:] if len(tokens) > 3 else []
        result.append("  ".join(coord_tokens + extra))

    return "\n".join(result)


# ============================================================
# 1.1  数据加载与对齐
# ============================================================
def load_and_align():
    """读取 Excel，与 VASP 文件对齐，返回只有匹配行的 DataFrame。"""
    print("[1/4] 读取 Excel ...")
    df = pd.read_excel(EXCEL_PATH)
    print(f"       Excel 原始行数: {len(df)}")

    # 获取可用的 VASP 文件 ID
    vasp_ids = set()
    for fname in os.listdir(VASP_DIR):
        if fname.endswith(".vasp"):
            vasp_ids.add(int(fname.replace(".vasp", "")))
    print(f"       VASP 文件数: {len(vasp_ids)}")

    # 只保留 structures 列值在 vasp_ids 中的行
    df = df[df["structures"].isin(vasp_ids)].reset_index(drop=True)
    print(f"       对齐后保留行数: {len(df)}")

    return df


# ============================================================
# 1.3  Prompt 构造
# ============================================================
def build_instruction() -> str:
    """返回统一的 instruction 指令文本。"""
    return (
        "You are a materials science expert. "
        "Given the physical and chemical properties of a hydrogen storage material, "
        "generate the corresponding atomic structure in VASP POSCAR format."
    )


def build_input_text(row: pd.Series) -> str:
    """根据一行数据构造 input 文本（特征描述）。"""
    parts = [f"Target ΔGH = {row[TARGET_COL]:.6f} eV"]
    for col in FEATURE_COLS:
        parts.append(f"{col} = {row[col]:.6f}")
    return "\n".join(parts)


def build_sample(row: pd.Series) -> dict:
    """构造一条 SFT 训练样本。"""
    struct_id = int(row["structures"])
    vasp_path = VASP_DIR / f"{struct_id}.vasp"
    vasp_str = vasp_to_string(str(vasp_path))

    return {
        "instruction": build_instruction(),
        "input": build_input_text(row),
        "output": vasp_str,
    }


# ============================================================
# 主流程
# ============================================================
def main():
    random.seed(RANDOM_SEED)

    # 1. 加载与对齐
    df = load_and_align()

    # 2. 序列化全部样本
    print("[2/4] 序列化 VASP 文件 & 构造 Prompt ...")
    samples = []
    for idx, row in df.iterrows():
        sample = build_sample(row)
        samples.append(sample)
        if (idx + 1) % 2000 == 0:
            print(f"       已处理 {idx + 1}/{len(df)} ...")
    print(f"       全部处理完成，共 {len(samples)} 条样本")

    # 3. 切分训练集 / 测试集
    print("[3/4] 切分训练集 / 测试集 ...")
    indices = list(range(len(samples)))
    random.shuffle(indices)
    split = int(len(samples) * TRAIN_RATIO)
    train_indices = sorted(indices[:split])
    test_indices = sorted(indices[split:])

    train_data = [samples[i] for i in train_indices]
    test_data = [samples[i] for i in test_indices]
    print(f"       训练集: {len(train_data)}  |  测试集: {len(test_data)}")

    # 4. 保存 JSON
    print("[4/4] 保存 JSON ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "dataset_train.json"
    test_path = OUTPUT_DIR / "dataset_test.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 训练集已保存: {train_path}")
    print(f"✓ 测试集已保存: {test_path}")
    print(f"\n示例样本 (train[0]):")
    print(json.dumps(train_data[0], ensure_ascii=False, indent=2)[:800])


if __name__ == "__main__":
    main()
