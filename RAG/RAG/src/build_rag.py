"""
Phase 2: RAG 检索系统搭建
- 2.1 对训练集特征归一化后建立 FAISS 向量索引
- 2.2 实现 retrieve_references(target_features, k=3)
- 2.3 将检索到的参考结构注入 Prompt，生成增强版 JSON 数据集

输入:
  - data/processed/dataset_train.json
  - data/processed/dataset_test.json
输出:
  - data/processed/dataset_train_rag.json
  - data/processed/dataset_test_rag.json
  - data/processed/faiss_index/  (索引 + 元数据)
"""

import json
import numpy as np
import faiss
import pickle
import re
from pathlib import Path

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = DATA_DIR / "faiss_index"

TRAIN_JSON = DATA_DIR / "dataset_train.json"
TEST_JSON = DATA_DIR / "dataset_test.json"

# 特征列名（与 preprocess.py 中一致）
FEATURE_NAMES = ["φ", "Nd1", "Np0", "Out_e0", "Out_e1", "ψ1",
                 "First_IE0", "CN", "L_bond", "R0"]
TARGET_NAME = "ΔGH"

TOP_K = 3


# ============================================================
# 工具函数
# ============================================================
def parse_features_from_input(input_text: str) -> dict:
    """从 input 字段文本中解析出特征字典。"""
    features = {}
    for line in input_text.strip().split("\n"):
        line = line.strip()
        match = re.match(r"^(?:Target\s+)?(.+?)\s*=\s*(.+?)(?:\s*eV)?$", line)
        if match:
            key = match.group(1).strip()
            val = float(match.group(2).strip())
            features[key] = val
    return features


def features_to_vector(features: dict) -> np.ndarray:
    """将特征字典转为固定顺序的 numpy 向量（含 ΔGH）。"""
    vec = []
    vec.append(features.get(TARGET_NAME, 0.0))
    for name in FEATURE_NAMES:
        vec.append(features.get(name, 0.0))
    return np.array(vec, dtype=np.float32)


# ============================================================
# 2.1  向量数据库初始化（FAISS）
# ============================================================
def build_index(train_data: list) -> tuple:
    """
    对训练集构建 FAISS 索引。
    返回: (faiss_index, vectors, mean, std)
    """
    print("[1/4] 构建特征向量 ...")
    vectors = []
    for sample in train_data:
        feat = parse_features_from_input(sample["input"])
        vec = features_to_vector(feat)
        vectors.append(vec)

    vectors = np.array(vectors, dtype=np.float32)
    print(f"       向量矩阵形状: {vectors.shape}")

    # 归一化（z-score）使欧氏距离有意义
    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0)
    std[std == 0] = 1.0  # 避免除零
    vectors_norm = (vectors - mean) / std

    print("[2/4] 构建 FAISS 索引 ...")
    dim = vectors_norm.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors_norm)
    print(f"       索引中向量数: {index.ntotal}")

    # 保存索引和归一化参数
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "train.index"))
    with open(INDEX_DIR / "norm_params.pkl", "wb") as f:
        pickle.dump({"mean": mean, "std": std}, f)
    print(f"       索引已保存: {INDEX_DIR}")

    return index, vectors_norm, mean, std


# ============================================================
# 2.2  检索逻辑
# ============================================================
def retrieve_references(query_features: dict,
                        index: faiss.IndexFlatL2,
                        train_data: list,
                        mean: np.ndarray,
                        std: np.ndarray,
                        k: int = TOP_K) -> list:
    """
    检索与目标特征最相似的 k 个训练样本。
    返回: [(距离, 样本dict), ...]
    """
    vec = features_to_vector(query_features).reshape(1, -1)
    vec_norm = (vec - mean) / std

    distances, indices = index.search(vec_norm, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append((float(dist), train_data[idx]))
    return results


# ============================================================
# 2.3  Prompt 增强
# ============================================================
def build_rag_instruction() -> str:
    """增强版 instruction。"""
    return (
        "You are a materials science expert. "
        "Given the physical and chemical properties of a hydrogen storage material "
        "and several reference structures with similar properties, "
        "generate the corresponding atomic structure in VASP POSCAR format."
    )


def augment_input_with_references(original_input: str,
                                  references: list) -> str:
    """将检索到的参考结构拼接到 input 中。"""
    parts = [original_input, ""]
    parts.append(f"Below are {len(references)} reference structures with similar properties:")
    parts.append("")

    for i, (dist, ref_sample) in enumerate(references, 1):
        parts.append(f"--- Reference {i} (L2 distance: {dist:.4f}) ---")
        parts.append(f"Properties:\n{ref_sample['input']}")
        parts.append(f"Structure:\n{ref_sample['output']}")
        parts.append("")

    return "\n".join(parts)


def augment_dataset(dataset: list,
                    index: faiss.IndexFlatL2,
                    train_data: list,
                    mean: np.ndarray,
                    std: np.ndarray,
                    is_train: bool = False) -> list:
    """对整个数据集进行 RAG 增强。"""
    augmented = []
    for i, sample in enumerate(dataset):
        feat = parse_features_from_input(sample["input"])

        if is_train:
            # 训练集检索 k+1 个，排除自身
            refs = retrieve_references(feat, index, train_data, mean, std, k=TOP_K + 1)
            # 排除与自身 output 完全相同的结果
            refs = [(d, s) for d, s in refs if s["output"] != sample["output"]][:TOP_K]
        else:
            refs = retrieve_references(feat, index, train_data, mean, std, k=TOP_K)

        new_sample = {
            "instruction": build_rag_instruction(),
            "input": augment_input_with_references(sample["input"], refs),
            "output": sample["output"],
        }
        augmented.append(new_sample)

        if (i + 1) % 2000 == 0:
            print(f"       已处理 {i + 1}/{len(dataset)} ...")

    return augmented


# ============================================================
# 主流程
# ============================================================
def main():
    # 加载 Phase 1 数据
    print("加载 Phase 1 数据集 ...")
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(TEST_JSON, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"  训练集: {len(train_data)} 条  |  测试集: {len(test_data)} 条")

    # 2.1 构建 FAISS 索引
    index, vectors_norm, mean, std = build_index(train_data)

    # 2.3 增强训练集
    print("[3/4] RAG 增强训练集 ...")
    train_rag = augment_dataset(train_data, index, train_data, mean, std, is_train=True)

    # 2.3 增强测试集
    print("[4/4] RAG 增强测试集 ...")
    test_rag = augment_dataset(test_data, index, train_data, mean, std, is_train=False)

    # 保存
    train_rag_path = DATA_DIR / "dataset_train_rag.json"
    test_rag_path = DATA_DIR / "dataset_test_rag.json"

    with open(train_rag_path, "w", encoding="utf-8") as f:
        json.dump(train_rag, f, ensure_ascii=False, indent=2)
    with open(test_rag_path, "w", encoding="utf-8") as f:
        json.dump(test_rag, f, ensure_ascii=False, indent=2)

    print(f"\n✓ RAG训练集已保存: {train_rag_path} ({len(train_rag)} 条)")
    print(f"✓ RAG测试集已保存: {test_rag_path} ({len(test_rag)} 条)")

    # 打印一条示例
    print("\n示例 (test_rag[0] input 前 600 字符):")
    print(test_rag[0]["input"][:600])


if __name__ == "__main__":
    main()
