"""RAG 检索可视化：展示生成时引用的参考结构"""
import json
import sys
from pathlib import Path
from typing import List

# Add project root to path for config import
APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
from config import RAG_DATA_DIR, RAG_TRAIN_DATA_PATH, RAG_NEAR_NEIGHBOR_PATH

def load_near_neighbors() -> list:
    """加载 near_neighbor_indices.json，返回索引列表（index i -> neighbor ref idx）。"""
    if not RAG_NEAR_NEIGHBOR_PATH.exists():
        return []
    with open(RAG_NEAR_NEIGHBOR_PATH) as f:
        data = json.load(f)
    # data 是 list，每项是该 query 的最近邻 ref 索引（单个 int）
    return data  # list of ints, data[i] = nearest neighbor for query i

def get_reference_structures(neighbor_indices: list) -> list:
    """给定邻居索引列表，返回对应的参考结构信息。"""
    if not RAG_TRAIN_DATA_PATH.exists():
        return []
    with open(RAG_TRAIN_DATA_PATH) as f:
        data = json.load(f)

    refs = []
    for nidx in neighbor_indices[:3]:
        if nidx < len(data):
            sample = data[nidx]
            output = sample.get("output", "")
            first_line = output.strip().split("\n")[0] if output else ""
            refs.append({
                "idx": nidx,
                "elements": first_line.strip(),
                "dg_h": sample.get("dg_h", None),
            })
    return refs

def format_rag_visualization(prompt: str, ref_indices: list) -> str:
    """将 prompt 和参考结构格式化为 markdown 展示文本。"""
    refs = get_reference_structures(ref_indices)
    if not refs:
        return f"**检索到的参考结构（无）**\n\n原始 prompt: {prompt[:100]}..."

    lines = ["**检索到的 3 个参考结构：**\n"]
    for i, ref in enumerate(refs, 1):
        dg_str = f"{ref['dg_h']:.4f}" if ref.get("dg_h") else "N/A"
        lines.append(f"- 参考 {i}（idx={ref['idx']}）: 元素={ref['elements']}, ΔG_H={dg_str} eV")
    lines.append(f"\n**当前 prompt 摘要：** {prompt[:80]}...")
    return "\n".join(lines)
