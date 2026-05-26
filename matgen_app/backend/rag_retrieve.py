"""
RAG 检索模块：从训练集检索相关参考结构，增强生成 prompt

检索策略：
1. TF-IDF 文本相似度（无需 embedding 模型）
2. 基于 ΔG_H 范围过滤
3. 返回 top-k 个参考结构和对应的 POSCAR

用法：
    from rag_retrieve import retrieve, build_rag_prompt
    refs = retrieve("在 Cu 表面吸附 H 原子", top_k=3)
    prompt = build_rag_prompt("在 Cu 表面吸附 H 原子", refs)
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter
import re

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
from config import RAG_DATA_DIR, RAG_TRAIN_DATA_PATH

# ── 加载数据 ────────────────────────────────────────────────────────────────
_train_data = None
_tfidf_matrix = None
_tfidf_vectorizer = None
_index_loaded = False


def _load_data() -> Tuple[list, list]:
    """Lazy load training data and near neighbor indices."""
    global _train_data, _index_loaded

    if _train_data is not None:
        return _train_data, []

    train_path = RAG_TRAIN_DATA_PATH
    if not train_path.exists():
        print(f"[WARN] RAG training data not found: {train_path}")
        _train_data = []
        return [], []

    with open(train_path, encoding='utf-8') as f:
        _train_data = json.load(f)

    _index_loaded = True
    return _train_data, []


def extract_dg_h(text: str) -> Optional[float]:
    """从文本中提取 ΔG_H 值（eV）。"""
    patterns = [
        r'ΔGH\s*=\s*([-+]?\d*\.?\d+)',
        r'Target\s+ΔGH\s*=\s*([-+]?\d*\.?\d+)',
        r'dg_h\s*[:=]\s*([-+]?\d*\.?\d+)',
        r'DG_H?\s*=\s*([-+]?\d*\.?\d+)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def extract_elements(poscar_text: str) -> List[str]:
    """从 POSCAR 中提取元素列表。"""
    try:
        lines = poscar_text.strip().split('\n')
        # Find element line (line 5 or 6 depending on selective dynamics)
        for i, line in enumerate(lines):
            stripped = line.strip()
            if i >= 4 and i <= 7:
                # Check if this looks like an element line
                parts = stripped.split()
                if len(parts) >= 2 and all(re.match(r'^[A-Z][a-z]?$', p) for p in parts[:min(3, len(parts))]):
                    # Might be element line - check next line for numbers
                    if i + 1 < len(lines):
                        next_parts = lines[i+1].strip().split()
                        if next_parts and all(re.match(r'^\d+$', p) for p in next_parts[:len(parts)]):
                            return parts
        # Fallback: try first non-numeric line
        for line in lines[5:8]:
            parts = line.strip().split()
            if parts and re.match(r'^[A-Z][a-z]?$', parts[0]):
                return parts
    except Exception:
        pass
    return []


def _build_tfidf_index():
    """Build TF-IDF index from training inputs."""
    global _tfidf_matrix, _tfidf_vectorizer

    train_data, _ = _load_data()
    if not train_data:
        return

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            print("[WARN] sklearn not available, using simple keyword matching")
            return

    # Build corpus from training inputs
    corpus = []
    for item in train_data:
        inp = item.get('input', '') or ''
        # Clean up the structured text
        inp_clean = inp.replace('\n', ' ').replace('φ', 'phi').replace('ΔGH', 'deltaGH')
        corpus.append(inp_clean)

    _tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
    )
    _tfidf_matrix = _tfidf_vectorizer.fit_transform(corpus)


def retrieve(query: str, top_k: int = 3, dg_h_range: tuple = (-1.0, 0.5)) -> List[Dict]:
    """
    检索与 query 最相似的 top_k 个参考结构。

    Returns:
        List of dicts with keys: 'input', 'output', 'dg_h', 'elements', 'similarity'
    """
    train_data, _ = _load_data()
    if not train_data:
        return []

    query_lower = query.lower()

    # Strategy 1: TF-IDF similarity (if available)
    if _tfidf_matrix is not None and _tfidf_vectorizer is not None:
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            query_vec = _tfidf_vectorizer.transform([query_lower])
            sims = cosine_similarity(query_vec, _tfidf_matrix).flatten()
            top_indices = sims.argsort()[-top_k * 3:][::-1]  # Get more candidates
        except Exception:
            top_indices = []
    else:
        # Strategy 2: Keyword-based scoring
        # Extract keywords from query
        keywords = set(re.findall(r'\b[A-Z][a-z]?\b', query))  # Element symbols
        keywords.update(re.findall(r'\b\d+\.?\d*\b', query))  # Numbers

        scored = []
        for i, item in enumerate(train_data):
            inp = (item.get('input', '') or '').lower()
            out = (item.get('output', '') or '').lower()

            # Score based on keyword overlap
            score = 0
            for kw in keywords:
                if kw.lower() in inp or kw.lower() in out:
                    score += 1
            scored.append((score, i))

        scored.sort(reverse=True)
        top_indices = [i for _, i in scored[:top_k * 3]]

    # Filter by ΔG_H range and get top_k
    results = []
    for idx in top_indices:
        if len(results) >= top_k:
            break
        item = train_data[idx]
        inp = item.get('input', '') or ''
        out = item.get('output', '') or ''

        # Extract ΔG_H
        dg_h = extract_dg_h(inp)

        # Filter by ΔG_H range
        if dg_h is not None:
            if dg_h < dg_h_range[0] or dg_h > dg_h_range[1]:
                continue

        # Extract elements from POSCAR
        elements = extract_elements(out)

        results.append({
            'index': idx,
            'input': inp,
            'output': out,
            'dg_h': dg_h,
            'elements': elements,
            'element_str': ' '.join(elements) if elements else 'Unknown',
        })

    return results


def build_rag_prompt(query: str, refs: List[Dict], include_output: bool = True) -> str:
    """
    将 query 和检索到的参考结构组合成增强 prompt。

    Args:
        query: 用户原始 prompt
        refs: retrieve() 返回的参考结构列表
        include_output: 是否在 prompt 中包含 POSCAR 输出示例

    Returns:
        增强后的完整 prompt
    """
    if not refs:
        return query

    # System prompt for RAG context
    system_part = """You are a materials science expert specializing in VASP POSCAR structure generation for hydrogen adsorption on high-entropy alloy surfaces.

参考结构（Retrieved Reference Structures）："""

    ref_lines = []
    for i, ref in enumerate(refs, 1):
        dg_h_str = f"{ref['dg_h']:.6f}" if ref['dg_h'] is not None else "N/A"
        elem_str = ref.get('element_str', 'Unknown')

        ref_lines.append(f"""
参考 {i}（元素: {elem_str}, ΔG_H: {dg_h_str} eV）:
=== Input Description ===
{ref['input'][:300]}
=== Generated POSCAR ===
{ref['output'][:500]}
---"""[1:])

    system_part += '\n'.join(ref_lines)

    system_part += f"""
=== Current Task ===
请参考上述 {len(refs)} 个相似结构，生成符合以下描述的 VASP POSCAR 格式结构：

{query}

要求：
1. 输出完整的 VASP POSCAR 格式（包含晶格常数、原子坐标等）
2. 确保元素种类和比例合理
3. 结构应具有物理合理性（无原子重叠、晶格常数合理）
"""

    return system_part


def build_rag_prompt_v2(query: str, refs: List[Dict]) -> str:
    """
    增强版 prompt：只提供输入描述，不直接给出完整 POSCAR（让模型自己生成）。
    """
    if not refs:
        return query

    ref_lines = []
    for i, ref in enumerate(refs, 1):
        dg_h_str = f"{ref['dg_h']:.6f}" if ref['dg_h'] is not None else "N/A"
        elem_str = ref.get('element_str', 'Unknown')

        ref_lines.append(f"参考 {i}: 元素={elem_str}, ΔG_H={dg_h_str} eV, 输入描述={ref['input'][:150]}")

    refs_text = '\n'.join(ref_lines)

    prompt = f"""参考以下相似结构及其性质，生成符合描述的 VASP POSCAR 结构：

{refs_text}

当前需求：{query}

请生成完整的 VASP POSCAR 格式结构。"""

    return prompt


# ── 简易版：仅提供元素信息 ──────────────────────────────────────────────────
def build_rag_prompt_minimal(query: str, refs: List[Dict]) -> str:
    """
    最小增强：仅提供参考结构的元素组成和 ΔG_H，引导生成方向。
    """
    if not refs:
        return query

    # Summarize element distributions
    all_elements = []
    dg_hs = []
    for ref in refs:
        if ref.get('elements'):
            all_elements.extend(ref['elements'])
        if ref.get('dg_h') is not None:
            dg_hs.append(ref['dg_h'])

    elem_counter = Counter(all_elements)
    top_elements = elem_counter.most_common(5)
    avg_dg_h = sum(dg_hs) / len(dg_hs) if dg_hs else -0.2

    elem_hint = ', '.join([f"{e}({c})" for e, c in top_elements])

    prompt = f"""[RAG Guidance] Similar structures use elements: {elem_hint}. Average ΔG_H: {avg_dg_h:.4f} eV. Target range: [-1.0, 0.5] eV.

User Query: {query}

Generate a valid VASP POSCAR structure following the similar element compositions and within the ΔG_H target range."""

    return prompt


def get_rag_stats() -> dict:
    """返回 RAG 数据集的统计信息。"""
    train_data, _ = _load_data()
    if not train_data:
        return {"total": 0}

    dg_hs = []
    elements_all = Counter()
    for item in train_data:
        inp = item.get('input', '') or ''
        out = item.get('output', '') or ''

        dg_h = extract_dg_h(inp)
        if dg_h is not None:
            dg_hs.append(dg_h)

        for elem in extract_elements(out):
            elements_all[elem] += 1

    return {
        "total": len(train_data),
        "dg_h_mean": round(sum(dg_hs) / len(dg_hs), 4) if dg_hs else None,
        "dg_h_std": round((sum((x - sum(dg_hs)/len(dg_hs))**2 for x in dg_hs) / len(dg_hs)) ** 0.5, 4) if dg_hs else None,
        "dg_h_min": round(min(dg_hs), 4) if dg_hs else None,
        "dg_h_max": round(max(dg_hs), 4) if dg_hs else None,
        "top_elements": dict(elements_all.most_common(10)),
    }


# ── 初始化索引（可选）───────────────────────────────────────────────────────
def init_index():
    """预加载 TF-IDF 索引（耗时约5-10秒）。"""
    import numpy as np
    _build_tfidf_index()
    print(f"[RAG] Index built: {_tfidf_matrix.shape if _tfidf_matrix is not None else 'N/A'}")


if __name__ == "__main__":
    # Quick test
    print("=== RAG Stats ===")
    stats = get_rag_stats()
    print(f"Total samples: {stats['total']}")
    print(f"ΔG_H mean: {stats.get('dg_h_mean')}")
    print(f"Top elements: {stats.get('top_elements')}")

    print("\n=== RAG Retrieval Test ===")
    test_query = "在 Ir(111) 表面吸附一个 H 原子"
    refs = retrieve(test_query, top_k=3)
    print(f"Query: {test_query}")
    print(f"Retrieved {len(refs)} refs:")
    for r in refs:
        print(f"  idx={r['index']}, dg_h={r['dg_h']}, elements={r.get('element_str')}")

    print("\n=== Prompt Build Test ===")
    prompt = build_rag_prompt_minimal(test_query, refs)
    print(f"Enhanced prompt:\n{prompt[:500]}")
