"""
真实结构库 —— 按目标性质检索真实参考结构（逆向设计的轻量实现）。

数据源：RAG 训练语料 dataset_train_rag.json（8301 条真实材料 POSCAR，每条 prompt
含 `Target ΔGH = X eV`）。给定用户目标 ΔG_H，从库中检索性质最接近的真实结构返回，
随后由真实 Equiformer 实时预测其 ΔG_H。

为何用它：真实 7B 生成模型(Qwen+LoRA)需 GPU/大内存，本机不可行；本库提供"真实模型
语料中的真实结构"，端到端可在 Mac 秒级运行，且符合 RAG 检索增强的系统叙事。
（需 GPU 服务器跑真实 LLM 生成时，设 MATGEN_USE_LLM=1 走 backend.rag_gen。）
"""
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import config as _cfg
from adapters.hea_gen_adapter import composition_from_poscar

_TARGET_RE = re.compile(r"Target\s*ΔGH\s*=\s*([-\d.]+)")

# 模块级缓存（35MB JSON 只解析一次）
_library: Optional[List[Dict[str, Any]]] = None


def _load_library(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """懒加载并缓存结构库：[{poscar, elements, source_target_dgH}, ...]，跳过不可解析项。"""
    global _library
    if _library is not None and path is None:
        return _library

    data_path = Path(path) if path else _cfg.RAG_TRAIN_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(
            f"结构库数据不存在: {data_path}\n"
            "请确认 RAG 训练语料就位，或设 MATGEN_DEMO=1 用轻量模式。"
        )

    with open(data_path, encoding="utf-8") as f:
        raw = json.load(f)

    lib = []
    for entry in raw:
        poscar = entry.get("output", "")
        m = _TARGET_RE.search(entry.get("input", ""))
        comp = composition_from_poscar(poscar)
        if poscar and m and comp:
            lib.append({
                "poscar": poscar,
                "elements": comp,
                "source_target_dgH": float(m.group(1)),
            })

    if path is None:
        _library = lib
    return lib


def sample_structures(target_dgH: float, batch_size: int = 10,
                      path: Optional[str] = None) -> List[Dict[str, Any]]:
    """检索性质最接近 target_dgH 的真实结构。

    返回 batch_size 条 {elements, poscar, target_dgH, source_target_dgH}。
    在"最接近的候选池"中随机取样，保证既贴近目标、又跨次有变化（非固定回显）。
    """
    lib = _load_library(path)
    if not lib:
        return []

    # 按与目标的偏差排序，取较近的候选池（池大小给随机性留空间）
    ranked = sorted(lib, key=lambda e: abs(e["source_target_dgH"] - target_dgH))
    pool_size = min(len(ranked), max(batch_size * 4, 40))
    pool = ranked[:pool_size]
    chosen = random.sample(pool, min(batch_size, len(pool)))

    return [{
        "elements": e["elements"],
        "poscar": e["poscar"],
        "target_dgH": target_dgH,
        "source_target_dgH": e["source_target_dgH"],
    } for e in chosen]
