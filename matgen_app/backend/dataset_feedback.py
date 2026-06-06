# backend/dataset_feedback.py
"""
P1-2: 数据集回灌 + 索引重建

飞轮闭环第六步：将人工验证通过的结构记录追加进训练数据集，
并基于 ΔG_H 数值近邻重建 RAG 检索索引。

对外接口：
    records_to_training_format(records, instruction) -> list
    append_to_dataset(new_records, dataset_path) -> dict
    rebuild_near_neighbor_index(dataset_path, index_path, top_k=3) -> dict
    feedback_pipeline(store, dataset_path, index_path, instruction) -> dict
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


# ─────────────────────────────────────────────────────────────────
# 1. 格式转换
# ─────────────────────────────────────────────────────────────────

def records_to_training_format(records: list, instruction: str) -> list:
    """
    把 StateStore.export_validated() 返回的记录列表转换为训练格式。

    训练格式：
        {"instruction": str, "input": str, "output": poscar_str, "dg_h": float}

    规则：
    - poscar 为空（None 或 ""）的记录跳过；
    - dg_h 优先取 dft_dg_h（DFT ground truth），回退到 predicted_dgH；
    - input 若记录已含 prompt 键则直接用，否则由 elements + target_dgH 自动构造。
    """
    result = []
    for rec in records:
        poscar = rec.get("poscar") or ""
        if not poscar.strip():
            # 跳过空 poscar
            continue

        # dg_h: dft_dg_h 优先，回退 predicted_dgH
        # 用 is not None 而非 or，避免 dft_dg_h=0.0（HER 最优值）被 falsy 吞掉
        dg_h = rec["dft_dg_h"] if rec.get("dft_dg_h") is not None else rec.get("predicted_dgH")

        # input: 已有 prompt 字段则直接用，否则自动生成
        if rec.get("prompt"):
            input_text = rec["prompt"]
        else:
            elements = rec.get("elements", "")
            target = rec.get("target_dgH")
            if target is not None:
                input_text = (
                    f"请为元素组成为 {elements} 的高熵合金生成 POSCAR 结构，"
                    f"目标 ΔG_H = {target:.4f} eV。"
                )
            else:
                input_text = (
                    f"请为元素组成为 {elements} 的高熵合金生成 POSCAR 结构。"
                )

        result.append({
            "instruction": instruction,
            "input": input_text,
            "output": poscar,
            "dg_h": dg_h,
        })
    return result


# ─────────────────────────────────────────────────────────────────
# 2. 追加写入数据集
# ─────────────────────────────────────────────────────────────────

def append_to_dataset(new_records: list, dataset_path: str) -> dict:
    """
    把训练格式记录追加进 dataset_path（JSON 数组文件）。

    - 若文件不存在则创建；
    - 按 output(poscar) 去重，已存在的跳过；
    - 写文件用 ensure_ascii=False, indent=2。

    返回：
        {"appended": n, "skipped_duplicates": m, "total": 文件最终条数}
    """
    # 读取现有数据
    path = Path(dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        with open(path, encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []

    # 按 output 建立去重集合
    existing_outputs: set[str] = {r.get("output", "") for r in existing}

    appended = 0
    skipped = 0
    for rec in new_records:
        out = rec.get("output", "")
        if out in existing_outputs:
            skipped += 1
        else:
            existing.append(rec)
            existing_outputs.add(out)
            appended += 1

    # 写回文件
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    return {
        "appended": appended,
        "skipped_duplicates": skipped,
        "total": len(existing),
    }


# ─────────────────────────────────────────────────────────────────
# 3. 重建近邻索引
# ─────────────────────────────────────────────────────────────────

def rebuild_near_neighbor_index(
    dataset_path: str,
    index_path: str,
    top_k: int = 3,
) -> dict:
    """
    基于 dg_h 数值距离为数据集中每条记录计算 top_k 近邻，写成索引文件。

    索引格式：{"0": [近邻idx, ...], "1": [...], ...}
    键为字符串，值为整数列表（不含自身）。

    不引入 faiss 等重型依赖，纯 numpy 实现。

    返回：{"num_entries": N, "index_path": str}
    """
    # 读取数据集
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    n = len(dataset)
    if n == 0:
        index: dict[str, list[int]] = {}
    else:
        # 过滤掉 dg_h 为 None 的记录，只对有效条目建索引
        # valid_indices: 原始 dataset 中 dg_h 非 None 的下标列表
        valid_indices = [i for i, r in enumerate(dataset) if r.get("dg_h") is not None]
        dg_h_values = np.array(
            [float(dataset[i]["dg_h"]) for i in valid_indices],
            dtype=np.float64,
        )

        # 对每条有效记录计算与其他有效记录的 |dg_h| 距离，取最近的 top_k（排除自身）
        m = len(valid_indices)  # 有效条目数
        index = {}
        for pos, orig_i in enumerate(valid_indices):
            if m <= 1:
                index[str(orig_i)] = []
                continue
            dists = np.abs(dg_h_values - dg_h_values[pos])
            dists[pos] = np.inf  # 排除自身
            k = min(top_k, m - 1)
            nearest_pos = np.argsort(dists)[:k].tolist()
            # 近邻下标转回原始 dataset 下标
            index[str(orig_i)] = [int(valid_indices[x]) for x in nearest_pos]

    # 写索引文件
    idx_path = Path(index_path)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    return {
        "num_entries": n,
        "index_path": str(index_path),
    }


# ─────────────────────────────────────────────────────────────────
# 4. 飞轮闭环 pipeline
# ─────────────────────────────────────────────────────────────────

def feedback_pipeline(
    store,
    dataset_path: str,
    index_path: str,
    instruction: str,
) -> dict:
    """
    飞轮回灌完整流程：
        store.export_validated()
        → records_to_training_format()
        → append_to_dataset()
        → rebuild_near_neighbor_index()
        → 返回汇总 dict

    参数：
        store       : StateStore 实例
        dataset_path: 训练数据集 JSON 文件路径
        index_path  : 近邻索引 JSON 文件路径
        instruction : 训练样本的 instruction 字段

    返回：
        {
            "validated_count": int,   # export_validated() 返回的原始记录数
            "appended": int,          # 实际追加条数
            "skipped_duplicates": int,
            "total": int,             # 数据集最终条数
            "num_entries": int,       # 索引覆盖条数
            "index_path": str,
        }
    """
    # ① 从状态库取出已验证记录
    raw_records = store.export_validated()

    # ② 转换为训练格式
    training_records = records_to_training_format(raw_records, instruction)

    # ③ 追加进数据集（含去重）
    append_result = append_to_dataset(training_records, dataset_path)

    # ④ 重建近邻索引（仅当数据集非空时有意义）
    if append_result["total"] > 0:
        idx_result = rebuild_near_neighbor_index(dataset_path, index_path)
    else:
        # 数据集为空：写空索引，保持接口一致
        idx_result = {"num_entries": 0, "index_path": str(index_path)}
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump({}, f)

    return {
        "validated_count": len(raw_records),
        "appended": append_result["appended"],
        "skipped_duplicates": append_result["skipped_duplicates"],
        "total": append_result["total"],
        "num_entries": idx_result["num_entries"],
        "index_path": idx_result["index_path"],
    }
