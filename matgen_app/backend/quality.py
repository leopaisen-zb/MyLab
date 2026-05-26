"""候选结构质量排序与结构合理性校验"""
from typing import List, Dict, Optional

def sort_candidates_by_quality(samples: list, target_dg_h: float = -0.2) -> list:
    """
    按 |ΔG_H - target_dg_h| 排序（越接近目标越好），返回排序后的样本列表。
    跳过 dg_h 为 None 的样本。
    """
    valid = [s for s in samples if s.get("dg_h") is not None]
    valid.sort(key=lambda s: abs(s["dg_h"] - target_dg_h))
    return valid

def validate_structure(poscar_text: str) -> dict:
    """
    校验 POSCAR 结构合理性，返回 {'valid': bool, 'warnings': list[str], 'info': dict, 'rejection_level': str}。
    rejection_level: 'pass' | 'text' | 'structure' | 'physics'
    """
    import ase.io
    import io
    import numpy as np
    warnings = []
    info = {}
    rejection_level = "pass"

    try:
        atoms = ase.io.read(io.StringIO(poscar_text), format="vasp")
    except Exception as e:
        return {"valid": False, "warnings": [f"解析失败: {e}"], "info": {}, "rejection_level": "structure"}

    numbers = atoms.numbers
    positions = atoms.positions
    lattice = atoms.cell.array

    # 1. 检查晶格常数非负
    if np.any(np.diag(lattice) <= 0):
        warnings.append("晶格常数存在负值或零")
        rejection_level = "physics"
    # 2. 检查原子间距过近（< 0.5 Å）
    from ase.neighborlist import neighbor_list
    n1, n2, d = neighbor_list('ijd', atoms, cutoff=10.0)
    min_dist = float(d.min()) if len(d) > 0 else float('inf')
    if min_dist < 0.5:
        warnings.append(f"最小原子间距 {min_dist:.3f} Å 过近（< 0.5 Å）")
        rejection_level = "physics"
    # 3. 元素统计
    elem_counts = {}
    for n in numbers:
        elem_counts[n] = elem_counts.get(n, 0) + 1
    info["num_atoms"] = len(numbers)
    info["num_elements"] = len(elem_counts)
    info["min_distance"] = round(min_dist, 3)

    return {
        "valid": len(warnings) == 0,
        "warnings": warnings,
        "info": info,
        "rejection_level": rejection_level,
    }
