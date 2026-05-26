"""统计计算与可视化数据准备"""
from pathlib import Path
from typing import Optional

def compute_summary_stats(samples: list, target_dg_h: float = -0.2) -> dict:
    """Compute pass_rate, avg_dg_h, generation_failure_rate from sample list."""
    if not samples:
        return {"pass_rate": 0, "avg_dg_h": None, "failure_rate": 0, "total": 0}
    total = len(samples)
    passed = sum(1 for s in samples if s.get("in_filter"))
    failed_poscar = sum(1 for s in samples if not s.get("poscar"))
    dg_hs = [s["dg_h"] for s in samples if s.get("dg_h") is not None]
    deviations = [abs(s["dg_h"] - target_dg_h) for s in samples if s.get("dg_h") is not None]
    return {
        "pass_rate": round(passed / total, 4),
        "avg_dg_h": round(sum(dg_hs) / len(dg_hs), 4) if dg_hs else None,
        "failure_rate": round(failed_poscar / total, 4),
        "total": total,
        "avg_deviation": round(sum(deviations) / len(deviations), 4) if deviations else None,
        "min_deviation": round(min(deviations), 4) if deviations else None,
    }

def prepare_histogram_data(distribution: dict) -> dict:
    """Return Streamlit bar_chart compatible dict from get_distribution output."""
    return {
        "bins": distribution["bins"],
        "counts": distribution["counts"],
    }

def prepare_filter_rate_chart(distribution: dict) -> dict:
    """Return Streamlit line_chart compatible dict for filter rate trend."""
    return {
        "bins": distribution["bins"],
        "filter_rates": distribution["filter_rates"],
    }

def prepare_scatter_data(samples: list, target_dg_h: float = -0.2) -> dict:
    """
    准备散点图数据：x=样本索引(时间序), y=|ΔG_H - target_dg_h|，用于观察偏差分布。
    返回 Streamlit scatter_chart 兼容格式。
    """
    points = []
    for i, s in enumerate(samples):
        if s.get("dg_h") is not None:
            deviation = abs(s["dg_h"] - target_dg_h)
            points.append({
                "index": i,
                "id": s.get("id", i),
                "dg_h": round(s["dg_h"], 4),
                "deviation": round(deviation, 4),
                "in_filter": s.get("in_filter", False),
            })
    return {
        "points": points,
        "target_dg_h": target_dg_h,
    }

def prepare_deviation_histogram(samples: list, target_dg_h: float = -0.2, bin_count: int = 15) -> dict:
    """
    准备|ΔG_H - target|偏差直方图数据。
    """
    import numpy as np
    deviations = [abs(s["dg_h"] - target_dg_h) for s in samples if s.get("dg_h") is not None]
    if not deviations:
        return {"bins": [], "counts": []}
    hist, bin_edges = np.histogram(deviations, bins=bin_count)
    return {
        "bins": [f"{bin_edges[i]:.2f}" for i in range(len(bin_edges)-1)],
        "counts": hist.tolist(),
        "deviations": deviations,
    }

def prepare_element_distribution(samples: list) -> dict:
    """
    从样本中统计元素出现频率分布。
    解析POSCAR提取元素种类。
    """
    from collections import Counter
    import ase.io
    import io
    element_counter = Counter()
    parseable = 0
    for s in samples:
        try:
            poscar = s.get("poscar", "")
            if poscar:
                atoms = ase.io.read(io.StringIO(poscar), format="vasp")
                for elem in atoms.get_chemical_symbols():
                    element_counter[elem] += 1
                parseable += 1
        except Exception:
            pass
    if not element_counter:
        return {"elements": [], "counts": [], "total_structures": 0}
    items = element_counter.most_common()
    return {
        "elements": [e for e, c in items],
        "counts": [c for e, c in items],
        "total_structures": parseable,
    }
