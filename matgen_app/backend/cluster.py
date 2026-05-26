"""多候选聚类与热力学一致性筛选"""
import numpy as np
from typing import List, Dict

def cluster_by_dg_h(samples: list, n_clusters: int = 3) -> List[List[Dict]]:
    """
    按 ΔG_H 对样本做简单分箱聚类，返回嵌套列表。
    每组内按 |ΔG_H - median| 排序（最接近组中位数的排第一）。
    """
    valid = [s for s in samples if s.get("dg_h") is not None]
    if len(valid) < n_clusters:
        return [valid] if valid else []

    vals = sorted([s["dg_h"] for s in valid])
    bin_edges = np.linspace(vals[0] - 0.01, vals[-1] + 0.01, n_clusters + 1)

    clusters = [[] for _ in range(n_clusters)]
    for s in valid:
        for i in range(n_clusters):
            if bin_edges[i] <= s["dg_h"] < bin_edges[i + 1]:
                clusters[i].append(s)
                break
        else:
            clusters[-1].append(s)

    for cluster in clusters:
        if len(cluster) > 1:
            median = np.median([s["dg_h"] for s in cluster])
            cluster.sort(key=lambda s: abs(s["dg_h"] - median))
    return clusters

def select_best_per_cluster(clusters: List[List[Dict]]) -> List[Dict]:
    """从每个聚类中选择最接近中位数的候选（热力学一致性最优代表）。"""
    best = []
    for cluster in clusters:
        if cluster:
            best.append(cluster[0])
    return best

def summarize_clusters(clusters: List[List[Dict]]) -> List[Dict]:
    """生成每个聚类的摘要：均值、标准差、计数、代表性候选ID。"""
    summaries = []
    for i, cluster in enumerate(clusters):
        if not cluster:
            continue
        vals = [s["dg_h"] for s in cluster]
        summaries.append({
            "cluster": i + 1,
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4) if len(vals) > 1 else 0,
            "count": len(cluster),
            "representative_id": cluster[0].get("id"),
            "dg_h_range": (round(min(vals), 4), round(max(vals), 4)),
        })
    return summaries
