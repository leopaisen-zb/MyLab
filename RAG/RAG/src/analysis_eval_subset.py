from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "rag_data"
    / "text2struct_rag"
)

TEST_JSON = DATA_DIR / "dataset_test_rag_code.json"
EVAL_FILES = [
    "eval_sandbox_A.json",
    "eval_sandbox_FINAL_promptB.json",
]
NEAR_NEIGHBOR_INDICES = DATA_DIR / "near_neighbor_indices.json"


@dataclass
class MetricsSummary:
    name: str
    n: int
    is_executable_ratio: Optional[float]
    is_valid_structure_ratio: Optional[float]
    is_composition_match_ratio: Optional[float]
    volume_error_mae: Optional[float]

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "name": self.name,
            "n": self.n,
            "is_executable": self.is_executable_ratio,
            "is_valid_structure": self.is_valid_structure_ratio,
            "is_composition_match": self.is_composition_match_ratio,
            "volume_error_MAE": self.volume_error_mae,
        }


def extract_near_neighbor_indices(
    max_l2_distance: float = 0.05,
) -> List[int]:
    """从测试集 JSON 中提取 L2 距离较小的“近邻子集”索引。"""
    if not TEST_JSON.exists():
        raise FileNotFoundError(f"Test JSON not found: {TEST_JSON}")

    with TEST_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    pattern = re.compile(r"Reference 1 .*L2 distance:\s*([0-9.]+)")
    chosen: List[int] = []
    for idx, sample in enumerate(data):
        input_text = sample.get("input", "")
        match = pattern.search(input_text)
        if not match:
            continue
        try:
            dist = float(match.group(1))
        except ValueError:
            continue
        if dist < max_l2_distance:
            chosen.append(idx)

    NEAR_NEIGHBOR_INDICES.parent.mkdir(parents=True, exist_ok=True)
    with NEAR_NEIGHBOR_INDICES.open("w", encoding="utf-8") as f:
        json.dump(chosen, f, ensure_ascii=False, indent=2)

    return chosen


def _load_indices() -> List[int]:
    if NEAR_NEIGHBOR_INDICES.exists():
        with NEAR_NEIGHBOR_INDICES.open("r", encoding="utf-8") as f:
            return json.load(f)
    return extract_near_neighbor_indices()


def _summarize_results(
    name: str,
    results: Iterable[Dict],
    subset_indices: Optional[Iterable[int]] = None,
) -> MetricsSummary:
    index_set = set(subset_indices) if subset_indices is not None else None

    filtered: List[Dict] = []
    for r in results:
        if index_set is not None and r.get("idx") not in index_set:
            continue
        filtered.append(r)

    n = len(filtered)
    if n == 0:
        return MetricsSummary(
            name=name,
            n=0,
            is_executable_ratio=None,
            is_valid_structure_ratio=None,
            is_composition_match_ratio=None,
            volume_error_mae=None,
        )

    exec_ok = sum(r.get("is_executable", 0) for r in filtered)
    valid_ok = sum(r.get("is_valid_structure", 0) for r in filtered)
    comp_ok = sum(r.get("is_composition_match", 0) for r in filtered)
    vol_errors = [
        r["volume_error"]
        for r in filtered
        if r.get("volume_error") is not None
    ]

    volume_mae: Optional[float]
    if vol_errors:
        volume_mae = sum(vol_errors) / len(vol_errors)
    else:
        volume_mae = None

    return MetricsSummary(
        name=name,
        n=n,
        is_executable_ratio=exec_ok / n,
        is_valid_structure_ratio=valid_ok / n,
        is_composition_match_ratio=comp_ok / n,
        volume_error_mae=volume_mae,
    )


def summarize_eval_files() -> Dict[str, List[Dict[str, Optional[float]]]]:
    """对 A/B 两个 eval 结果做全量 + 近邻子集统计。"""
    indices = _load_indices()

    summaries_overall: List[MetricsSummary] = []
    summaries_subset: List[MetricsSummary] = []

    for fname in EVAL_FILES:
        path = DATA_DIR / fname
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            results = json.load(f).get("results", [])
        summaries_overall.append(
            _summarize_results(fname, results, subset_indices=None),
        )
        summaries_subset.append(
            _summarize_results(
                f"{fname} (near-neighbor subset)",
                results,
                subset_indices=indices,
            ),
        )

    return {
        "overall": [s.to_dict() for s in summaries_overall],
        "near_neighbor_subset": [s.to_dict() for s in summaries_subset],
    }


def _pretty_print_summaries(data: Dict[str, List[Dict[str, Optional[float]]]]) -> None:
    def _fmt(x: Optional[float]) -> str:
        if x is None:
            return "N/A"
        return f"{x:.4f}"

    print("=" * 70)
    print("Overall metrics")
    print("=" * 70)
    for s in data.get("overall", []):
        print(f"- {s['name']}")
        print(f"  n                    : {s['n']}")
        print(f"  is_executable        : {_fmt(s['is_executable'])}")
        print(f"  is_valid_structure   : {_fmt(s['is_valid_structure'])}")
        print(f"  is_composition_match : {_fmt(s['is_composition_match'])}")
        print(f"  volume_error_MAE     : {_fmt(s['volume_error_MAE'])}")
        print()

    print("=" * 70)
    print("Near-neighbor subset metrics")
    print("=" * 70)
    for s in data.get("near_neighbor_subset", []):
        print(f"- {s['name']}")
        print(f"  n                    : {s['n']}")
        print(f"  is_executable        : {_fmt(s['is_executable'])}")
        print(f"  is_valid_structure   : {_fmt(s['is_valid_structure'])}")
        print(f"  is_composition_match : {_fmt(s['is_composition_match'])}")
        print(f"  volume_error_MAE     : {_fmt(s['volume_error_MAE'])}")
        print()


def main() -> None:
    print("分析近邻子集样本，并对 A/B 结果做对比统计...")
    indices = extract_near_neighbor_indices()
    print(f"近邻子集样本数: {len(indices)}")
    summary = summarize_eval_files()

    # 保存成 JSON，方便论文画表
    out_path = DATA_DIR / "eval_subset_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _pretty_print_summaries(summary)
    print(f"\n统计结果已写入: {out_path}")


if __name__ == "__main__":
    main()

