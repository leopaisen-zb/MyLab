"""
数据一致性不变式测试 —— 防止"显示/派生字段脱离真实数据"这类盲区 bug 复发。

背景：曾出现 elements 字段是占位假串（与 POSCAR 实际组成不符）、predict
fallback 返回与结构无关的常数等 bug。这类问题代码逻辑 review 与普通单元测试
都难发现，只有"跑通真实流水线 + 核对输出数据自洽"才能暴露。

本模块把那条"肉眼核对"固化为自动断言：任何流经系统的结构记录，其派生/显示字段
必须与其真实数据源（POSCAR）严格一致。新增任何"展示用字段"时，应在此补一条不变式。
"""
import os
import re
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# 整个模块在 DEMO 模式运行，避免加载大模型
os.environ.setdefault("MATGEN_DEMO", "1")

from core.task_orchestrator import TaskOrchestrator
from persistence.state_store import StateStore
from adapters.hea_gen_adapter import generate_fake_poscar, composition_from_poscar


def _total_atoms(comp: str) -> int:
    return sum(int(n) for n in re.findall(r"\d+", comp))


def _run_pipeline(batch_size=5):
    """用隔离库跑一遍真实编排流水线（fake 生成 + toy_mlp 预测），返回结构记录列表。

    逆向设计：仅以 target_dgH 为输入，元素组分由系统生成（不作为输入）。
    """
    orch = TaskOrchestrator()
    orch.state_store = StateStore(db_path=os.path.join(tempfile.mkdtemp(), "consistency.db"))
    orch.create_task("c1", {
        "target_dgH": -0.5, "tolerance": 2.0,
        "batch_size": batch_size, "gen_model_id": "fake", "pred_model_id": "toy_mlp",
    })
    orch.execute_task("c1")
    return list(orch.tasks["c1"]["structures"].values())


class TestElementsPoscarConsistency:
    """INV: 结构记录的 elements 字段必须 == 其 POSCAR 真实组成（曾经的占位串 bug）。"""

    def test_elements_equals_poscar_composition(self):
        records = _run_pipeline()
        assert records
        for r in records:
            assert r["elements"] == composition_from_poscar(r["poscar"]), \
                f"elements {r['elements']} 与 POSCAR 组成不符"

    def test_elements_atom_count_matches_poscar(self):
        records = _run_pipeline()
        for r in records:
            assert _total_atoms(r["elements"]) == _total_atoms(composition_from_poscar(r["poscar"]))

    def test_composition_generated_not_echoed(self):
        """INV(逆向设计): 元素组分由系统生成——必须来自 HEA 元素池，且跨候选有变化，
        而非把某个固定输入"组分进=组分出"地回显（这正是导师质疑的来源）。"""
        from adapters.hea_gen_adapter import ELEMENT_SYMBOLS
        records = _run_pipeline(batch_size=12)
        assert records
        seen = set()
        for r in records:
            els = set(re.findall(r"[A-Z][a-z]?", r["elements"]))
            assert els, f"组分为空: {r['elements']}"
            assert els <= set(ELEMENT_SYMBOLS), f"{els} 含元素池之外的元素"
            seen.add(frozenset(els))
        # 跨候选至少出现两种不同组分 → 证明是"生成"而非固定回显
        assert len(seen) >= 2, "全批次组分完全相同，疑似回显固定输入而非生成"


class TestPredictionNotPlaceholder:
    """INV: 预测值必须存在且有限，不是 None/占位/写死常数。"""

    def test_predicted_dgh_is_finite_number(self):
        records = _run_pipeline()
        predicted = [r for r in records if r.get("status") in ("predicted", "filtered_in", "filtered_out")]
        assert predicted, "应有结构完成预测"
        for r in predicted:
            v = r.get("predicted_dgH")
            assert isinstance(v, float) and v == v and abs(v) != float("inf")

    def test_fallback_prediction_is_structure_dependent(self, monkeypatch):
        """real 模型失败时, fallback 也须随结构变化(曾返回与结构无关的常数)。"""
        import backend.eq_predict as ep
        monkeypatch.setattr(ep, "predict",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("forced")))
        from adapters.eq_adapter import EQAdapter
        adapter = EQAdapter()
        v1 = adapter.predict(generate_fake_poscar(["Ir", "Pd", "Pt", "Rh", "Ru"], 20))
        v2 = adapter.predict(generate_fake_poscar(["Fe", "Co", "Ni"], 9))
        assert v1 != v2, "fallback 对不同组成返回了相同值（疑似与结构无关的常数）"
