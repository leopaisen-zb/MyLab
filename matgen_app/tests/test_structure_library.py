"""真实结构库测试 —— 按目标检索 + 数据一致性（elements 必须来自真实 POSCAR）。"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config as _cfg
from backend.structure_library import sample_structures, _load_library
from adapters.hea_gen_adapter import composition_from_poscar

# 库依赖真实语料；缺失则跳过（不同环境可能没有 RAG 数据）
_HAS_DATA = _cfg.RAG_TRAIN_DATA_PATH.exists()
pytestmark = pytest.mark.skipif(not _HAS_DATA, reason="RAG 训练语料不在，跳过结构库测试")


def _total_atoms(comp: str) -> int:
    return sum(int(n) for n in re.findall(r"\d+", comp))


class TestSampleStructures:
    def test_returns_requested_count(self):
        res = sample_structures(target_dgH=-0.5, batch_size=5)
        assert len(res) == 5

    def test_each_has_real_poscar_and_target(self):
        res = sample_structures(target_dgH=-0.5, batch_size=3)
        for r in res:
            assert r["poscar"] and ("Cartesian" in r["poscar"] or "Direct" in r["poscar"])
            assert r["target_dgH"] == -0.5
            assert "source_target_dgH" in r

    def test_elements_consistent_with_poscar(self):
        """数据完整性不变式：elements 必须 == 真实 POSCAR 组成。"""
        res = sample_structures(target_dgH=-0.3, batch_size=8)
        for r in res:
            assert r["elements"] == composition_from_poscar(r["poscar"])
            assert _total_atoms(r["elements"]) > 0

    def test_retrieval_is_target_conditioned(self):
        """检索应贴近目标：取到的结构源目标应集中在请求目标附近。"""
        target = -0.5
        res = sample_structures(target_dgH=target, batch_size=10)
        # 候选池是"最接近的 N 条"，故偏差应较小（放宽到 0.5 eV 容忍随机池）
        for r in res:
            assert abs(r["source_target_dgH"] - target) < 0.5

    def test_different_targets_give_different_structures(self):
        low = {r["elements"] for r in sample_structures(target_dgH=-1.5, batch_size=10)}
        high = {r["elements"] for r in sample_structures(target_dgH=1.5, batch_size=10)}
        # 极端不同的目标应检索到基本不同的结构集
        assert low != high
