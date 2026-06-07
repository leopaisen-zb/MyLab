"""展示层中文映射测试 —— 含完整性不变式（每个状态都必须有中文标签）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.state_machine import StructureState
from labels import (
    STATE_LABELS, state_label, field_label, localize_record, localize_distribution,
)


class TestStateLabelCompleteness:
    """不变式：状态机里每个状态都必须有对应中文标签（新增状态忘了补会红）。"""

    def test_every_state_has_chinese_label(self):
        for s in StructureState:
            assert s.value in STATE_LABELS, f"状态 {s.value} 缺少中文标签"

    def test_no_orphan_labels(self):
        valid = {s.value for s in StructureState}
        for key in STATE_LABELS:
            assert key in valid, f"STATE_LABELS 多了状态机不存在的键 {key}"


class TestLabelHelpers:
    def test_state_label_known(self):
        assert state_label("dft_verified") == "DFT已验证"

    def test_state_label_unknown_passthrough(self):
        assert state_label("xyz") == "xyz"

    def test_field_label_known(self):
        assert field_label("predicted_dgH") == "预测ΔG_H(eV)"

    def test_field_label_unknown_passthrough(self):
        assert field_label("foo") == "foo"


class TestLocalize:
    def test_localize_record_translates_keys_and_status(self):
        rec = {"uuid": "abc", "status": "filtered_in", "predicted_dgH": -0.5}
        out = localize_record(rec)
        assert out["结构ID"] == "abc"
        assert out["状态"] == "初筛通过"
        assert out["预测ΔG_H(eV)"] == -0.5
        # 原记录不被修改
        assert rec["status"] == "filtered_in"

    def test_localize_distribution(self):
        out = localize_distribution({"filtered_in": 4, "validated": 2})
        assert out == {"初筛通过": 4, "专家确认": 2}
