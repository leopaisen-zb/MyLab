# tests/test_hea_gen_adapter.py
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.hea_gen_adapter import (
    HEAGenAdapter, generate_fake_poscar, composition_from_poscar, ELEMENT_SYMBOLS,
)


def _total_atoms(comp: str) -> int:
    """从组成串(如 Ir4Pd4Pt4)累加原子数。"""
    import re
    return sum(int(n) for n in re.findall(r"\d+", comp))


class TestCompositionFromPoscar:
    """组成提取助手：elements 字段必须与 POSCAR 实际原子一致（修复 elements 占位 bug）。"""

    def test_matches_fake_poscar_5_elements(self):
        poscar = generate_fake_poscar(["Ir", "Pd", "Pt", "Rh", "Ru"], num_atoms=20)
        comp = composition_from_poscar(poscar)
        # 5 个元素全在，且总原子数 = 20（不是只取前3个/随机）
        for el in ["Ir", "Pd", "Pt", "Rh", "Ru"]:
            assert el in comp
        assert _total_atoms(comp) == 20

    def test_total_atoms_match_num_atoms(self):
        poscar = generate_fake_poscar(["Ir", "Pd", "Pt"], num_atoms=12)
        assert _total_atoms(composition_from_poscar(poscar)) == 12

    def test_invalid_poscar_returns_empty(self):
        assert composition_from_poscar("not a poscar") == ""


class TestGenerateBatchElementsConsistency:
    """generate_batch 产出的 elements 字段必须与其 poscar 真实组成一致。"""

    def test_elements_field_matches_poscar(self, monkeypatch):
        import os
        os.environ["MATGEN_DEMO"] = "0"  # 强制走 fallback 假生成器(forward 内 backend 不可用时)
        adapter = HEAGenAdapter()
        batch = adapter.generate_batch(target_dgH=-0.5, batch_size=3)
        for item in batch:
            expected = composition_from_poscar(item["poscar"])
            assert item["elements"] == expected
            # 原子总数应与 poscar 一致，而非随机 3 元素小数
            assert _total_atoms(item["elements"]) == _total_atoms(expected)


class TestGenerateFakePoscar:
    """Test POSCAR generation helper function."""

    def test_generates_valid_poscar_format(self):
        """Generated POSCAR should have correct format."""
        poscar = generate_fake_poscar(["Ir", "Pd", "Pt"], num_atoms=6)
        lines = poscar.split("\n")

        assert len(lines) >= 7  # minimum lines for valid POSCAR
        assert lines[0] == "Generated HEA structure"
        assert "Cartesian" in lines

    def test_respects_num_atoms(self):
        """Number of coordinate lines should match num_atoms."""
        poscar = generate_fake_poscar(["Ir", "Pd"], num_atoms=10)
        lines = poscar.split("\n")

        # Find Cartesian section
        cart_idx = next(i for i, l in enumerate(lines) if l == "Cartesian")
        coord_lines = [l for l in lines[cart_idx+1:] if l.strip()]
        assert len(coord_lines) == 10

    def test_lattice_values_reasonable(self):
        """Lattice vectors should have at least one component in reasonable range (5-15 Angstrom)."""
        poscar = generate_fake_poscar(["Ir", "Pd", "Pt"], num_atoms=6)
        lines = poscar.split("\n")

        # Parse lattice lines (lines 2-4)
        for line in lines[2:5]:
            parts = line.split()
            if len(parts) == 3:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                # Check that at least one component of each lattice vector is reasonable
                # (0.0 components are allowed for non-orthogonal lattices)
                non_zero = [v for v in [x, y, z] if v != 0.0]
                if non_zero:
                    for v in non_zero:
                        assert 5 <= v <= 15, f"Lattice component {v} out of range"


class TestHEAGenAdapter:
    """Test HEAGenAdapter class."""

    def test_init_default_config(self):
        """Test adapter initializes with default config."""
        adapter = HEAGenAdapter()
        assert adapter.config["max_length"] == 512
        assert adapter.config["temperature"] == 0.7
        assert adapter._initialized is False

    def test_init_custom_config(self):
        """Test adapter initializes with custom config."""
        adapter = HEAGenAdapter({"max_length": 1024, "temperature": 0.5})
        assert adapter.config["max_length"] == 1024
        assert adapter.config["temperature"] == 0.5

    def test_initialize_returns_true(self):
        """Test initialize returns True even without real model."""
        adapter = HEAGenAdapter()
        result = adapter.initialize()
        assert result is True
        assert adapter.is_initialized() is True

    def test_initialize_twice_is_idempotent(self):
        """Calling initialize twice should not fail."""
        adapter = HEAGenAdapter()
        assert adapter.initialize() is True
        assert adapter.initialize() is True

    def test_validate_input_valid(self):
        """逆向设计：以目标性质为输入即合法。"""
        adapter = HEAGenAdapter()
        assert adapter.validate_input({"target_dgH": -0.5}) is True

    def test_validate_input_missing_target(self):
        """缺少 target_dgH 应失败。"""
        adapter = HEAGenAdapter()
        assert adapter.validate_input({}) is False

    def test_validate_input_not_dict(self):
        """Non-dict input should fail."""
        adapter = HEAGenAdapter()
        assert adapter.validate_input("not a dict") is False
        assert adapter.validate_input(None) is False

    def test_forward_returns_poscar_string(self):
        """forward() should return POSCAR string."""
        adapter = HEAGenAdapter()
        result = adapter.forward({"target_dgH": -0.5})
        assert isinstance(result, str)
        assert "Cartesian" in result

    def test_generate_batch_returns_correct_count(self):
        """generate_batch should return batch_size structures."""
        adapter = HEAGenAdapter()
        results = adapter.generate_batch(target_dgH=-0.5, batch_size=20)
        assert len(results) == 20

    def test_generate_batch_results_have_required_fields(self):
        """Each result should have elements, poscar, and target_dgH."""
        adapter = HEAGenAdapter()
        results = adapter.generate_batch(target_dgH=-0.5, batch_size=5)
        for r in results:
            assert "elements" in r
            assert "poscar" in r
            assert "target_dgH" in r
            assert r["target_dgH"] == -0.5

    def test_parse_structure_returns_dict_when_pymatgen_available(self):
        """parse_structure should return parsed structure dict."""
        adapter = HEAGenAdapter()
        poscar = generate_fake_poscar(["Ir", "Pd", "Pt"], num_atoms=6)
        result = adapter.parse_structure(poscar)

        assert result is not None
        assert "parsed" in result
        assert result["parsed"] is True
        assert "composition" in result

    def test_parse_structure_returns_none_for_invalid_poscar(self):
        """parse_structure should return None for invalid POSCAR."""
        adapter = HEAGenAdapter()
        result = adapter.parse_structure("not a valid poscar content at all")
        assert result is None

    def test_reload_updates_config(self):
        """reload should update config and reset initialization."""
        adapter = HEAGenAdapter()
        adapter.initialize()

        result = adapter.reload({"max_length": 2048})
        assert result is True
        assert adapter.config["max_length"] == 2048
        assert adapter._initialized is True
