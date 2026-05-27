# tests/test_eq_adapter.py
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.eq_adapter import EQAdapter


class TestEQAdapter:
    """Test EQAdapter for ΔG_H prediction."""

    def test_init_default_config(self):
        """Test adapter initializes with default config."""
        adapter = EQAdapter()
        assert adapter.config["batch_size"] == 32
        assert adapter.model_eager is None
        assert adapter.model_scripted is None

    def test_init_custom_config(self):
        """Test adapter initializes with custom config."""
        adapter = EQAdapter({"batch_size": 64, "device": "cpu"})
        assert adapter.config["batch_size"] == 64
        assert adapter.config["device"] == "cpu"

    def test_initialize_returns_true(self):
        """Test initialize returns True even without real model."""
        adapter = EQAdapter()
        result = adapter.initialize()
        assert result is True
        assert adapter.is_initialized() is True

    def test_validate_input_valid(self):
        """Valid parsed structure dict should pass validation."""
        adapter = EQAdapter()
        valid_input = {"parsed": True, "composition": "Ir2Pd2"}
        assert adapter.validate_input(valid_input) is True

    def test_validate_input_missing_parsed_flag(self):
        """Input without parsed flag should fail."""
        adapter = EQAdapter()
        assert adapter.validate_input({"composition": "Ir2Pd2"}) is False

    def test_validate_input_not_dict(self):
        """Non-dict input should fail."""
        adapter = EQAdapter()
        assert adapter.validate_input("not a dict") is False
        assert adapter.validate_input(None) is False

    def test_forward_returns_float(self):
        """forward() should return a float value."""
        adapter = EQAdapter()
        result = adapter.forward({"parsed": True})
        assert isinstance(result, float)

    def test_forward_value_in_reasonable_range(self):
        """Predicted ΔG_H should be in reasonable range (-2 to 0 eV for hydrogen)."""
        adapter = EQAdapter()
        results = [adapter.forward({"parsed": True}) for _ in range(100)]
        for r in results:
            assert -2.0 <= r <= 0.0, f"ΔG_H {r} outside reasonable range"

    def test_predict_alias_for_forward(self):
        """predict() should be alias for forward()."""
        adapter = EQAdapter()
        struct_data = {"parsed": True, "composition": "Ir2Pd2"}
        assert adapter.predict(struct_data) == adapter.forward(struct_data)

    def test_predict_batch(self):
        """predict_batch should return list of predictions."""
        adapter = EQAdapter()
        structures = [{"parsed": True} for _ in range(10)]
        results = adapter.predict_batch(structures)

        assert len(results) == 10
        assert all(isinstance(r, float) for r in results)

    def test_reload_updates_config(self):
        """reload should update config."""
        adapter = EQAdapter()
        adapter.initialize()

        result = adapter.reload({"batch_size": 128})
        assert result is True
        assert adapter.config["batch_size"] == 128
