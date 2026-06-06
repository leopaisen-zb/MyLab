# tests/test_model_registry_versions.py
"""
P1-3: ModelVersionRegistry 持久化版本注册表 的 TDD 测试套件。
运行方式: MATGEN_DEMO=1 python -m pytest tests/test_model_registry_versions.py -q
"""
import json
import os
import pytest
from pathlib import Path


# ─────────────────────────────────────────────────────────────────
# 1. 基本构造与空注册表
# ─────────────────────────────────────────────────────────────────

class TestModelVersionRegistryInit:
    """构造 ModelVersionRegistry，文件不存在时视为空注册表。"""

    def test_import_model_version_registry(self):
        """ModelVersionRegistry 应能从 backend.model_registry 导入。"""
        from backend.model_registry import ModelVersionRegistry
        assert ModelVersionRegistry is not None

    def test_create_registry_with_temp_path(self, tmp_path):
        """可传入临时路径构造 registry，文件不存在时不报错。"""
        from backend.model_registry import ModelVersionRegistry
        path = tmp_path / "versions.json"
        registry = ModelVersionRegistry(path=str(path))
        assert registry is not None

    def test_list_all_empty_when_no_file(self, tmp_path):
        """文件不存在时 list_all() 应返回空 dict。"""
        from backend.model_registry import ModelVersionRegistry
        path = tmp_path / "versions.json"
        registry = ModelVersionRegistry(path=str(path))
        result = registry.list_all()
        assert result == {}

    def test_get_versions_empty_when_no_file(self, tmp_path):
        """文件不存在时 get_versions('some_id') 应返回空 list。"""
        from backend.model_registry import ModelVersionRegistry
        registry = ModelVersionRegistry(path=str(tmp_path / "v.json"))
        assert registry.get_versions("nonexistent_model") == []

    def test_get_latest_version_returns_none_when_empty(self, tmp_path):
        """文件不存在时 get_latest_version() 应返回 None。"""
        from backend.model_registry import ModelVersionRegistry
        registry = ModelVersionRegistry(path=str(tmp_path / "v.json"))
        assert registry.get_latest_version("nonexistent_model") is None


# ─────────────────────────────────────────────────────────────────
# 2. register_version 版本自增
# ─────────────────────────────────────────────────────────────────

class TestRegisterVersion:
    """register_version 应追加版本、自增版本号、记录必要字段。"""

    def test_register_first_version(self, tmp_path):
        """第一次注册应返回版本号 v1。"""
        from backend.model_registry import ModelVersionRegistry
        registry = ModelVersionRegistry(path=str(tmp_path / "v.json"))
        result = registry.register_version(
            model_id="equiformer_v2",
            checkpoint_path="/path/to/ckpt_v1.pt",
        )
        assert result["version"] == "v1"

    def test_register_second_version_increments(self, tmp_path):
        """第二次注册同一 model_id 应自增到 v2。"""
        from backend.model_registry import ModelVersionRegistry
        registry = ModelVersionRegistry(path=str(tmp_path / "v.json"))
        registry.register_version("equiformer_v2", "/ckpt_v1.pt")
        result = registry.register_version("equiformer_v2", "/ckpt_v2.pt")
        assert result["version"] == "v2"

    def test_register_version_records_checkpoint_path(self, tmp_path):
        """注册返回值应包含正确的 checkpoint_path。"""
        from backend.model_registry import ModelVersionRegistry
        registry = ModelVersionRegistry(path=str(tmp_path / "v.json"))
        result = registry.register_version("equiformer_v2", "/some/ckpt.pt")
        assert result["checkpoint_path"] == "/some/ckpt.pt"

    def test_register_version_records_created_at(self, tmp_path):
        """注册返回值应包含 created_at（ISO 格式字符串）。"""
        from backend.model_registry import ModelVersionRegistry
        registry = ModelVersionRegistry(path=str(tmp_path / "v.json"))
        result = registry.register_version("equiformer_v2", "/ckpt.pt")
        assert "created_at" in result
        assert isinstance(result["created_at"], str)
        assert "T" in result["created_at"]  # ISO 8601 含 T

    def test_register_version_records_notes(self, tmp_path):
        """传入 notes 时注册返回值应含正确 notes。"""
        from backend.model_registry import ModelVersionRegistry
        registry = ModelVersionRegistry(path=str(tmp_path / "v.json"))
        result = registry.register_version(
            "qwen_lora", "/lora.pt", notes="demo retrain"
        )
        assert result["notes"] == "demo retrain"

    def test_register_version_records_metrics(self, tmp_path):
        """传入 metrics dict 时注册返回值应含 metrics。"""
        from backend.model_registry import ModelVersionRegistry
        registry = ModelVersionRegistry(path=str(tmp_path / "v.json"))
        metrics = {"appended": 5, "rmse": 0.12}
        result = registry.register_version(
            "equiformer_v2", "/ckpt.pt", metrics=metrics
        )
        assert result["metrics"] == metrics

    def test_register_different_model_ids_independent(self, tmp_path):
        """不同 model_id 的版本号各自独立自增。"""
        from backend.model_registry import ModelVersionRegistry
        registry = ModelVersionRegistry(path=str(tmp_path / "v.json"))
        r1 = registry.register_version("equiformer_v2", "/a.pt")
        r2 = registry.register_version("qwen_lora", "/b.pt")
        r3 = registry.register_version("equiformer_v2", "/c.pt")
        assert r1["version"] == "v1"
        assert r2["version"] == "v1"  # 不同 model_id，从 v1 开始
        assert r3["version"] == "v2"


# ─────────────────────────────────────────────────────────────────
# 3. get_versions / get_latest_version / list_all
# ─────────────────────────────────────────────────────────────────

class TestQueryMethods:
    """查询方法行为。"""

    def test_get_versions_returns_all(self, tmp_path):
        """get_versions 应返回所有已注册版本列表。"""
        from backend.model_registry import ModelVersionRegistry
        registry = ModelVersionRegistry(path=str(tmp_path / "v.json"))
        registry.register_version("equiformer_v2", "/a.pt")
        registry.register_version("equiformer_v2", "/b.pt")
        registry.register_version("equiformer_v2", "/c.pt")
        versions = registry.get_versions("equiformer_v2")
        assert len(versions) == 3

    def test_get_versions_ordered_by_version(self, tmp_path):
        """get_versions 返回列表应按版本号升序排列。"""
        from backend.model_registry import ModelVersionRegistry
        registry = ModelVersionRegistry(path=str(tmp_path / "v.json"))
        for i in range(4):
            registry.register_version("equiformer_v2", f"/ckpt_{i}.pt")
        versions = registry.get_versions("equiformer_v2")
        assert [v["version"] for v in versions] == ["v1", "v2", "v3", "v4"]

    def test_get_latest_version_returns_newest(self, tmp_path):
        """get_latest_version 应返回版本号最大的版本。"""
        from backend.model_registry import ModelVersionRegistry
        registry = ModelVersionRegistry(path=str(tmp_path / "v.json"))
        registry.register_version("equiformer_v2", "/a.pt")
        registry.register_version("equiformer_v2", "/b.pt")
        latest = registry.get_latest_version("equiformer_v2")
        assert latest["version"] == "v2"
        assert latest["checkpoint_path"] == "/b.pt"

    def test_list_all_contains_all_model_ids(self, tmp_path):
        """list_all 应包含所有注册过的 model_id。"""
        from backend.model_registry import ModelVersionRegistry
        registry = ModelVersionRegistry(path=str(tmp_path / "v.json"))
        registry.register_version("equiformer_v2", "/a.pt")
        registry.register_version("qwen_lora", "/b.pt")
        all_data = registry.list_all()
        assert "equiformer_v2" in all_data
        assert "qwen_lora" in all_data

    def test_list_all_values_are_lists(self, tmp_path):
        """list_all 的值应为版本列表。"""
        from backend.model_registry import ModelVersionRegistry
        registry = ModelVersionRegistry(path=str(tmp_path / "v.json"))
        registry.register_version("equiformer_v2", "/a.pt")
        all_data = registry.list_all()
        assert isinstance(all_data["equiformer_v2"], list)
        assert len(all_data["equiformer_v2"]) == 1


# ─────────────────────────────────────────────────────────────────
# 4. JSON 持久化（写后新建实例能读到）
# ─────────────────────────────────────────────────────────────────

class TestPersistence:
    """写入后新建 Registry 实例，仍能读到数据。"""

    def test_versions_persist_across_instances(self, tmp_path):
        """写入3个版本后新建 registry 实例，仍能 get_versions。"""
        from backend.model_registry import ModelVersionRegistry
        path = str(tmp_path / "v.json")
        reg1 = ModelVersionRegistry(path=path)
        for i in range(3):
            reg1.register_version("equiformer_v2", f"/ckpt_{i}.pt")

        # 新实例读取
        reg2 = ModelVersionRegistry(path=path)
        versions = reg2.get_versions("equiformer_v2")
        assert len(versions) == 3
        assert versions[-1]["version"] == "v3"

    def test_json_file_is_valid_json(self, tmp_path):
        """生成的 JSON 文件应为合法 JSON，可用 json.load 读取。"""
        from backend.model_registry import ModelVersionRegistry
        path = tmp_path / "v.json"
        registry = ModelVersionRegistry(path=str(path))
        registry.register_version("equiformer_v2", "/ckpt.pt", notes="test")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "equiformer_v2" in data

    def test_subsequent_register_increments_correctly_after_reload(self, tmp_path):
        """跨实例累计注册，版本号应连续递增。"""
        from backend.model_registry import ModelVersionRegistry
        path = str(tmp_path / "v.json")
        ModelVersionRegistry(path=path).register_version("qwen_lora", "/a.pt")
        ModelVersionRegistry(path=path).register_version("qwen_lora", "/b.pt")
        reg3 = ModelVersionRegistry(path=path)
        result = reg3.register_version("qwen_lora", "/c.pt")
        assert result["version"] == "v3"

    def test_file_uses_non_ascii_safe_encoding(self, tmp_path):
        """JSON 文件应用 ensure_ascii=False 写入（支持中文 notes）。"""
        from backend.model_registry import ModelVersionRegistry
        path = tmp_path / "v.json"
        registry = ModelVersionRegistry(path=str(path))
        registry.register_version("qwen_lora", "/b.pt", notes="演示重训版本")
        content = path.read_text(encoding="utf-8")
        assert "演示重训版本" in content  # 中文未被转义
