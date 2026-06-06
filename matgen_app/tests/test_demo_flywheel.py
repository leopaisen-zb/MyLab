# tests/test_demo_flywheel.py
"""
P2: 飞轮闭环演示脚本 TDD 测试套件。

运行方式: MATGEN_DEMO=1 python -m pytest tests/test_demo_flywheel.py -q

测试策略：
  - 把脚本主逻辑封装在 run_demo(...)  中，此文件直接 import 并调用。
  - 全部使用 temp 路径（tmp_path fixture），绝不写真实 rag_data。
  - 验证 run_demo 返回的汇总 dict 包含预期字段（new_version、状态分布等）。
"""
import sys
import os
import uuid
import pytest
from pathlib import Path

# 确保 matgen_app 根目录在 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRunDemoImport:
    """脚本可导入，主函数签名正确。"""

    def test_run_demo_importable(self):
        """scripts/demo_flywheel.py 应能导入 run_demo 函数。"""
        # 把 scripts 目录加进路径
        scripts_dir = Path(__file__).parent.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))
        try:
            from demo_flywheel import run_demo
            assert callable(run_demo)
        finally:
            sys.path.remove(str(scripts_dir))


class TestRunDemoEndToEnd:
    """run_demo(...) 端到端飞轮闭环测试。"""

    def _run(self, tmp_path):
        """辅助：用 temp 路径调用 run_demo，返回汇总 dict。"""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))
        try:
            from demo_flywheel import run_demo
            return run_demo(
                db_path=str(tmp_path / "test_demo.db"),
                registry_path=str(tmp_path / "registry.json"),
                dataset_path=str(tmp_path / "demo_train.json"),
                index_path=str(tmp_path / "demo_index.json"),
            )
        finally:
            if str(scripts_dir) in sys.path:
                sys.path.remove(str(scripts_dir))

    def test_run_demo_returns_dict(self, tmp_path):
        """run_demo 应返回 dict。"""
        result = self._run(tmp_path)
        assert isinstance(result, dict)

    def test_run_demo_has_new_version(self, tmp_path):
        """run_demo 返回值应包含 new_version（重训+注册步骤完成）。"""
        result = self._run(tmp_path)
        assert "new_version" in result
        assert isinstance(result["new_version"], dict)
        assert "version" in result["new_version"]

    def test_run_demo_has_state_distribution(self, tmp_path):
        """run_demo 返回值应包含 state_distribution（状态分布）。"""
        result = self._run(tmp_path)
        assert "state_distribution" in result
        assert isinstance(result["state_distribution"], dict)

    def test_run_demo_validated_count_positive(self, tmp_path):
        """run_demo 结束时应至少有 1 个 validated 结构。"""
        result = self._run(tmp_path)
        dist = result["state_distribution"]
        validated_count = dist.get("validated", 0) + dist.get("exported_for_training", 0)
        assert validated_count >= 1, f"应有 validated 结构，实际分布: {dist}"

    def test_run_demo_has_retrain_feedback(self, tmp_path):
        """run_demo 返回值应包含 feedback（数据集回灌统计）。"""
        result = self._run(tmp_path)
        assert "feedback" in result
        fb = result["feedback"]
        assert "validated_count" in fb

    def test_run_demo_does_not_write_real_rag_data(self, tmp_path):
        """run_demo 绝不写真实 rag_data 路径。"""
        import config as _cfg
        real_dataset = _cfg.RAG_TRAIN_DATA_PATH
        real_index = _cfg.RAG_NEAR_NEIGHBOR_PATH

        # 记录运行前 mtime（若文件存在）
        before_ds = real_dataset.stat().st_mtime if real_dataset.exists() else None
        before_idx = real_index.stat().st_mtime if real_index.exists() else None

        self._run(tmp_path)

        # 文件 mtime 不变（未被修改）
        after_ds = real_dataset.stat().st_mtime if real_dataset.exists() else None
        after_idx = real_index.stat().st_mtime if real_index.exists() else None

        assert before_ds == after_ds, "run_demo 不应修改真实训练集"
        assert before_idx == after_idx, "run_demo 不应修改真实近邻索引"

    def test_run_demo_no_exception(self, tmp_path):
        """run_demo 全流程不应抛出异常。"""
        # 如果会抛，上面 _run 就已捕获到了；此测试单独跑一遍确认不抛
        try:
            self._run(tmp_path)
        except Exception as e:
            pytest.fail(f"run_demo 抛出了异常: {e}")

    def test_run_demo_structure_uuid_in_result(self, tmp_path):
        """run_demo 返回值应包含至少一个演示结构的 UUID。"""
        result = self._run(tmp_path)
        # 可能以 demo_structure_uuid 或 structure_uuids 列表形式给出
        has_uuid = (
            "demo_structure_uuid" in result
            or "structure_uuids" in result
        )
        assert has_uuid, f"返回值应含结构 UUID 信息: {list(result.keys())}"

    def test_state_history_contains_validated_transition(self, tmp_path):
        """H-1 修复验证：溯源链应包含 dft_verified→validated 转移（专家确认步骤不再断裂）。"""
        result = self._run(tmp_path)
        assert "state_history" in result, "返回值应含 state_history 键"
        history = result["state_history"]
        assert isinstance(history, list), "state_history 应为列表"
        assert any(
            h["to_state"] == "validated" for h in history
        ), f"溯源链应含 validated 转移，实际历史: {history}"
        assert any(
            h["to_state"] == "dft_verified" for h in history
        ), f"溯源链应含 dft_verified 转移，实际历史: {history}"


class TestRunDemoIsolation:
    """确保演示脚本完全隔离，使用 temp 路径。"""

    def test_run_demo_writes_to_temp_db(self, tmp_path):
        """run_demo 应将数据写入 temp db，不使用系统默认路径。"""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))
        try:
            from demo_flywheel import run_demo
            temp_db = str(tmp_path / "isolated_test.db")
            run_demo(
                db_path=temp_db,
                registry_path=str(tmp_path / "reg.json"),
                dataset_path=str(tmp_path / "ds.json"),
                index_path=str(tmp_path / "idx.json"),
            )
            assert Path(temp_db).exists(), "temp db 应被创建"
        finally:
            if str(scripts_dir) in sys.path:
                sys.path.remove(str(scripts_dir))
