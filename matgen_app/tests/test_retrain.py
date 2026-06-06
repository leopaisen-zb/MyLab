# tests/test_retrain.py
"""
P1-3: run_retrain 端到端 TDD 测试套件。
运行方式: MATGEN_DEMO=1 python -m pytest tests/test_retrain.py -q
"""
import json
import os
import uuid
import pytest

from persistence.state_store import StateStore


# ─────────────────────────────────────────────────────────────────
# 辅助：向 store 插入 validated 记录
# ─────────────────────────────────────────────────────────────────

def _populate_store(db_path: str, n: int = 3) -> StateStore:
    """插入 n 条 validated 状态的结构记录。"""
    store = StateStore(db_path=db_path)
    for i in range(n):
        uid = str(uuid.uuid4())
        store.save_record(uid, {
            "task_id": "task-retrain",
            "status": "validated",
            "elements": f"IrPd{i}",
            "poscar": f"POSCAR_RETRAIN_{i}",
            "predicted_dgH": -0.2 * (i + 1),
            "target_dgH": -0.3,
            "created_at": "2026-06-06T10:00:00",
        })
    return store


# ─────────────────────────────────────────────────────────────────
# 1. run_retrain demo 模式（核心端到端）
# ─────────────────────────────────────────────────────────────────

class TestRunRetrainDemo:
    """demo=True 时，run_retrain 跳过真实训练，直接注册 mock 版本。"""

    def test_import_run_retrain(self):
        """run_retrain 应能从 backend.retrain 导入。"""
        from backend.retrain import run_retrain
        assert run_retrain is not None

    def test_demo_prediction_model_registers_new_version(self, tmp_path, temp_db):
        """demo 模式对 prediction 模型应注册一个新版本。"""
        from backend.retrain import run_retrain
        from backend.model_registry import ModelVersionRegistry

        store = _populate_store(temp_db, n=3)
        registry = ModelVersionRegistry(path=str(tmp_path / "versions.json"))
        dataset_path = str(tmp_path / "train.json")
        index_path = str(tmp_path / "index.json")

        result = run_retrain(
            model_kind="prediction",
            store=store,
            registry=registry,
            demo=True,
            dataset_path=dataset_path,
            index_path=index_path,
        )

        # 应有新版本被注册
        import config as _cfg
        versions = registry.get_versions(_cfg.DEFAULT_PRED_MODEL_ID)
        assert len(versions) == 1
        assert versions[0]["version"] == "v1"

    def test_demo_generation_model_registers_new_version(self, tmp_path, temp_db):
        """demo 模式对 generation 模型应注册一个新版本。"""
        from backend.retrain import run_retrain
        from backend.model_registry import ModelVersionRegistry
        import config as _cfg

        store = _populate_store(temp_db, n=2)
        registry = ModelVersionRegistry(path=str(tmp_path / "versions.json"))
        dataset_path = str(tmp_path / "train.json")
        index_path = str(tmp_path / "index.json")

        result = run_retrain(
            model_kind="generation",
            store=store,
            registry=registry,
            demo=True,
            dataset_path=dataset_path,
            index_path=index_path,
        )

        versions = registry.get_versions(_cfg.DEFAULT_GEN_MODEL_ID)
        assert len(versions) == 1
        assert versions[0]["version"] == "v1"

    def test_demo_result_contains_new_version_info(self, tmp_path, temp_db):
        """run_retrain 返回值应包含 new_version 信息。"""
        from backend.retrain import run_retrain
        from backend.model_registry import ModelVersionRegistry

        store = _populate_store(temp_db, n=2)
        registry = ModelVersionRegistry(path=str(tmp_path / "versions.json"))
        dataset_path = str(tmp_path / "train.json")
        index_path = str(tmp_path / "index.json")

        result = run_retrain(
            model_kind="prediction",
            store=store,
            registry=registry,
            demo=True,
            dataset_path=dataset_path,
            index_path=index_path,
        )

        assert "new_version" in result
        assert result["new_version"]["version"] == "v1"

    def test_demo_result_contains_feedback_stats(self, tmp_path, temp_db):
        """run_retrain 返回值应包含 feedback 统计（appended / validated_count）。"""
        from backend.retrain import run_retrain
        from backend.model_registry import ModelVersionRegistry

        store = _populate_store(temp_db, n=3)
        registry = ModelVersionRegistry(path=str(tmp_path / "versions.json"))
        dataset_path = str(tmp_path / "train.json")
        index_path = str(tmp_path / "index.json")

        result = run_retrain(
            model_kind="prediction",
            store=store,
            registry=registry,
            demo=True,
            dataset_path=dataset_path,
            index_path=index_path,
        )

        assert "feedback" in result
        fb = result["feedback"]
        assert "appended" in fb
        assert "validated_count" in fb
        assert fb["validated_count"] == 3

    def test_demo_version_notes_marked_as_demo(self, tmp_path, temp_db):
        """demo 模式注册的版本 notes 应标注包含 'demo'。"""
        from backend.retrain import run_retrain
        from backend.model_registry import ModelVersionRegistry
        import config as _cfg

        store = _populate_store(temp_db, n=1)
        registry = ModelVersionRegistry(path=str(tmp_path / "versions.json"))
        dataset_path = str(tmp_path / "train.json")
        index_path = str(tmp_path / "index.json")

        run_retrain(
            model_kind="prediction",
            store=store,
            registry=registry,
            demo=True,
            dataset_path=dataset_path,
            index_path=index_path,
        )

        latest = registry.get_latest_version(_cfg.DEFAULT_PRED_MODEL_ID)
        assert "demo" in latest["notes"].lower()

    def test_demo_env_var_triggers_demo_mode(self, tmp_path, temp_db, monkeypatch):
        """MATGEN_DEMO=1 环境变量也应触发 demo 模式（即使 demo=False）。"""
        monkeypatch.setenv("MATGEN_DEMO", "1")
        from backend.retrain import run_retrain
        from backend.model_registry import ModelVersionRegistry
        import config as _cfg
        import importlib
        import backend.retrain as retrain_mod
        importlib.reload(retrain_mod)  # 重新加载使 env var 生效
        run_retrain_fn = retrain_mod.run_retrain

        store = _populate_store(temp_db, n=1)
        registry = ModelVersionRegistry(path=str(tmp_path / "versions.json"))
        dataset_path = str(tmp_path / "train.json")
        index_path = str(tmp_path / "index.json")

        result = run_retrain_fn(
            model_kind="prediction",
            store=store,
            registry=registry,
            demo=False,  # 显式关闭，但 env 为 1
            dataset_path=dataset_path,
            index_path=index_path,
        )

        # 应该走 demo 路径（不真实跑训练）
        versions = registry.get_versions(_cfg.DEFAULT_PRED_MODEL_ID)
        assert len(versions) == 1

    def test_demo_feedback_pipeline_runs(self, tmp_path, temp_db):
        """demo 模式应先跑 feedback_pipeline（数据集文件应存在）。"""
        from backend.retrain import run_retrain
        from backend.model_registry import ModelVersionRegistry

        store = _populate_store(temp_db, n=2)
        registry = ModelVersionRegistry(path=str(tmp_path / "versions.json"))
        dataset_path = str(tmp_path / "train.json")
        index_path = str(tmp_path / "index.json")

        run_retrain(
            model_kind="prediction",
            store=store,
            registry=registry,
            demo=True,
            dataset_path=dataset_path,
            index_path=index_path,
        )

        assert os.path.exists(dataset_path), "feedback_pipeline 应生成数据集文件"

    def test_demo_version_metrics_contains_feedback_count(self, tmp_path, temp_db):
        """demo 注册的版本 metrics 应包含回灌条数统计。"""
        from backend.retrain import run_retrain
        from backend.model_registry import ModelVersionRegistry
        import config as _cfg

        store = _populate_store(temp_db, n=4)
        registry = ModelVersionRegistry(path=str(tmp_path / "versions.json"))
        dataset_path = str(tmp_path / "train.json")
        index_path = str(tmp_path / "index.json")

        run_retrain(
            model_kind="prediction",
            store=store,
            registry=registry,
            demo=True,
            dataset_path=dataset_path,
            index_path=index_path,
        )

        latest = registry.get_latest_version(_cfg.DEFAULT_PRED_MODEL_ID)
        assert latest["metrics"] is not None
        # metrics 里应有 appended 或 validated_count 之类的数字
        assert any(isinstance(v, (int, float)) for v in latest["metrics"].values())


# ─────────────────────────────────────────────────────────────────
# 2. run_retrain 非 demo 模式（mock subprocess）
# ─────────────────────────────────────────────────────────────────

class TestRunRetrainNonDemo:
    """非 demo 模式时应构造并调用真实训练命令（mock subprocess 不真跑）。"""

    def test_non_demo_prediction_calls_subprocess(
        self, tmp_path, temp_db, monkeypatch
    ):
        """非 demo prediction 重训应调用 subprocess（命令含 train_equiformer）。"""
        import subprocess
        from backend.retrain import run_retrain
        from backend.model_registry import ModelVersionRegistry

        called_cmds = []

        def mock_run(cmd, *args, **kwargs):
            called_cmds.append(cmd)
            # 模拟成功返回
            class FakeResult:
                returncode = 0
                stdout = ""
                stderr = ""
            return FakeResult()

        monkeypatch.setenv("MATGEN_DEMO", "0")
        monkeypatch.setattr("subprocess.run", mock_run)

        store = _populate_store(temp_db, n=1)
        registry = ModelVersionRegistry(path=str(tmp_path / "versions.json"))
        dataset_path = str(tmp_path / "train.json")
        index_path = str(tmp_path / "index.json")

        run_retrain(
            model_kind="prediction",
            store=store,
            registry=registry,
            demo=False,
            dataset_path=dataset_path,
            index_path=index_path,
        )

        assert len(called_cmds) >= 1
        # 命令列表中应包含 train_equiformer
        all_cmd_strs = " ".join(str(c) for c in called_cmds[0])
        assert "train_equiformer" in all_cmd_strs

    def test_non_demo_generation_calls_subprocess(
        self, tmp_path, temp_db, monkeypatch
    ):
        """非 demo generation 重训应调用 subprocess（命令含 lora 或 finetune）。"""
        from backend.retrain import run_retrain
        from backend.model_registry import ModelVersionRegistry

        called_cmds = []

        def mock_run(cmd, *args, **kwargs):
            called_cmds.append(cmd)
            class FakeResult:
                returncode = 0
                stdout = ""
                stderr = ""
            return FakeResult()

        monkeypatch.setenv("MATGEN_DEMO", "0")
        monkeypatch.setattr("subprocess.run", mock_run)

        store = _populate_store(temp_db, n=1)
        registry = ModelVersionRegistry(path=str(tmp_path / "versions.json"))
        dataset_path = str(tmp_path / "train.json")
        index_path = str(tmp_path / "index.json")

        run_retrain(
            model_kind="generation",
            store=store,
            registry=registry,
            demo=False,
            dataset_path=dataset_path,
            index_path=index_path,
        )

        assert len(called_cmds) >= 1

    def test_non_demo_registers_version_after_success(
        self, tmp_path, temp_db, monkeypatch
    ):
        """subprocess 成功返回后应注册新版本。"""
        from backend.retrain import run_retrain
        from backend.model_registry import ModelVersionRegistry
        import config as _cfg

        def mock_run(cmd, *args, **kwargs):
            class FakeResult:
                returncode = 0
                stdout = ""
                stderr = ""
            return FakeResult()

        monkeypatch.setenv("MATGEN_DEMO", "0")
        monkeypatch.setattr("subprocess.run", mock_run)

        store = _populate_store(temp_db, n=1)
        registry = ModelVersionRegistry(path=str(tmp_path / "versions.json"))
        dataset_path = str(tmp_path / "train.json")
        index_path = str(tmp_path / "index.json")

        run_retrain(
            model_kind="prediction",
            store=store,
            registry=registry,
            demo=False,
            dataset_path=dataset_path,
            index_path=index_path,
        )

        versions = registry.get_versions(_cfg.DEFAULT_PRED_MODEL_ID)
        assert len(versions) == 1


# ─────────────────────────────────────────────────────────────────
# 3. invalid model_kind
# ─────────────────────────────────────────────────────────────────

class TestRunRetrainValidation:
    def test_invalid_model_kind_raises(self, tmp_path, temp_db):
        """model_kind 不在合法值集合时应抛 ValueError。"""
        from backend.retrain import run_retrain
        from backend.model_registry import ModelVersionRegistry

        store = StateStore(db_path=temp_db)
        registry = ModelVersionRegistry(path=str(tmp_path / "versions.json"))

        with pytest.raises(ValueError, match="model_kind"):
            run_retrain(
                model_kind="unknown_kind",
                store=store,
                registry=registry,
                demo=True,
                dataset_path=str(tmp_path / "train.json"),
                index_path=str(tmp_path / "index.json"),
            )


# ─────────────────────────────────────────────────────────────────
# 4. HIGH-1: demo retrain 不写真实 RAG 训练集
# ─────────────────────────────────────────────────────────────────

class TestDemoIsolation:
    """HIGH-1：demo retrain 必须写 demo 隔离路径，不得写真实 RAG 训练集。"""

    def test_demo_retrain_uses_demo_path_not_real_path(
        self, tmp_path, temp_db, monkeypatch
    ):
        """
        断言：demo=True 时，feedback_pipeline 接收的 dataset_path
        是 demo 隔离路径（config.DEMO_TRAIN_DATA_PATH），不是真实路径
        （config.RAG_TRAIN_DATA_PATH）。
        """
        import config as _cfg
        from backend.retrain import run_retrain
        from backend.model_registry import ModelVersionRegistry

        captured = {}

        def mock_feedback_pipeline(store, dataset_path, index_path, instruction):
            captured["dataset_path"] = dataset_path
            captured["index_path"] = index_path
            return {
                "validated_count": 0,
                "appended": 0,
                "skipped_duplicates": 0,
                "total": 0,
                "num_entries": 0,
                "index_path": index_path,
            }

        monkeypatch.setattr("backend.retrain.feedback_pipeline", mock_feedback_pipeline)

        store = StateStore(db_path=temp_db)
        registry = ModelVersionRegistry(path=str(tmp_path / "versions.json"))
        demo_dataset = str(tmp_path / "demo_train.json")
        demo_index = str(tmp_path / "demo_index.json")

        run_retrain(
            model_kind="prediction",
            store=store,
            registry=registry,
            demo=True,
            dataset_path=demo_dataset,
            index_path=demo_index,
        )

        # 断言：传入 feedback_pipeline 的路径是 demo 路径，不是真实 RAG 路径
        assert captured["dataset_path"] == demo_dataset
        assert captured["index_path"] == demo_index
        assert captured["dataset_path"] != str(_cfg.RAG_TRAIN_DATA_PATH)
        assert captured["index_path"] != str(_cfg.RAG_NEAR_NEIGHBOR_PATH)

    def test_api_demo_retrain_uses_config_demo_paths(self, monkeypatch):
        """
        HIGH-1：通过 API 调用 /models/retrain 时，run_retrain 收到的
        dataset_path 必须是 config.DEMO_TRAIN_DATA_PATH（demo 隔离），
        而非 config.RAG_TRAIN_DATA_PATH（真实训练集）。
        """
        import config as _cfg

        captured = {}

        def mock_run_retrain(**kwargs):
            captured.update(kwargs)
            return {
                "model_kind": kwargs.get("model_kind"),
                "demo": True,
                "feedback": {"validated_count": 0, "appended": 0,
                             "skipped_duplicates": 0, "total": 0,
                             "num_entries": 0, "index_path": ""},
                "new_version": {"version": "v1", "checkpoint_path": "mock",
                                "created_at": "2026-01-01", "notes": "demo",
                                "metrics": {}},
            }

        monkeypatch.setattr("api.routes.run_retrain", mock_run_retrain)

        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.routes import router

        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        client = TestClient(app)

        resp = client.post("/api/v1/models/retrain", json={"model_kind": "prediction"})
        assert resp.status_code == 200

        assert captured.get("dataset_path") == str(_cfg.DEMO_TRAIN_DATA_PATH)
        assert captured.get("index_path") == str(_cfg.DEMO_NEAR_NEIGHBOR_PATH)
        assert captured.get("dataset_path") != str(_cfg.RAG_TRAIN_DATA_PATH)


# ─────────────────────────────────────────────────────────────────
# 5. HIGH-2: 非 demo 模式训练命令含传入的 dataset_path
# ─────────────────────────────────────────────────────────────────

class TestDatasetPathPropagation:
    """HIGH-2：非 demo 训练命令必须包含传入的 dataset_path，而非硬编码路径。"""

    def test_prediction_cmd_contains_dataset_path(
        self, tmp_path, temp_db, monkeypatch
    ):
        """非 demo prediction 训练命令应包含传入的 dataset_path。"""
        from backend.retrain import run_retrain
        from backend.model_registry import ModelVersionRegistry

        called_cmds = []

        def mock_run(cmd, *args, **kwargs):
            called_cmds.append(cmd)
            class FakeResult:
                returncode = 0
                stdout = ""
                stderr = ""
            return FakeResult()

        monkeypatch.setenv("MATGEN_DEMO", "0")
        monkeypatch.setattr("subprocess.run", mock_run)

        store = _populate_store(temp_db, n=1)
        registry = ModelVersionRegistry(path=str(tmp_path / "versions.json"))
        custom_dataset = str(tmp_path / "custom_train.json")
        index_path = str(tmp_path / "index.json")

        run_retrain(
            model_kind="prediction",
            store=store,
            registry=registry,
            demo=False,
            dataset_path=custom_dataset,
            index_path=index_path,
        )

        assert len(called_cmds) >= 1
        all_cmd_str = " ".join(str(c) for c in called_cmds[0])
        # 命令应包含自定义 dataset 路径，而非硬编码的 RAG_TRAIN_DATA_PATH
        assert custom_dataset in all_cmd_str

    def test_generation_cmd_contains_dataset_path(
        self, tmp_path, temp_db, monkeypatch
    ):
        """非 demo generation 训练命令应包含传入的 dataset_path。"""
        from backend.retrain import run_retrain
        from backend.model_registry import ModelVersionRegistry

        called_cmds = []

        def mock_run(cmd, *args, **kwargs):
            called_cmds.append(cmd)
            class FakeResult:
                returncode = 0
                stdout = ""
                stderr = ""
            return FakeResult()

        monkeypatch.setenv("MATGEN_DEMO", "0")
        monkeypatch.setattr("subprocess.run", mock_run)

        store = _populate_store(temp_db, n=1)
        registry = ModelVersionRegistry(path=str(tmp_path / "versions.json"))
        custom_dataset = str(tmp_path / "custom_lora_train.json")
        index_path = str(tmp_path / "index.json")

        run_retrain(
            model_kind="generation",
            store=store,
            registry=registry,
            demo=False,
            dataset_path=custom_dataset,
            index_path=index_path,
        )

        assert len(called_cmds) >= 1
        all_cmd_str = " ".join(str(c) for c in called_cmds[0])
        assert custom_dataset in all_cmd_str

    def test_non_demo_subprocess_has_cwd(self, tmp_path, temp_db, monkeypatch):
        """MEDIUM-3：非 demo subprocess.run 应传 cwd=PROJECT_ROOT。"""
        import config as _cfg
        from backend.retrain import run_retrain
        from backend.model_registry import ModelVersionRegistry

        captured_kwargs = {}

        def mock_run(cmd, *args, **kwargs):
            captured_kwargs.update(kwargs)
            class FakeResult:
                returncode = 0
                stdout = ""
                stderr = ""
            return FakeResult()

        monkeypatch.setenv("MATGEN_DEMO", "0")
        monkeypatch.setattr("subprocess.run", mock_run)

        store = _populate_store(temp_db, n=1)
        registry = ModelVersionRegistry(path=str(tmp_path / "versions.json"))

        run_retrain(
            model_kind="prediction",
            store=store,
            registry=registry,
            demo=False,
            dataset_path=str(tmp_path / "train.json"),
            index_path=str(tmp_path / "index.json"),
        )

        assert "cwd" in captured_kwargs
        assert captured_kwargs["cwd"] == str(_cfg.PROJECT_ROOT)
