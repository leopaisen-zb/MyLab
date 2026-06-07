# tests/test_model_registry.py
"""
P0-2: 模型注册表 + 多模型切换 的测试套件。
运行方式: MATGEN_DEMO=1 python -m pytest tests/test_model_registry.py -q
"""
import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# 确保整个测试模块在 DEMO 模式下运行，防止在有 transformers 的服务器上挂起/OOM
os.environ.setdefault("MATGEN_DEMO", "1")


# ── 1. 注册表存在性 ──────────────────────────────────────────────────────────

class TestRegistryExists:
    """config.py 里必须包含两个模型注册表和两个默认 id 常量（只含真实模型）。"""

    def test_gen_models_is_list(self):
        """GEN_MODELS 应是非空 list。"""
        from config import GEN_MODELS
        assert isinstance(GEN_MODELS, list)
        assert len(GEN_MODELS) >= 1

    def test_pred_models_is_list(self):
        """PRED_MODELS 应是非空 list。"""
        from config import PRED_MODELS
        assert isinstance(PRED_MODELS, list)
        assert len(PRED_MODELS) >= 1

    def test_gen_model_entries_have_required_fields(self):
        """每个 GEN_MODELS 条目必须含 id / name / description / type 字段。"""
        from config import GEN_MODELS
        required = {"id", "name", "description", "type"}
        for entry in GEN_MODELS:
            assert required.issubset(entry.keys()), f"缺少字段: {entry}"

    def test_pred_model_entries_have_required_fields(self):
        """每个 PRED_MODELS 条目必须含 id / name / description / type 字段。"""
        from config import PRED_MODELS
        required = {"id", "name", "description", "type"}
        for entry in PRED_MODELS:
            assert required.issubset(entry.keys()), f"缺少字段: {entry}"

    def test_gen_model_ids_are_unique(self):
        """GEN_MODELS 的 id 不应重复。"""
        from config import GEN_MODELS
        ids = [e["id"] for e in GEN_MODELS]
        assert len(ids) == len(set(ids))

    def test_pred_model_ids_are_unique(self):
        """PRED_MODELS 的 id 不应重复。"""
        from config import PRED_MODELS
        ids = [e["id"] for e in PRED_MODELS]
        assert len(ids) == len(set(ids))

    def test_gen_registry_contains_only_real_models(self):
        """GEN_MODELS 只含真实模型，不应有 demo 条目（如 fake）。"""
        from config import GEN_MODELS
        ids = [e["id"] for e in GEN_MODELS]
        assert "qwen_lora" in ids, "qwen_lora 应在 GEN_MODELS 中"
        assert "fake" not in ids, "fake 已移出注册表，由 MATGEN_DEMO=1 环境变量驱动"

    def test_pred_registry_contains_only_real_models(self):
        """PRED_MODELS 只含真实模型，不应有 demo 条目（如 toy_mlp）。"""
        from config import PRED_MODELS
        ids = [e["id"] for e in PRED_MODELS]
        assert "equiformer_v2" in ids, "equiformer_v2 应在 PRED_MODELS 中"
        assert "toy_mlp" not in ids, "toy_mlp 已移出注册表，由 MATGEN_DEMO=1 环境变量驱动"


class TestDefaultIds:
    """DEFAULT_GEN_MODEL_ID / DEFAULT_PRED_MODEL_ID 必须存在且在注册表中。"""

    def test_default_gen_model_id_exists(self):
        from config import DEFAULT_GEN_MODEL_ID, GEN_MODELS
        assert isinstance(DEFAULT_GEN_MODEL_ID, str)
        ids = [e["id"] for e in GEN_MODELS]
        assert DEFAULT_GEN_MODEL_ID in ids

    def test_default_pred_model_id_exists(self):
        from config import DEFAULT_PRED_MODEL_ID, PRED_MODELS
        assert isinstance(DEFAULT_PRED_MODEL_ID, str)
        ids = [e["id"] for e in PRED_MODELS]
        assert DEFAULT_PRED_MODEL_ID in ids

    def test_default_gen_model_is_qwen_lora(self):
        """默认生成模型应是 qwen_lora（真实模型）。"""
        from config import DEFAULT_GEN_MODEL_ID
        assert DEFAULT_GEN_MODEL_ID == "qwen_lora"

    def test_default_pred_model_is_equiformer_v2(self):
        """默认预测模型应是 equiformer_v2（真实模型）。"""
        from config import DEFAULT_PRED_MODEL_ID
        assert DEFAULT_PRED_MODEL_ID == "equiformer_v2"


# ── 2. Schema 字段 ───────────────────────────────────────────────────────────

class TestSchemaModelIdFields:
    """GenerationRequest 必须支持 gen_model_id / pred_model_id 可选字段（多模型切换架构保留）。"""

    def test_gen_model_id_optional_defaults_none(self):
        from api.schemas import GenerationRequest
        req = GenerationRequest(target_dgH=-0.5)
        assert req.gen_model_id is None

    def test_pred_model_id_optional_defaults_none(self):
        from api.schemas import GenerationRequest
        req = GenerationRequest(target_dgH=-0.5)
        assert req.pred_model_id is None

    def test_gen_model_id_can_be_set_to_real_model(self):
        from api.schemas import GenerationRequest
        req = GenerationRequest(target_dgH=-0.5, gen_model_id="qwen_lora")
        assert req.gen_model_id == "qwen_lora"

    def test_pred_model_id_can_be_set_to_real_model(self):
        from api.schemas import GenerationRequest
        req = GenerationRequest(target_dgH=-0.5, pred_model_id="equiformer_v2")
        assert req.pred_model_id == "equiformer_v2"

    def test_model_dump_includes_model_ids(self):
        """model_dump() 应包含 gen_model_id / pred_model_id，供 orchestrator 读取。"""
        from api.schemas import GenerationRequest
        req = GenerationRequest(target_dgH=-0.5, gen_model_id="qwen_lora", pred_model_id="equiformer_v2")
        data = req.model_dump()
        assert "gen_model_id" in data
        assert "pred_model_id" in data
        assert data["gen_model_id"] == "qwen_lora"
        assert data["pred_model_id"] == "equiformer_v2"


# ── 3. Orchestrator 按 model_id 选择行为 ─────────────────────────────────────

class TestOrchestratorModelSelection:
    """在 MATGEN_DEMO=1 环境下（模块顶部已设），orchestrator 应能完成端到端流程。
    轻量执行由 MATGEN_DEMO=1 驱动，与 model_id 无关。"""

    def _make_orchestrator(self):
        from core.task_orchestrator import TaskOrchestrator
        return TaskOrchestrator()

    def test_execute_in_demo_mode_completes(self):
        """MATGEN_DEMO=1 模式：不依赖大模型，整个 pipeline 应正常完成。"""
        orch = self._make_orchestrator()
        task_id = "test-model-selection-001"
        config = {
            "target_dgH": -0.5,
            "tolerance": 0.1,
            "elements": ["Ir", "Pd", "Pt"],
            "batch_size": 2,
        }
        orch.create_task(task_id, config)
        orch.execute_task(task_id)

        status = orch.get_task_status(task_id)
        assert status is not None
        assert status["status"] == "completed"
        assert status["total"] == 2
        assert status["processed"] == 2

    def test_execute_in_demo_mode_produces_poscar(self):
        """DEMO 模式：生成的结构应包含有效 poscar 字段。"""
        orch = self._make_orchestrator()
        task_id = "test-model-selection-002"
        config = {
            "target_dgH": -0.5,
            "tolerance": 0.1,
            "elements": ["Ir", "Pd"],
            "batch_size": 1,
        }
        orch.create_task(task_id, config)
        orch.execute_task(task_id)

        status = orch.get_task_status(task_id)
        results = status["results"]
        assert len(results) >= 1
        struct = results[0]
        assert "poscar" in struct
        assert len(struct["poscar"]) > 0

    def test_execute_without_model_id_uses_defaults(self):
        """不传 model_id 时行为不变：任务正常完成（向后兼容）。"""
        orch = self._make_orchestrator()
        task_id = "test-model-selection-003"
        config = {
            "target_dgH": -0.5,
            "tolerance": 0.5,  # 宽松容差，避免全部 filtered_out
            "elements": ["Ir", "Pd"],
            "batch_size": 2,
            # 不传 gen_model_id / pred_model_id
        }
        orch.create_task(task_id, config)
        orch.execute_task(task_id)

        status = orch.get_task_status(task_id)
        assert status["status"] == "completed"
        assert status["total"] == 2


# ── 4. 未知 model_id 回退 ────────────────────────────────────────────────────

class TestUnknownModelIdFallback:
    """传入未知 model_id 时应回退到默认，不应崩溃。"""

    def test_unknown_gen_model_id_does_not_crash(self):
        from core.task_orchestrator import TaskOrchestrator
        orch = TaskOrchestrator()
        task_id = "test-unknown-gen-001"
        config = {
            "target_dgH": -0.5,
            "tolerance": 0.5,
            "elements": ["Ir", "Pd"],
            "batch_size": 1,
            "gen_model_id": "bogus_gen_model_xyz",  # 不存在的 id，应回退到默认
        }
        orch.create_task(task_id, config)
        orch.execute_task(task_id)  # 不应抛异常

        status = orch.get_task_status(task_id)
        assert status is not None
        assert status["status"] == "completed"

    def test_unknown_pred_model_id_does_not_crash(self):
        from core.task_orchestrator import TaskOrchestrator
        orch = TaskOrchestrator()
        task_id = "test-unknown-pred-001"
        config = {
            "target_dgH": -0.5,
            "tolerance": 0.5,
            "elements": ["Ir", "Pd"],
            "batch_size": 1,
            "pred_model_id": "bogus_pred_model_xyz",  # 不存在的 id，应回退到默认
        }
        orch.create_task(task_id, config)
        orch.execute_task(task_id)  # 不应抛异常

        status = orch.get_task_status(task_id)
        assert status is not None
        assert status["status"] == "completed"


# ── 5. GET /models 端点 ──────────────────────────────────────────────────────

class TestModelsEndpoint:
    """GET /models 应返回可用模型清单。"""

    def _make_client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.routes import router
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        return TestClient(app)

    def test_models_endpoint_returns_200(self):
        client = self._make_client()
        response = client.get("/api/v1/models")
        assert response.status_code == 200

    def test_models_endpoint_contains_gen_and_pred(self):
        client = self._make_client()
        response = client.get("/api/v1/models")
        data = response.json()
        assert "gen_models" in data
        assert "pred_models" in data

    def test_models_endpoint_gen_models_nonempty(self):
        client = self._make_client()
        response = client.get("/api/v1/models")
        data = response.json()
        assert len(data["gen_models"]) >= 1

    def test_models_endpoint_pred_models_nonempty(self):
        client = self._make_client()
        response = client.get("/api/v1/models")
        data = response.json()
        assert len(data["pred_models"]) >= 1

    def test_models_endpoint_gen_models_only_real(self):
        """GET /models 返回的 gen_models 只含真实模型，不含 fake。"""
        client = self._make_client()
        response = client.get("/api/v1/models")
        data = response.json()
        ids = [m["id"] for m in data["gen_models"]]
        assert "qwen_lora" in ids
        assert "fake" not in ids

    def test_models_endpoint_pred_models_only_real(self):
        """GET /models 返回的 pred_models 只含真实模型，不含 toy_mlp。"""
        client = self._make_client()
        response = client.get("/api/v1/models")
        data = response.json()
        ids = [m["id"] for m in data["pred_models"]]
        assert "equiformer_v2" in ids
        assert "toy_mlp" not in ids

    def test_generate_with_model_ids_accepted(self):
        """POST /generate 应接受 gen_model_id / pred_model_id 字段（真实 id）。"""
        client = self._make_client()
        response = client.post("/api/v1/generate", json={
            "target_dgH": -0.5,
            "batch_size": 2,
            "gen_model_id": "qwen_lora",
            "pred_model_id": "equiformer_v2",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
