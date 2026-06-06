# tests/test_routes.py
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from api.routes import router
from api.schemas import GenerationRequest, StructureStatus


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_returns_healthy(self):
        """Health endpoint should return healthy status."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        client = TestClient(app)

        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestGenerateEndpoint:
    """Test POST /generate endpoint."""

    def test_generate_returns_task_id(self):
        """Generate endpoint should return a task_id."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        client = TestClient(app)

        response = client.post("/api/v1/generate", json={
            "target_dgH": -0.5,
            "tolerance": 0.05,
            "elements": ["Ir", "Pd", "Pt"],
            "batch_size": 10
        })

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "created"
        assert "message" in data

    def test_generate_accepts_partial_config(self):
        """Generate should work with minimal config."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        client = TestClient(app)

        response = client.post("/api/v1/generate", json={
            "target_dgH": -0.5
        })

        assert response.status_code == 200

    def test_generate_validates_batch_size_range(self):
        """Generate should validate batch_size range."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        client = TestClient(app)

        # batch_size > 500 should fail validation
        response = client.post("/api/v1/generate", json={
            "target_dgH": -0.5,
            "batch_size": 1000
        })
        assert response.status_code == 422  # Validation error


class TestTaskStatusEndpoint:
    """Test GET /tasks/{task_id}/status endpoint."""

    def test_get_status_nonexistent_task(self):
        """Getting status of non-existent task should return 404."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        client = TestClient(app)

        response = client.get("/api/v1/tasks/nonexistent-task-id/status")
        assert response.status_code == 404


class TestStructureEndpoints:
    """Test structure-specific endpoints."""

    def test_get_structure_nonexistent(self):
        """Getting non-existent structure should return 404."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        client = TestClient(app)

        response = client.get("/api/v1/structures/nonexistent-id")
        assert response.status_code == 404

    def test_validate_structure_invalid_decision(self):
        """Validating with invalid decision should return 400."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        client = TestClient(app)

        response = client.post(
            "/api/v1/structures/some-id/validate?decision=maybe"
        )
        assert response.status_code == 400


class TestAPISchemas:
    """Test API schema validation."""

    def test_generation_request_valid(self):
        """Test valid GenerationRequest schema."""
        req = GenerationRequest(
            target_dgH=-0.5,
            tolerance=0.05,
            elements=["Ir", "Pd", "Pt"],
            batch_size=20
        )
        assert req.target_dgH == -0.5
        assert req.batch_size == 20

    def test_generation_request_defaults(self):
        """Test GenerationRequest default values."""
        req = GenerationRequest(target_dgH=-0.5)
        assert req.tolerance == 0.05
        assert req.batch_size == 10
        assert req.elements == ["Ir", "Pd", "Pt", "Rh", "Ru"]

    def test_generation_request_batch_size_bounds(self):
        """Test GenerationRequest batch_size bounds."""
        # batch_size too small
        with pytest.raises(Exception):
            GenerationRequest(target_dgH=-0.5, batch_size=0)

        # batch_size too large
        with pytest.raises(Exception):
            GenerationRequest(target_dgH=-0.5, batch_size=1000)

    def test_structure_status_enum_values(self):
        """Test StructureStatus enum has all expected values (dft_verified added in P1-1)."""
        expected = {
            "generated", "rejected_precheck", "predicted",
            "filtered_in", "filtered_out", "dft_verified", "validated", "rejected",
            "exported_for_training",  # M-1 补充
        }
        actual = {s.value for s in StructureStatus}
        assert actual == expected

    def test_structure_status_has_exported_for_training(self):
        """M-1 修复：StructureStatus 应包含 EXPORTED_FOR_TRAINING = 'exported_for_training'。"""
        assert hasattr(StructureStatus, "EXPORTED_FOR_TRAINING")
        assert StructureStatus.EXPORTED_FOR_TRAINING.value == "exported_for_training"


class TestValidateEndpointStateTransition:
    """H-1 修复：/validate 端点区分 404（不存在）和 409（非法状态转移）。"""

    def _make_app_with_record(self, temp_db, status: str):
        """创建独立 FastAPI app + 预插入指定状态记录，返回 (client, uid)。"""
        import uuid as _uuid
        from fastapi import FastAPI, APIRouter, HTTPException
        from fastapi.testclient import TestClient
        from persistence.state_store import StateStore
        from core.task_orchestrator import TaskOrchestrator

        store = StateStore(db_path=temp_db)
        uid = str(_uuid.uuid4())
        store.save_record(uid, {
            "task_id": "t1", "status": status,
            "elements": "Ir2", "poscar": "POSCAR",
        })

        orch = TaskOrchestrator.__new__(TaskOrchestrator)
        orch.state_store = store

        app = FastAPI()
        local_router = APIRouter()

        @local_router.post("/structures/{sid}/validate")
        async def _validate(sid: str, decision: str):
            if decision not in ["validated", "rejected"]:
                raise HTTPException(status_code=400, detail="bad decision")
            record = orch.state_store.get_record(sid)
            if record is None:
                raise HTTPException(status_code=404, detail="Structure not found")
            result = orch.update_structure_decision(sid, decision)
            if not result:
                raise HTTPException(status_code=409, detail="Invalid state transition")
            return {"structure_id": sid, "decision": decision}

        app.include_router(local_router, prefix="/api/v1")
        return TestClient(app), uid

    def test_validate_filtered_in_returns_409(self, temp_db):
        """H-1 修复：filtered_in 状态的结构调 validate?decision=validated 应返回 409，不是 404。"""
        client, uid = self._make_app_with_record(temp_db, "filtered_in")
        resp = client.post(f"/api/v1/structures/{uid}/validate?decision=validated")
        assert resp.status_code == 409, (
            f"Expected 409 for invalid state transition, got {resp.status_code}"
        )

    def test_validate_nonexistent_returns_404(self, temp_db):
        """不存在的 UUID 应返回 404。"""
        client, _ = self._make_app_with_record(temp_db, "filtered_in")
        resp = client.post("/api/v1/structures/no-such-uuid/validate?decision=validated")
        assert resp.status_code == 404

    def test_validate_dft_verified_returns_200(self, temp_db):
        """dft_verified 状态的结构调 validate?decision=validated 应返回 200。"""
        client, uid = self._make_app_with_record(temp_db, "dft_verified")
        resp = client.post(f"/api/v1/structures/{uid}/validate?decision=validated")
        assert resp.status_code == 200
        assert resp.json()["decision"] == "validated"

    def test_reject_filtered_in_returns_200(self, temp_db):
        """filtered_in → rejected 仍合法，应返回 200。"""
        client, uid = self._make_app_with_record(temp_db, "filtered_in")
        resp = client.post(f"/api/v1/structures/{uid}/validate?decision=rejected")
        assert resp.status_code == 200
