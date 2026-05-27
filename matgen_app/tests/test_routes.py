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
        """Test StructureStatus enum has all expected values."""
        expected = {
            "generated", "rejected_precheck", "predicted",
            "filtered_in", "filtered_out", "validated", "rejected"
        }
        actual = {s.value for s in StructureStatus}
        assert actual == expected
