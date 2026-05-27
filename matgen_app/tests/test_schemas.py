# tests/test_schemas.py
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schemas import (
    GenerationRequest, GenerationResponse,
    TaskStatusResponse, StructureRecord, StructureStatus
)


class TestGenerationRequest:
    """Test GenerationRequest Pydantic model."""

    def test_valid_request(self):
        """Test creating a valid GenerationRequest."""
        req = GenerationRequest(
            target_dgH=-0.5,
            tolerance=0.05,
            elements=["Ir", "Pd"],
            batch_size=10
        )
        assert req.target_dgH == -0.5
        assert req.tolerance == 0.05

    def test_default_values(self):
        """Test default values are applied."""
        req = GenerationRequest(target_dgH=-0.5)
        assert req.tolerance == 0.05
        assert req.batch_size == 10
        assert req.elements == ["Ir", "Pd", "Pt", "Rh", "Ru"]

    def test_batch_size_validation(self):
        """Test batch_size must be between 1 and 500."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            GenerationRequest(target_dgH=-0.5, batch_size=0)

        with pytest.raises(ValidationError):
            GenerationRequest(target_dgH=-0.5, batch_size=600)


class TestGenerationResponse:
    """Test GenerationResponse Pydantic model."""

    def test_response_creation(self):
        """Test creating a GenerationResponse."""
        resp = GenerationResponse(
            task_id="abc-123",
            status="created",
            message="Task created"
        )
        assert resp.task_id == "abc-123"
        assert resp.status == "created"


class TestTaskStatusResponse:
    """Test TaskStatusResponse Pydantic model."""

    def test_status_response(self):
        """Test TaskStatusResponse structure."""
        resp = TaskStatusResponse(
            task_id="abc",
            status="running",
            total=100,
            processed=50,
            success=45,
            failed=5,
            results=[]
        )
        assert resp.total == 100
        assert resp.processed == 50

    def test_status_response_with_results(self):
        """Test status response with structure results."""
        resp = TaskStatusResponse(
            task_id="abc",
            status="completed",
            total=10,
            processed=10,
            success=8,
            failed=2,
            results=[
                {"uuid": "s1", "status": "filtered_in"},
                {"uuid": "s2", "status": "filtered_out"},
            ]
        )
        assert len(resp.results) == 2


class TestStructureRecord:
    """Test StructureRecord Pydantic model."""

    def test_record_with_all_fields(self):
        """Test StructureRecord with all fields populated."""
        record = StructureRecord(
            uuid="struct-001",
            status=StructureStatus.FILTERED_IN,
            elements="Ir2Pd2Pt2",
            predicted_dgH=-0.52,
            target_dgH=-0.50,
            error=None
        )
        assert record.predicted_dgH == -0.52
        assert record.error is None

    def test_record_with_error(self):
        """Test StructureRecord with error field."""
        record = StructureRecord(
            uuid="struct-002",
            status=StructureStatus.REJECTED_PRECHECK,
            elements="Ir2Pd2",
            error="Parse error: invalid lattice"
        )
        assert record.error is not None
        assert "Parse error" in record.error


class TestStructureStatus:
    """Test StructureStatus enum."""

    def test_all_statuses_are_strings(self):
        """All status values should be lowercase strings."""
        for status in StructureStatus:
            assert isinstance(status.value, str)
            assert status.value == status.value.lower()

    def test_status_value_matches_enum_name(self):
        """Enum value should match enum name in lowercase."""
        for status in StructureStatus:
            assert status.value == status.name.lower()
