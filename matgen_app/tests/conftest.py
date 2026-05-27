# tests/conftest.py
import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory for tests."""
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)

@pytest.fixture
def temp_db(temp_workspace):
    """Create a temporary database path for tests."""
    return os.path.join(temp_workspace, "test_matgen_eq.db")

@pytest.fixture
def sample_poscar():
    """Sample POSCAR content for testing."""
    return """Generated HEA structure
1.0
8.123456 0.0 0.0
0.0 8.123456 0.0
0.0 0.0 8.123456
Ir Pd Pt Rh Ru
2 2 2 2 2
Cartesian
0.0 0.0 0.0
1.0 1.0 0.0
2.0 2.0 0.0
3.0 3.0 0.0
4.0 4.0 0.0
0.0 0.0 1.0
1.0 1.0 1.0
2.0 2.0 1.0
3.0 3.0 1.0
4.0 4.0 1.0
"""

@pytest.fixture
def sample_structure_record():
    """Sample structure record for testing."""
    import uuid
    return {
        "uuid": str(uuid.uuid4()),
        "task_id": "test-task-001",
        "status": "generated",
        "elements": "Ir2Pd2Pt2Rh2Ru2",
        "poscar": "test poscar content",
        "predicted_dgH": -0.52,
        "target_dgH": -0.50,
        "created_at": "2026-05-27T10:00:00"
    }
