# tests/test_workspace.py
import pytest
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from persistence.workspace import Workspace


class TestWorkspaceInit:
    """Test Workspace initialization."""

    def test_init_creates_root_directory(self, temp_workspace):
        """Test that Workspace creates root directory."""
        ws = Workspace(root=temp_workspace)
        assert os.path.exists(temp_workspace)

    def test_init_creates_subdirectories(self, temp_workspace):
        """Test that Workspace creates runs/models/data subdirs."""
        ws = Workspace(root=temp_workspace)
        assert os.path.isdir(os.path.join(temp_workspace, "runs"))
        assert os.path.isdir(os.path.join(temp_workspace, "models"))
        assert os.path.isdir(os.path.join(temp_workspace, "data"))


class TestWorkspaceTaskWorkspace:
    """Test per-task workspace creation."""

    def test_create_task_workspace(self, temp_workspace):
        """Test creating a dedicated task workspace."""
        ws = Workspace(root=temp_workspace)
        task_id = "task-123"

        task_dir = ws.create_task_workspace(task_id)

        assert os.path.exists(task_dir)
        assert os.path.isdir(os.path.join(task_dir, "structures"))
        assert os.path.isdir(os.path.join(task_dir, "logs"))
        assert os.path.isdir(os.path.join(task_dir, "checkpoints"))
        assert os.path.isdir(os.path.join(task_dir, "exports"))

    def test_get_task_workspace_existing(self, temp_workspace):
        """Test getting workspace for existing task."""
        ws = Workspace(root=temp_workspace)
        task_id = "task-456"

        ws.create_task_workspace(task_id)
        task_dir = ws.get_task_workspace(task_id)

        assert task_dir is not None
        assert os.path.exists(task_dir)

    def test_get_task_workspace_nonexistent(self, temp_workspace):
        """Test getting workspace for non-existent task returns None."""
        ws = Workspace(root=temp_workspace)
        result = ws.get_task_workspace("nonexistent")
        assert result is None


class TestWorkspaceSaveStructure:
    """Test structure file saving."""

    def test_save_structure(self, temp_workspace, sample_poscar):
        """Test saving a structure to workspace."""
        ws = Workspace(root=temp_workspace)
        task_id = "task-789"
        struct_id = "struct-001"

        ws.create_task_workspace(task_id)
        path = ws.save_structure(
            task_id=task_id,
            structure_id=struct_id,
            poscar=sample_poscar,
            metadata={"elements": "Ir2Pd2", "predicted_dgH": -0.5}
        )

        assert os.path.exists(path)
        assert os.path.exists(os.path.join(path, "structure.poscar"))
        assert os.path.exists(os.path.join(path, "metadata.json"))

    def test_save_structure_creates_parent_dirs(self, temp_workspace, sample_poscar):
        """Test that save_structure creates parent directories if needed."""
        ws = Workspace(root=temp_workspace)
        path = ws.save_structure(
            task_id="brand-new-task",
            structure_id="struct-new",
            poscar=sample_poscar,
            metadata={}
        )
        assert os.path.exists(path)


class TestWorkspaceListRuns:
    """Test listing run directories."""

    def test_list_runs_empty(self, temp_workspace):
        """Test listing runs when none exist."""
        ws = Workspace(root=temp_workspace)
        runs = ws.list_runs()
        assert runs == []

    def test_list_runs_returns_all(self, temp_workspace):
        """Test listing all run directories."""
        ws = Workspace(root=temp_workspace)

        ws.create_task_workspace("task-a")
        ws.create_task_workspace("task-b")
        ws.create_task_workspace("task-c")

        runs = ws.list_runs()
        assert len(runs) == 3


class TestWorkspaceCleanup:
    """Test workspace cleanup functionality."""

    def test_cleanup_old_runs_keeps_recent(self, temp_workspace):
        """Test that cleanup keeps the specified number of recent runs."""
        ws = Workspace(root=temp_workspace)

        for i in range(15):
            ws.create_task_workspace(f"task-{i:02d}")

        ws.cleanup_old_runs(keep_last=5)

        runs = ws.list_runs()
        assert len(runs) == 5

    def test_cleanup_old_runs_removes_oldest(self, temp_workspace):
        """Test that cleanup removes oldest runs first."""
        ws = Workspace(root=temp_workspace)

        ws.create_task_workspace("old-task")
        import time
        time.sleep(0.01)
        ws.create_task_workspace("new-task")

        ws.cleanup_old_runs(keep_last=1)

        runs = ws.list_runs()
        assert len(runs) == 1
        assert runs[0]["task_id"] == "new-task"
