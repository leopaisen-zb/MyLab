# tests/test_checkpoint.py
import pytest
import os
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.checkpoint import CheckpointManager


class TestCheckpointManager:
    """Test CheckpointManager for task state persistence."""

    def test_save_and_load_checkpoint(self, temp_workspace):
        """Test saving and loading a checkpoint."""
        cm = CheckpointManager(workspace_root=temp_workspace)
        task_id = "test-task-123"
        data = {"status": "running", "progress": 50, "structures": ["s1", "s2"]}

        filepath = cm.save_checkpoint(task_id, data)
        assert os.path.exists(filepath)

        loaded = cm.load_checkpoint(task_id)
        assert loaded == data

    def test_load_nonexistent_checkpoint_returns_none(self, temp_workspace):
        """Loading non-existent checkpoint returns None."""
        cm = CheckpointManager(workspace_root=temp_workspace)
        result = cm.load_checkpoint("nonexistent-task")
        assert result is None

    def test_get_latest_checkpoint(self, temp_workspace):
        """Test get_latest_checkpoint is alias for load_checkpoint."""
        cm = CheckpointManager(workspace_root=temp_workspace)
        task_id = "test-task-456"
        data = {"status": "completed"}

        cm.save_checkpoint(task_id, data)
        latest = cm.get_latest_checkpoint(task_id)

        assert latest == data

    def test_delete_checkpoint(self, temp_workspace):
        """Test checkpoint deletion."""
        cm = CheckpointManager(workspace_root=temp_workspace)
        task_id = "test-task-789"
        cm.save_checkpoint(task_id, {"data": "test"})

        assert cm.delete_checkpoint(task_id) is True
        assert cm.load_checkpoint(task_id) is None

    def test_delete_nonexistent_returns_false(self, temp_workspace):
        """Deleting non-existent checkpoint returns False."""
        cm = CheckpointManager(workspace_root=temp_workspace)
        assert cm.delete_checkpoint("nonexistent") is False

    def test_checkpoint_contains_metadata(self, temp_workspace):
        """Saved checkpoint should contain task_id and timestamp."""
        cm = CheckpointManager(workspace_root=temp_workspace)
        task_id = "test-task-meta"
        data = {"status": "done"}

        filepath = cm.save_checkpoint(task_id, data)

        with open(filepath, "r", encoding="utf-8") as f:
            record = json.load(f)

        assert record["task_id"] == task_id
        assert "timestamp" in record
        assert record["data"] == data

    def test_list_checkpoints(self, temp_workspace):
        """Test listing all checkpoints."""
        cm = CheckpointManager(workspace_root=temp_workspace)
        cm.save_checkpoint("task-a", {"a": 1})
        cm.save_checkpoint("task-b", {"b": 2})

        checkpoints = cm.list_checkpoints()
        filenames = [p.name for p in checkpoints]

        assert any("task-a_checkpoint.json" in f for f in filenames)
        assert any("task-b_checkpoint.json" in f for f in filenames)
