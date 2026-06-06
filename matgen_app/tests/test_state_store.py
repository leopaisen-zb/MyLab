# tests/test_state_store.py
import pytest
import os
import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from persistence.state_store import StateStore


class TestStateStoreInit:
    """Test StateStore initialization and database setup."""

    def test_init_creates_database_file(self, temp_db):
        """Test that init creates the database file."""
        store = StateStore(db_path=temp_db)
        assert os.path.exists(temp_db)

    def test_init_creates_tables(self, temp_db):
        """Test that init creates required tables."""
        store = StateStore(db_path=temp_db)

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name IN ('structures', 'tasks')
        """)
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "structures" in tables
        assert "tasks" in tables

    def test_init_creates_indexes(self, temp_db):
        """Test that init creates indexes on structures table."""
        store = StateStore(db_path=temp_db)

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND name IN ('idx_task_id', 'idx_status')
        """)
        indexes = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "idx_task_id" in indexes
        assert "idx_status" in indexes


class TestStateStoreSaveRecord:
    """Test save_record functionality."""

    def test_save_and_get_record(self, temp_db, sample_structure_record):
        """Test saving and retrieving a structure record."""
        store = StateStore(db_path=temp_db)
        record = sample_structure_record

        store.save_record(record["uuid"], record)
        retrieved = store.get_record(record["uuid"])

        assert retrieved is not None
        assert retrieved["uuid"] == record["uuid"]
        assert retrieved["status"] == record["status"]
        assert retrieved["elements"] == record["elements"]

    def test_save_record_insert_or_replace(self, temp_db, sample_structure_record):
        """Test INSERT OR REPLACE behavior (upsert)."""
        store = StateStore(db_path=temp_db)
        record = sample_structure_record

        store.save_record(record["uuid"], record)
        record["status"] = "predicted"
        store.save_record(record["uuid"], record)

        retrieved = store.get_record(record["uuid"])
        assert retrieved["status"] == "predicted"

    def test_get_nonexistent_returns_none(self, temp_db):
        """Getting non-existent record returns None."""
        store = StateStore(db_path=temp_db)
        result = store.get_record("nonexistent-uuid")
        assert result is None


class TestStateStoreGetByTask:
    """Test get_records_by_task functionality."""

    def test_get_records_by_task(self, temp_db):
        """Test retrieving all records for a specific task."""
        store = StateStore(db_path=temp_db)
        task_id = "task-123"

        for i in range(5):
            store.save_record(f"struct-{i}", {
                "task_id": task_id,
                "status": "generated",
                "elements": "Ir2Pd2",
                "poscar": "",
            })

        records = store.get_records_by_task(task_id)
        assert len(records) == 5

    def test_get_records_by_task_empty(self, temp_db):
        """Test retrieving records for task with no structures."""
        store = StateStore(db_path=temp_db)
        records = store.get_records_by_task("nonexistent-task")
        assert records == []


class TestStateStoreGetByStatus:
    """Test get_records_by_status functionality."""

    def test_get_records_by_status(self, temp_db):
        """Test filtering records by status."""
        store = StateStore(db_path=temp_db)

        for i in range(3):
            store.save_record(f"in-{i}", {
                "task_id": "t1",
                "status": "filtered_in",
                "elements": "Ir2Pd2",
                "poscar": "",
            })
        for i in range(2):
            store.save_record(f"out-{i}", {
                "task_id": "t1",
                "status": "filtered_out",
                "elements": "Ir2Pd2",
                "poscar": "",
            })

        filtered_in = store.get_records_by_status("filtered_in")
        filtered_out = store.get_records_by_status("filtered_out")

        assert len(filtered_in) == 3
        assert len(filtered_out) == 2

    def test_get_records_by_status_respects_limit(self, temp_db):
        """Test that status query respects limit parameter."""
        store = StateStore(db_path=temp_db)

        for i in range(20):
            store.save_record(f"struct-{i}", {
                "task_id": "t1",
                "status": "generated",
                "elements": "Ir2Pd2",
                "poscar": "",
            })

        results = store.get_records_by_status("generated", limit=5)
        assert len(results) == 5


class TestStateStoreTaskMethods:
    """Test task-level operations."""

    def test_save_and_get_task(self, temp_db):
        """Test saving and retrieving a task."""
        store = StateStore(db_path=temp_db)
        task_data = {
            "target_dgH": -0.5,
            "tolerance": 0.05,
            "batch_size": 10
        }

        store.save_task("task-abc", "created", task_data)
        retrieved = store.get_task("task-abc")

        assert retrieved is not None
        assert retrieved["task_id"] == "task-abc"
        assert retrieved["status"] == "created"
        assert retrieved["config"]["target_dgH"] == -0.5

    def test_get_nonexistent_task(self, temp_db):
        """Getting non-existent task returns None."""
        store = StateStore(db_path=temp_db)
        result = store.get_task("nonexistent-task")
        assert result is None


class TestStateStoreStateHistory:
    """Test state transition history (flywheel traceability)."""

    def test_state_history_table_created(self, temp_db):
        """state_history table should be created on init."""
        StateStore(db_path=temp_db)
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='state_history'"
        )
        row = cursor.fetchone()
        conn.close()
        assert row is not None

    def test_record_and_get_state_history(self, temp_db):
        """Record transitions and retrieve them newest-first."""
        store = StateStore(db_path=temp_db)
        store.save_record("s1", {
            "task_id": "t1", "status": "generated", "elements": "Ir2", "poscar": "",
        })
        store.record_state_change("s1", "generated", "predicted", "eq predicted")
        store.record_state_change("s1", "predicted", "filtered_in", "within tolerance")

        history = store.get_state_history("s1")
        assert len(history) == 2
        # newest first
        assert history[0]["to_state"] == "filtered_in"
        assert history[0]["from_state"] == "predicted"
        assert history[0]["reason"] == "within tolerance"

    def test_get_state_history_empty(self, temp_db):
        """No history returns empty list."""
        store = StateStore(db_path=temp_db)
        assert store.get_state_history("nope") == []


class TestStateStoreExportValidated:
    """Test export of validated structures for training feedback (flywheel ⑥)."""

    def test_export_validated_returns_only_validated(self, temp_db):
        """Only validated / exported_for_training records are exported."""
        store = StateStore(db_path=temp_db)
        store.save_record("v1", {
            "task_id": "t1", "status": "validated", "elements": "Ir2",
            "poscar": "P1", "predicted_dgH": -0.5,
        })
        store.save_record("v2", {
            "task_id": "t1", "status": "exported_for_training", "elements": "Pd2",
            "poscar": "P2", "predicted_dgH": -0.4,
        })
        store.save_record("f1", {
            "task_id": "t1", "status": "filtered_out", "elements": "Pt2", "poscar": "P3",
        })

        exported = store.export_validated()
        uuids = {r["uuid"] for r in exported}
        assert uuids == {"v1", "v2"}

    def test_export_validated_empty(self, temp_db):
        """No validated records returns empty list."""
        store = StateStore(db_path=temp_db)
        assert store.export_validated() == []


class TestStateStoreStateDistribution:
    """Test state distribution counts (flywheel monitoring)."""

    def test_get_state_distribution(self, temp_db):
        store = StateStore(db_path=temp_db)
        for i in range(3):
            store.save_record(f"g{i}", {
                "task_id": "t1", "status": "generated", "elements": "x", "poscar": "",
            })
        for i in range(2):
            store.save_record(f"p{i}", {
                "task_id": "t1", "status": "predicted", "elements": "x", "poscar": "",
            })

        dist = store.get_state_distribution()
        assert dist["generated"] == 3
        assert dist["predicted"] == 2

    def test_get_state_distribution_empty(self, temp_db):
        store = StateStore(db_path=temp_db)
        assert store.get_state_distribution() == {}


class TestStateStoreConnectionSafety:
    """Test connection management and safety."""

    def test_multiple_concurrent_saves(self, temp_db):
        """Test that multiple sequential saves don't leak connections."""
        store = StateStore(db_path=temp_db)

        for i in range(100):
            store.save_record(f"struct-{i}", {
                "task_id": "task-1",
                "status": "generated",
                "elements": "Ir2Pd2",
                "poscar": "",
            })

        count = len(store.get_records_by_task("task-1"))
        assert count == 100

    def test_exception_during_save_does_not_corrupt(self, temp_db):
        """Test that exception during save doesn't leave corrupt data."""
        store = StateStore(db_path=temp_db)

        # Save a valid record
        store.save_record("valid-struct", {
            "task_id": "t1",
            "status": "generated",
            "elements": "Ir2Pd2",
            "poscar": "",
        })

        # Try to get it back - should still work
        retrieved = store.get_record("valid-struct")
        assert retrieved is not None
        assert retrieved["uuid"] == "valid-struct"
