# persistence/state_store.py
import json
import os
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import contextmanager


class StateStore:
    def __init__(self, db_path: str = "./workspace/matgen_eq.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS structures (
                    uuid TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    elements TEXT,
                    poscar TEXT,
                    predicted_dgH REAL,
                    target_dgH REAL,
                    error TEXT,
                    decision TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    metadata TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    config TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_id ON structures(task_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON structures(status)
            """)
            conn.commit()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def save_record(self, uuid: str, record: Dict[str, Any]):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute("""
                INSERT OR REPLACE INTO structures
                (uuid, task_id, status, elements, poscar, predicted_dgH, target_dgH, error, decision, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                uuid,
                record.get("task_id", ""),
                record.get("status", ""),
                record.get("elements", ""),
                record.get("poscar", ""),
                record.get("predicted_dgH"),
                record.get("target_dgH"),
                record.get("error"),
                record.get("decision"),
                record.get("created_at", now),
                record.get("updated_at", now),
                json.dumps(record.get("metadata", {}), ensure_ascii=False)
            ))
            conn.commit()

    def get_record(self, uuid: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM structures WHERE uuid = ?", (uuid,))
            row = cursor.fetchone()
            if row is None:
                return None
            columns = ["uuid", "task_id", "status", "elements", "poscar", "predicted_dgH",
                       "target_dgH", "error", "decision", "created_at", "updated_at", "metadata"]
            record = dict(zip(columns, row))
            if record["metadata"]:
                record["metadata"] = json.loads(record["metadata"])
            return record

    def get_records_by_task(self, task_id: str) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM structures WHERE task_id = ?", (task_id,))
            rows = cursor.fetchall()
            columns = ["uuid", "task_id", "status", "elements", "poscar", "predicted_dgH",
                       "target_dgH", "error", "decision", "created_at", "updated_at", "metadata"]
            records = []
            for row in rows:
                record = dict(zip(columns, row))
                if record["metadata"]:
                    record["metadata"] = json.loads(record["metadata"])
                records.append(record)
            return records

    def get_records_by_status(self, status: str, limit: int = 100) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM structures WHERE status = ? LIMIT ?", (status, limit))
            rows = cursor.fetchall()
            columns = ["uuid", "task_id", "status", "elements", "poscar", "predicted_dgH",
                       "target_dgH", "error", "decision", "created_at", "updated_at", "metadata"]
            records = []
            for row in rows:
                record = dict(zip(columns, row))
                if record["metadata"]:
                    record["metadata"] = json.loads(record["metadata"])
                records.append(record)
            return records

    def save_task(self, task_id: str, status: str, config: dict):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute("""
                INSERT OR REPLACE INTO tasks (task_id, status, config, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (task_id, status, json.dumps(config, ensure_ascii=False), now, now))
            conn.commit()

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            columns = ["task_id", "status", "config", "created_at", "updated_at"]
            record = dict(zip(columns, row))
            if record["config"]:
                record["config"] = json.loads(record["config"])
            return record
