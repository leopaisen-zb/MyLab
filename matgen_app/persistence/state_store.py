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
            # P1-1：幂等迁移——为已有 DB 添加 DFT 结果列
            for col_sql in [
                "ALTER TABLE structures ADD COLUMN dft_dg_h REAL",
                "ALTER TABLE structures ADD COLUMN dft_energy REAL",
                "ALTER TABLE structures ADD COLUMN dft_status TEXT",
            ]:
                try:
                    cursor.execute(col_sql)
                except Exception:
                    pass  # 列已存在时忽略
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
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS state_history (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    structure_uuid TEXT NOT NULL,
                    from_state   TEXT,
                    to_state     TEXT NOT NULL,
                    changed_at   TEXT NOT NULL,
                    reason       TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (structure_uuid) REFERENCES structures(uuid)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_uuid ON state_history(structure_uuid)
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
                (uuid, task_id, status, elements, poscar, predicted_dgH, target_dgH,
                 error, decision, created_at, updated_at, metadata,
                 dft_dg_h, dft_energy, dft_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(record.get("metadata", {}), ensure_ascii=False),
                record.get("dft_dg_h"),
                record.get("dft_energy"),
                record.get("dft_status"),
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
                       "target_dgH", "error", "decision", "created_at", "updated_at", "metadata",
                       "dft_dg_h", "dft_energy", "dft_status"]
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
                       "target_dgH", "error", "decision", "created_at", "updated_at", "metadata",
                       "dft_dg_h", "dft_energy", "dft_status"]
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
                       "target_dgH", "error", "decision", "created_at", "updated_at", "metadata",
                       "dft_dg_h", "dft_energy", "dft_status"]
            records = []
            for row in rows:
                record = dict(zip(columns, row))
                if record["metadata"]:
                    record["metadata"] = json.loads(record["metadata"])
                records.append(record)
            return records

    def backfill_dft(self, uuid: str, dft_dg_h: float,
                     dft_energy: Optional[float] = None,
                     dft_status: str = "completed") -> bool:
        """原子回填 DFT 结果并同步将状态从 filtered_in 改为 dft_verified。
        只在 status='filtered_in' 时执行，避免并发 TOCTOU 竞态。
        返回 True 表示找到 filtered_in 记录并成功更新，False 表示记录不存在或状态非 filtered_in。"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute("""
                UPDATE structures
                SET dft_dg_h=?, dft_energy=?, dft_status=?, status='dft_verified', updated_at=?
                WHERE uuid=? AND status='filtered_in'
            """, (dft_dg_h, dft_energy, dft_status, now, uuid))
            conn.commit()
            return cursor.rowcount > 0

    def record_state_change(self, structure_uuid: str, from_state: Optional[str],
                            to_state: str, reason: str = "") -> int:
        """Record a state transition for traceability. Returns the new history row id."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO state_history (structure_uuid, from_state, to_state, changed_at, reason)
                VALUES (?, ?, ?, ?, ?)
            """, (structure_uuid, from_state or "", to_state, datetime.now().isoformat(), reason))
            conn.commit()
            return cursor.lastrowid

    def get_state_history(self, structure_uuid: str) -> List[Dict[str, Any]]:
        """Return transition history for a structure, newest first."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT structure_uuid, from_state, to_state, changed_at, reason "
                "FROM state_history WHERE structure_uuid = ? ORDER BY id DESC",
                (structure_uuid,),
            )
            columns = ["structure_uuid", "from_state", "to_state", "changed_at", "reason"]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def export_validated(self) -> List[Dict[str, Any]]:
        """Return records that passed human validation (flywheel ⑥ feedback export)."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM structures WHERE status IN ('validated', 'exported_for_training')"
            )
            rows = cursor.fetchall()
            columns = ["uuid", "task_id", "status", "elements", "poscar", "predicted_dgH",
                       "target_dgH", "error", "decision", "created_at", "updated_at", "metadata",
                       "dft_dg_h", "dft_energy", "dft_status"]
            records = []
            for row in rows:
                record = dict(zip(columns, row))
                if record["metadata"]:
                    record["metadata"] = json.loads(record["metadata"])
                records.append(record)
            return records

    def get_state_distribution(self) -> Dict[str, int]:
        """Return {state: count} across all structures (flywheel monitoring)."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT status, COUNT(*) FROM structures GROUP BY status"
            )
            return {row[0]: row[1] for row in cursor.fetchall()}

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
