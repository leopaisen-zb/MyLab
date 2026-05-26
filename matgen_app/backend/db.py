"""
SQLite persistence for batch results and human review.
Database file: matgen_app/results.db
"""

import sqlite3
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any

DB_PATH = Path(__file__).resolve().parent.parent / "results.db"

# Must match the instruction field used in rag_gen.py training-data export
SYSTEM_INSTRUCTION = (
    "You are a materials science expert. Given the physical and chemical properties "
    "of a hydrogen storage material and several reference structures with similar properties, "
    "generate the corresponding atomic structure in VASP POSCAR format."
)

# ── 完整7状态定义（对应论文第五章）──────────────────────────────────────────────
# generated | rejected_precheck | predicted | filtered_in | filtered_out | validated | rejected | exported_for_training
ALL_STATES = [
    "generated",
    "rejected_precheck",
    "predicted",
    "filtered_in",
    "filtered_out",
    "validated",
    "rejected",
    "exported_for_training",
]

# 状态分组展示用
STATE_GROUPS = {
    "进行中": ["generated", "rejected_precheck", "predicted"],
    "已初筛": ["filtered_in", "filtered_out"],
    "已终审": ["validated", "rejected", "exported_for_training"],
}

# 有效的状态转移（单向，不可逆）
VALID_TRANSITIONS = {
    "generated": ["rejected_precheck", "predicted"],
    "rejected_precheck": [],           # 终态
    "predicted": ["filtered_in", "filtered_out"],
    "filtered_in": ["validated", "rejected"],
    "filtered_out": [],                # 终态
    "validated": ["exported_for_training"],
    "rejected": [],                    # 终态
    "exported_for_training": [],       # 终态
}


def _conn() -> sqlite3.Connection:
    """Return a connection with row_factory=sqlite3.Row."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    """Create tables if not exist. Call at app startup."""
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS batch_samples (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid            TEXT    NOT NULL,
                created_at       TEXT    NOT NULL,
                prompt           TEXT    NOT NULL,
                poscar           TEXT    NOT NULL,
                dg_h             REAL,
                in_filter        INTEGER NOT NULL DEFAULT 0,
                human_status     TEXT    NOT NULL DEFAULT 'pending',
                validation_status TEXT   NOT NULL DEFAULT 'pending',
                notes            TEXT    NOT NULL DEFAULT '',
                rejection_level  TEXT    NOT NULL DEFAULT '',
                current_state    TEXT    NOT NULL DEFAULT 'generated'
            )
        """)
        # Migration: add columns if not exist (for existing DBs)
        for col_sql in [
            "ALTER TABLE batch_samples ADD COLUMN uuid TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE batch_samples ADD COLUMN rejection_level TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE batch_samples ADD COLUMN current_state TEXT NOT NULL DEFAULT 'generated'",
        ]:
            try:
                con.execute(col_sql)
            except Exception:
                pass

        # State transition history table for full traceability
        con.execute("""
            CREATE TABLE IF NOT EXISTS state_history (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id    INTEGER NOT NULL,
                from_state   TEXT,
                to_state     TEXT    NOT NULL,
                changed_at   TEXT    NOT NULL,
                reason       TEXT    NOT NULL DEFAULT '',
                FOREIGN KEY (sample_id) REFERENCES batch_samples(id)
            )
        """)


def save_sample(prompt: str, poscar: str, dg_h: Optional[float], in_filter: bool, rejection_level: str = '', current_state: str = '') -> int:
    """Insert a new sample. Returns the new row id."""
    sample_uuid = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    # Determine initial state based on rejection_level
    if rejection_level == 'text':
        initial_state = 'rejected_precheck'
    elif rejection_level == 'structure':
        initial_state = 'rejected_precheck'
    elif rejection_level == 'physics':
        initial_state = 'rejected_precheck'
    elif rejection_level == 'pass' and poscar:
        initial_state = 'generated'
    else:
        initial_state = current_state or 'generated'

    with _conn() as con:
        cur = con.execute(
            """
            INSERT INTO batch_samples (created_at, prompt, poscar, dg_h, in_filter, rejection_level, uuid, current_state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                prompt,
                poscar,
                dg_h,
                1 if in_filter else 0,
                rejection_level,
                sample_uuid,
                initial_state,
            ),
        )
        sample_id = cur.lastrowid

        # Record initial state in history
        con.execute(
            """
            INSERT INTO state_history (sample_id, from_state, to_state, changed_at, reason)
            VALUES (?, ?, ?, ?, ?)
            """,
            (sample_id, '', initial_state, now, 'Initial creation'),
        )
        return sample_id


def get_all_samples() -> List[Dict[str, Any]]:
    """Return all samples as list of dicts, newest first."""
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM batch_samples ORDER BY id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_sample(sample_id: int) -> Optional[Dict[str, Any]]:
    """Return one sample by id, or None."""
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM batch_samples WHERE id = ?", (sample_id,)
        ).fetchone()
    return dict(row) if row is not None else None


def update_review(
    sample_id: int,
    human_status: str,
    validation_status: str,
    notes: str = "",
) -> bool:
    """Update human_status, validation_status, notes. Returns True if row was found."""
    with _conn() as con:
        cur = con.execute(
            """
            UPDATE batch_samples
            SET human_status = ?, validation_status = ?, notes = ?
            WHERE id = ?
            """,
            (human_status, validation_status, notes, sample_id),
        )
    return cur.rowcount > 0


def export_validated() -> List[Dict[str, Any]]:
    """Return all samples where validation_status='validated' as training-format dicts.

    Each dict: {'instruction': ..., 'input': prompt, 'output': poscar, 'dg_h': dg_h}
    """
    with _conn() as con:
        rows = con.execute(
            "SELECT prompt, poscar, dg_h FROM batch_samples WHERE validation_status = 'validated'"
        ).fetchall()
    return [
        {
            "instruction": SYSTEM_INSTRUCTION,
            "input": row["prompt"],
            "output": row["poscar"],
            "dg_h": row["dg_h"],
        }
        for row in rows
    ]


def delete_sample(sample_id: int) -> bool:
    """Delete a sample. Returns True if found."""
    with _conn() as con:
        cur = con.execute(
            "DELETE FROM batch_samples WHERE id = ?", (sample_id,)
        )
    return cur.rowcount > 0


def record_state_change(sample_id: int, from_state: str, to_state: str, reason: str = '') -> int:
    """Record a state transition. Returns the new row id."""
    with _conn() as con:
        cur = con.execute(
            """
            INSERT INTO state_history (sample_id, from_state, to_state, changed_at, reason)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                sample_id,
                from_state or '',
                to_state,
                datetime.utcnow().isoformat(),
                reason,
            ),
        )
        return cur.lastrowid


def get_state_history(sample_id: int) -> List[Dict[str, Any]]:
    """Return state transition history for a sample, newest first."""
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM state_history WHERE sample_id = ? ORDER BY changed_at DESC",
            (sample_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_stats() -> Dict[str, Any]:
    """Return summary stats: total, pending, approved, rejected, validated, in_filter_count."""
    with _conn() as con:
        row = con.execute(
            """
            SELECT
                COUNT(*)                                              AS total,
                SUM(human_status = 'pending')                        AS pending,
                SUM(human_status = 'approved')                       AS approved,
                SUM(human_status = 'rejected')                       AS rejected,
                SUM(validation_status = 'validated')                 AS validated,
                SUM(in_filter = 1)                                   AS in_filter_count
            FROM batch_samples
            """
        ).fetchone()
    return dict(row)



def query_samples(
    dg_h_min: Optional[float] = None,
    dg_h_max: Optional[float] = None,
    human_status: Optional[str] = None,
    validation_status: Optional[str] = None,
    limit: int = 100,
) -> list:
    """按条件过滤样本，返回匹配记录（最新优先）。"""
    conditions = []
    params = []
    if dg_h_min is not None:
        conditions.append("dg_h >= ?")
        params.append(dg_h_min)
    if dg_h_max is not None:
        conditions.append("dg_h <= ?")
        params.append(dg_h_max)
    if human_status is not None and human_status != "all":
        conditions.append("human_status = ?")
        params.append(human_status)
    if validation_status is not None and validation_status != "all":
        conditions.append("validation_status = ?")
        params.append(validation_status)
    where = " AND ".join(conditions) if conditions else "1=1"
    query = f"SELECT * FROM batch_samples WHERE {where} ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with _conn() as con:
        rows = con.execute(query, params).fetchall()
    return [dict(row) for row in rows]

def get_distribution(bin_count: int = 20) -> dict:
    """Return {'bins': [...], 'counts': [...], 'filter_rates': [...]} for ΔG_H histogram."""
    with _conn() as con:
        rows = con.execute(
            "SELECT dg_h FROM batch_samples WHERE dg_h IS NOT NULL ORDER BY created_at"
        ).fetchall()
    if not rows:
        return {"bins": [], "counts": [], "filter_rates": []}
    values = [r["dg_h"] for r in rows]
    import numpy as np
    hist, bin_edges = np.histogram(values, bins=bin_count)
    # filter rate per bin
    filter_rates = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_range = [v for v in values if lo <= v < hi]
        if in_range:
            total = len(in_range)
            passed = sum(1 for v in in_range if -1.0 <= v <= 0.5)
            filter_rates.append(passed / total)
        else:
            filter_rates.append(0.0)
    return {
        "bins": [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)],
        "counts": hist.tolist(),
        "filter_rates": [round(x, 3) for x in filter_rates],
        "total": len(values),
    }

def get_rejection_level_stats() -> dict:
    """Return rejection level distribution: {level: count, ...}"""
    with _conn() as con:
        rows = con.execute(
            """
            SELECT rejection_level, COUNT(*) as count
            FROM batch_samples
            GROUP BY rejection_level
            ORDER BY count DESC
            """
        ).fetchall()
    result = {}
    total = 0
    for row in rows:
        level = row["rejection_level"] or "unknown"
        count = row["count"]
        result[level] = count
        total += count
    result["_total"] = total
    return result

def transition_state(sample_id: int, to_state: str, reason: str = '') -> bool:
    """
    执行单向状态转移。验证转移合法性并记录历史。
    返回是否成功（失败时返回False，如非法转移）。
    """
    sample = get_sample(sample_id)
    if sample is None:
        return False
    from_state = sample.get("current_state", "generated")
    if to_state not in VALID_TRANSITIONS.get(from_state, []):
        return False
    now = datetime.utcnow().isoformat()
    with _conn() as con:
        con.execute(
            "UPDATE batch_samples SET current_state = ? WHERE id = ?",
            (to_state, sample_id),
        )
        con.execute(
            """
            INSERT INTO state_history (sample_id, from_state, to_state, changed_at, reason)
            VALUES (?, ?, ?, ?, ?)
            """,
            (sample_id, from_state, to_state, now, reason),
        )
    return True

def get_samples_by_state(state: str, limit: int = 100) -> list:
    """按当前状态筛选样本"""
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM batch_samples WHERE current_state = ? ORDER BY id DESC LIMIT ?",
            (state, limit),
        ).fetchall()
    return [dict(row) for row in rows]

def get_filtered_in_samples(limit: int = 100) -> list:
    """获取所有 filtered_in 状态（待专家审查）的样本"""
    return get_samples_by_state("filtered_in", limit)

def get_state_distribution() -> dict:
    """返回各状态的样本计数分布"""
    with _conn() as con:
        rows = con.execute(
            """
            SELECT current_state, COUNT(*) as count
            FROM batch_samples
            GROUP BY current_state
            ORDER BY count DESC
            """
        ).fetchall()
    return {row["current_state"]: row["count"] for row in rows}

def update_sample_uuid(sample_id: int, new_uuid: str) -> bool:
    """为已有样本更新UUID（仅用于历史数据迁移）"""
    with _conn() as con:
        cur = con.execute(
            "UPDATE batch_samples SET uuid = ? WHERE id = ?",
            (new_uuid, sample_id),
        )
    return cur.rowcount > 0

def add_missing_uuids() -> int:
    """为所有缺少UUID的样本补充UUID，返回更新数量"""
    count = 0
    with _conn() as con:
        rows = con.execute(
            "SELECT id FROM batch_samples WHERE uuid IS NULL OR uuid = ''"
        ).fetchall()
    for row in rows:
        new_uuid = str(uuid.uuid4())
        con.execute("UPDATE batch_samples SET uuid = ? WHERE id = ?", (new_uuid, row["id"]))
        count += 1
    return count
