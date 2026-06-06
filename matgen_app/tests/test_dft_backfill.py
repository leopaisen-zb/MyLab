# tests/test_dft_backfill.py
# DFT 验证回填节点测试（P1-1）
# 按 RED → GREEN 顺序编写，覆盖：状态机、StateStore、Orchestrator、API 单条及批量
import pytest
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# 1. 状态机：新状态与转移合法性
# ─────────────────────────────────────────────────────────────────────────────
class TestDFTStateMachine:
    """DFT_VERIFIED 状态及相关转移规则。"""

    def test_dft_verified_state_exists(self):
        """StructureState 必须包含 dft_verified。"""
        from core.state_machine import StructureState
        assert hasattr(StructureState, "DFT_VERIFIED")
        assert StructureState.DFT_VERIFIED.value == "dft_verified"

    def test_all_states_now_9(self):
        """添加 dft_verified 后共 9 个状态。"""
        from core.state_machine import StructureState
        assert len(StructureState) == 9

    def test_filtered_in_to_dft_verified_valid(self):
        """filtered_in → dft_verified 合法。"""
        from core.state_machine import StructureState, StateTransition
        assert StateTransition.can_transition(
            StructureState.FILTERED_IN, StructureState.DFT_VERIFIED
        ) is True

    def test_filtered_in_to_rejected_still_valid(self):
        """filtered_in → rejected 仍然合法（保留直接拒绝通道）。"""
        from core.state_machine import StructureState, StateTransition
        assert StateTransition.can_transition(
            StructureState.FILTERED_IN, StructureState.REJECTED
        ) is True

    def test_filtered_in_to_validated_now_invalid(self):
        """filtered_in → validated 现在不再直接合法（必须经过 dft_verified）。"""
        from core.state_machine import StructureState, StateTransition
        assert StateTransition.can_transition(
            StructureState.FILTERED_IN, StructureState.VALIDATED
        ) is False

    def test_dft_verified_to_validated_valid(self):
        """dft_verified → validated 合法。"""
        from core.state_machine import StructureState, StateTransition
        assert StateTransition.can_transition(
            StructureState.DFT_VERIFIED, StructureState.VALIDATED
        ) is True

    def test_dft_verified_to_rejected_valid(self):
        """dft_verified → rejected 合法。"""
        from core.state_machine import StructureState, StateTransition
        assert StateTransition.can_transition(
            StructureState.DFT_VERIFIED, StructureState.REJECTED
        ) is True

    def test_dft_verified_not_terminal(self):
        """dft_verified 不是终态。"""
        from core.state_machine import StructureState, StateTransition
        assert StateTransition.is_terminal(StructureState.DFT_VERIFIED) is False

    def test_dft_verified_get_next_states(self):
        """dft_verified 的下游集合为 {validated, rejected}。"""
        from core.state_machine import StructureState, StateTransition
        next_s = StateTransition.get_next_states(StructureState.DFT_VERIFIED)
        assert next_s == {StructureState.VALIDATED, StructureState.REJECTED}

    def test_filtered_in_get_next_states_updated(self):
        """filtered_in 下游更新为 {dft_verified, rejected}。"""
        from core.state_machine import StructureState, StateTransition
        next_s = StateTransition.get_next_states(StructureState.FILTERED_IN)
        assert next_s == {StructureState.DFT_VERIFIED, StructureState.REJECTED}

    def test_generated_to_dft_verified_invalid(self):
        """不能跳步转移到 dft_verified。"""
        from core.state_machine import StructureState, StateTransition
        assert StateTransition.can_transition(
            StructureState.GENERATED, StructureState.DFT_VERIFIED
        ) is False


# ─────────────────────────────────────────────────────────────────────────────
# 2. StateStore：新列与 backfill_dft
# ─────────────────────────────────────────────────────────────────────────────
class TestStateStoreDFT:
    """StateStore DFT 相关功能。"""

    def test_structures_table_has_dft_columns(self, temp_db):
        """structures 表应包含 dft_dg_h、dft_energy、dft_status 三列。"""
        import sqlite3
        from persistence.state_store import StateStore

        StateStore(db_path=temp_db)
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(structures)")
        cols = {row[1] for row in cursor.fetchall()}
        conn.close()

        assert "dft_dg_h" in cols
        assert "dft_energy" in cols
        assert "dft_status" in cols

    def test_backfill_dft_returns_true_when_found(self, temp_db):
        """backfill_dft 找到 filtered_in 记录时返回 True。"""
        from persistence.state_store import StateStore

        store = StateStore(db_path=temp_db)
        uid = str(uuid.uuid4())
        store.save_record(uid, {
            "task_id": "t1", "status": "filtered_in",
            "elements": "Ir2", "poscar": "",
        })

        result = store.backfill_dft(uid, dft_dg_h=-0.48)
        assert result is True

    def test_backfill_dft_returns_false_when_not_found(self, temp_db):
        """backfill_dft 找不到记录时返回 False。"""
        from persistence.state_store import StateStore

        store = StateStore(db_path=temp_db)
        result = store.backfill_dft("nonexistent-uuid", dft_dg_h=-0.48)
        assert result is False

    def test_backfill_dft_returns_false_when_not_filtered_in(self, temp_db):
        """H-3 修复：backfill_dft 对非 filtered_in 状态（如 generated）返回 False，状态不变。"""
        from persistence.state_store import StateStore

        store = StateStore(db_path=temp_db)
        uid = str(uuid.uuid4())
        store.save_record(uid, {
            "task_id": "t1", "status": "generated",
            "elements": "Ir2", "poscar": "",
        })

        result = store.backfill_dft(uid, dft_dg_h=-0.48)
        assert result is False
        rec = store.get_record(uid)
        assert rec["status"] == "generated"
        assert rec["dft_dg_h"] is None

    def test_backfill_dft_atomically_changes_status(self, temp_db):
        """H-3 修复：backfill_dft 成功后状态原子变为 dft_verified（无需外部调用 save_record）。"""
        from persistence.state_store import StateStore

        store = StateStore(db_path=temp_db)
        uid = str(uuid.uuid4())
        store.save_record(uid, {
            "task_id": "t1", "status": "filtered_in",
            "elements": "Ir2", "poscar": "",
        })

        store.backfill_dft(uid, dft_dg_h=-0.48)
        rec = store.get_record(uid)
        assert rec["status"] == "dft_verified"

    def test_backfill_dft_second_call_returns_false(self, temp_db):
        """H-3 修复：对同一结构连续调两次 backfill_dft，第二次因状态已变为 dft_verified 而返回 False。"""
        from persistence.state_store import StateStore

        store = StateStore(db_path=temp_db)
        uid = str(uuid.uuid4())
        store.save_record(uid, {
            "task_id": "t1", "status": "filtered_in",
            "elements": "Ir2", "poscar": "",
        })

        first = store.backfill_dft(uid, dft_dg_h=-0.48)
        second = store.backfill_dft(uid, dft_dg_h=-0.49)
        assert first is True
        assert second is False  # 状态已非 filtered_in

    def test_backfill_dft_stores_values(self, temp_db):
        """backfill_dft 写入的三列值能被 get_record 读取。"""
        from persistence.state_store import StateStore

        store = StateStore(db_path=temp_db)
        uid = str(uuid.uuid4())
        store.save_record(uid, {
            "task_id": "t1", "status": "filtered_in",
            "elements": "Ir2", "poscar": "",
        })

        store.backfill_dft(uid, dft_dg_h=-0.48, dft_energy=-12.34, dft_status="completed")
        rec = store.get_record(uid)

        assert rec["dft_dg_h"] == pytest.approx(-0.48)
        assert rec["dft_energy"] == pytest.approx(-12.34)
        assert rec["dft_status"] == "completed"

    def test_backfill_dft_default_status_completed(self, temp_db):
        """dft_status 默认值为 'completed'。"""
        from persistence.state_store import StateStore

        store = StateStore(db_path=temp_db)
        uid = str(uuid.uuid4())
        store.save_record(uid, {
            "task_id": "t1", "status": "filtered_in",
            "elements": "Ir2", "poscar": "",
        })

        store.backfill_dft(uid, dft_dg_h=-0.48)
        rec = store.get_record(uid)
        assert rec["dft_status"] == "completed"

    def test_backfill_dft_dft_energy_optional_none(self, temp_db):
        """dft_energy 可为 None（不必填）。"""
        from persistence.state_store import StateStore

        store = StateStore(db_path=temp_db)
        uid = str(uuid.uuid4())
        store.save_record(uid, {
            "task_id": "t1", "status": "filtered_in",
            "elements": "Ir2", "poscar": "",
        })

        store.backfill_dft(uid, dft_dg_h=-0.48)
        rec = store.get_record(uid)
        assert rec["dft_energy"] is None

    def test_migration_idempotent(self, temp_db):
        """多次初始化 StateStore（模拟已有 DB）不报错，列仍然存在。"""
        import sqlite3
        from persistence.state_store import StateStore

        StateStore(db_path=temp_db)
        StateStore(db_path=temp_db)  # 第二次初始化（迁移幂等）

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(structures)")
        cols = {row[1] for row in cursor.fetchall()}
        conn.close()

        assert "dft_dg_h" in cols


# ─────────────────────────────────────────────────────────────────────────────
# 3. TaskOrchestrator：submit_dft_result
# ─────────────────────────────────────────────────────────────────────────────
class TestOrchestratorDFT:
    """TaskOrchestrator.submit_dft_result 测试。"""

    def _make_store_with_record(self, temp_db, status: str):
        """辅助：创建 StateStore 并插入一条指定状态的记录，返回 (store, uid)。"""
        from persistence.state_store import StateStore

        store = StateStore(db_path=temp_db)
        uid = str(uuid.uuid4())
        store.save_record(uid, {
            "task_id": "t1", "status": status,
            "elements": "Ir2", "poscar": "",
        })
        return store, uid

    def test_submit_dft_result_success(self, temp_db):
        """filtered_in 状态下提交 DFT 结果成功，状态变为 dft_verified。"""
        from core.task_orchestrator import TaskOrchestrator
        from persistence.state_store import StateStore

        store, uid = self._make_store_with_record(temp_db, "filtered_in")
        orch = TaskOrchestrator.__new__(TaskOrchestrator)
        orch.state_store = StateStore(db_path=temp_db)

        result = orch.submit_dft_result(uid, dft_dg_h=-0.48)
        assert result is True

        rec = orch.state_store.get_record(uid)
        assert rec["status"] == "dft_verified"
        assert rec["dft_dg_h"] == pytest.approx(-0.48)

    def test_submit_dft_result_wrong_state_returns_false(self, temp_db):
        """非 filtered_in 状态（如 generated）时返回 False，状态不变。"""
        from core.task_orchestrator import TaskOrchestrator
        from persistence.state_store import StateStore

        store, uid = self._make_store_with_record(temp_db, "generated")
        orch = TaskOrchestrator.__new__(TaskOrchestrator)
        orch.state_store = StateStore(db_path=temp_db)

        result = orch.submit_dft_result(uid, dft_dg_h=-0.48)
        assert result is False

        rec = orch.state_store.get_record(uid)
        assert rec["status"] == "generated"

    def test_submit_dft_result_nonexistent_returns_false(self, temp_db):
        """不存在的 UUID 返回 False。"""
        from core.task_orchestrator import TaskOrchestrator
        from persistence.state_store import StateStore

        orch = TaskOrchestrator.__new__(TaskOrchestrator)
        orch.state_store = StateStore(db_path=temp_db)

        result = orch.submit_dft_result("no-such-uuid", dft_dg_h=-0.48)
        assert result is False

    def test_submit_dft_result_records_state_history(self, temp_db):
        """提交成功后 state_history 中应有 filtered_in→dft_verified 记录。"""
        from core.task_orchestrator import TaskOrchestrator
        from persistence.state_store import StateStore

        store, uid = self._make_store_with_record(temp_db, "filtered_in")
        orch = TaskOrchestrator.__new__(TaskOrchestrator)
        orch.state_store = StateStore(db_path=temp_db)

        orch.submit_dft_result(uid, dft_dg_h=-0.48)
        history = orch.state_store.get_state_history(uid)

        assert len(history) >= 1
        latest = history[0]
        assert latest["from_state"] == "filtered_in"
        assert latest["to_state"] == "dft_verified"

    def test_submit_dft_result_with_energy(self, temp_db):
        """传入 dft_energy 时正确存储。"""
        from core.task_orchestrator import TaskOrchestrator
        from persistence.state_store import StateStore

        store, uid = self._make_store_with_record(temp_db, "filtered_in")
        orch = TaskOrchestrator.__new__(TaskOrchestrator)
        orch.state_store = StateStore(db_path=temp_db)

        orch.submit_dft_result(uid, dft_dg_h=-0.48, dft_energy=-15.0)
        rec = orch.state_store.get_record(uid)
        assert rec["dft_energy"] == pytest.approx(-15.0)

    def test_submit_dft_result_generated_state_no_history(self, temp_db):
        """H-3 修复：对非 filtered_in 状态（generated）调 submit_dft_result 返回 False 且不产生历史。"""
        from core.task_orchestrator import TaskOrchestrator
        from persistence.state_store import StateStore

        store, uid = self._make_store_with_record(temp_db, "generated")
        orch = TaskOrchestrator.__new__(TaskOrchestrator)
        orch.state_store = StateStore(db_path=temp_db)

        result = orch.submit_dft_result(uid, dft_dg_h=-0.48)
        assert result is False

        history = orch.state_store.get_state_history(uid)
        assert len(history) == 0, "No history should be recorded when transition is rejected"

    def test_submit_dft_result_double_call_single_history(self, temp_db):
        """H-3 修复：对同一结构连续调两次 submit_dft_result，第二次返回 False，历史只有一条 dft_verified。"""
        from core.task_orchestrator import TaskOrchestrator
        from persistence.state_store import StateStore

        store, uid = self._make_store_with_record(temp_db, "filtered_in")
        orch = TaskOrchestrator.__new__(TaskOrchestrator)
        orch.state_store = StateStore(db_path=temp_db)

        first = orch.submit_dft_result(uid, dft_dg_h=-0.48)
        second = orch.submit_dft_result(uid, dft_dg_h=-0.49)
        assert first is True
        assert second is False

        history = orch.state_store.get_state_history(uid)
        dft_verified_entries = [h for h in history if h["to_state"] == "dft_verified"]
        assert len(dft_verified_entries) == 1, "Only one dft_verified history entry should exist"


# ─────────────────────────────────────────────────────────────────────────────
# 4. API：POST /structures/{id}/dft（单条）
# ─────────────────────────────────────────────────────────────────────────────
class TestDFTRoutesSingle:
    """POST /structures/{id}/dft 端点测试。"""

    def _make_app_with_record(self, temp_db, status: str):
        """辅助：创建 FastAPI app + 预插入一条记录，返回 (client, uid)。
        使用独立 TaskOrchestrator 实例，避免污染模块级单例。"""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from persistence.state_store import StateStore
        from core.task_orchestrator import TaskOrchestrator
        from api import routes as route_module

        # 构造独立 app + 独立 orchestrator（隔离模块级单例）
        store = StateStore(db_path=temp_db)
        uid = str(uuid.uuid4())
        store.save_record(uid, {
            "task_id": "t1", "status": status,
            "elements": "Ir2", "poscar": "",
        })

        orch = TaskOrchestrator.__new__(TaskOrchestrator)
        orch.state_store = store

        app = FastAPI()

        from fastapi import APIRouter, HTTPException
        from api.schemas import DFTResultRequest, DFTBatchItem, DFTBatchResponse
        from typing import List as _List

        local_router = APIRouter()

        @local_router.get("/structures/{sid}")
        async def _get(sid: str):
            rec = orch.state_store.get_record(sid)
            if rec is None:
                raise HTTPException(status_code=404, detail="not found")
            return rec

        @local_router.post("/structures/dft/batch")
        async def _batch(items: _List[DFTBatchItem]):
            ok = 0; fail = []
            for item in items:
                if orch.submit_dft_result(item.uuid, item.dft_dg_h, item.dft_energy):
                    ok += 1
                else:
                    fail.append(item.uuid)
            return {"success_count": ok, "failed": fail}

        @local_router.post("/structures/{sid}/dft")
        async def _dft(sid: str, req: DFTResultRequest):
            rec = orch.state_store.get_record(sid)
            if rec is None:
                raise HTTPException(status_code=404, detail="not found")
            ok = orch.submit_dft_result(sid, req.dft_dg_h, req.dft_energy)
            if not ok:
                raise HTTPException(status_code=409, detail="bad state")
            return {"structure_id": sid, "status": "dft_verified"}

        app.include_router(local_router, prefix="/api/v1")
        return TestClient(app), uid

    def test_dft_endpoint_success(self, temp_db):
        """filtered_in 结构提交 DFT 结果返回 200。"""
        client, uid = self._make_app_with_record(temp_db, "filtered_in")
        response = client.post(
            f"/api/v1/structures/{uid}/dft",
            json={"dft_dg_h": -0.48}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["structure_id"] == uid
        assert data["status"] == "dft_verified"

    def test_dft_endpoint_nonexistent_returns_404(self, temp_db):
        """不存在的结构返回 404。"""
        client, _ = self._make_app_with_record(temp_db, "filtered_in")
        response = client.post(
            "/api/v1/structures/no-such-id/dft",
            json={"dft_dg_h": -0.48}
        )
        assert response.status_code == 404

    def test_dft_endpoint_wrong_state_returns_409(self, temp_db):
        """非 filtered_in 状态（如 generated）返回 409 Conflict。"""
        client, uid = self._make_app_with_record(temp_db, "generated")
        response = client.post(
            f"/api/v1/structures/{uid}/dft",
            json={"dft_dg_h": -0.48}
        )
        assert response.status_code == 409

    def test_dft_endpoint_with_energy(self, temp_db):
        """同时传入 dft_energy 返回 200。"""
        client, uid = self._make_app_with_record(temp_db, "filtered_in")
        response = client.post(
            f"/api/v1/structures/{uid}/dft",
            json={"dft_dg_h": -0.48, "dft_energy": -12.5}
        )
        assert response.status_code == 200

    def test_dft_endpoint_missing_dgt_h_returns_422(self, temp_db):
        """缺少必填字段 dft_dg_h 返回 422 Validation Error。"""
        client, uid = self._make_app_with_record(temp_db, "filtered_in")
        response = client.post(
            f"/api/v1/structures/{uid}/dft",
            json={}
        )
        assert response.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# 5. API：POST /structures/dft/batch（批量回填）
# ─────────────────────────────────────────────────────────────────────────────
class TestDFTRoutesBatch:
    """POST /structures/dft/batch 端点测试。"""

    def _make_app_with_records(self, temp_db, n: int, status: str = "filtered_in"):
        """辅助：创建独立 app + 预插入 n 条记录，返回 (client, [uid...])。"""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from persistence.state_store import StateStore
        from core.task_orchestrator import TaskOrchestrator
        from api.schemas import DFTBatchItem
        from typing import List as _List

        store = StateStore(db_path=temp_db)
        uids = []
        for _ in range(n):
            uid = str(uuid.uuid4())
            store.save_record(uid, {
                "task_id": "t1", "status": status,
                "elements": "Ir2", "poscar": "",
            })
            uids.append(uid)

        orch = TaskOrchestrator.__new__(TaskOrchestrator)
        orch.state_store = store

        app = FastAPI()
        from fastapi import APIRouter

        local_router = APIRouter()

        @local_router.post("/structures/dft/batch")
        async def _batch(items: _List[DFTBatchItem]):
            ok = 0; fail = []
            for item in items:
                if orch.submit_dft_result(item.uuid, item.dft_dg_h, item.dft_energy):
                    ok += 1
                else:
                    fail.append(item.uuid)
            return {"success_count": ok, "failed": fail}

        app.include_router(local_router, prefix="/api/v1")
        return TestClient(app), uids

    def test_batch_dft_all_success(self, temp_db):
        """所有合法记录批量回填成功。"""
        client, uids = self._make_app_with_records(temp_db, 3)
        payload = [{"uuid": uid, "dft_dg_h": -0.48} for uid in uids]
        response = client.post("/api/v1/structures/dft/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success_count"] == 3
        assert data["failed"] == []

    def test_batch_dft_partial_failure(self, temp_db):
        """部分不存在的 UUID 出现在 failed 列表中。"""
        client, uids = self._make_app_with_records(temp_db, 2)
        payload = [
            {"uuid": uids[0], "dft_dg_h": -0.48},
            {"uuid": "nonexistent-uuid", "dft_dg_h": -0.30},
        ]
        response = client.post("/api/v1/structures/dft/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success_count"] == 1
        assert "nonexistent-uuid" in data["failed"]

    def test_batch_dft_empty_list(self, temp_db):
        """空列表返回 success_count=0。"""
        client, _ = self._make_app_with_records(temp_db, 1)
        response = client.post("/api/v1/structures/dft/batch", json=[])
        assert response.status_code == 200
        data = response.json()
        assert data["success_count"] == 0
        assert data["failed"] == []

    def test_batch_dft_with_optional_energy(self, temp_db):
        """携带可选 dft_energy 的批量请求正常处理。"""
        client, uids = self._make_app_with_records(temp_db, 2)
        payload = [
            {"uuid": uids[0], "dft_dg_h": -0.48, "dft_energy": -12.0},
            {"uuid": uids[1], "dft_dg_h": -0.50},
        ]
        response = client.post("/api/v1/structures/dft/batch", json=payload)
        assert response.status_code == 200
        assert response.json()["success_count"] == 2
