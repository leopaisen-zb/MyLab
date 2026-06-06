# tests/test_dataset_feedback.py
"""
TDD 测试套件：backend/dataset_feedback.py
P1-2: 数据集回灌 + 索引重建（飞轮闭环第六步）
"""
import json
import os
import uuid
import pytest

from backend.dataset_feedback import (
    records_to_training_format,
    append_to_dataset,
    rebuild_near_neighbor_index,
    feedback_pipeline,
)
from persistence.state_store import StateStore


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

INSTRUCTION = "You are a materials science expert. Generate POSCAR."


def _make_record(poscar="POSCAR content", predicted_dg=-0.3, target_dg=-0.5,
                 elements="IrPdPt", dft_dg_h=None, status="validated"):
    """构造一条模拟 StateStore.export_validated() 返回的记录。"""
    r = {
        "uuid": str(uuid.uuid4()),
        "task_id": "task-001",
        "status": status,
        "elements": elements,
        "poscar": poscar,
        "predicted_dgH": predicted_dg,
        "target_dgH": target_dg,
        "created_at": "2026-06-06T10:00:00",
    }
    if dft_dg_h is not None:
        r["dft_dg_h"] = dft_dg_h
    return r


# ─────────────────────────────────────────────────────────────────
# 1. records_to_training_format
# ─────────────────────────────────────────────────────────────────

class TestRecordsToTrainingFormat:
    def test_basic_conversion(self):
        """正常记录应转换为含4个键的训练格式 dict。"""
        records = [_make_record(poscar="POSCAR_A", predicted_dg=-0.4)]
        result = records_to_training_format(records, INSTRUCTION)
        assert len(result) == 1
        item = result[0]
        assert item["instruction"] == INSTRUCTION
        assert item["output"] == "POSCAR_A"
        assert item["dg_h"] == pytest.approx(-0.4)
        assert "input" in item
        assert isinstance(item["input"], str) and len(item["input"]) > 0

    def test_dft_dg_h_priority_over_predicted(self):
        """若记录含 dft_dg_h，dg_h 应优先取 dft_dg_h 而非 predicted_dgH。"""
        rec = _make_record(predicted_dg=-0.3, dft_dg_h=-0.99)
        result = records_to_training_format([rec], INSTRUCTION)
        assert result[0]["dg_h"] == pytest.approx(-0.99)

    def test_no_dft_falls_back_to_predicted(self):
        """无 dft_dg_h 时，dg_h 应取 predicted_dgH。"""
        rec = _make_record(predicted_dg=-0.55)
        result = records_to_training_format([rec], INSTRUCTION)
        assert result[0]["dg_h"] == pytest.approx(-0.55)

    def test_skip_empty_poscar(self):
        """poscar 为空字符串或 None 的记录应被跳过。"""
        records = [
            _make_record(poscar=""),
            _make_record(poscar=None),
            _make_record(poscar="VALID_POSCAR"),
        ]
        result = records_to_training_format(records, INSTRUCTION)
        assert len(result) == 1
        assert result[0]["output"] == "VALID_POSCAR"

    def test_use_prompt_key_if_present(self):
        """若记录已含 prompt 键，input 应直接用它。"""
        rec = _make_record(poscar="P")
        rec["prompt"] = "Custom prompt text"
        result = records_to_training_format([rec], INSTRUCTION)
        assert result[0]["input"] == "Custom prompt text"

    def test_empty_records_returns_empty_list(self):
        assert records_to_training_format([], INSTRUCTION) == []

    def test_input_contains_elements(self):
        """自动生成的 input 应包含元素信息。"""
        rec = _make_record(elements="IrPdPt", poscar="P")
        result = records_to_training_format([rec], INSTRUCTION)
        assert "IrPdPt" in result[0]["input"]

    def test_dft_dg_h_zero_not_replaced_by_predicted(self):
        """C-1 修复：dft_dg_h=0.0（HER 最优值，falsy）时 dg_h 必须是 0.0，不能被 predicted_dgH 替换。"""
        rec = _make_record(poscar="POSCAR_ZERO", predicted_dg=-0.5, dft_dg_h=0.0)
        result = records_to_training_format([rec], INSTRUCTION)
        assert len(result) == 1
        assert result[0]["dg_h"] == pytest.approx(0.0), (
            "dft_dg_h=0.0 should not be treated as falsy and replaced by predicted_dgH"
        )

    def test_dft_dg_h_none_falls_back_to_predicted(self):
        """dft_dg_h 键不存在时，回退到 predicted_dgH（确保 None 判断正确）。"""
        rec = _make_record(poscar="POSCAR_NDF", predicted_dg=-0.77)
        # 确保 rec 中没有 dft_dg_h 键
        rec.pop("dft_dg_h", None)
        result = records_to_training_format([rec], INSTRUCTION)
        assert result[0]["dg_h"] == pytest.approx(-0.77)


# ─────────────────────────────────────────────────────────────────
# 2. append_to_dataset
# ─────────────────────────────────────────────────────────────────

class TestAppendToDataset:
    def _sample_records(self, n=2):
        return [
            {"instruction": INSTRUCTION, "input": f"prompt_{i}",
             "output": f"POSCAR_{i}", "dg_h": -0.1 * i}
            for i in range(n)
        ]

    def test_creates_new_file(self, temp_workspace):
        """目标文件不存在时应自动创建。"""
        path = os.path.join(temp_workspace, "dataset.json")
        records = self._sample_records(2)
        result = append_to_dataset(records, path)
        assert os.path.exists(path)
        assert result["appended"] == 2
        assert result["total"] == 2
        assert result["skipped_duplicates"] == 0

    def test_appends_to_existing_file(self, temp_workspace):
        """追加时应把新记录加到原有数据后面。"""
        path = os.path.join(temp_workspace, "dataset.json")
        # 先写入2条
        initial = self._sample_records(2)
        append_to_dataset(initial, path)
        # 再追加3条（output不重复）
        more = [
            {"instruction": INSTRUCTION, "input": "p", "output": f"NEW_{i}", "dg_h": -0.9}
            for i in range(3)
        ]
        result = append_to_dataset(more, path)
        assert result["appended"] == 3
        assert result["total"] == 5
        assert result["skipped_duplicates"] == 0

    def test_deduplication_by_output(self, temp_workspace):
        """已存在相同 output(poscar) 的记录不应重复追加。"""
        path = os.path.join(temp_workspace, "dataset.json")
        records = self._sample_records(2)
        append_to_dataset(records, path)
        # 再次追加相同记录
        result = append_to_dataset(records, path)
        assert result["appended"] == 0
        assert result["skipped_duplicates"] == 2
        assert result["total"] == 2

    def test_partial_deduplication(self, temp_workspace):
        """部分重复时只追加不重复的部分。"""
        path = os.path.join(temp_workspace, "dataset.json")
        records = self._sample_records(2)
        append_to_dataset(records, path)
        mixed = records[:1] + [
            {"instruction": INSTRUCTION, "input": "new", "output": "BRAND_NEW", "dg_h": -0.7}
        ]
        result = append_to_dataset(mixed, path)
        assert result["appended"] == 1
        assert result["skipped_duplicates"] == 1
        assert result["total"] == 3

    def test_file_is_valid_json_array(self, temp_workspace):
        """写入后文件应为合法 JSON 数组。"""
        path = os.path.join(temp_workspace, "dataset.json")
        append_to_dataset(self._sample_records(2), path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 2

    def test_empty_new_records(self, temp_workspace):
        """传入空列表时文件不变（或新建空数组）。"""
        path = os.path.join(temp_workspace, "dataset.json")
        result = append_to_dataset([], path)
        assert result["appended"] == 0
        assert result["total"] == 0


# ─────────────────────────────────────────────────────────────────
# 3. rebuild_near_neighbor_index
# ─────────────────────────────────────────────────────────────────

class TestRebuildNearNeighborIndex:
    def _write_dataset(self, path, n=5):
        records = [
            {"instruction": INSTRUCTION, "input": f"p{i}",
             "output": f"POSCAR_{i}", "dg_h": float(i) * 0.1}
            for i in range(n)
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f)
        return records

    def test_creates_index_file(self, temp_workspace):
        """应在指定路径生成索引文件。"""
        dataset_path = os.path.join(temp_workspace, "dataset.json")
        index_path = os.path.join(temp_workspace, "index.json")
        self._write_dataset(dataset_path, n=5)
        result = rebuild_near_neighbor_index(dataset_path, index_path, top_k=3)
        assert os.path.exists(index_path)
        assert result["num_entries"] == 5

    def test_index_structure(self, temp_workspace):
        """索引文件应为 {str(idx): [近邻idx列表]} 结构。"""
        dataset_path = os.path.join(temp_workspace, "dataset.json")
        index_path = os.path.join(temp_workspace, "index.json")
        self._write_dataset(dataset_path, n=5)
        rebuild_near_neighbor_index(dataset_path, index_path, top_k=2)
        with open(index_path, encoding="utf-8") as f:
            idx = json.load(f)
        # 键应为字符串数字
        assert "0" in idx
        assert "4" in idx
        # 每条最多 top_k 个近邻
        for v in idx.values():
            assert len(v) <= 2
            assert isinstance(v, list)

    def test_neighbor_count_equals_top_k(self, temp_workspace):
        """数据条数 > top_k 时每条应有恰好 top_k 个近邻。"""
        dataset_path = os.path.join(temp_workspace, "dataset.json")
        index_path = os.path.join(temp_workspace, "index.json")
        self._write_dataset(dataset_path, n=6)
        rebuild_near_neighbor_index(dataset_path, index_path, top_k=3)
        with open(index_path, encoding="utf-8") as f:
            idx = json.load(f)
        for v in idx.values():
            assert len(v) == 3

    def test_neighbor_excludes_self(self, temp_workspace):
        """近邻列表中不应包含自身索引。"""
        dataset_path = os.path.join(temp_workspace, "dataset.json")
        index_path = os.path.join(temp_workspace, "index.json")
        self._write_dataset(dataset_path, n=5)
        rebuild_near_neighbor_index(dataset_path, index_path, top_k=3)
        with open(index_path, encoding="utf-8") as f:
            idx = json.load(f)
        for k, v in idx.items():
            assert int(k) not in v

    def test_small_dataset_top_k_capped(self, temp_workspace):
        """数据条数 <= top_k 时，近邻数应 < top_k（不含自身，最多 n-1 条）。"""
        dataset_path = os.path.join(temp_workspace, "dataset.json")
        index_path = os.path.join(temp_workspace, "index.json")
        self._write_dataset(dataset_path, n=2)
        rebuild_near_neighbor_index(dataset_path, index_path, top_k=5)
        with open(index_path, encoding="utf-8") as f:
            idx = json.load(f)
        for v in idx.values():
            assert len(v) == 1  # n=2, 除自身只剩1条

    def test_return_dict_contains_index_path(self, temp_workspace):
        """返回值应包含 index_path 键。"""
        dataset_path = os.path.join(temp_workspace, "dataset.json")
        index_path = os.path.join(temp_workspace, "index.json")
        self._write_dataset(dataset_path, n=3)
        result = rebuild_near_neighbor_index(dataset_path, index_path)
        assert result["index_path"] == index_path

    def test_none_dg_h_excluded_from_index(self, temp_workspace):
        """M-2 修复：dg_h=None 的记录不进入近邻索引（不被当作 0.0 污染检索结果）。"""
        dataset_path = os.path.join(temp_workspace, "dataset.json")
        index_path = os.path.join(temp_workspace, "index.json")
        records = [
            {"instruction": INSTRUCTION, "input": "p0", "output": "POSCAR_0", "dg_h": -0.5},
            {"instruction": INSTRUCTION, "input": "p1", "output": "POSCAR_1", "dg_h": None},
            {"instruction": INSTRUCTION, "input": "p2", "output": "POSCAR_2", "dg_h": -0.6},
        ]
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(records, f)
        rebuild_near_neighbor_index(dataset_path, index_path, top_k=3)
        with open(index_path, encoding="utf-8") as f:
            idx = json.load(f)
        # 索引中应无 dg_h=None 的记录（索引键 "1" 不存在）
        assert "1" not in idx, "Record with dg_h=None must not appear in index"
        # 有效记录 "0" 和 "2" 应在索引中，且互为近邻
        assert "0" in idx
        assert "2" in idx
        assert 2 in idx["0"]
        assert 0 in idx["2"]
        # 近邻中不能包含 None 记录（index 1）
        for neighbors in idx.values():
            assert 1 not in neighbors, "dg_h=None record (idx=1) must not appear as neighbor"


# ─────────────────────────────────────────────────────────────────
# 4. feedback_pipeline 端到端
# ─────────────────────────────────────────────────────────────────

class TestFeedbackPipeline:
    def _populate_store(self, db_path, n=3):
        """向 StateStore 插入 n 条 validated 状态的记录。"""
        store = StateStore(db_path=db_path)
        for i in range(n):
            uid = str(uuid.uuid4())
            store.save_record(uid, {
                "task_id": "task-pipe",
                "status": "validated",
                "elements": f"IrPd{i}",
                "poscar": f"POSCAR_PIPE_{i}",
                "predicted_dgH": -0.2 * (i + 1),
                "target_dgH": -0.3,
                "created_at": "2026-06-06T10:00:00",
            })
        return store

    def test_pipeline_writes_dataset(self, temp_db, temp_workspace):
        """pipeline 应把 validated 记录写入数据集文件。"""
        store = self._populate_store(temp_db, n=3)
        dataset_path = os.path.join(temp_workspace, "train.json")
        index_path = os.path.join(temp_workspace, "index.json")
        result = feedback_pipeline(store, dataset_path, index_path, INSTRUCTION)
        assert os.path.exists(dataset_path)
        with open(dataset_path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 3

    def test_pipeline_writes_index(self, temp_db, temp_workspace):
        """pipeline 应生成近邻索引文件。"""
        store = self._populate_store(temp_db, n=3)
        dataset_path = os.path.join(temp_workspace, "train.json")
        index_path = os.path.join(temp_workspace, "index.json")
        feedback_pipeline(store, dataset_path, index_path, INSTRUCTION)
        assert os.path.exists(index_path)

    def test_pipeline_returns_summary(self, temp_db, temp_workspace):
        """pipeline 返回值应包含 appended/total/num_entries 等汇总信息。"""
        store = self._populate_store(temp_db, n=2)
        dataset_path = os.path.join(temp_workspace, "train.json")
        index_path = os.path.join(temp_workspace, "index.json")
        result = feedback_pipeline(store, dataset_path, index_path, INSTRUCTION)
        assert "appended" in result
        assert "total" in result
        assert "num_entries" in result
        assert result["appended"] == 2

    def test_pipeline_empty_store(self, temp_db, temp_workspace):
        """空 store 时 pipeline 应正常运行，数据集为空数组。"""
        store = StateStore(db_path=temp_db)
        dataset_path = os.path.join(temp_workspace, "train.json")
        index_path = os.path.join(temp_workspace, "index.json")
        result = feedback_pipeline(store, dataset_path, index_path, INSTRUCTION)
        assert result["appended"] == 0
        assert result["total"] == 0

    def test_pipeline_idempotent(self, temp_db, temp_workspace):
        """重复运行 pipeline 不应导致重复追加（去重保证幂等）。"""
        store = self._populate_store(temp_db, n=2)
        dataset_path = os.path.join(temp_workspace, "train.json")
        index_path = os.path.join(temp_workspace, "index.json")
        feedback_pipeline(store, dataset_path, index_path, INSTRUCTION)
        result2 = feedback_pipeline(store, dataset_path, index_path, INSTRUCTION)
        assert result2["appended"] == 0
        assert result2["skipped_duplicates"] == 2
        assert result2["total"] == 2
