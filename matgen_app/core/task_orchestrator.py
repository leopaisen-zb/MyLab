# core/task_orchestrator.py
import os
import uuid
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
from core.state_machine import StructureState, StateTransition
from core.checkpoint import CheckpointManager
from adapters.hea_gen_adapter import HEAGenAdapter, generate_fake_poscar, composition_from_poscar
from adapters.eq_adapter import EQAdapter
from persistence.workspace import Workspace
from persistence.state_store import StateStore
from backend.model_registry import ModelVersionRegistry
import config as _cfg

class TaskOrchestrator:
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.checkpointer = CheckpointManager()
        self.workspace = Workspace()
        self.state_store = StateStore()
        self.hea_gen_adapter = HEAGenAdapter()
        self.eq_adapter = EQAdapter()
        self._lock = threading.Lock()
        # P1-3: 版本注册表（绝对路径，避免 cwd 敏感）
        self.model_registry = ModelVersionRegistry(path=str(_cfg.REGISTRY_PATH))

    def task_exists(self, task_id: str) -> bool:
        return task_id in self.tasks

    def create_task(self, task_id: str, config: Dict[str, Any]):
        with self._lock:
            self.tasks[task_id] = {
                "task_id": task_id,
                "config": config,
                "status": "created",
                "created_at": datetime.now().isoformat(),
                "structures": {},
                "stats": {"total": 0, "processed": 0, "success": 0, "failed": 0}
            }
            self.checkpointer.save_checkpoint(task_id, self.tasks[task_id])

    # ── 模型 id 校验辅助 ─────────────────────────────────────────────────────

    @staticmethod
    def _resolve_gen_model_id(gen_model_id: Optional[str]) -> str:
        """校验 gen_model_id，未知时回退到默认。"""
        valid_ids = {e["id"] for e in _cfg.GEN_MODELS}
        if gen_model_id and gen_model_id in valid_ids:
            return gen_model_id
        if gen_model_id and gen_model_id not in valid_ids:
            print(f"[TaskOrchestrator] 未知 gen_model_id='{gen_model_id}'，回退到默认 '{_cfg.DEFAULT_GEN_MODEL_ID}'")
        return _cfg.DEFAULT_GEN_MODEL_ID

    @staticmethod
    def _resolve_pred_model_id(pred_model_id: Optional[str]) -> str:
        """校验 pred_model_id，未知时回退到默认。"""
        valid_ids = {e["id"] for e in _cfg.PRED_MODELS}
        if pred_model_id and pred_model_id in valid_ids:
            return pred_model_id
        if pred_model_id and pred_model_id not in valid_ids:
            print(f"[TaskOrchestrator] 未知 pred_model_id='{pred_model_id}'，回退到默认 '{_cfg.DEFAULT_PRED_MODEL_ID}'")
        return _cfg.DEFAULT_PRED_MODEL_ID

    def _generate_batch_with_model(
        self,
        gen_model_id: str,
        target_dgH: float,
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        """根据 MATGEN_DEMO 环境变量选择生成方式，产出与 HEAGenAdapter.generate_batch 相同格式。

        逆向设计：仅以 target_dgH 为条件，元素种类与配比由生成器自主产出。
        gen_model_id 参数保留，供未来真实多模型切换使用。
        """
        if os.environ.get("MATGEN_DEMO") == "1":
            # 轻量模式：本地快速合成，不加载大模型/不依赖语料（测试/无 GPU）
            results = []
            for _ in range(batch_size):
                poscar = generate_fake_poscar(num_atoms=20, target_dgH=target_dgH)
                results.append({
                    "elements": composition_from_poscar(poscar),
                    "poscar": poscar,
                    "target_dgH": target_dgH,
                })
            return results
        if os.environ.get("MATGEN_USE_LLM") == "1":
            # GPU 服务器：真实 Qwen-7B + LoRA 生成
            return self.hea_gen_adapter.generate_batch(
                target_dgH=target_dgH,
                batch_size=batch_size,
            )
        # 默认（本机真实路径）：从真实结构库按目标性质检索（端到端真实模型，Mac 可跑）
        from backend.structure_library import sample_structures
        return sample_structures(target_dgH=target_dgH, batch_size=batch_size)

    def _predict_with_model(self, pred_model_id: str, poscar_text: str, parsed: dict = None) -> float:
        """根据 MATGEN_DEMO 环境变量选择预测方式。

        pred_model_id 参数保留，供未来真实多模型切换使用。
        """
        if os.environ.get("MATGEN_DEMO") == "1":
            # DEMO 模式：走 toy MLP（EQAdapter.forward），不加载大型 checkpoint
            features = {**parsed, "parsed": True} if parsed else {"composition": "", "num_sites": 20, "parsed": True}
            return self.eq_adapter.forward(features)
        # 真实模式：走 EQAdapter.predict（equiformer_v2 或其他真实模型，内部有 fallback）
        return self.eq_adapter.predict(poscar_text)

    def execute_task(self, task_id: str):
        task = self.tasks.get(task_id)
        if not task:
            return

        task["status"] = "running"
        config = task["config"]
        batch_size = config.get("batch_size", 10)

        # 解析模型选择（未知 id 自动回退）
        gen_model_id  = self._resolve_gen_model_id(config.get("gen_model_id"))
        pred_model_id = self._resolve_pred_model_id(config.get("pred_model_id"))

        # P1-3: 取当前预测模型版本（无版本时记为 "base"）
        _latest_ver = self.model_registry.get_latest_version(pred_model_id)
        pred_model_version = _latest_ver["version"] if _latest_ver else "base"

        if config.get("target_dgH") is None:
            task["status"] = "failed"
            task["error"] = "target_dgH is required"
            return

        try:
            structures = self._generate_batch_with_model(
                gen_model_id=gen_model_id,
                target_dgH=config.get("target_dgH"),
                batch_size=batch_size,
            )

            task["stats"]["total"] = len(structures)

            for struct_data in structures:
                struct_id = str(uuid.uuid4())
                struct_record = {
                    "uuid": struct_id,
                    "status": StructureState.GENERATED.value,
                    "elements": struct_data.get("elements", ""),
                    "poscar": struct_data.get("poscar", ""),
                    "target_dgH": config.get("target_dgH"),
                    "created_at": datetime.now().isoformat(),
                    "task_id": task_id,
                    # P1-3: 在 metadata 中记录本次预测所用模型 id 和版本
                    "metadata": {
                        "pred_model_id": pred_model_id,
                        "pred_model_version": pred_model_version,
                    },
                }

                self.state_store.save_record(struct_id, struct_record)
                task["structures"][struct_id] = struct_record
                task["stats"]["processed"] += 1

                try:
                    parsed = self.hea_gen_adapter.parse_structure(struct_data["poscar"])
                    if parsed is None:
                        self._transition_state(struct_id, StructureState.REJECTED_PRECHECK, task)
                        task["stats"]["failed"] += 1
                        continue

                    struct_record["parsed_structure"] = parsed
                    predicted = self._predict_with_model(pred_model_id, struct_data["poscar"], parsed)
                    struct_record["predicted_dgH"] = predicted
                    self.state_store.save_record(struct_id, struct_record)
                    self._transition_state(struct_id, StructureState.PREDICTED, task)

                    tolerance = config.get("tolerance", 0.05)
                    target = config.get("target_dgH")
                    if abs(predicted - target) <= tolerance:
                        self._transition_state(struct_id, StructureState.FILTERED_IN, task)
                        task["stats"]["success"] += 1
                    else:
                        self._transition_state(struct_id, StructureState.FILTERED_OUT, task)

                except Exception as e:
                    struct_record["error"] = str(e)
                    task["stats"]["failed"] += 1

            task["status"] = "completed"
            self.checkpointer.save_checkpoint(task_id, task)

        except Exception as e:
            task["status"] = "failed"
            task["error"] = str(e)

    def _transition_state(self, struct_id: str, new_state: StructureState, task: Dict):
        record = self.state_store.get_record(struct_id)
        if record:
            current = StructureState(record["status"])
            if StateTransition.can_transition(current, new_state):
                record["status"] = new_state.value
                record["updated_at"] = datetime.now().isoformat()
                self.state_store.save_record(struct_id, record)
                self.state_store.record_state_change(
                    struct_id, current.value, new_state.value
                )
                task["structures"][struct_id] = record

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        task = self.tasks.get(task_id)
        if not task:
            checkpoint = self.checkpointer.get_latest_checkpoint(task_id)
            if checkpoint:
                task = checkpoint
        if not task:
            return None
        return {
            "task_id": task["task_id"],
            "status": task.get("status", "unknown"),
            "total": task.get("stats", {}).get("total", 0),
            "processed": task.get("stats", {}).get("processed", 0),
            "success": task.get("stats", {}).get("success", 0),
            "failed": task.get("stats", {}).get("failed", 0),
            "results": list(task.get("structures", {}).values())
        }

    def get_structure_record(self, structure_id: str) -> Optional[Dict[str, Any]]:
        return self.state_store.get_record(structure_id)

    def submit_dft_result(self, structure_id: str, dft_dg_h: float,
                          dft_energy: Optional[float] = None) -> bool:
        """将外部 DFT 计算结果原子回填到结构记录，并将状态从 filtered_in 推进到 dft_verified。
        找不到记录或当前状态非 filtered_in 时返回 False，且不产生历史记录。"""
        # backfill_dft 原子执行：仅在 status='filtered_in' 时写 DFT 字段 + 改状态，避免并发 TOCTOU
        success = self.state_store.backfill_dft(
            structure_id, dft_dg_h=dft_dg_h, dft_energy=dft_energy
        )
        if not success:
            return False
        # 写入成功后记录一条状态历史（不重复改状态，仅留审计轨迹）
        self.state_store.record_state_change(
            structure_id, "filtered_in", "dft_verified", reason="DFT backfill"
        )
        return True

    def update_structure_decision(self, structure_id: str, decision: str) -> bool:
        record = self.state_store.get_record(structure_id)
        if not record:
            return False
        current = StructureState(record["status"])
        target = StructureState.VALIDATED if decision == "validated" else StructureState.REJECTED
        if StateTransition.can_transition(current, target):
            record["status"] = target.value
            record["decision"] = decision
            record["updated_at"] = datetime.now().isoformat()
            self.state_store.save_record(structure_id, record)
            self.state_store.record_state_change(
                structure_id, current.value, target.value, reason="expert decision"
            )
            return True
        return False