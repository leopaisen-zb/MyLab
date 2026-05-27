# core/task_orchestrator.py
import uuid
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
from core.state_machine import StructureState, StateTransition
from core.checkpoint import CheckpointManager
from adapters.hea_gen_adapter import HEAGenAdapter
from adapters.eq_adapter import EQAdapter
from persistence.workspace import Workspace
from persistence.state_store import StateStore

class TaskOrchestrator:
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.checkpointer = CheckpointManager()
        self.workspace = Workspace()
        self.state_store = StateStore()
        self.hea_gen_adapter = HEAGenAdapter()
        self.eq_adapter = EQAdapter()
        self._lock = threading.Lock()

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

    def execute_task(self, task_id: str):
        task = self.tasks.get(task_id)
        if not task:
            return

        task["status"] = "running"
        config = task["config"]
        batch_size = config.get("batch_size", 10)

        try:
            structures = self.hea_gen_adapter.generate_batch(
                target_dgH=config.get("target_dgH"),
                elements=config.get("elements", ["Ir", "Pd", "Pt", "Rh", "Ru"]),
                batch_size=batch_size
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
                    "task_id": task_id
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
                    predicted = self.eq_adapter.predict(parsed)
                    struct_record["predicted_dgH"] = predicted
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
            return True
        return False