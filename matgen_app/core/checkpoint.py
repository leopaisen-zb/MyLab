# core/checkpoint.py
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class CheckpointManager:
    def __init__(self, workspace_root: str = "./workspace"):
        self.workspace_root = Path(workspace_root)
        self.checkpoint_dir = self.workspace_root / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, task_id: str, checkpoint_data: Dict[str, Any]) -> str:
        checkpoint_file = self.checkpoint_dir / f"{task_id}_checkpoint.json"
        checkpoint_record = {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "data": checkpoint_data
        }
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_record, f, ensure_ascii=False, indent=2)
        return str(checkpoint_file)

    def load_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        checkpoint_file = self.checkpoint_dir / f"{task_id}_checkpoint.json"
        if not checkpoint_file.exists():
            return None
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            record = json.load(f)
        return record.get("data")

    def get_latest_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self.load_checkpoint(task_id)

    def delete_checkpoint(self, task_id: str) -> bool:
        checkpoint_file = self.checkpoint_dir / f"{task_id}_checkpoint.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            return True
        return False

    def list_checkpoints(self):
        return list(self.checkpoint_dir.glob("*_checkpoint.json"))