# persistence/workspace.py
import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import uuid

class Workspace:
    def __init__(self, root: str = "./workspace"):
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.models_dir = self.root / "models"
        self.data_dir = self.root / "data"
        self._ensure_dirs()

    def _ensure_dirs(self):
        for d in [self.root, self.runs_dir, self.models_dir, self.data_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def create_task_workspace(self, task_id: str) -> Path:
        task_dir = self.runs_dir / f"run_{task_id}"
        (task_dir / "structures").mkdir(parents=True, exist_ok=True)
        (task_dir / "logs").mkdir(parents=True, exist_ok=True)
        (task_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (task_dir / "exports").mkdir(parents=True, exist_ok=True)
        return task_dir

    def get_task_workspace(self, task_id: str) -> Optional[Path]:
        task_dir = self.runs_dir / f"run_{task_id}"
        return task_dir if task_dir.exists() else None

    def save_structure(self, task_id: str, structure_id: str, poscar: str, metadata: dict) -> Path:
        task_dir = self.get_task_workspace(task_id)
        if task_dir is None:
            task_dir = self.create_task_workspace(task_id)
        struct_dir = task_dir / "structures" / structure_id
        struct_dir.mkdir(parents=True, exist_ok=True)
        poscar_file = struct_dir / "structure.poscar"
        meta_file = struct_dir / "metadata.json"
        with open(poscar_file, "w", encoding="utf-8") as f:
            f.write(poscar)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return struct_dir

    def list_runs(self) -> List[dict]:
        runs = []
        for run_dir in sorted(self.runs_dir.glob("run_*"), key=lambda x: x.stat().st_mtime, reverse=True):
            meta_file = run_dir / "meta.json"
            runs.append({
                "task_id": run_dir.name.replace("run_", ""),
                "path": str(run_dir),
                "created": datetime.fromtimestamp(run_dir.stat().st_ctime).isoformat()
            })
        return runs

    def cleanup_old_runs(self, keep_last: int = 10):
        runs = sorted(self.runs_dir.glob("run_*"), key=lambda x: x.stat().st_mtime, reverse=True)
        for run_dir in runs[keep_last:]:
            shutil.rmtree(run_dir, ignore_errors=True)