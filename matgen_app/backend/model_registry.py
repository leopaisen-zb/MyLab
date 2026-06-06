# backend/model_registry.py
"""
P1-3: 持久化模型版本注册表

将模型版本信息持久化到 JSON 文件，支持跨会话读取。
与 config.py 中的静态注册表互补：
  - config.py 的 GEN_MODELS/PRED_MODELS：静态 metadata（id/name/description/type）
  - ModelVersionRegistry：动态版本历史（checkpoint/metrics/notes）

用法示例：
    registry = ModelVersionRegistry(path="workspace/model_versions.json")
    v = registry.register_version("equiformer_v2", "/path/to/ckpt.pt", notes="demo retrain")
    latest = registry.get_latest_version("equiformer_v2")
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ModelVersionRegistry:
    """
    JSON 文件支持的模型版本注册表。

    文件格式（顶层为 model_id -> [版本列表]）：
    {
      "equiformer_v2": [
        {"version": "v1", "checkpoint_path": "...", "created_at": "...", "notes": "...", "metrics": {...}},
        ...
      ]
    }
    """

    def __init__(self, path: str = "workspace/model_versions.json"):
        self._path = Path(path)

    # ─────────────────────────────────────────────────────────────
    # 内部：读写 JSON
    # ─────────────────────────────────────────────────────────────

    def _load(self) -> Dict[str, List[Dict[str, Any]]]:
        """从文件加载注册表；文件不存在或为空时返回空 dict。"""
        if not self._path.exists() or self._path.stat().st_size == 0:
            return {}
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self, data: Dict[str, List[Dict[str, Any]]]) -> None:
        """将注册表写回文件（ensure_ascii=False, indent=2）。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ─────────────────────────────────────────────────────────────
    # 公开接口
    # ─────────────────────────────────────────────────────────────

    def register_version(
        self,
        model_id: str,
        checkpoint_path: str,
        notes: str = "",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        为 model_id 追加一个新版本，版本号自增（v1, v2, ...）。

        返回新注册的版本 dict：
            {version, checkpoint_path, created_at, notes, metrics}
        """
        data = self._load()
        existing_versions: List[Dict[str, Any]] = data.get(model_id, [])

        # 版本号 = 当前条数 + 1
        next_num = len(existing_versions) + 1
        version_tag = f"v{next_num}"

        new_entry: Dict[str, Any] = {
            "version": version_tag,
            "checkpoint_path": checkpoint_path,
            "created_at": datetime.now().isoformat(),
            "notes": notes,
            "metrics": metrics if metrics is not None else {},
        }

        existing_versions.append(new_entry)
        data[model_id] = existing_versions
        self._save(data)
        return new_entry

    def get_versions(self, model_id: str) -> List[Dict[str, Any]]:
        """返回该 model_id 的所有版本列表（按版本号升序）。"""
        data = self._load()
        return list(data.get(model_id, []))

    def get_latest_version(self, model_id: str) -> Optional[Dict[str, Any]]:
        """返回该 model_id 的最新版本；无版本时返回 None。"""
        versions = self.get_versions(model_id)
        return versions[-1] if versions else None

    def list_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """返回全部注册表内容：{model_id: [versions...]}。"""
        return self._load()
