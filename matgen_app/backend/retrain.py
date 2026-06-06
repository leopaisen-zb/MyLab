# backend/retrain.py
"""
P1-3: 一键重训编排（飞轮闭环第⑧步）

把"导出回灌 → 训练 → 注册新版本"串成一个入口函数 run_retrain。

支持两种模式：
  - demo 模式（demo=True 或环境变量 MATGEN_DEMO=1）：
      跳过真实训练，直接注册一个 mock 版本，适合答辩演示。
  - 非 demo 模式：
      调用真实训练脚本（需 GPU 环境），成功后注册新版本。

CLI 入口（argparse）：
    python -m backend.retrain --model-kind prediction --demo
    python -m backend.retrain --model-kind generation
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import config as _cfg
from backend.dataset_feedback import feedback_pipeline
from backend.model_registry import ModelVersionRegistry
from persistence.state_store import StateStore


# ─────────────────────────────────────────────────────────────────
# 内部：判断是否处于 demo 模式
# ─────────────────────────────────────────────────────────────────

def _is_demo(demo_flag: bool) -> bool:
    """demo=True 或环境变量 MATGEN_DEMO=1 时进入 demo 模式。"""
    return demo_flag or os.environ.get("MATGEN_DEMO", "0").strip() == "1"


# ─────────────────────────────────────────────────────────────────
# 内部：构造训练命令
# ─────────────────────────────────────────────────────────────────

def _build_train_cmd_prediction(dataset_path: str) -> list:
    """构造 Equiformer 预测模型训练命令。

    注意：--data-path 参数名需与 train_equiformer.py 的 argparse 保持一致。
    """
    train_script = _cfg.EQ_SRC / "train_equiformer.py"
    return [
        sys.executable,
        str(train_script),
        "--checkpoint", str(_cfg.EQ_CKPT),
        "--data-path", dataset_path,
    ]


def _build_train_cmd_generation(dataset_path: str) -> list:
    """构造 Qwen LoRA 生成模型训练命令。

    注意：--data-path 参数名需与 train_lora.py 的 argparse 保持一致。
    """
    lora_script = _cfg.PROJECT_ROOT / "RAG" / "RAG" / "train_lora.py"
    return [
        sys.executable,
        str(lora_script),
        "--base-model", str(_cfg.BASE_MODEL_PATH),
        "--lora-checkpoint", str(_cfg.LORA_CKPT),
        "--data-path", dataset_path,
    ]


# ─────────────────────────────────────────────────────────────────
# 公开接口
# ─────────────────────────────────────────────────────────────────

def run_retrain(
    model_kind: str,
    store: StateStore,
    registry: ModelVersionRegistry,
    demo: bool = False,
    dataset_path: Optional[str] = None,
    index_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    飞轮⑧：导出回灌 → 训练 → 注册新版本。

    参数：
        model_kind    : "prediction" 或 "generation"
        store         : StateStore 实例（取 validated 记录）
        registry      : ModelVersionRegistry 实例（写版本）
        demo          : True 时跳过真实训练，直接注册 mock 版本
        dataset_path  : 训练数据集路径（None 时用 config 默认路径）
        index_path    : 近邻索引路径（None 时用 config 默认路径）

    返回：
        {
          "model_kind": str,
          "demo": bool,
          "feedback": dict,       # feedback_pipeline 返回值
          "new_version": dict,    # 注册的新版本 dict
        }
    """
    # 参数校验
    if model_kind not in ("prediction", "generation"):
        raise ValueError(
            f"model_kind 必须是 'prediction' 或 'generation'，got: '{model_kind}'"
        )

    # 路径默认值
    dataset_path = dataset_path or str(_cfg.RAG_TRAIN_DATA_PATH)
    index_path = index_path or str(_cfg.RAG_NEAR_NEIGHBOR_PATH)

    # ① 数据集回灌（无论 demo 与否都跑，保证数据同步）
    fb_result = feedback_pipeline(
        store=store,
        dataset_path=dataset_path,
        index_path=index_path,
        instruction=_cfg.SYSTEM_INSTRUCTION,
    )

    # ② 确定 model_id 和 checkpoint 路径
    if model_kind == "prediction":
        model_id = _cfg.DEFAULT_PRED_MODEL_ID
        base_ckpt = str(_cfg.EQ_CKPT)
    else:
        model_id = _cfg.DEFAULT_GEN_MODEL_ID
        base_ckpt = str(_cfg.LORA_CKPT)

    # ③ 训练 + 版本注册
    use_demo = _is_demo(demo)

    if use_demo:
        # Demo 模式：跳过真实训练，注册 mock 版本
        notes = "demo retrain（跳过真实训练，仅用于演示）"
        metrics = {
            "validated_count": fb_result["validated_count"],
            "appended": fb_result["appended"],
            "total_dataset": fb_result["total"],
        }
        # mock checkpoint 路径（带版本后缀，便于识别）
        existing = registry.get_versions(model_id)
        next_v = len(existing) + 1
        mock_ckpt = f"{base_ckpt}.demo_v{next_v}"
        new_version = registry.register_version(
            model_id=model_id,
            checkpoint_path=mock_ckpt,
            notes=notes,
            metrics=metrics,
        )
    else:
        # 真实训练模式
        if model_kind == "prediction":
            cmd = _build_train_cmd_prediction(dataset_path)
        else:
            cmd = _build_train_cmd_generation(dataset_path)

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(_cfg.PROJECT_ROOT),
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"训练脚本失败（returncode={proc.returncode}）:\n{proc.stderr}"
            )

        # 真实训练产出的 checkpoint 路径（此处用 base_ckpt 代表更新后的位置）
        notes = f"retrain after feedback: appended={fb_result['appended']}"
        metrics = {
            "validated_count": fb_result["validated_count"],
            "appended": fb_result["appended"],
            "total_dataset": fb_result["total"],
        }
        new_version = registry.register_version(
            model_id=model_id,
            checkpoint_path=base_ckpt,
            notes=notes,
            metrics=metrics,
        )

    return {
        "model_kind": model_kind,
        "demo": use_demo,
        "feedback": fb_result,
        "new_version": new_version,
    }


# ─────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="MatGen-Eq 一键重训（飞轮⑧）\n"
                    "Demo 用法: python -m backend.retrain --model-kind prediction --demo\n"
                    "真实训练（需 GPU）: python -m backend.retrain --model-kind prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-kind",
        choices=["prediction", "generation"],
        default="prediction",
        help="要重训的模型类型（prediction=Equiformer, generation=Qwen LoRA）",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Demo 模式：跳过真实训练，直接注册 mock 版本（答辩用）",
    )
    parser.add_argument(
        "--db-path",
        default=str(_cfg.DB_PATH),
        help="StateStore 数据库路径",
    )
    parser.add_argument(
        "--registry-path",
        default="workspace/model_versions.json",
        help="版本注册表 JSON 路径",
    )
    args = parser.parse_args()

    store = StateStore(db_path=args.db_path)
    registry = ModelVersionRegistry(path=args.registry_path)

    print(f"[retrain] model_kind={args.model_kind}, demo={args.demo}")
    result = run_retrain(
        model_kind=args.model_kind,
        store=store,
        registry=registry,
        demo=args.demo,
    )

    import json as _json
    print("[retrain] 完成：")
    print(_json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    _cli()
