# api/routes.py
from typing import List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from api.schemas import (
    GenerationRequest, GenerationResponse, TaskStatusResponse, StructureStatus,
    DFTResultRequest, DFTBatchItem, DFTBatchResponse,
)
from core.task_orchestrator import TaskOrchestrator
import uuid
import config as _cfg

router = APIRouter()
orchestrator = TaskOrchestrator()

@router.post("/generate", response_model=GenerationResponse)
async def create_generation_task(req: GenerationRequest):
    task_id = str(uuid.uuid4())
    try:
        orchestrator.create_task(task_id, req.model_dump())
        return GenerationResponse(
            task_id=task_id,
            status="created",
            message=f"Task created successfully. Submit to /tasks/{task_id}/execute to start."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tasks/{task_id}/execute")
async def execute_task(task_id: str, background_tasks: BackgroundTasks):
    if not orchestrator.task_exists(task_id):
        raise HTTPException(status_code=404, detail="Task not found")
    background_tasks.add_task(orchestrator.execute_task, task_id)
    return {"task_id": task_id, "status": "executing", "message": "Task started in background"}

@router.get("/tasks/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    status = orchestrator.get_task_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return status

@router.get("/structures/{structure_id}")
async def get_structure(structure_id: str):
    record = orchestrator.get_structure_record(structure_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Structure not found")
    return record

@router.post("/structures/{structure_id}/validate")
async def validate_structure(structure_id: str, decision: str):
    if decision not in ["validated", "rejected"]:
        raise HTTPException(status_code=400, detail="Decision must be 'validated' or 'rejected'")
    record = orchestrator.get_structure_record(structure_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Structure not found")
    result = orchestrator.update_structure_decision(structure_id, decision)
    if not result:
        raise HTTPException(
            status_code=409,
            detail="Invalid state transition: structure must be in 'dft_verified' to validate "
                   "(or 'filtered_in'/'dft_verified' to reject)"
        )
    return {"structure_id": structure_id, "decision": decision}

@router.post("/structures/dft/batch", response_model=DFTBatchResponse)
async def batch_submit_dft(items: List[DFTBatchItem]):
    """P1-1：批量 DFT 结果回填。接收 JSON 数组，返回成功数与失败 UUID 列表。"""
    success_count = 0
    failed: List[str] = []
    for item in items:
        ok = orchestrator.submit_dft_result(
            item.uuid, dft_dg_h=item.dft_dg_h, dft_energy=item.dft_energy
        )
        if ok:
            success_count += 1
        else:
            failed.append(item.uuid)
    return DFTBatchResponse(success_count=success_count, failed=failed)


@router.post("/structures/{structure_id}/dft")
async def submit_dft_result(structure_id: str, req: DFTResultRequest):
    """P1-1：单条 DFT 结果回填。将状态从 filtered_in 推进到 dft_verified。"""
    record = orchestrator.get_structure_record(structure_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Structure not found")
    success = orchestrator.submit_dft_result(
        structure_id, dft_dg_h=req.dft_dg_h, dft_energy=req.dft_energy
    )
    if not success:
        raise HTTPException(
            status_code=409,
            detail="State transition failed: structure must be in filtered_in state"
        )
    return {"structure_id": structure_id, "status": "dft_verified"}


@router.get("/models")
async def list_models():
    """返回可用的生成模型与预测模型清单（从 config 注册表读取）。"""
    return {
        "gen_models": _cfg.GEN_MODELS,
        "pred_models": _cfg.PRED_MODELS,
        "default_gen_model_id": _cfg.DEFAULT_GEN_MODEL_ID,
        "default_pred_model_id": _cfg.DEFAULT_PRED_MODEL_ID,
    }

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "MatGen-Eq API"}
