# api/routes.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from api.schemas import GenerationRequest, GenerationResponse, TaskStatusResponse, StructureStatus
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
    result = orchestrator.update_structure_decision(structure_id, decision)
    if not result:
        raise HTTPException(status_code=404, detail="Structure not found")
    return {"structure_id": structure_id, "decision": decision}

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
