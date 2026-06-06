# api/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class StructureStatus(str, Enum):
    GENERATED = "generated"
    REJECTED_PRECHECK = "rejected_precheck"
    PREDICTED = "predicted"
    FILTERED_IN = "filtered_in"
    FILTERED_OUT = "filtered_out"
    VALIDATED = "validated"
    REJECTED = "rejected"

class GenerationRequest(BaseModel):
    target_dgH: float = Field(..., description="Target ΔG_H value in eV")
    tolerance: float = Field(default=0.05, description="Tolerance range in eV")
    elements: List[str] = Field(default=["Ir", "Pd", "Pt", "Rh", "Ru"], description="Element types")
    batch_size: int = Field(default=10, ge=1, le=500, description="Number of structures to generate")
    element_percentages: Optional[dict] = Field(default=None, description="Element percentage constraints")
    gen_model_id: Optional[str] = Field(default=None, description="生成模型 id，不传则使用默认（见 config.GEN_MODELS）")
    pred_model_id: Optional[str] = Field(default=None, description="预测模型 id，不传则使用默认（见 config.PRED_MODELS）")

class GenerationResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    total: int
    processed: int
    success: int
    failed: int
    results: List[dict]

class StructureRecord(BaseModel):
    uuid: str
    status: StructureStatus
    elements: str
    predicted_dgH: Optional[float] = None
    target_dgH: Optional[float] = None
    error: Optional[str] = None
    created_at: Optional[str] = None