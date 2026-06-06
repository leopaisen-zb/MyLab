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
    DFT_VERIFIED = "dft_verified"          # P1-1：DFT 回填中间态
    VALIDATED = "validated"
    REJECTED = "rejected"
    EXPORTED_FOR_TRAINING = "exported_for_training"  # 已导出进训练集

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

class DFTResultRequest(BaseModel):
    """单条 DFT 结果回填请求体。"""
    dft_dg_h: float = Field(..., description="DFT 计算的 ΔG_H 值（eV），必填")
    dft_energy: Optional[float] = Field(default=None, description="DFT 总能量（eV），可选")


class DFTBatchItem(BaseModel):
    """批量回填中的单条条目。"""
    uuid: str = Field(..., description="结构 UUID")
    dft_dg_h: float = Field(..., description="DFT ΔG_H（eV）")
    dft_energy: Optional[float] = Field(default=None, description="DFT 总能量（eV），可选")


class DFTBatchResponse(BaseModel):
    """批量回填响应。"""
    success_count: int
    failed: List[str]   # 回填失败的 UUID 列表


class StructureRecord(BaseModel):
    uuid: str
    status: StructureStatus
    elements: str
    predicted_dgH: Optional[float] = None
    target_dgH: Optional[float] = None
    error: Optional[str] = None
    created_at: Optional[str] = None