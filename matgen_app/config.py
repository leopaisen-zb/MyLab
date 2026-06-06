"""
MatGen-Eq system configuration.
All paths are relative to PROJECT_ROOT for cross-platform compatibility.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # mylab(1)/

# ── Eqv2-Lite ──────────────────────────────────────────────────────────────
EQ_SRC       = PROJECT_ROOT / "equiformer" / "src"
EQ_CKPT      = PROJECT_ROOT / "equiformer" / "checkpionts" / "best_standalone_equiformer_v2_model.pt"
NORM_STATS   = PROJECT_ROOT / "RAG" / "RAG" / "rag_datasets" / "custom_hydrogen" / "normalization_stats.json"

EQ_MAX_RADIUS    = 12.0
EQ_MAX_NEIGHBORS = 20

# ── HEA-Gen (qwen_vasp_lora direct POSCAR generation) ──────────────────────
LORA_CKPT = PROJECT_ROOT / "RAG" / "RAG" / "checkpoints" / "qwen_vasp_lora" / "checkpoint-3114"

# Base model: use local server path if available, else HuggingFace id
_SERVER_BASE = Path("/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
BASE_MODEL_PATH: str = str(_SERVER_BASE) if _SERVER_BASE.exists() else "Qwen/Qwen2.5-7B-Instruct"

# ── Database ────────────────────────────────────────────────────────────────
DB_PATH = PROJECT_ROOT / "matgen_app" / "results.db"

# ── RAG data paths ─────────────────────────────────────────────────────────
RAG_DATA_DIR = PROJECT_ROOT / "RAG" / "RAG" / "rag_data"
RAG_TRAIN_DATA_PATH = RAG_DATA_DIR / "text_rag" / "dataset_train_rag.json"
RAG_NEAR_NEIGHBOR_PATH = RAG_DATA_DIR / "text2struct_rag" / "near_neighbor_indices.json"

# ── Generation defaults ─────────────────────────────────────────────────────
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE    = 0.0
DEFAULT_TOP_P          = 1.0

# ── Closed-loop batch defaults ──────────────────────────────────────────────
DEFAULT_FILTER_LOW  = -1.00
DEFAULT_FILTER_HIGH =  0.50

# ── 模型注册表 ──────────────────────────────────────────────────────────────
# 生成模型（结构生成 / POSCAR 生成）
GEN_MODELS = [
    {
        "id": "qwen_lora",
        "name": "Qwen2.5-7B + VASP LoRA",
        "description": "基于 Qwen2.5-7B 微调的 LoRA 模型，通过 RAG 检索增强生成 VASP POSCAR 结构",
        "type": "llm_lora",
    },
    {
        "id": "fake",
        "name": "Fake POSCAR (demo)",
        "description": "随机生成合法 POSCAR 格式的占位结构，无需 GPU，用于演示和测试",
        "type": "demo",
    },
]

# 预测模型（ΔG_H 能量预测）
PRED_MODELS = [
    {
        "id": "equiformer_v2",
        "name": "Eqv2-Lite Equiformer",
        "description": "等变图神经网络 Equiformer V2 轻量版，预测 ΔG_H 吸附能（eV）",
        "type": "gnn",
    },
    {
        "id": "toy_mlp",
        "name": "Toy MLP (demo)",
        "description": "基于组成特征的轻量 MLP 演示模型，无需大型 checkpoint，用于测试",
        "type": "demo",
    },
]

# 默认选用的模型 id（不传时的行为）
DEFAULT_GEN_MODEL_ID  = "qwen_lora"
DEFAULT_PRED_MODEL_ID = "equiformer_v2"

SYSTEM_INSTRUCTION = (
    "You are a materials science expert. "
    "Given the physical and chemical properties of a hydrogen storage material "
    "and several reference structures with similar properties, "
    "generate the corresponding atomic structure in VASP POSCAR format."
)
