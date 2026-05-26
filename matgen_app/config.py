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

SYSTEM_INSTRUCTION = (
    "You are a materials science expert. "
    "Given the physical and chemical properties of a hydrogen storage material "
    "and several reference structures with similar properties, "
    "generate the corresponding atomic structure in VASP POSCAR format."
)
