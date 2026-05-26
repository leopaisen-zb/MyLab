import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

LORA_PATH = Path("/home/ubuntu/HZH/RAG/RAG/checkpoints/qwen_vasp_code_lora/checkpoint-6225")
SAVE_DIR = Path("/home/ubuntu/HZH/RAG/RAG/checkpoints/qwen_merged_bfloat16")
HF_HUB_DIR = Path.home() / ".cache" / "huggingface" / "hub"


def _is_valid_snapshot(snapshot_dir: Path) -> bool:
    if not (snapshot_dir / "config.json").exists():
        return False
    has_sharded = any(snapshot_dir.glob("model-00001-of-*.safetensors"))
    has_single = (snapshot_dir / "model.safetensors").exists()
    has_index = (snapshot_dir / "model.safetensors.index.json").exists()
    return has_sharded or has_single or has_index


def _find_local_snapshot(repo_id: str) -> Path | None:
    cache_root = HF_HUB_DIR / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not cache_root.exists():
        return None

    snapshots = [p for p in cache_root.iterdir() if p.is_dir()]
    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for snapshot in snapshots:
        if _is_valid_snapshot(snapshot):
            return snapshot
    return None


if not LORA_PATH.exists():
    raise FileNotFoundError(f"LoRA 路径不存在: {LORA_PATH}")

adapter_config_path = LORA_PATH / "adapter_config.json"
if not adapter_config_path.exists():
    raise FileNotFoundError(f"缺少 adapter_config.json: {adapter_config_path}")

with open(adapter_config_path, "r", encoding="utf-8") as f:
    adapter_cfg = json.load(f)

base_model_name = adapter_cfg.get("base_model_name_or_path")
if not base_model_name:
    raise RuntimeError("adapter_config.json 中缺少 base_model_name_or_path")

local_snapshot = _find_local_snapshot(base_model_name)
base_model_source = str(local_snapshot) if local_snapshot else base_model_name
local_only = local_snapshot is not None

if local_only:
    print(f"1. 使用本地缓存底座模型: {base_model_source}")
else:
    print(f"1. 未找到本地缓存，将从远端加载: {base_model_name}")

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_source,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=local_only,
)

print("2. 正在加载 LoRA/DoRA 适配器 ...")
peft_model = PeftModel.from_pretrained(base_model, str(LORA_PATH), local_files_only=True)

print("3. 正在合并 LoRA 到底座 (merge_and_unload) ...")
merged_model = peft_model.merge_and_unload()

print("4. 保存合并后的完整模型 ...")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
merged_model.save_pretrained(str(SAVE_DIR), safe_serialization=True)

tokenizer_source = str(LORA_PATH) if (LORA_PATH / "tokenizer_config.json").exists() else base_model_source
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_source,
    trust_remote_code=True,
    local_files_only=True if tokenizer_source == str(LORA_PATH) or local_only else False,
)
tokenizer.save_pretrained(str(SAVE_DIR))

print(f"\n✅ 合并完成！模型已保存至: {SAVE_DIR}")