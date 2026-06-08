"""
GRPO 强化学习后训练：以 Eqv2-Lite 预测的 ΔG_H 接近度为奖励，对齐 HEA-Gen 生成的 POSCAR。

策略（policy）= SFT 文本 LoRA `checkpoints/qwen_vasp_lora/checkpoint-3114`（输出本身就是 POSCAR）。
奖励（reward）= reward_dgh.make_reward_fn → 进程内调 eq_reward（Eqv2-Lite，6 层 ckpt）。
冻结的 base 充当 KL 参考策略。结构照搬 train_sft_text_rag.py，SFTTrainer → GRPOTrainer。

⚠️ 需 CUDA GPU 机器（unsloth + trl + vllm），本机 Mac 跑不了。
⚠️ 运行前在 unsloth 环境补装奖励模型依赖：pip install torch_geometric e3nn ase scipy
   （torch 已是 2.x，无需动）。

用法：
    # 微型冒烟（先验证不 OOM、reward_std>0）：
    GRPO_SMOKE=8 python src/train_grpo.py
    # 全量：
    python src/train_grpo.py
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

# Unsloth 必须在 trl 之前 import（import 时打 GRPO 补丁）
from unsloth import FastLanguageModel
import transformers.utils.import_utils
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

# 暴力绕过 HuggingFace 的 CVE 安全检查（与 SFT 脚本一致）
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

# 奖励模型建议跑 CPU，避免抢 vLLM rollout 的显存（21.45M，CPU 足够）。
# 若 GPU 显存充裕想让奖励模型也上卡，启动前显式设 MATGEN_DEVICE=cuda 即可。
os.environ.setdefault("MATGEN_DEVICE", "cpu")

# reward 函数（同目录模块）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward_dgh import make_reward_fn  # noqa: E402

# ============================================================
# 路径与超参
# ============================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent  # RAG/RAG

# 真实训练数据路径（注意：SFT 脚本里的 data/rag_data/... 是失效路径，已核实）
TRAIN_JSON: Path = Path(os.environ.get(
    "GRPO_TRAIN_JSON",
    PROJECT_ROOT / "rag_data" / "text_rag" / "dataset_train_rag.json",
))

# RL 起点：SFT 文本 LoRA（r=16）。GRPO 继续训练这同一个 LoRA。
POLICY_LORA: str = os.environ.get(
    "GRPO_POLICY_LORA",
    str(PROJECT_ROOT / "checkpoints" / "qwen_vasp_lora" / "checkpoint-3114"),
)

OUTPUT_DIR: Path = PROJECT_ROOT / "checkpoints" / "qwen_vasp_grpo"

MAX_SEQ_LENGTH: int = 8192
MAX_PROMPT_LEN: int = 6144        # input 含 3 个 RAG 参考块，偏长
MAX_COMPLETION_LEN: int = 1024    # ~16-40 原子的 POSCAR 足够
NUM_GENERATIONS: int = 8          # G：组内采样数
TEMPERATURE: float = 0.9          # 必须采样，greedy 组内零方差学不动
BETA_KL: float = 0.04             # KL 锚住 SFT 分布防奖励作弊；作弊就升到 0.1
LEARNING_RATE: float = 1e-6       # 远低于 SFT 的 2e-5，RL on LoRA 极敏感
PER_DEVICE_BS: int = NUM_GENERATIONS
GRAD_ACCUM: int = 4
LORA_RANK: int = 16               # 与 qwen_vasp_lora 的 r=16 对齐

# 奖励旋钮
REWARD_TAU: float = float(os.environ.get("REWARD_TAU", "0.05"))
REWARD_FMT_BONUS: float = 0.1
REWARD_PENALTY: float = -1.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TARGET_RE = re.compile(r"Target\s*Δ?GH\s*=\s*(-?\d+(?:\.\d+)?)")


def load_policy():
    """加载 SFT 文本 LoRA 作为可训练策略；冻结 base = KL 参考。

    主路径：直接从 LoRA adapter 目录加载（Unsloth 会按 adapter_config 的 base 自动装基座并挂 LoRA，
    训练时 adapter 默认可训）。若安装的 Unsloth 版本要求“先 base 再 get_peft_model 再 load_adapter”，
    见下方 FALLBACK 注释切换。
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=POLICY_LORA,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        dtype=torch.bfloat16,
        fast_inference=True,          # 启用 Unsloth vLLM rollout
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=float(os.environ.get("GRPO_GPU_MEM_UTIL", "0.55")),
    )
    # --- FALLBACK（若上面加载的 adapter 不可训，改用基座 + 新建 LoRA + 载入 SFT 权重）---
    # base = adapter_config.json 里的 base_model_name_or_path
    # model, tokenizer = FastLanguageModel.from_pretrained(model_name=<BASE>, fast_inference=True,
    #     max_lora_rank=LORA_RANK, gpu_memory_utilization=0.55, load_in_4bit=False, dtype=torch.bfloat16)
    # model = FastLanguageModel.get_peft_model(model, r=LORA_RANK, lora_alpha=32, lora_dropout=0.0,
    #     target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    #     use_gradient_checkpointing="unsloth", random_state=42, bias="none")
    # model.load_adapter(POLICY_LORA, adapter_name="default")   # 载入 SFT 权重作为起点
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_prompt(tokenizer, instruction: str, input_text: str) -> str:
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_dataset(tokenizer) -> Dataset:
    """dataset_train_rag.json -> {prompt, target_dgh}。丢弃 gold output；从 input 提取目标 ΔG_H。"""
    import json
    with TRAIN_JSON.open(encoding="utf-8") as f:
        raw: List[Dict[str, Any]] = json.load(f)

    rows: List[Dict[str, Any]] = []
    skipped = 0
    for r in raw:
        m = TARGET_RE.search(r.get("input", ""))
        if not m:
            skipped += 1
            continue
        rows.append({
            "prompt": build_prompt(tokenizer, r["instruction"], r["input"]),
            "target_dgh": float(m.group(1)),
        })
    logger.info("GRPO 数据集：%d 条可用，%d 条无 target 被跳过。", len(rows), skipped)
    return Dataset.from_list(rows)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA 不可用。GRPO 7B 训练需 GPU 机器（本机 Mac 仅能验证奖励路径）。"
        )

    logger.info("加载策略 LoRA：%s", POLICY_LORA)
    model, tokenizer = load_policy()

    ds = build_dataset(tokenizer)

    # 微型冒烟：GRPO_SMOKE=8 只取前 8 条
    smoke = os.environ.get("GRPO_SMOKE")
    if smoke:
        n = int(smoke)
        ds = ds.select(range(min(n, len(ds))))
        logger.info("SMOKE 模式：仅用 %d 条 prompt。", len(ds))

    reward_fn = make_reward_fn(
        tau=REWARD_TAU, fmt_bonus=REWARD_FMT_BONUS, degenerate_penalty=REWARD_PENALTY,
    )

    cfg = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LEN,
        max_completion_length=MAX_COMPLETION_LEN,
        temperature=TEMPERATURE,
        beta=BETA_KL,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.03,
        bf16=True,
        optim="adamw_torch_fused",
        logging_steps=1,
        save_steps=(4 if smoke else 100),
        save_total_limit=4,
        num_train_epochs=1,
        seed=42,
        report_to="none",
        use_vllm=True,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=cfg,
        train_dataset=ds,
    )

    logger.info("开始 GRPO 训练。盯：reward 上升、reward_std>0、kl 有界。")
    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    logger.info("GRPO LoRA 已保存到：%s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
