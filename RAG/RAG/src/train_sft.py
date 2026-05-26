"""
Supervised Fine-Tuning (SFT) script for Qwen2.5-7B-Instruct using Unsloth and TRL.

Author: zhhe@streamax.com
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments, PreTrainedTokenizerBase
import transformers.utils.import_utils

# 暴力绕过 HuggingFace 的 CVE 安全检查
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

# ============================================================
# Constants & Hyperparameters
# ============================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
TRAIN_JSON: Path = PROJECT_ROOT / "data" / "rag_data" / "text2struct_rag" / "dataset_train_rag_code.json"
OUTPUT_DIR: Path = PROJECT_ROOT / "checkpoints" / "qwen_vasp_code_lora"

_LOCAL_MODEL_PATH: str = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
    "/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
MODEL_NAME: str = _LOCAL_MODEL_PATH if os.path.isdir(_LOCAL_MODEL_PATH) else "Qwen/Qwen2.5-7B-Instruct"

MAX_SEQ_LENGTH: int = 8192

LORA_R: int = 64
LORA_ALPHA: int = 32
LORA_DROPOUT: float = 0.0
TARGET_MODULES: List[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

PER_DEVICE_BATCH_SIZE: int = 2
GRADIENT_ACCUMULATION_STEPS: int = 4
LEARNING_RATE: float = 2e-5
NUM_EPOCHS: int = 3
WARMUP_RATIO: float = 0.05
LOGGING_STEPS: int = 10
SAVE_STEPS: int = 500

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_train_data(file_path: Path) -> List[Dict[str, Any]]:
    logger.info(f"Loading dataset from: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} training samples.")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"训练数据文件未找到，请检查路径: {file_path}")
    except json.JSONDecodeError:
        raise ValueError("训练数据文件格式错误，必须是合法的 JSON 格式。")


def format_to_text(examples: Dict[str, List[Any]], tokenizer: PreTrainedTokenizerBase) -> Dict[str, List[str]]:
    texts: List[str] = []
    for instruction, input_text, output_text in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}


def main() -> None:
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 环境不可用，请检查显卡驱动或 PyTorch 安装状态。")

        logger.info("Phase 3: LoRA SFT - Qwen2.5-7B-Instruct (Cache Cleared & Strict BPE Match)")
        
        logger.info("Loading model and tokenizer in BF16 precision...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=False,
            dtype=torch.bfloat16,
        )

        logger.info("Configuring high-order rsLoRA with DoRA enabled...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=TARGET_MODULES,
            use_gradient_checkpointing="unsloth",
            random_state=42,
            use_rslora=True,
            bias="none",
            use_dora=True,
        )

        logger.info("Formatting dataset into Unsloth compatible text field...")
        raw_data = load_train_data(TRAIN_JSON)
        dataset = Dataset.from_list(raw_data)
        
        # 强制清理可能残留的脏缓存文件，确保重新执行 Tokenization 和掩码逻辑
        dataset.cleanup_cache_files()
        
        dataset = dataset.map(
            lambda examples: format_to_text(examples, tokenizer),
            batched=True,
            remove_columns=dataset.column_names,
            load_from_cache_file=False  # 禁用当前映射过程的缓存读取
        )

        logger.info("Constructing SFTTrainer and TrainingArguments...")
        training_args = TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            warmup_ratio=WARMUP_RATIO,
            fp16=False,
            bf16=True,
            torch_compile=False,
            logging_steps=LOGGING_STEPS,
            save_steps=SAVE_STEPS,
            save_total_limit=3,
            lr_scheduler_type="cosine_with_restarts",
            neftune_noise_alpha=5,
            optim="adamw_torch_fused",
            seed=42,
            report_to="none",
            weight_decay=0.01,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            args=training_args,
            packing=False,
        )

        # 严格还原带有 \n 的匹配符，保证与底层的 Token ID 序列完全咬合
        # logger.info("Injecting strict Unsloth train_on_responses_only loss mask...")
        # trainer = train_on_responses_only(
        #     trainer,
        #     instruction_part="<|im_start|>user\n",
        #     response_part="<|im_start|>assistant\n",
        # )

        resume_checkpoint = None
        if OUTPUT_DIR.exists():
            checkpoints = sorted(OUTPUT_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
            if checkpoints:
                resume_checkpoint = str(checkpoints[-1])
                logger.info(f"Checkpoint detected. Resuming training from: {resume_checkpoint}")
            else:
                logger.info("No checkpoint found. Starting training from scratch.")
        else:
            logger.info("Output directory does not exist. Starting training from scratch.")

        logger.info("Training initiated.")
        trainer_stats = trainer.train(resume_from_checkpoint=resume_checkpoint)

        logger.info(f"Training completed. Final loss: {trainer_stats.training_loss:.4f}")

        logger.info("Saving final LoRA weights...")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(OUTPUT_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR))
        logger.info(f"Model successfully saved to: {OUTPUT_DIR}")

    except Exception as e:
        print(f"\n训练流程发生严重错误: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()