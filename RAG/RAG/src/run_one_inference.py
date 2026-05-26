#!/usr/bin/env python3
"""
单条推理脚本：对比三种权重在测试集第一条样本上的生成效果。

支持的三种权重形式:
1. LoRA 适配器权重: qwen_vasp_code_lora
2. 合并后完整模型: qwen_merged_bfloat16
3. 合并后完整模型（修复版）: qwen_merged_fixed

用法:
    python run_one_inference.py
"""

from typing import Any
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 修复 torchao 对 torch.int1~7 的依赖
for _n in range(1, 8):
    if not hasattr(torch, f"int{_n}"):
        setattr(torch, f"int{_n}", torch.int8)

# 修复 checkpoint rng_state.pth 含 numpy 对象时 weights_only=True 报错
torch.serialization.add_safe_globals(
    [
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        *[v for v in vars(np.dtypes).values() if isinstance(v, type)],
    ]
)

# 路径配置
LORA_PATH: str = "/home/ubuntu/HZH/RAG/RAG/checkpoints/qwen_vasp_code_lora/checkpoint-723"
MERGED_BFLOAT16_PATH: str = "/home/ubuntu/HZH/RAG/RAG/checkpoints/qwen_merged_bfloat16"
MERGED_FIXED_PATH: str = "/home/ubuntu/HZH/RAG/RAG/checkpoints/qwen_merged_fixed"
TEST_JSON: str = (
    "/home/ubuntu/HZH/RAG/RAG/data/rag_data/text2struct_rag/dataset_test_rag_code.json"
)

# 生成参数（与 eval_sandbox 对齐）
CODE_PREFIX: str = "```python\ndef generate_structure():\n"
MAX_NEW_TOKENS: int = 1024
MAX_SEQ_LENGTH: int = 4096


def build_code_prompt(instruction: str, user_input: str, tokenizer: Any) -> str:
    """构建代码生成 Prompt，并注入函数前缀。

    Args:
        instruction: 测试样本中的系统指令文本。
        user_input: 测试样本中的用户输入描述。
        tokenizer: 已加载的分词器实例，需支持 apply_chat_template。

    Returns:
        拼接好前缀的完整 Prompt 字符串。
    """
    # 与 eval_sandbox 中逻辑保持一致，追加“只输出代码”的约束。
    messages = [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": user_input + "\n\n请只输出 Python 代码，不要解释文字。",
        },
    ]
    prompt: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt += CODE_PREFIX
    return prompt


def run_with_lora(instruction: str, user_input: str, gt_output: str) -> None:
    """使用 LoRA 适配器权重执行单条推理并打印结果。

    Args:
        instruction: 测试样本中的系统指令文本。
        user_input: 测试样本中的用户输入描述。
        gt_output: 测试样本中的标准答案字符串。
    """
    from unsloth import FastLanguageModel

    if not os.path.isdir(LORA_PATH):
        raise FileNotFoundError(f"LoRA 权重目录不存在: {LORA_PATH}")

    print("=" * 60)
    print("模型一：LoRA 适配器 (qwen_vasp_code_lora)")
    print("=" * 60)
    print(f"LoRA 路径: {LORA_PATH}")

    print("\n[1/3] 加载 LoRA 模型与 tokenizer (Unsloth + LoRA) ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LORA_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  模型加载完成。")

    prompt: str = build_code_prompt(instruction, user_input, tokenizer)
    print(f"\n[2/3] Prompt 长度（字符）: {len(prompt)}")

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len: int = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_len:]
    generated_text: str = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )

    print("\n[3/3] 生成结果 (LoRA)")
    print("-" * 60)
    print("【模型生成 - LoRA】")
    print("-" * 60)
    print(CODE_PREFIX + generated_text)
    print("-" * 60)
    print("【标准答案 (output)】")
    print("-" * 60)
    print(gt_output[:2000] + ("..." if len(gt_output) > 2000 else ""))
    print("-" * 60)


def run_with_merged_model(
    model_dir: str,
    model_name_for_log: str,
    instruction: str,
    user_input: str,
    gt_output: str,
) -> None:
    """使用合并后的完整模型执行单条推理并打印结果。

    Args:
        model_dir: 合并模型所在目录绝对路径。
        model_name_for_log: 日志输出中使用的模型名称。
        instruction: 测试样本中的系统指令文本。
        user_input: 测试样本中的用户输入描述。
        gt_output: 测试样本中的标准答案字符串。
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"合并模型目录不存在: {model_dir}")

    print("=" * 60)
    print(f"模型二/三：合并后完整模型 ({model_name_for_log})")
    print("=" * 60)
    print(f"模型路径: {model_dir}")

    print("\n[1/3] 加载合并模型与 tokenizer (transformers) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  模型加载完成。")

    prompt: str = build_code_prompt(instruction, user_input, tokenizer)
    print(f"\n[2/3] Prompt 长度（字符）: {len(prompt)}")

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len: int = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_len:]
    generated_text: str = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )

    print(f"\n[3/3] 生成结果 ({model_name_for_log})")
    print("-" * 60)
    print(f"【模型生成 - {model_name_for_log}】")
    print("-" * 60)
    print(CODE_PREFIX + generated_text)
    print("-" * 60)
    print("【标准答案 (output)】")
    print("-" * 60)
    print(gt_output[:2000] + ("..." if len(gt_output) > 2000 else ""))
    print("-" * 60)


def main() -> None:
    """对测试集第一条数据分别用三种权重执行一次推理并打印结果。

    Raises:
        FileNotFoundError: 当测试集文件或任一模型目录不存在时抛出，错误信息为中文。
    """
    if not os.path.isfile(TEST_JSON):
        raise FileNotFoundError(f"测试集不存在: {TEST_JSON}")
    # 1. 加载数据（只取第一条）
    with open(TEST_JSON, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    sample = test_data[0]
    instruction: str = sample.get("instruction", "")
    user_input: str = sample.get("input", "")
    gt_output: str = sample.get("output", "")

    print(
        f"\n已加载测试集第 1 条 "
        f"(instruction 长度: {len(instruction)}, input 长度: {len(user_input)})"
    )

    # 2. 依次用三种权重执行推理
    # 先运行两个合并模型，避免在导入 Unsloth 之后再触发其对 transformers 的全局补丁逻辑。
    run_with_merged_model(
        MERGED_BFLOAT16_PATH,
        "qwen_merged_bfloat16",
        instruction,
        user_input,
        gt_output,
    )
    run_with_merged_model(
        MERGED_FIXED_PATH,
        "qwen_merged_fixed",
        instruction,
        user_input,
        gt_output,
    )
    # 最后再运行 LoRA 版本，此时导入 Unsloth 不会影响前面两个模型的推理。
    run_with_lora(instruction, user_input, gt_output)


if __name__ == "__main__":
    main()
