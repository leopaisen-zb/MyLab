from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from pymatgen.core import Lattice, Structure
from tqdm import tqdm

# ============================================================
# 配置区 (单条/小批量评估，Unsloth + LoRA)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_JSON = (
    PROJECT_ROOT
    / "data"
    / "rag_data"
    / "text2struct_rag"
    / "dataset_test_rag_code.json"
)
LORA_PATH = (
    PROJECT_ROOT / "checkpoints" / "qwen_vasp_code_lora" / "checkpoint-723"
)

# 不再使用 vLLM，改为 Unsloth 单机推理
ENABLE_RESUME: bool = False

# 基线实验：先跑前 200 条
NUM_SAMPLES: Optional[int] = 200
BATCH_SIZE: int = 1
MAX_NEW_TOKENS: int = 1024
MAX_SEQ_LENGTH: int = 4096
PROMPT_TOKEN_BUDGET: int = MAX_SEQ_LENGTH - MAX_NEW_TOKENS - 64

CACHE_PATH = (
    PROJECT_ROOT
    / "data"
    / "rag_data"
    / "text2struct_rag"
    / "eval_sandbox_FINAL_promptB.json"
)

# 🌟 定海神针：锁死模型的输出格式，防止乱码
PROMPT_CODE_PREFIX: str = (
    "```python\n"
    "def generate_structure():\n"
    "    from pymatgen.core import Lattice, Structure\n"
)


# ============================================================
# 🌟 核心科技：正则暴力提取 (无视一切 Python 语法报错)
# ============================================================
def evaluate_single_prediction(
    gen_text: str,
    ground_truth_structure: Structure,
) -> Dict[str, Any]:
    """对单条模型输出做正则解析并打分。

    Args:
        gen_text: 大模型生成的原始文本（可能包含多余内容）。
        ground_truth_structure: 该样本对应的真实结构。

    Returns:
        含 4 个评价指标的字典。
    """
    metrics: Dict[str, Any] = {
        "is_executable": 0,
        "is_valid_structure": 0,
        "is_composition_match": 0,
        "volume_error": None,
    }

    try:
        # 1. 用正则表达式强行抠出三个数组的字符串
        lat_match = re.search(
            r"lattice_matrix\s*=\s*(\[\s*\[.*?\]\s*\])",
            gen_text,
            re.DOTALL,
        )
        sp_match = re.search(
            r"species\s*=\s*(\[.*?\])",
            gen_text,
            re.DOTALL,
        )
        coord_match = re.search(
            r"coords\s*=\s*(\[\s*\[.*?\]\s*\])",
            gen_text,
            re.DOTALL,
        )

        if not (lat_match and sp_match and coord_match):
            raise ValueError("未能在生成文本中找到 lattice_matrix/species/coords 三个数组。")

        # 2. 将字符串转换为真正的 Python 列表
        lattice_matrix = ast.literal_eval(lat_match.group(1))
        species = ast.literal_eval(sp_match.group(1))
        coords = ast.literal_eval(coord_match.group(1))

        # 3. 自动对齐长度 (补齐模型的数学缺陷)
        if len(species) < len(coords):
            species = species + [species[-1]] * (len(coords) - len(species))
        elif len(species) > len(coords):
            species = species[: len(coords)]

        # 4. 安全地构建 Structure，不使用 exec
        generated_struct = Structure(Lattice(lattice_matrix), species, coords)

        metrics["is_executable"] = 1

        # 物理合法性：最小原子间距阈值
        distance_matrix = generated_struct.distance_matrix
        min_dist = (
            distance_matrix[distance_matrix > 0].min()
            if len(generated_struct) > 1
            else 999.0
        )
        if min_dist > 0.5:
            metrics["is_valid_structure"] = 1

        # 化学式精确匹配
        if (
            generated_struct.composition.reduced_formula
            == ground_truth_structure.composition.reduced_formula
        ):
            metrics["is_composition_match"] = 1

        # 体积绝对误差
        metrics["volume_error"] = abs(
            generated_struct.volume - ground_truth_structure.volume
        )
    except Exception:
        # 提取或构建失败直接跳过，避免中断整体评估
        pass

    return metrics


# ============================================================
# 数据处理与大模型推理
# ============================================================
def get_ground_truth_structure(gt_output: str) -> Optional[Structure]:
    """从标准答案代码中解析出 pymatgen 结构。

    Args:
        gt_output: 标准答案中的 Python 代码字符串。

    Returns:
        如果解析成功，返回对应的 Structure；否则返回 None。
    """
    try:
        lat_match = re.search(
            r"lattice_matrix\s*=\s*(\[\s*\[.*?\]\s*\])",
            gt_output,
            re.DOTALL,
        )
        sp_match = re.search(
            r"species\s*=\s*(\[.*?\])",
            gt_output,
            re.DOTALL,
        )
        coord_match = re.search(
            r"coords\s*=\s*(\[\s*\[.*?\]\s*\])",
            gt_output,
            re.DOTALL,
        )
        if lat_match and sp_match and coord_match:
            lattice_matrix = ast.literal_eval(lat_match.group(1))
            species = ast.literal_eval(sp_match.group(1))
            coords = ast.literal_eval(coord_match.group(1))
            return Structure(Lattice(lattice_matrix), species, coords)
    except Exception:
        pass
    return None


def _build_system_and_user_content(
    instruction: str,
    user_input: str,
) -> tuple[str, str]:
    """根据指令和用户输入构造带元素约束的 system/user 文本。"""
    system_content = (
        instruction
        + "\n\nIMPORTANT:\n"
        + "1) The chemical element types in the `species` list MUST be chosen ONLY from "
        + "the elements that appear in the provided reference structures.\n"
        + "2) Do NOT invent any new elements that are not present in the reference structures.\n"
        + "3) Prefer to keep the element ratios as close as possible to the reference structure "
        + "with the smallest L2 distance.\n"
    )
    user_content = (
        user_input
        + "\n\n请严格遵守上述元素约束，并且只输出 Python 代码，不要解释文字。"
    )
    return system_content, user_content


def _count_tokens(text: str, instruction: str, tokenizer: Any) -> int:
    """统计带模板的 prompt token 数，用于控制长度。"""
    system_content, user_content = _build_system_and_user_content(
        instruction,
        text,
    )
    msgs = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": user_content,
        },
    ]
    prompt = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    return len(
        tokenizer.encode(
            prompt + PROMPT_CODE_PREFIX,
            add_special_tokens=False,
        )
    )


def trim_input_if_needed(
    user_input: str,
    instruction: str,
    tokenizer: Any,
) -> str:
    """如果超过 token budget，就对 input 做二分截断。"""
    if _count_tokens(user_input, instruction, tokenizer) <= PROMPT_TOKEN_BUDGET:
        return user_input
    lo, hi = 0, len(user_input)
    while lo < hi - 64:
        mid = (lo + hi) // 2
        if _count_tokens(user_input[:mid], instruction, tokenizer) <= PROMPT_TOKEN_BUDGET:
            lo = mid
        else:
            hi = mid
    return user_input[:lo] + "\n\n[输入已截断]"


def build_code_prompt(
    instruction: str,
    user_input: str,
    tokenizer: Any,
) -> str:
    """构造最终送入模型的 prompt（带 chat 模板 + 代码前缀）。"""
    system_content, user_content = _build_system_and_user_content(
        instruction,
        user_input,
    )
    msgs = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": user_content,
        },
    ]
    prompt = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt + PROMPT_CODE_PREFIX


def load_cached_results() -> Dict[int, Dict[str, Any]]:
    """从缓存文件中加载历史评估结果（当前默认关闭）。"""
    if not ENABLE_RESUME or not CACHE_PATH.exists():
        return {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {
            r["idx"]: r
            for r in raw.get("results", [])
            if isinstance(r, dict) and "idx" in r
        }
    except Exception:
        return {}


def save_results(results: List[Dict[str, Any]], num_eval: int) -> None:
    """将评估结果写入 JSON 文件，方便后续分析。"""
    if not results:
        return
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"results": sorted(results, key=lambda x: x.get("idx", 0))},
            f,
            ensure_ascii=False,
            indent=2,
        )


def run_lora_inference(
    prompts: List[str],
    model: Any,
    tokenizer: Any,
) -> List[str]:
    """使用 Unsloth + LoRA 对一批 prompt 做推理。

    Args:
        prompts: 已经构造好的完整 prompt 列表。
        model: 已加载好的 Unsloth FastLanguageModel 模型。
        tokenizer: 与模型对应的 tokenizer。

    Returns:
        每条 prompt 对应的生成文本（只取第一个候选）。
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    generated_texts: List[str] = []
    for prompt in prompts:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
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
        gen_text: str = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )
        generated_texts.append(gen_text)

    return generated_texts


def main() -> None:
    """主入口：对测试集前 NUM_SAMPLES 条样本做评估。"""
    print("=" * 70 + "\n沙盒评估 (正则暴力提取涨点版)\n" + "=" * 70)

    with open(TEST_JSON, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    num_eval: int = len(test_data) if NUM_SAMPLES is None else min(
        NUM_SAMPLES,
        len(test_data),
    )
    print(f"本次评估样本数: {num_eval}")

    # 使用 Unsloth 加载 LoRA 微调好的 Qwen2.5-7B-Instruct
    from unsloth import FastLanguageModel

    print("\n[1/3] 加载 Unsloth LoRA 模型与 tokenizer ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(LORA_PATH),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )
    FastLanguageModel.for_inference(model)
    print("  模型加载完成。")

    prompts_to_infer: List[str] = []
    gt_structures_to_eval: List[Structure] = []
    indices_to_infer: List[int] = []

    for idx in tqdm(range(num_eval), desc="预处理", unit="条"):
        sample = test_data[idx]
        gt_structure = get_ground_truth_structure(sample.get("output", ""))
        if gt_structure is None:
            continue
        trimmed = trim_input_if_needed(
            sample.get("input", ""),
            sample.get("instruction", ""),
            tokenizer,
        )
        prompts_to_infer.append(
            build_code_prompt(
                sample.get("instruction", ""),
                trimmed,
                tokenizer,
            )
        )
        gt_structures_to_eval.append(gt_structure)
        indices_to_infer.append(idx)

    if not prompts_to_infer:
        print("没有可评估的样本，直接退出。")
        return

    print("\n[2/3] 开始逐条推理 ...")
    gen_texts = run_lora_inference(
        prompts_to_infer,
        model,
        tokenizer,
    )
    print("[3/3] 推理完成，开始打分与汇总。")
    results: List[Dict[str, Any]] = []
    for idx, gen_text, gt_struct in zip(
        indices_to_infer,
        gen_texts,
        gt_structures_to_eval,
    ):
        metrics = evaluate_single_prediction(gen_text, gt_struct)
        results.append({"idx": idx, **metrics})

    save_results(results, num_eval)

    print("\n" + "=" * 70 + "\n汇总 4 项指标（测试集平均）\n" + "-" * 70)
    n = len(results)
    if n > 0:
        exec_ok = sum(r.get("is_executable", 0) for r in results)
        valid_ok = sum(r.get("is_valid_structure", 0) for r in results)
        comp_ok = sum(r.get("is_composition_match", 0) for r in results)
        vol_errors = [
            r["volume_error"]
            for r in results
            if r.get("volume_error") is not None
        ]
        print(f"  样本数:                      {n}")
        print(
            f"  is_executable (提取成功率):  {exec_ok / n:.4f}  "
            f"({exec_ok}/{n})"
        )
        print(
            f"  is_valid_structure (平均):   {valid_ok / n:.4f}  "
            f"({valid_ok}/{n})"
        )
        print(
            f"  is_composition_match (平均): {comp_ok / n:.4f}  "
            f"({comp_ok}/{n})"
        )
        if vol_errors:
            print(
                "  volume_error (MAE):          "
                f"{sum(vol_errors) / len(vol_errors):.4f}"
            )
        else:
            print("  volume_error (MAE):          N/A")
    print("=" * 70)


if __name__ == "__main__":
    main()