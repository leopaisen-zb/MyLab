"""
评估脚本：在测试集上使用微调后的 Qwen2.5-7B-Instruct 生成 VASP POSCAR，
并计算三类指标：
1. 格式有效率 (Validity Rate)
2. 化学成分匹配率 (Composition Accuracy)
3. 晶格体积平均绝对误差 (Volume MAE)

输入:  data/processed/dataset_test_rag.json
输出:  data/processed/evaluation_metrics.json
"""

from __future__ import annotations

import atexit
import gc
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import unsloth  # 必须在 transformers 之前，否则优化不生效
from unsloth import FastLanguageModel
from transformers import AutoTokenizer  # noqa: F401  保留以防需要

# ============================================================
# 全局状态：供紧急保存使用
# ============================================================
_global_results: List[Dict[str, Any]] = []
_global_output_path: Optional[Path] = None


def _emergency_save() -> None:
    """进程退出时（正常或异常）自动保存当前进度。"""
    global _global_results, _global_output_path
    if not _global_results or _global_output_path is None:
        return
    try:
        total = len(_global_results)
        v = sum(1 for r in _global_results if r.get("validity"))
        e = sum(1 for r in _global_results if r.get("composition_match_exact"))
        rx = sum(1 for r in _global_results if r.get("composition_match_relaxed"))
        vol_errors = [r["volume_abs_error"] for r in _global_results
                      if r.get("volume_abs_error") is not None]
        payload = {
            "summary": {
                "total_samples_evaluated": total,
                "validity_rate": v / total if total else 0.0,
                "composition_accuracy_exact": e / total if total else 0.0,
                "composition_accuracy_relaxed": rx / total if total else 0.0,
                "volume_mae": sum(vol_errors) / len(vol_errors) if vol_errors else None,
                "_note": "emergency save on exit",
            },
            "results": _global_results,
        }
        with open(_global_output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[紧急保存] 已将 {total} 条结果写入 {_global_output_path}")
    except Exception as ex:
        print(f"[紧急保存失败] {ex}")


atexit.register(_emergency_save)


def _signal_handler(sig, frame):
    print(f"\n[信号 {sig}] 收到终止信号，触发紧急保存 ...")
    _emergency_save()
    sys.exit(1)


for _sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(_sig, _signal_handler)
    except (OSError, ValueError):
        pass


# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TEST_RAG_JSON = DATA_DIR / "dataset_test_rag.json"

# 只评估微调后的模型
LORA_PATH = PROJECT_ROOT / "checkpoints" / "qwen_vasp_lora"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# 设为 None 表示使用全部测试集；也可手动改成整数做快速调试
NUM_SAMPLES: Optional[int] = None
MAX_NEW_TOKENS = 1024

# 留给生成的 token 余量：prompt_token 上限 = max_seq_length - MAX_NEW_TOKENS - 安全余量
PROMPT_TOKEN_BUDGET = 4096 - MAX_NEW_TOKENS - 64  # ≈ 3008


# ============================================================
# 工具函数：RAG Prompt 智能裁剪
# ============================================================
def _remove_structure_blocks(input_text: str) -> str:
    """
    从 RAG input 中移除每个参考条目里的 'Structure:\n...' 块，
    只保留 Properties 部分，以大幅缩短 token 占用。
    """
    import re
    # 匹配 "Structure:\n" 直到下一个 "--- Reference" 或字符串末尾
    pattern = re.compile(
        r"Structure:\n.*?(?=\n--- Reference|\Z)",
        re.DOTALL,
    )
    return pattern.sub("Structure: [省略以适应上下文长度]", input_text)


def _extract_target_block(input_text: str) -> str:
    """提取 input 中的目标特征块（第一个 '--- Reference' 之前的部分）。"""
    marker = "\n--- Reference"
    idx = input_text.find(marker)
    if idx == -1:
        return input_text
    return input_text[:idx].strip()


def _extract_references(input_text: str) -> List[str]:
    """
    将 RAG input 拆分为若干参考条目列表。
    每个元素对应一整个 '--- Reference N ... ---' 块（含尾部内容）。
    """
    import re
    # 按 "--- Reference N" 分割
    parts = re.split(r"(?=--- Reference \d)", input_text)
    refs = [p.strip() for p in parts if p.strip().startswith("--- Reference")]
    return refs


def smart_trim_input(input_text: str, instruction: str, tokenizer, budget: int) -> str:
    """
    按分级策略裁剪 RAG input，确保 chat_template 后总 token 数不超过 budget。

    裁剪优先级（信息损失从少到多）：
      Level 0: 原始 prompt（不裁剪）
      Level 1: 保留全部参考的 Properties，移除 Structure 坐标块
      Level 2: 只保留 2 个参考（无 Structure）
      Level 3: 只保留 1 个参考（无 Structure）
      Level 4: 移除全部参考，仅保留目标特征
    """
    def _count_tokens(inp: str) -> int:
        msgs = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": inp},
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return len(tokenizer.encode(text, add_special_tokens=False))

    # Level 0: 无需裁剪
    if _count_tokens(input_text) <= budget:
        return input_text

    # Level 1: 移除所有 Structure 块
    level1 = _remove_structure_blocks(input_text)
    if _count_tokens(level1) <= budget:
        print("    [Prompt 裁剪] Level-1: 移除参考结构坐标块")
        return level1

    # Level 2-3: 逐步减少参考条目数量（已移除 Structure）
    target_block = _extract_target_block(input_text)
    refs_no_struct = _extract_references(level1)
    for keep in (2, 1):
        if len(refs_no_struct) <= keep:
            continue
        kept_refs = refs_no_struct[:keep]
        candidate = (
            target_block
            + f"\n\nBelow are {keep} reference structures with similar properties:\n\n"
            + "\n\n".join(kept_refs)
        )
        if _count_tokens(candidate) <= budget:
            print(f"    [Prompt 裁剪] Level-{4 - keep}: 保留 {keep} 个参考（无坐标）")
            return candidate

    # Level 4: 仅保留目标特征
    print("    [Prompt 裁剪] Level-4: 仅保留目标特征（无 RAG 参考）")
    return target_block


# ============================================================
# 工具函数：VASP 解析相关
# ============================================================
def _split_non_empty_lines(text: str) -> List[str]:
    """按行拆分并去掉首尾空白，过滤空行。"""
    return [line.strip() for line in text.splitlines() if line.strip()]


def extract_pure_vasp(raw_text: str) -> str:
    """
    专门针对大模型的“话痨”行为进行剥壳，提取纯净的 VASP 文本。
    """
    # 策略 1：优先处理带有 Markdown 代码块的情况
    if "```" in raw_text:
        parts = raw_text.split("```")
        # 一般格式为：前言 + ```[lang]\nPOSCAR...\n```，取第 2 段
        if len(parts) >= 2:
            content = parts[1]
            # 去掉可选的语言标记，如 ```vasp
            stripped = content.lstrip()
            if stripped.lower().startswith("vasp"):
                stripped = stripped[4:]
            return stripped.strip()

    # 策略 2：没有 ``` 时，寻找 VASP 特有的缩放因子行（通常是 1.0...）
    lines = raw_text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("1.000"):
            # 通常上一行为注释 / 系统名
            start_idx = max(0, i - 1)
            return "\n".join(lines[start_idx:]).strip()

    # 兜底返回原文本裁剪版
    return raw_text.strip()


def check_validity(gen_text: str) -> bool:
    """
    检查生成文本是否符合基本 VASP POSCAR 格式。

    条件：
    - 至少有 8 行（去除空行后计数）
    - 第 3–5 行能解析出 3 个浮点数（晶格向量）
    - 包含 "Cartesian" 或 "Direct"（不区分大小写）
    - 其后至少有一行能解析为 3 个浮点坐标
    """
    try:
        lines = _split_non_empty_lines(gen_text)
        if len(lines) < 8:
            return False

        # 第 3-5 行晶格向量
        lattice_ok = 0
        for i in range(2, 5):
            if i >= len(lines):
                break
            tokens = lines[i].split()
            if len(tokens) < 3:
                continue
            try:
                _ = [float(t) for t in tokens[:3]]
                lattice_ok += 1
            except Exception:
                continue
        if lattice_ok != 3:
            return False

        # 坐标类型行
        coord_type_idx = None
        for idx, line in enumerate(lines):
            lower = line.lower()
            if lower == "cartesian" or lower == "direct":
                coord_type_idx = idx
                break
        if coord_type_idx is None:
            return False

        # 坐标行：紧随坐标类型行之后，至少有一行是 3 个浮点
        has_coord_line = False
        for line in lines[coord_type_idx + 1 :]:
            tokens = line.split()
            if len(tokens) < 3:
                continue
            try:
                _ = [float(t) for t in tokens[:3]]
                has_coord_line = True
                break
            except Exception:
                continue

        return has_coord_line
    except Exception:
        # 极端乱码直接视为非法
        return False


def check_composition_match(gen_text: str, gt_text: str, debug_print: bool = False) -> dict:
    """
    智能化学成分评估：同时返回“严格匹配”和“同族容忍匹配”两个结果。
    """
    try:
        from typing import List

        # 内部提取非空行的辅助逻辑
        def get_lines(text: str) -> List[str]:
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            return lines

        gen_lines = get_lines(gen_text)
        gt_lines = get_lines(gt_text)

        # 锚点倒推寻找元素和数量行
        def find_elem_and_count(lines: List[str]):
            for i, line in enumerate(lines):
                if line.strip().lower() in ["cartesian", "direct"]:
                    if i >= 2:
                        return lines[i - 2].split(), lines[i - 1].split()
            return [], []

        gen_elem, gen_counts = find_elem_and_count(gen_lines)
        gt_elem, gt_counts = find_elem_and_count(gt_lines)

        if not gen_elem or not gt_elem:
            return {"exact": False, "relaxed": False}

        # 化学同族字典 (等价取代映射)
        CHEM_GROUPS = {
            "IIB": ["Zn", "Cd", "Hg"],
            "IIIA": ["B", "Al", "Ga", "In", "Tl"],
            "IVB": ["Ti", "Zr", "Hf"],
            "VB": ["V", "Nb", "Ta"],
            "VIB": ["Cr", "Mo", "W"],
            "VIIB": ["Mn", "Tc", "Re"],
            "VIII": ["Fe", "Ru", "Os", "Co", "Rh", "Ir", "Ni", "Pd", "Pt"],
            "IB": ["Cu", "Ag", "Au"],
        }

        def get_group(element: str) -> str:
            for group_name, elements in CHEM_GROUPS.items():
                if element in elements:
                    return group_name
            # 未在字典中的元素，用自身符号作为 key
            return element

        def to_chem_dict(elems, counts, use_group: bool = False):
            chem_dict: Dict[str, int] = {}
            for e, c in zip(elems, counts):
                key = get_group(e) if use_group else e
                chem_dict[key] = chem_dict.get(key, 0) + int(c)
            return chem_dict

        # 计算双重匹配度
        gen_chem_exact = to_chem_dict(gen_elem, gen_counts, use_group=False)
        gt_chem_exact = to_chem_dict(gt_elem, gt_counts, use_group=False)
        exact_match = gen_chem_exact == gt_chem_exact

        gen_chem_relaxed = to_chem_dict(gen_elem, gen_counts, use_group=True)
        gt_chem_relaxed = to_chem_dict(gt_elem, gt_counts, use_group=True)
        relaxed_match = gen_chem_relaxed == gt_chem_relaxed

        if debug_print:
            print(f"    [严格对比] 生成: {gen_chem_exact}  | 真实: {gt_chem_exact}")
            print(f"    [同族对比] 生成: {gen_chem_relaxed} | 真实: {gt_chem_relaxed}")

        return {"exact": exact_match, "relaxed": relaxed_match}

    except Exception:
        return {"exact": False, "relaxed": False}


def _parse_lattice_matrix(lines: List[str]) -> np.ndarray:
    """
    从非空行列表中解析第 3-5 行为 3x3 晶格矩阵。
    失败则抛出异常，由上层捕获决定是否跳过该样本。
    """
    if len(lines) < 5:
        raise ValueError("not enough lines for lattice")

    lattice_rows: List[List[float]] = []
    for i in range(2, 5):
        tokens = lines[i].split()
        if len(tokens) < 3:
            raise ValueError(f"lattice line {i+1} has <3 tokens")
        try:
            row = [float(t) for t in tokens[:3]]
        except Exception as exc:
            raise ValueError(f"cannot parse lattice line {i+1}") from exc
        lattice_rows.append(row)

    return np.array(lattice_rows, dtype=float)


def calculate_volume_error(gen_text: str, gt_text: str) -> float:
    """
    计算生成晶胞体积与真实晶胞体积的绝对误差。

    - 体积 = |det(3x3 晶格矩阵)|
    - 若生成文本解析失败，调用方应跳过该样本。
    """
    gen_lines = _split_non_empty_lines(gen_text)
    gt_lines = _split_non_empty_lines(gt_text)

    gen_lattice = _parse_lattice_matrix(gen_lines)
    gt_lattice = _parse_lattice_matrix(gt_lines)

    gen_volume = float(abs(np.linalg.det(gen_lattice)))
    gt_volume = float(abs(np.linalg.det(gt_lattice)))

    return abs(gen_volume - gt_volume)


# ============================================================
# 主评估流程
# ============================================================
def main() -> None:
    if not TEST_RAG_JSON.exists():
        raise FileNotFoundError(f"测试集不存在: {TEST_RAG_JSON}")

    if not LORA_PATH.exists():
        raise FileNotFoundError(f"LoRA 权重路径不存在: {LORA_PATH}")

    print("=" * 70)
    print("微调模型 VASP 结构评估 (Validity / Composition / Volume MAE)")
    print("=" * 70)
    print(f"测试集: {TEST_RAG_JSON}")
    print(f"LoRA 权重: {LORA_PATH}")
    print(f"基础模型: {MODEL_NAME}")

    # 1. 加载测试数据
    print("\n[1/4] 加载测试数据 ...")
    with open(TEST_RAG_JSON, "r", encoding="utf-8") as f:
        test_data: List[Dict[str, Any]] = json.load(f)
    total_available = len(test_data)

    if NUM_SAMPLES is None or NUM_SAMPLES <= 0 or NUM_SAMPLES > total_available:
        num_eval = total_available
    else:
        num_eval = NUM_SAMPLES

    print(f"  测试集总数: {total_available}，本次评估: {num_eval} 条")

    # 2. 加载模型 (Unsloth 4-bit 极速推理)
    print("\n[2/4] 加载模型 (Unsloth 4-bit 极速推理) ...")
    if torch.cuda.is_available():
        try:
            print(f"  使用 GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            print("  使用 GPU: 已启用 CUDA")
    else:
        print("  警告: 当前未检测到 GPU，推理可能非常慢。")

    # 这里显式指定基础模型 + LoRA 适配器，避免在不同机器上复用训练时的缓存绝对路径
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        adapter_name=str(LORA_PATH),
        max_seq_length=4096,
        load_in_4bit=True,
        local_files_only=False,  # 需要时允许从 Hub 下载基础模型
    )
    FastLanguageModel.for_inference(model)
    print(f"  模型已加载: {MODEL_NAME} (+ LoRA: {LORA_PATH})")

    # 3. 逐条生成 + 计算指标
    print(f"\n[3/4] 开始推理与指标计算 ({num_eval} 条) ...\n")

    global _global_output_path, _global_results
    output_path = DATA_DIR / "evaluation_metrics.json"
    _global_output_path = output_path
    results: List[Dict[str, Any]] = []
    start_idx = 0

    # 断点续传：若已有结果文件，则从历史进度继续
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_payload = json.load(f)
                results = existing_payload.get("results", [])
                start_idx = len(results)
            if start_idx > 0:
                print(
                    f"\n[断点续传] 检测到已有进度，已完成 {start_idx} 条，"
                    f"将从第 {start_idx + 1} 条继续运行..."
                )
        except Exception as e:
            print(f"\n[警告] 读取历史进度失败: {e}，将从头开始。")
            results = []
            start_idx = 0

    # 这些计数器仅用于本轮运行时的即时统计；最终汇总会基于全部 results 重新计算
    valid_count = 0
    exact_match_count = 0
    relaxed_match_count = 0
    volume_abs_error_sum = 0.0
    volume_count = 0

    for idx in range(start_idx, num_eval):
        sample = test_data[idx]
        instruction = sample.get("instruction", "")
        user_input = sample.get("input", "")
        gt_text = sample.get("output", "")

        print(f"[样本 {idx + 1}/{num_eval}]")

        gen_text: str = ""
        generation_error: Optional[str] = None

        try:
            # 智能裁剪：确保 prompt 不超过 token 预算，同时最大程度保留有效信息
            trimmed_input = smart_trim_input(
                user_input, instruction, tokenizer, PROMPT_TOKEN_BUDGET
            )
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": trimmed_input},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated_ids = outputs[0][prompt_len:]
            gen_text_raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
            gen_text = extract_pure_vasp(gen_text_raw)
        except Exception as exc:
            generation_error = repr(exc)
            print(f"  生成失败，跳过该样本。错误: {generation_error}")

        sample_record: Dict[str, Any] = {
            "index": idx,
            "generation_error": generation_error,
        }

        if generation_error is None:
            # 1) 格式有效性
            is_valid = check_validity(gen_text)
            if is_valid:
                valid_count += 1

            # 2) 化学成分匹配 (双重指标)
            comp_result = check_composition_match(
                gen_text, gt_text, debug_print=(idx < 5)
            )
            is_exact = comp_result["exact"]
            is_relaxed = comp_result["relaxed"]

            if is_exact:
                exact_match_count += 1
            if is_relaxed:
                relaxed_match_count += 1

            # 3) 体积误差
            volume_error: Optional[float] = None
            try:
                volume_error = float(calculate_volume_error(gen_text, gt_text))
                volume_abs_error_sum += volume_error
                volume_count += 1
            except Exception:
                volume_error = None

            sample_record.update(
                {
                    "validity": bool(is_valid),
                    "composition_match_exact": bool(is_exact),
                    "composition_match_relaxed": bool(is_relaxed),
                    "volume_abs_error": volume_error,
                    "generated": gen_text,
                    "ground_truth": gt_text,
                }
            )

            print(
                f"  Valid: {is_valid} | "
                f"ExactComp: {is_exact} | "
                f"RelaxedComp: {is_relaxed} | "
                f"VolErr: {volume_error if volume_error is not None else 'N/A'}"
            )
        else:
            sample_record.update(
                {
                    "validity": False,
                    "composition_match_exact": False,
                    "composition_match_relaxed": False,
                    "volume_abs_error": None,
                    "generated": None,
                    "ground_truth": gt_text,
                }
            )

        results.append(sample_record)
        _global_results = results  # 同步全局，供 atexit 紧急保存使用

        # === 每条增量保存 + 显存垃圾回收 ===
        try:
            current_total = len(results)
            v_count = sum(1 for r in results if r.get("validity"))
            e_count = sum(1 for r in results if r.get("composition_match_exact"))
            r_count = sum(1 for r in results if r.get("composition_match_relaxed"))
            vol_errors = [
                r.get("volume_abs_error")
                for r in results
                if r.get("volume_abs_error") is not None
            ]

            temp_summary = {
                "total_samples_evaluated": current_total,
                "validity_rate": v_count / current_total if current_total > 0 else 0.0,
                "composition_accuracy_exact": (
                    e_count / current_total if current_total > 0 else 0.0
                ),
                "composition_accuracy_relaxed": (
                    r_count / current_total if current_total > 0 else 0.0
                ),
                "volume_mae": (
                    sum(vol_errors) / len(vol_errors) if vol_errors else None
                ),
            }
            payload = {
                "summary": temp_summary,
                "results": results,
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            # 每10条打印一次保存提示，避免刷屏
            if current_total % 10 == 0 or current_total == num_eval:
                print(f"    [进度保存] 已将前 {current_total} 条结果写入磁盘。")
        except Exception as save_exc:
            print(f"    [警告] 进度保存失败，但继续运行: {save_exc}")

        # 每条样本后清理显存碎片，降低 OOM 风险
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # 4. 汇总与保存
    print("\n[4/4] 汇总结果 ...\n")

    # 最终基于全部 results 重新计算整体指标，兼容断点续传场景
    total = len(results)
    validity_rate = (
        sum(1 for r in results if r.get("validity")) / total if total > 0 else 0.0
    )
    exact_accuracy = (
        sum(1 for r in results if r.get("composition_match_exact")) / total
        if total > 0
        else 0.0
    )
    relaxed_accuracy = (
        sum(1 for r in results if r.get("composition_match_relaxed")) / total
        if total > 0
        else 0.0
    )
    vol_errors_all = [
        r.get("volume_abs_error")
        for r in results
        if r.get("volume_abs_error") is not None
    ]
    volume_mae = (
        sum(vol_errors_all) / len(vol_errors_all) if vol_errors_all else None
    )

    print("=" * 70)
    print("评估汇总：")
    print("-" * 70)
    print(f"  总样本数 (评估):        {total}")
    print(f"  格式有效 (Validity):    {valid_count} / {total} "
          f"= {validity_rate:.4f}")
    print(
        f"  严格成分匹配 (Exact):   {exact_match_count} / {total} "
        f"= {exact_accuracy:.4f}"
    )
    print(
        f"  同族容忍匹配 (Relaxed): {relaxed_match_count} / {total} "
        f"= {relaxed_accuracy:.4f}"
    )
    if volume_mae is not None:
        print(
            f"  体积 MAE (Volume MAE):  {volume_mae:.6f} "
            f"(基于 {volume_count} 条可解析样本)"
        )
    else:
        print("  体积 MAE (Volume MAE):  无法计算（无有效样本）")
    print("=" * 70)

    summary = {
        "total_samples_evaluated": total,
        "validity_rate": validity_rate,
        "composition_accuracy_exact": exact_accuracy,
        "composition_accuracy_relaxed": relaxed_accuracy,
        "volume_mae": volume_mae,
        "volume_sample_count": volume_count,
        "config": {
            "model_name": MODEL_NAME,
            "lora_path": str(LORA_PATH),
            "num_samples": NUM_SAMPLES,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
    }

    output_path = DATA_DIR / "evaluation_metrics.json"
    payload = {
        "summary": summary,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 详细对比结果与汇总指标已保存到: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[用户中断] 脚本已停止，进度已由 atexit 自动保存。")
        sys.exit(0)
    except torch.cuda.OutOfMemoryError as oom:
        print(f"\n[CUDA OOM] GPU 显存不足: {oom}")
        print("提示: 可尝试减小 MAX_NEW_TOKENS 或将 NUM_SAMPLES 分批运行。")
        sys.exit(1)
    except Exception as exc:
        import traceback
        print(f"\n[致命错误] 脚本因未捕获异常终止:")
        traceback.print_exc()
        print(f"\n已通过 atexit 自动保存当前进度。")
        sys.exit(1)

