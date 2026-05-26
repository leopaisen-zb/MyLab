"""
评估脚本：在 RAG 增强测试集上评估 Qwen2.5-7B-Instruct。
支持基线（未微调）或加载 LoRA 微调权重进行对比。

输入: data/processed/dataset_test_rag.json
输出: 打印到终端 + 保存到 data/processed/baseline_results.json 或 finetuned_results.json
"""

import json
import torch
from pathlib import Path
import unsloth  # 必须在 transformers 之前，否则优化不生效
from unsloth import FastLanguageModel
from transformers import AutoTokenizer  # 保留这个，以防万一

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TEST_RAG_JSON = DATA_DIR / "dataset_test_rag.json"

# 设为非 None 则加载微调 LoRA，否则为基线评估
LORA_PATH = PROJECT_ROOT / "checkpoints" / "qwen_vasp_lora"  # None = 基线

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
NUM_SAMPLES = 5          # 测试几条样本（节省时间）
MAX_NEW_TOKENS = 512     # 生成的最大 token 数


# ============================================================
# 简单的结构健全性检查
# ============================================================
def check_vasp_sanity(text: str) -> dict:
    """对生成的文本做基本 VASP 格式检查。"""
    lines = [l for l in text.strip().split("\n") if l.strip()]
    report = {
        "total_lines": len(lines),
        "has_lattice": False,
        "has_coord_type": False,
        "coord_lines": 0,
        "parseable": False,
    }

    if len(lines) < 8:
        return report

    # 检查第 3-5 行是否为 3 个浮点数（晶格向量）
    lattice_ok = 0
    for i in range(2, min(5, len(lines))):
        tokens = lines[i].split()
        try:
            if len(tokens) >= 3:
                [float(t) for t in tokens[:3]]
                lattice_ok += 1
        except ValueError:
            pass
    report["has_lattice"] = lattice_ok == 3

    # 检查是否有 Cartesian / Direct
    for line in lines:
        if line.strip().lower() in ("cartesian", "direct"):
            report["has_coord_type"] = True
            break

    # 统计坐标行数（能解析出 3 个浮点数的行）
    coord_start = False
    for line in lines:
        if line.strip().lower() in ("cartesian", "direct"):
            coord_start = True
            continue
        if coord_start:
            tokens = line.split()
            try:
                if len(tokens) >= 3:
                    [float(t) for t in tokens[:3]]
                    report["coord_lines"] += 1
            except ValueError:
                pass

    report["parseable"] = (
        report["has_lattice"]
        and report["has_coord_type"]
        and report["coord_lines"] > 0
    )
    return report


# ============================================================
# 主流程
# ============================================================
def main():
    use_lora = LORA_PATH is not None and Path(LORA_PATH).exists()
    result_path = DATA_DIR / ("finetuned_results.json" if use_lora else "baseline_results.json")

    print("=" * 60)
    if use_lora:
        print("微调模型评估: Qwen2.5-7B-Instruct + LoRA")
        print(f"  LoRA: {LORA_PATH}")
    else:
        print("基线评估: Qwen2.5-7B-Instruct (未微调)")
    print("=" * 60)

    # 1. 加载测试数据
    print("\n[1/3] 加载测试数据 ...")
    with open(TEST_RAG_JSON, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"  测试集总数: {len(test_data)}，本次评估: {NUM_SAMPLES} 条")

    # 2. 加载模型 (Unsloth 4-bit 极速推理)
    print("\n[2/3] 加载模型 (Unsloth 4-bit 极速推理) ...")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    model_load_path = str(LORA_PATH) if use_lora else MODEL_NAME

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_load_path,
        max_seq_length=4096,
        load_in_4bit=True,
        local_files_only=use_lora,  # 本地 LoRA 不访问 Hub，避免断网/墙报错
    )
    FastLanguageModel.for_inference(model)  # 开启原生 2x 极速推理模式
    print(f"  已加载模型: {model_load_path}")
    print("  模型加载完成！")

    # 3. 逐条推理
    print(f"\n[3/3] 开始推理 ({NUM_SAMPLES} 条) ...\n")
    results = []

    for i in range(NUM_SAMPLES):
        sample = test_data[i]
        print(f"{'='*60}")
        print(f"样本 {i+1}/{NUM_SAMPLES}")
        print(f"{'='*60}")

        # 构造 chat messages（不含 assistant 回复）
        messages = [
            {"role": "system", "content": sample["instruction"]},
            {"role": "user", "content": sample["input"]},
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        print(f"  Prompt tokens: {prompt_len}")

        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,        # greedy，结果可复现
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = outputs[0][prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 健全性检查
        sanity = check_vasp_sanity(generated_text)

        # 真实输出的前几行（对照用）
        gt_preview = "\n".join(sample["output"].split("\n")[:5])

        print(f"\n  --- 生成结果 (前 10 行) ---")
        gen_lines = generated_text.strip().split("\n")
        for line in gen_lines[:10]:
            print(f"  {line}")
        if len(gen_lines) > 10:
            print(f"  ... (共 {len(gen_lines)} 行)")

        print(f"\n  --- 真实结构 (前 5 行) ---")
        for line in gt_preview.split("\n"):
            print(f"  {line}")

        print(f"\n  --- 健全性检查 ---")
        print(f"  晶格向量: {'✓' if sanity['has_lattice'] else '✗'}")
        print(f"  坐标类型: {'✓' if sanity['has_coord_type'] else '✗'}")
        print(f"  坐标行数: {sanity['coord_lines']}")
        print(f"  可解析:   {'✓' if sanity['parseable'] else '✗'}")

        results.append({
            "index": i,
            "generated": generated_text,
            "ground_truth": sample["output"],
            "sanity": sanity,
        })
        print()

    # 汇总统计
    parseable_count = sum(1 for r in results if r["sanity"]["parseable"])
    print("=" * 60)
    print(f"汇总: {parseable_count}/{NUM_SAMPLES} 条生成结果可解析为 VASP 格式")
    print("=" * 60)

    # 保存结果
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 详细结果已保存: {result_path}")


if __name__ == "__main__":
    main()
