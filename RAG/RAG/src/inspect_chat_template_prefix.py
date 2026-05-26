import json
from pathlib import Path

from unsloth import FastLanguageModel

# 和 train_sft.py 保持一致
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_JSON = PROJECT_ROOT / "data" / "rag_data" / "text2struct_rag" / "dataset_train_rag_code.json"

_LOCAL_MODEL_PATH = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots" / "a09a35458c702b33eeacc393d103063234e8bc28"
MODEL_NAME = str(_LOCAL_MODEL_PATH) if _LOCAL_MODEL_PATH.is_dir() else "Qwen/Qwen2.5-7B-Instruct"


def main() -> None:
    # 1. 读数据
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"样本数: {len(data)}")

    sample_idx = 0  # 你也可以改成别的索引
    sample = data[sample_idx]
    print(f"查看样本索引: {sample_idx}")

    # 2. 加载 tokenizer（用 FastLanguageModel 确保和训练一致）
    print(f"加载 tokenizer: {MODEL_NAME}")
    _, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=8192,   # 和 train_sft.py 相同即可
        load_in_4bit=False,
    )

    # 3. 用 chat_template 构造文本
    messages = [
        {"role": "system", "content": sample["instruction"]},
        {"role": "user", "content": sample["input"]},
        {"role": "assistant", "content": sample["output"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # 4. 打印前一段内容（用 repr 看到所有换行和特殊 token）
    print("\n==== chat 模板前 400 字符 repr ====")
    print(repr(text[:400]))
        # ==== 精确找出分隔符 ====
    print("\n==== 查找各 role 的分隔符（前后 80 字符 repr） ====")
    for role in ["system", "user", "assistant"]:
        marker = f"<|im_start|>{role}"
        idx = text.find(marker)
        if idx == -1:
            print(f"{role}: 未找到 '{marker}'")
            continue
        # 从 marker 开始，往后截一小段出来看（包含 \n 等）
        snippet = text[idx:idx + 80]
        print(f"{role} 起始位置: {idx}")
        print(repr(snippet))

    # 额外：单独把 user / assistant 的“分隔符 + 换行”截出来，方便直接复制
    print("\n==== 可直接用于 train_on_responses_only 的分隔符猜测 ====")
    for role in ["user", "assistant"]:
        marker = f"<|im_start|>{role}"
        idx = text.find(marker)
        if idx == -1:
            continue
        # 从 marker 开始，一直到第一个换行符（含 \n）
        end_idx = text.find("\n", idx)
        if end_idx == -1:
            end_idx = idx + 40  # 兜底
        delim = text[idx:end_idx + 1]
        print(f"{role}_part = {repr(delim)}")

    # 顺带把完整文本的前 400 字符原样打印一份，方便肉眼看
    print("\n==== chat 模板前 400 字符（原样） ====")
    print(text[:400])


if __name__ == "__main__":
    main()