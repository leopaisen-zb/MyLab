import json
from transformers import AutoTokenizer

LOCAL_MODEL_PATH = "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
TRAIN_JSON = "/home/ubuntu/HZH/RAG/RAG/data/rag_data/text2struct_rag/dataset_train_rag_code.json"

print("正在加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)

print("正在加载测试数据...")
with open(TRAIN_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)
sample = data[0]

messages = [
    {"role": "system", "content": sample["instruction"]},
    {"role": "user", "content": sample["input"]},
    {"role": "assistant", "content": sample["output"]},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

# 模拟 CustomCompletionCollator 的逻辑
response_template = "<|im_start|>assistant\n"
response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)

print(f"\n[目标模板 Token IDs]: {response_token_ids}")

input_ids = tokenizer.encode(text, add_special_tokens=False)
labels = input_ids.copy()

search_len = len(response_token_ids)
found_idx = -1
for j in range(len(labels) - search_len + 1):
    if labels[j:j+search_len] == response_token_ids:
        found_idx = j + search_len
        break

print(f"[查找结果]: found_idx = {found_idx}")

if found_idx != -1:
    labels[:found_idx] = [-100] * found_idx
    print("\n[成功] 找到了 assistant 的起始位置。")
else:
    print("\n[致命错误] 无法在输入序列中找到 response_template 的 Token 序列！")

# 提取真正计算 Loss 的部分并解码
valid_labels = [l for l in labels if l != -100]
decoded_loss_text = tokenizer.decode(valid_labels)

print("-" * 60)
print("模型实际计算 Loss 的文本内容：")
print("-" * 60)
print(decoded_loss_text)
print("-" * 60)