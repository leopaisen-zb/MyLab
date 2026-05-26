import json
from pathlib import Path

def fix_dataset(file_path: Path):
    if not file_path.exists():
        print(f"找不到文件: {file_path}")
        return
    
    print(f"正在处理: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    new_instruction = (
        "You are an expert AI in materials science and computational chemistry. "
        "Given the physical and chemical properties of a target hydrogen storage material "
        "and several reference structures, write a complete Python function named `generate_structure()` "
        "using the `pymatgen` library to build and return the corresponding atomic structure. "
        "Output ONLY the Python code block."
    )
    
    modified_count = 0
    for item in data:
        # 1. 替换 instruction
        item["instruction"] = new_instruction
        
        # 2. 给 output 加上 Markdown 代码块包裹（如果还没有的话）
        out_text = item.get("output", "").strip()
        if not out_text.startswith("```python"):
            # 如果有残留的裸露 ``` 开头，先剥掉
            if out_text.startswith("```"):
                out_text = out_text.strip("`").strip()
            item["output"] = f"```python\n{out_text}\n```"
            modified_count += 1
            
    # 覆盖保存
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"处理完成。成功修正了 {len(data)} 条指令，补充了 {modified_count} 条 Markdown 格式。\n")

if __name__ == "__main__":
    # 使用你指定的绝对路径
    train_path = Path(r"D:\mylab\RAG\RAG\data\rag_data\text2struct_rag\dataset_train_rag_code.json")
    test_path = Path(r"D:\mylab\RAG\RAG\data\rag_data\text2struct_rag\dataset_test_rag_code.json")
    
    fix_dataset(train_path)
    fix_dataset(test_path)