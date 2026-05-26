import json
from pathlib import Path

# Check RAG data structures
DATA_DIR = Path(r'h:\BaiduNetdiskDownload\mylab(1)\mylab(1)\RAG\RAG\rag_data')

# 1. Check near_neighbor_indices
nn_path = DATA_DIR / 'text2struct_rag' / 'near_neighbor_indices.json'
with open(nn_path) as f:
    nn_data = json.load(f)
print(f"near_neighbor_indices: {len(nn_data)} entries")
print(f"  first 3: {nn_data[:3]}")
print(f"  entry 0 type: {type(nn_data[0])}")

# 2. Check training data
train_path = DATA_DIR / 'text_rag' / 'dataset_train_rag.json'
with open(train_path) as f:
    train_data = json.load(f)
print(f"\ndataset_train_rag: {len(train_data)} entries")
print(f"  entry 0 keys: {list(train_data[0].keys())}")
print(f"  entry 0 sample:")
for k, v in list(train_data[0].items())[:4]:
    v_str = str(v)[:100]
    print(f"    {k}: {v_str}")

# 3. Check text2struct RAG
text2struct_path = DATA_DIR / 'text2struct_rag' / 'dataset_train_rag_code.json'
with open(text2struct_path) as f:
    t2s_data = json.load(f)
print(f"\ndataset_train_rag_code: {len(t2s_data)} entries")
print(f"  entry 0 keys: {list(t2s_data[0].keys())}")

# 4. How many reference structures per query?
print(f"\nNeighbor info:")
print(f"  Total queries: {len(nn_data)}")
print(f"  For entry 0 (nearest neighbor = {nn_data[0]}):")
if nn_data[0] < len(train_data):
    ref = train_data[nn_data[0]]
    print(f"    ref elements: {ref.get('output', '')[:50]}")
