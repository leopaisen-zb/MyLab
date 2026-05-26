import sys
import os
sys.path.insert(0, r'h:\BaiduNetdiskDownload\mylab(1)\mylab(1)\matgen_app')
os.environ["PYTHONIOENCODING"] = "utf-8"

results = []

# Test RAG retrieval
try:
    from backend.rag_retrieve import retrieve, build_rag_prompt_minimal, get_rag_stats, init_index

    stats = get_rag_stats()
    results.append(f"RAG Stats: {stats}")

    # Init index
    results.append("Building TF-IDF index...")
    init_index()
    results.append("Index built!")

    # Test queries
    queries = [
        "在 Ir(111) 表面吸附一个 H 原子",
        "生成 IrPdPt 三元合金表面结构",
        "Cu(111) surface with H adsorption",
    ]

    for q in queries:
        refs = retrieve(q, top_k=3)
        results.append(f"\nQuery: {q}")
        results.append(f"Retrieved {len(refs)} refs")
        for r in refs:
            results.append(f"  idx={r['index']}, dg_h={r['dg_h']}, elements={r.get('element_str')}")

        # Build enhanced prompt
        enhanced = build_rag_prompt_minimal(q, refs)
        results.append(f"  Enhanced prompt length: {len(enhanced)} chars")

    results.append("\nSUCCESS: RAG retrieval works!")
except Exception as e:
    results.append(f"ERROR: {e}")
    import traceback
    results.append(traceback.format_exc())

output_path = r'h:\BaiduNetdiskDownload\mylab(1)\mylab(1)\rag_test_output.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(results))
print("Done - check rag_test_output.txt")
