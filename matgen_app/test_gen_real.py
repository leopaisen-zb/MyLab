import os
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['MATGEN_DEVICE'] = 'cpu'
os.environ['MATGEN_DEMO'] = '0'

output_file = PROJECT_ROOT / "gen_real_output.txt"
with open(output_file, "w") as f:
    f.write("=== REAL LLM GENERATION TEST ===\n")
    f.write(f"Time: {time.strftime('%H:%M:%S')}\n")
    f.flush()

    from backend.rag_gen import generate, load_model
    from config import BASE_MODEL_PATH

    f.write(f"Base model: {BASE_MODEL_PATH}\n")
    f.write("Loading model...\n")
    f.flush()

    t0 = time.time()
    try:
        model, tokenizer = load_model(BASE_MODEL_PATH)
        f.write(f"Model loaded in {time.time()-t0:.1f}s\n")
        f.write(f"Model type: {type(model)}\n")
        f.flush()
    except Exception as e:
        f.write(f"Model loading error: {e}\n")
        import traceback
        traceback.print_exc(file=f)
        f.flush()
        sys.exit(1)

    test_prompt = "Generate H adsorption structure on Ir(111) surface with reasonable lattice and coordinates."
    f.write(f"\nGenerating with prompt: {test_prompt}\n")
    f.flush()

    t0 = time.time()
    try:
        result = generate(test_prompt, base_model_name_or_path=BASE_MODEL_PATH)
        elapsed = time.time() - t0
        f.write(f"Generation done in {elapsed:.1f}s\n")
        f.write(f"Result length: {len(result)} chars\n")
        f.write(f"Result preview:\n{result[:500]}\n")
        f.flush()
    except Exception as e:
        f.write(f"Generation error: {e}\n")
        import traceback
        traceback.print_exc(file=f)
        f.flush()

print(f"Output written to {output_file}")
