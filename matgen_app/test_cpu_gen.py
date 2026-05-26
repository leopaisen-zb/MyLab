import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use CPU instead of DirectML
os.environ['MATGEN_DEVICE'] = 'cpu'

output_file = PROJECT_ROOT / "cpu_test_output.txt"
with open(output_file, "w") as f:
    f.write("=== CPU MODE TEST ===\n")
    f.flush()

    import torch
    f.write(f"PyTorch version: {torch.__version__}\n")
    f.flush()

    from backend.rag_gen import generate, load_model

    f.write("Loading model (CPU mode)...\n")
    f.flush()
    try:
        model, tokenizer = load_model()
        f.write(f"Model loaded: {type(model)}\n")
        f.flush()
    except Exception as e:
        f.write(f"Model loading error: {e}\n")
        import traceback
        traceback.print_exc(file=f)
        f.flush()
        sys.exit(1)

    f.write("Testing generation...\n")
    f.flush()
    try:
        result = generate('Generate H adsorption structure on Ir(111) surface.')
        f.write(f"Generation result length: {len(result)}\n")
        f.write(f"Result preview: {result[:300]}\n")
        f.flush()
    except Exception as e:
        f.write(f"Generation error: {e}\n")
        import traceback
        traceback.print_exc(file=f)
        f.flush()

print(f"Output written to {output_file}")