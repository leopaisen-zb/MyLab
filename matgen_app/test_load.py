import os
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['MATGEN_DEVICE'] = 'directml'

output_file = PROJECT_ROOT / "load_test.txt"
with open(output_file, "w") as f:
    f.write("=== MODEL LOADING TEST ===\n")
    f.write(f"Time: {time.strftime('%H:%M:%S')}\n")
    f.flush()

    from backend.rag_gen import load_model, get_device
    dml = get_device()
    f.write(f"Device: {dml}\n")
    f.flush()

    f.write("Loading model...\n")
    f.flush()
    t0 = time.time()
    try:
        model, tokenizer = load_model()
        f.write(f"Model loaded in {time.time()-t0:.1f}s\n")
        f.write(f"Model type: {type(model)}\n")
        f.write(f"Model device: {next(model.parameters()).device}\n")
        f.flush()
    except Exception as e:
        f.write(f"Model loading error: {e}\n")
        import traceback
        traceback.print_exc(file=f)
        f.flush()

print(f"Output: {output_file}")
