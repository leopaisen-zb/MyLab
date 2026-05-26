import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['MATGEN_DEVICE'] = 'directml'

# Write output to file
output_file = PROJECT_ROOT / "test_output.txt"
with open(output_file, "w") as f:
    f.write("Starting test...\n")
    f.flush()

    from backend.rag_gen import _DEMO_POSCAR
    f.write(f"Demo POSCAR loaded, length: {len(_DEMO_POSCAR)}\n")
    f.flush()

    from backend.quality import validate_structure
    f.write("Validating structure...\n")
    f.flush()
    check = validate_structure(_DEMO_POSCAR)
    f.write(f"Validation result: {check}\n")
    f.flush()

    from backend.eq_predict import predict, load_model
    f.write("Loading model...\n")
    f.flush()
    try:
        load_model()
        f.write("Model loaded successfully\n")
        f.flush()
    except Exception as e:
        f.write(f"Model loading error: {e}\n")
        import traceback
        traceback.print_exc(file=f)
        f.flush()
        sys.exit(1)

    f.write("Testing EQ prediction...\n")
    f.flush()
    try:
        res = predict(_DEMO_POSCAR)
        f.write(f'Prediction result: {res}\n')
        f.flush()
    except Exception as e:
        f.write(f'Prediction error: {e}\n')
        import traceback
        traceback.print_exc(file=f)
        f.flush()

print(f"Output written to {output_file}")