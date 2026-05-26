import os
import sys
from pathlib import Path

# Add matgen_app to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['MATGEN_DEVICE'] = 'directml'
os.environ['MATGEN_DEMO'] = '1'  # Use demo POSCAR

from backend.rag_gen import generate

print("Testing generation with DEMO mode...")
try:
    result = generate('Generate H adsorption structure on Ir(111) surface with reasonable lattice and coordinates.')
    print(f'RESULT: {repr(result[:500] if result else "EMPTY")}')
except Exception as e:
    print(f'Generation error: {e}')
    import traceback
    traceback.print_exc()

print("\nTesting quality validation...")
try:
    from backend.quality import validate_structure
    check = validate_structure(result)
    print(f'Validation result: {check}')
except Exception as e:
    print(f'Validation error: {e}')
    import traceback
    traceback.print_exc()

print("\nTesting EQ prediction...")
try:
    from backend.eq_predict import predict
    res = predict(result)
    print(f'Prediction result: {res}')
except Exception as e:
    print(f'Prediction error: {e}')
    import traceback
    traceback.print_exc()