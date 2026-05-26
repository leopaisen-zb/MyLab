import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['MATGEN_DEVICE'] = 'directml'

from backend.rag_gen import _DEMO_POSCAR
from backend.eq_predict import predict, load_model

print("Loading model...")
try:
    load_model()
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Testing EQ prediction...")
try:
    res = predict(_DEMO_POSCAR)
    print(f'Prediction result: {res}')
except Exception as e:
    print(f'Prediction error: {e}')
    import traceback
    traceback.print_exc()