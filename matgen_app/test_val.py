import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['MATGEN_DEVICE'] = 'directml'

from backend.rag_gen import _DEMO_POSCAR
from backend.quality import validate_structure

print("Testing quality validation...")
try:
    check = validate_structure(_DEMO_POSCAR)
    print(f'Validation result: {check}')
except Exception as e:
    print(f'Validation error: {e}')
    import traceback
    traceback.print_exc()