import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['MATGEN_DEVICE'] = 'directml'
os.environ['MATGEN_DEMO'] = '1'

from backend.rag_gen import generate

print("Testing DEMO generation...")
result = generate('test prompt')
print(f'Result length: {len(result)}')
print(f'Result: {result[:200]}')