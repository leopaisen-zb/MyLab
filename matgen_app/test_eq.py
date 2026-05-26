import sys, os, time
sys.path.insert(0, 'matgen_app')
os.environ['MATGEN_DEVICE'] = 'directml'

from backend.rag_gen import _DEMO_POSCAR
from backend.eq_predict import predict

print("Testing Eq prediction...")
print(f"Demo POSCAR: {len(_DEMO_POSCAR)} chars")
t0 = time.time()
res = predict(_DEMO_POSCAR)
print(f"Time: {time.time()-t0:.1f}s")
print(f"Result: {res}")