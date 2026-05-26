import sys, os, time
sys.path.insert(0, 'matgen_app')
os.environ['MATGEN_DEVICE'] = 'directml'

output = open('matgen_app/eq_test.txt', 'w')
output.write("Testing Eq prediction...\n")
output.flush()

from backend.rag_gen import _DEMO_POSCAR
output.write(f"Demo POSCAR: {len(_DEMO_POSCAR)} chars\n")
output.flush()

from backend.eq_predict import predict
output.write("Calling predict()...\n")
output.flush()

t0 = time.time()
res = predict(_DEMO_POSCAR)
elapsed = time.time() - t0
output.write(f"Time: {elapsed:.1f}s\n")
output.write(f"Result: {res}\n")
output.flush()
output.close()
print(f"Done in {elapsed:.1f}s. Result: {res}")