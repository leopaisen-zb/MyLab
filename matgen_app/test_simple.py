import os
os.environ['MATGEN_DEVICE'] = 'directml'

from backend.rag_gen import _DEMO_POSCAR
print("Demo POSCAR length:", len(_DEMO_POSCAR))
print("Demo POSCAR:", _DEMO_POSCAR[:200])