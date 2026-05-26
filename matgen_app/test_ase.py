from ase import Atoms
import ase.neighborlist

# Check what methods are available on Atoms
atoms = Atoms()
methods = [m for m in dir(atoms) if 'distance' in m.lower() or 'min' in m.lower()]
print("Methods with 'distance' or 'min':", methods)

# Check get_minimum_distances
try:
    from ase.geometry import get_minimum_distance
    print("get_minimum_distance exists in ase.geometry")
except ImportError:
    print("get_minimum_distance NOT in ase.geometry")