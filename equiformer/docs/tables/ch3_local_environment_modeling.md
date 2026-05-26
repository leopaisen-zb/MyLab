# Local Chemical Environment Modeling Evidence

The project uses `max_radius=12.0 Å`, not `Rc=6.0 Å`. The `Rc=6.0 Å` statement in the thesis draft should be corrected.

## Graph Construction

- `datasets/custom_data_processor_simplified.py` builds a distance graph from atomic coordinates.
- Each atom connects to neighbors within `max_radius=12.0 Å`.
- Neighbor count is capped by `max_neighbors`.
- Prepared dataset summary records `max_radius=12.0` and `max_neighbors=50`.
- Model training and benchmark configurations use `max_radius=12.0` and `max_neighbors=20`.

## Chemical Identity Encoding

- Each structure stores `atomic_numbers`.
- `src/standalone_equiformer_v2.py` maps atomic numbers through `sphere_embedding`.
- EquiformerV2 edge blocks support source/target atomic embeddings through `use_atom_edge_embedding=True`.
- These embeddings let the model distinguish multi-element local coordination environments.

## Local Geometry Encoding

- Edge vectors and edge distances are computed from atomic positions.
- Distances are expanded by Gaussian radial basis functions.
- SO3/SO2 blocks process equivariant angular information from local neighborhoods.

## Tabular Local Descriptors

The prepared dataset also contains local descriptor columns such as:

- `CN`
- `L_bond`
- `R0`, `R1`, `R`
- element-shell descriptors including `Nd`, `Np`, `Ns`, `Out_e`, `First_IE`

These descriptors are used in the structure-plus-tabular fusion experiments and can be cited as auxiliary evidence that the workflow explicitly represents high-entropy-alloy local coordination and chemical heterogeneity.
