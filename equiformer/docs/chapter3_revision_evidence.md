# Chapter 3 Revision Evidence

Naming note: historical scripts and artifacts use `EnhancedEquiformerV2`, but the implementation directly wraps the original `EquiformerV2_OC20` class. Thesis-facing text should refer to this line of experiments as `Original Eqv2 / EquiformerV2`.

## 1. Lightweight Design Necessity

The existing results show that the original Eqv2 / EquiformerV2 baseline is not the best choice for this medium-sized hydrogen adsorption dataset. Compared with the original Eqv2 / EquiformerV2 baseline, Eqv2-Lite reduces parameter count from 38.82M to 4.69M, improves inference throughput from 3.13 to 86.99 samples/s, and improves test performance from MAE 0.1320 eV / RMSE 0.2325 eV / R2 0.8814 to MAE 0.0868 eV / RMSE 0.1798 eV / R2 0.9334.

This supports the revision argument that the lightweight architecture is not only an efficiency optimization, but also a better capacity match for the current high-entropy-alloy hydrogen adsorption dataset.

Table source: `docs/tables/ch3_param_perf_speed.csv`.

## 2. Architecture Pruning Rationale

The existing ablations support the pruning choices:

- Angular order: Lmax=4 gives the best balance. Lmax=5 and Lmax=6 increase parameters to 27.46M and 36.91M but degrade MAE/RMSE.
- Depth: 3 layers outperform 1, 2, and 4 layers in the Eqv2-Lite depth ablation, showing that simply deepening the model is not beneficial.
- Radial basis: 256 Gaussian bases gives the best radial result in the existing sweep.
- Module switches: attention renormalization and separable S2 operations improve the tested baseline, while separate layer normalization is not necessary in this dataset.

Table source: `docs/tables/ch3_architecture_ablation.csv`.

## 3. Adaptation To High-Entropy-Alloy HER Surface Configurations

The model is adapted to the target system through graph-based local environment encoding. Atomic coordinates define local neighborhoods, atomic numbers encode multi-element chemical identity, and distance/edge features encode local coordination geometry. The dataset also includes local coordination descriptors such as CN, bond length, and atomic-radius-related features, which are used by the fusion branch as auxiliary descriptors.

The thesis should correct `Rc=6.0 Å` to `max_radius=12.0 Å` for the current project evidence, and explain that the graph uses a radius cutoff plus neighbor cap to cover local coordination shells around adsorption structures.

Detailed source anchors: `docs/tables/ch3_local_environment_modeling.md`.
