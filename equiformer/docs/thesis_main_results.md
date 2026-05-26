# Thesis Main Experimental Results

This file collects the main thesis-facing experimental results in one place. It is a citation index and interpretation guide; original result files remain the source of record.

Naming note: historical code and result filenames use `EnhancedEquiformerV2` / `enhanced_equiformer_v2`, but the implementation is a thin wrapper around the original `EquiformerV2_OC20` class. In thesis-facing text, this is referred to as `Original Eqv2 / EquiformerV2`. Existing directory names are left unchanged for traceability.

## Overall Conclusion

The experiments support Eqv2-Lite as the main model for hydrogen adsorption energy prediction on the current high-entropy-alloy surface dataset. Compared with the original Eqv2 / EquiformerV2 baseline, Eqv2-Lite uses far fewer parameters, runs much faster, and achieves better predictive accuracy. The ablations further show that the task benefits from a capacity-matched equivariant model rather than simply increasing model size.

The latest parameter reallocation experiment does not support a strong claim that FFN-heavy reallocation is optimal. In the current single-seed run, increasing edge/local-geometry capacity gives the best result among the reallocation variants.

## 1. Main Accuracy And Efficiency Result

| Model | Role | Params | R2 | MAE | RMSE | Batch size | Total samples | Runs | Avg time / sample | Throughput | Speedup | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Eqv2-Lite | Main method | 4.69M | 0.9334 | 0.0868 | 0.1798 | 64 | 1037 | 17 | 0.0115 s | 86.99 samples/s | 27.78x | `/home/leo494/mylab/实验结果/实验结果摘要.csv`; `result/speed_benchmark/infer_speed_results.json` |
| Original Eqv2 / EquiformerV2 | Original baseline, untuned | 38.82M | 0.8814 | 0.1320 | 0.2325 | 64 | 1037 | 17 | 0.3193 s | 3.13 samples/s | 1.00x | `/home/leo494/mylab/实验结果/实验结果摘要.csv`; `result/speed_benchmark/infer_speed_results.json` |
| Original Eqv2 / EquiformerV2, adapted config | Tuned/adapted run for fusion base | not benchmarked | 0.8951 | 0.1108 | 0.2186 | not benchmarked | not benchmarked | not benchmarked | not benchmarked | not benchmarked | not benchmarked | `/home/leo494/mylab/实验结果/实验结果摘要.csv`; `experiments/2025-09-10_run1/ablation/num_layers=2__sphere_channels=64__num_heads=4__grid_resolution=16__edge_channels=64__REAL_FUSION_BASE/seed=42/logs/enhanced_equiformer_v2_test_results.json` |

Thesis-ready interpretation:

Eqv2-Lite reduces parameters by about 8.29x and improves throughput by about 27.78x relative to the original untuned Eqv2 / EquiformerV2 baseline, while also improving test MAE from 0.1320 eV to 0.0868 eV. The adapted original Eqv2 run improves over the untuned original baseline, but still underperforms Eqv2-Lite. This supports the claim that the lightweight architecture is a better capacity match for the current dataset, not only an efficiency optimization.

The adapted original Eqv2 run is labeled as a tuned/adapted configuration rather than pretrained fine-tuning. `src/train_enhanced_equiformer_v2.py` exposes `--pretrained` and `--pretrained_ckpt` arguments, but the training path used for these results does not load a pretrained checkpoint before training.

## 2. Fusion Branch Comparison

| Model / Experiment | R2 | MAE | RMSE | Source |
| --- | ---: | ---: | ---: | --- |
| Eqv2-Lite | 0.9334 | 0.0868 | 0.1798 | `/home/leo494/mylab/实验结果/实验结果摘要.csv` |
| Original Eqv2 / EquiformerV2, adapted config | 0.8951 | 0.1108 | 0.2186 | `experiments/2025-09-10_run1/ablation/num_layers=2__sphere_channels=64__num_heads=4__grid_resolution=16__edge_channels=64__REAL_FUSION_BASE/seed=42/logs/enhanced_equiformer_v2_test_results.json` |
| Eqv2 + tabular branch, gate fusion | 0.9093 | 0.1024 | 0.1857 | `experiments/20251021_165820_tabfusion_run_gate_fusion_100epochs/metrics.csv` |
| Eqv2 + tabular branch, concat fusion | 0.9083 | 0.1049 | 0.1867 | `experiments/20251021_160614_tabfusion_run_real_equiformer_with_loss/metrics.csv` |
| Original Eqv2 / EquiformerV2 | 0.8814 | 0.1320 | 0.2325 | `/home/leo494/mylab/实验结果/实验结果摘要.csv` |

Thesis-ready interpretation:

The adapted original Eqv2 configuration improves over the untuned original baseline. The tabular fusion branches further improve over the original Eqv2 structural branch but do not surpass Eqv2-Lite. This supports using Eqv2-Lite as the primary Chapter 3 model and treating the original Eqv2 tuning and fusion branch as comparative extensions rather than the final main method.

## 3. Depth Ablation

| Setting | R2 | MAE | RMSE | Params | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| `num_layers=1` | 0.9055 | 0.1035 | 0.2142 | 3.57M | `result/eq_lite_ablation/depth/summary.csv` |
| `num_layers=2` | 0.9174 | 0.1025 | 0.2002 | 6.75M | `result/eq_lite_ablation/depth/summary.csv` |
| `num_layers=3` | 0.9228 | 0.0943 | 0.1936 | 9.94M | `result/eq_lite_ablation/depth/summary.csv` |
| `num_layers=4` | 0.9101 | 0.0981 | 0.2088 | 13.12M | `result/eq_lite_ablation/depth/summary.csv` |

Thesis-ready interpretation:

Three layers give the best result in this depth sweep. Increasing to four layers adds parameters but reduces performance, supporting the conclusion that excessive depth is not beneficial for this dataset.

## 4. Angular Order Ablation

| Setting | R2 | MAE | RMSE | Params | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| `lmax=3` | 0.9087 | 0.1041 | 0.2105 | 12.99M | `result/eq_lite_ablation/lmax/summary.csv` |
| `lmax=4` | 0.9280 | 0.0996 | 0.1870 | 19.49M | `result/eq_lite_ablation/lmax/summary.csv` |
| `lmax=5` | 0.9119 | 0.1076 | 0.2068 | 27.46M | `result/eq_lite_ablation/lmax/summary.csv` |
| `lmax=6` | 0.9104 | 0.1138 | 0.2085 | 36.91M | `result/eq_lite_ablation/lmax/summary.csv` |

Thesis-ready interpretation:

`lmax=4` gives the best balance in the angular-order sweep. Higher angular orders substantially increase parameter count but degrade MAE/RMSE, supporting the model-pruning rationale.

## 5. Radial Basis Ablation

| Setting | R2 | MAE | RMSE | Params | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| `num_gaussians=32` | 0.9228 | 0.1023 | 0.1936 | 19.40M | `result/eq_lite_ablation/radial/summary.csv` |
| `num_gaussians=64` | 0.8970 | 0.1132 | 0.2235 | 19.43M | `result/eq_lite_ablation/radial/summary.csv` |
| `num_gaussians=128` | 0.9179 | 0.0971 | 0.1996 | 19.49M | `result/eq_lite_ablation/radial/summary.csv` |
| `num_gaussians=256` | 0.9270 | 0.0954 | 0.1882 | 19.61M | `result/eq_lite_ablation/radial/summary.csv` |

Thesis-ready interpretation:

The radial basis sweep supports using a sufficiently expressive radial representation, with `num_gaussians=256` giving the best result among tested settings. The non-monotonic trend also shows that performance is not determined by parameter count alone.

## 6. Channel Ablation

| Setting | R2 | MAE | RMSE | Params | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| `sphere_channels=32` | 0.9118 | 0.1022 | 0.2070 | 8.12M | `result/eq_lite_ablation/channels/summary.csv` |
| `sphere_channels=64` | 0.9213 | 0.0981 | 0.1955 | 11.91M | `result/eq_lite_ablation/channels/summary.csv` |
| `sphere_channels=128` | 0.9348 | 0.0887 | 0.1779 | 19.49M | `result/eq_lite_ablation/channels/summary.csv` |

Thesis-ready interpretation:

Increasing sphere channels improves this sweep, but the result should be discussed together with the main model and efficiency table because the thesis objective is accuracy-efficiency balance, not maximum parameter count.

## 7. Parameter Reallocation Ablation

| Variant | R2 | MAE | RMSE | Params | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| `edge_heavy` | 0.9325 | 0.0961 | 0.1811 | 5.85M | `result/eq_lite_ablation/parameter_reallocation/summary.csv` |
| `ffn_light` | 0.9263 | 0.1040 | 0.1891 | 4.54M | `result/eq_lite_ablation/parameter_reallocation/summary.csv` |
| `ffn_heavy` | 0.9204 | 0.0991 | 0.1965 | 5.18M | `result/eq_lite_ablation/parameter_reallocation/summary.csv` |
| `attention_heavy` | 0.9183 | 0.1015 | 0.1991 | 7.82M | `result/eq_lite_ablation/parameter_reallocation/summary.csv` |
| `balanced` | 0.9165 | 0.1046 | 0.2013 | 6.77M | `result/eq_lite_ablation/parameter_reallocation/summary.csv` |
| `radial_heavy` | 0.8958 | 0.1132 | 0.2249 | 4.99M | `result/eq_lite_ablation/parameter_reallocation/summary.csv` |

Thesis-ready interpretation:

The reallocation experiment supports emphasizing local edge/geometric information more than simply increasing FFN capacity. The current result should be described as single-seed evidence unless additional seeds are run.

## Recommended Thesis Claims

Supported:

- Eqv2-Lite achieves a better accuracy-efficiency trade-off than the original Eqv2 / EquiformerV2 baseline on the current hydrogen adsorption dataset.
- The lightweight design is justified by both performance and inference efficiency.
- Moderate architecture settings are preferable to simply increasing model complexity.
- Local geometric/edge representation is important for the task.

Avoid or weaken:

- Do not claim that FFN-heavy parameter reallocation is the best design. The current reallocation experiment favors `edge_heavy`.
- Do not claim that larger equivariant models always improve performance. Several ablations show degradation after increasing capacity.

## Related Index Files

- `docs/results_manifest.md`
- `docs/chapter3_revision_evidence.md`
- `docs/tables/ch3_param_perf_speed.csv`
- `docs/tables/ch3_architecture_ablation.csv`
- `docs/tables/ch3_local_environment_modeling.md`
