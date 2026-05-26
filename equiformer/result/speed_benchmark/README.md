# EquiformerV2 Inference Speed Benchmark

This directory keeps the Chapter 3 inference-speed comparison between the full Enhanced EquiformerV2 baseline and Eqv2-Lite.

## Files

- `eqv2_best.json`: full Enhanced EquiformerV2 baseline configuration.
- `eq_lite.json`: Eqv2-Lite configuration.
- `infer_speed_test.py`: inference-speed benchmark script.
- `infer_speed_results.json`: preserved inference-speed benchmark output.
- `generate_comparison.py`: generates `speed_compare.csv` from inference-speed results.
- `speed_compare.csv`: optional summary table generated from `infer_speed_results.json`.

## Scope

Chapter 3 uses inference speed as the maintained efficiency comparison. This directory should not be used as evidence for model optimization during training.

## Usage

To regenerate the summary table from the existing inference-speed JSON:

```bash
python generate_comparison.py
```

To rerun the inference benchmark manually:

```bash
python infer_speed_test.py
```

Existing result files should be preserved unless a new benchmark run is intentional and documented.

## Output Fields

`infer_speed_results.json` contains:

- `model_name`: model identifier.
- `num_parameters`: parameter count.
- `batch_size`: benchmark batch size.
- `avg_time`: average inference time per batch.
- `std_time`: standard deviation of batch inference time.
- `avg_time_per_sample`: average inference time per sample.
- `samples_per_second`: inference throughput.
- `total_samples`: number of samples used in the benchmark.
- `num_runs`: number of timed batches/runs.
