#!/usr/bin/env python3
"""Generate an inference-speed summary CSV from infer_speed_results.json."""

import json
from pathlib import Path

import pandas as pd


def load_inference_results():
    benchmark_dir = Path(__file__).parent
    infer_results_file = benchmark_dir / "infer_speed_results.json"

    if not infer_results_file.exists():
        raise FileNotFoundError(
            "Missing infer_speed_results.json. Run infer_speed_test.py only if a new "
            "inference benchmark is intended."
        )

    with open(infer_results_file, "r", encoding="utf-8") as file:
        return json.load(file)


def create_comparison_table(infer_results):
    rows = []

    for result in infer_results:
        num_params = result.get("num_parameters", 0)
        batch_size = result.get("batch_size", 64)
        avg_time_ms = result.get("avg_time_per_sample", 0) * 1000
        infer_speed = result.get("samples_per_second", 0)

        rows.append(
            {
                "Model": result.get("model_name", "unknown"),
                "Parameters (M)": f"{num_params / 1e6:.2f}",
                f"Infer Time BS={batch_size} (ms/sample)": (
                    f"{avg_time_ms:.2f}" if avg_time_ms > 0 else "N/A"
                ),
                f"Infer Speed BS={batch_size} (samples/s)": (
                    f"{infer_speed:.2f}" if infer_speed > 0 else "N/A"
                ),
                "Total Samples": result.get("total_samples", "N/A"),
                "Runs": result.get("num_runs", "N/A"),
            }
        )

    return pd.DataFrame(rows)


def print_inference_speedup(df):
    if len(df) < 2:
        return

    original_row = None
    lite_row = None

    for idx, row in df.iterrows():
        model_name = str(row["Model"]).lower()
        if "original" in model_name or "eqv2_best" in model_name:
            original_row = idx
        elif "lite" in model_name:
            lite_row = idx

    if original_row is None or lite_row is None:
        return

    infer_time_cols = [col for col in df.columns if col.startswith("Infer Time")]
    if not infer_time_cols:
        return

    infer_time_col = infer_time_cols[0]
    try:
        original_time = float(df.loc[original_row, infer_time_col])
        lite_time = float(df.loc[lite_row, infer_time_col])
    except (TypeError, ValueError):
        return

    if lite_time <= 0:
        return

    batch_size = "unknown"
    if "BS=" in infer_time_col:
        batch_size = infer_time_col.split("BS=", 1)[1].split(" ", 1)[0]

    speedup = original_time / lite_time
    print(f"Inference speedup (BS={batch_size}): {speedup:.2f}x")


def main():
    benchmark_dir = Path(__file__).parent

    print("Loading inference-speed results...")
    infer_results = load_inference_results()

    print("Generating inference-speed comparison table...")
    df = create_comparison_table(infer_results)

    output_file = benchmark_dir / "speed_compare.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"[OK] Summary saved to: {output_file}")

    print("\n" + "=" * 100)
    print("Inference-Speed Summary")
    print("=" * 100)
    print(df.to_string(index=False))

    print_inference_speedup(df)


if __name__ == "__main__":
    main()
