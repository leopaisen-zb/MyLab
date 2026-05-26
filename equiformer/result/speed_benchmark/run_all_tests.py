#!/usr/bin/env python3
"""Run the maintained Chapter 3 inference-speed benchmark utilities."""

import subprocess
import sys
import time
from pathlib import Path


def run_script(script_name):
    script_path = Path(__file__).parent / script_name

    if not script_path.exists():
        print(f"[SKIP] Missing script: {script_name}")
        return False

    print("\n" + "=" * 60)
    print(f"Running: {script_name}")
    print("=" * 60)

    start_time = time.time()
    try:
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=Path(__file__).parent,
            check=True,
            capture_output=False,
        )
    except subprocess.CalledProcessError as exc:
        elapsed_time = time.time() - start_time
        print(f"[FAIL] {script_name} failed after {elapsed_time:.1f}s: {exc}")
        return False
    except KeyboardInterrupt:
        print(f"[STOP] {script_name} interrupted")
        return False

    elapsed_time = time.time() - start_time
    print(f"[OK] {script_name} finished in {elapsed_time:.1f}s")
    return True


def main():
    print("=" * 60)
    print("EquiformerV2 inference-speed utilities")
    print("=" * 60)

    scripts = [
        "infer_speed_test.py",
        "generate_comparison.py",
    ]

    results = []
    total_start_time = time.time()

    for script in scripts:
        results.append((script, run_script(script)))

    total_time = time.time() - total_start_time

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for script, success in results:
        status = "OK" if success else "FAILED"
        print(f"{script}: {status}")
    print(f"Total time: {total_time:.1f}s")

    if not all(success for _, success in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
