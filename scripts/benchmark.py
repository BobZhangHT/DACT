"""Reproduce the single-process runtime benchmark reported in the paper."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from dgp import generate_dataset
from methods import build_c_backend, run_test


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results_conditional_delay" / "benchmark.json"


def benchmark(*, datasets: int = 100, samples: int = 1000,
              seed: int = 20260809) -> dict:
    """Time DACT, supLM, and CvM on the same pre-generated datasets."""
    build_c_backend()
    data_rng = np.random.default_rng(seed)
    generated = [
        generate_dataset(
            "A1", data_rng, days=180, change_fraction=0.50,
            change_magnitude=0.025,
        )
        for _ in range(datasets)
    ]

    # Load the shared library and numerical kernels before starting the clock.
    warm = generated[0]
    run_test(
        "dact", warm.exposure, warm.deaths, warm.age_scores,
        np.random.default_rng(seed + 1), backend="c", samples=10,
    )

    timings: dict[str, list[float]] = {}
    for method_index, method in enumerate(("dact", "sup_lm", "cvm")):
        elapsed = []
        for replication, data in enumerate(generated):
            rng = np.random.default_rng(
                seed + 100_000 * (method_index + 1) + replication
            )
            started = time.perf_counter()
            run_test(
                method, data.exposure, data.deaths, data.age_scores, rng,
                backend="c", samples=samples,
            )
            elapsed.append(time.perf_counter() - started)
        timings[method] = elapsed

    summary = {
        method: {
            "median_seconds": float(np.median(values)),
            "q1_seconds": float(np.quantile(values, 0.25)),
            "q3_seconds": float(np.quantile(values, 0.75)),
        }
        for method, values in timings.items()
    }
    result = {
        "design": {
            "datasets": datasets,
            "days": 180,
            "scenario": "A1",
            "change_fraction": 0.50,
            "change_magnitude": 0.025,
            "gaussian_paths": samples,
            "backend": "c",
            "processes": 1,
            "seed": seed,
        },
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "summary": summary,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=int, default=100)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()
    result = benchmark(datasets=args.datasets, samples=args.samples, seed=args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
