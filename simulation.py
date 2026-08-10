"""Run the daily conditional-CFR simulations with resume and parallelism."""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import CONFIG, RAW_DIR, Backend, Mode, ensure_directories
from dgp import generate_dataset
from methods import (
    bootstrap_change_intervals,
    build_c_backend,
    gradient_change_test,
)

EXPERIMENTS = ("testing", "intervals", "sensitivity")


def _seed(label: str, cell: int, replication: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([
        CONFIG.simulation.master_seed,
        zlib.crc32(label.encode("utf-8")) & 0xFFFFFFFF,
        cell, replication,
    ])


def _apply_test(data, method, rng, backend):
    exposure = data.cases.astype(float) if method == "no_delay" else data.exposure
    active = "proposed_gaussian" if method == "no_delay" else method
    return gradient_change_test(
        exposure, data.deaths, data.age_scores, active, rng,
        backend=backend,
        gaussian_samples=CONFIG.simulation.gaussian_calibration_samples,
    )


def _run_task(task: dict) -> dict:
    started = time.perf_counter()
    rng = np.random.default_rng(_seed(task["experiment"] + ":dgp", task["cell"], task["replication"]))
    method_rng = np.random.default_rng(_seed(task["experiment"] + ":method", task["cell"], task["replication"]))
    try:
        if task["experiment"] == "testing":
            scenario = task["scenario"]
            kwargs = {
                "days": task["days"], "death_scale": task["death_scale"],
                "change_fraction": task["change_fraction"],
            }
            if task["magnitude"] > 0:
                kwargs["change_magnitude"] = task["magnitude"]
            data = generate_dataset(scenario, rng, **kwargs)
            result = _apply_test(data, task["method"], method_rng, task["backend"])
            record = {
                "scenario": scenario, "method": task["method"], "days": task["days"],
                "death_scale": task["death_scale"], "magnitude": task["magnitude"],
                "change_fraction": task["change_fraction"], "rejected": result.rejected,
                "statistic": result.statistic, "p_value": result.p_value,
                "estimated_change": result.estimated_change, "true_change": data.true_change,
                "location_error": None if data.true_change is None else abs(result.estimated_change - data.true_change),
                "failure": not result.null_fit.success,
            }
        elif task["experiment"] == "intervals":
            data = generate_dataset(
                task["scenario"], rng, days=task["days"],
                change_magnitude=task["magnitude"], change_fraction=task["change_fraction"],
            )
            boot = bootstrap_change_intervals(
                data.exposure, data.deaths, data.age_scores, method_rng,
                backend=task["backend"], bootstrap_samples=CONFIG.method.bootstrap_samples,
            )
            estimate = np.asarray(boot["estimate"], dtype=float)
            truth = np.array([
                data.true_change / data.days,
                data.true_beta[data.true_change - 1], data.true_beta[data.true_change],
                data.true_beta[data.true_change] - data.true_beta[data.true_change - 1],
                np.exp(data.true_beta[data.true_change] - data.true_beta[data.true_change - 1]),
            ])
            boot_lower = np.asarray(boot["lower"], dtype=float)
            boot_upper = np.asarray(boot["upper"], dtype=float)
            record = {
                "scenario": task["scenario"], "days": task["days"],
                "magnitude": task["magnitude"], "change_fraction": task["change_fraction"],
                "labels": list(boot["labels"]), "truth": truth.tolist(),
                "estimate": estimate.tolist(), "se": np.asarray(boot["se"]).tolist(),
                "lower": boot_lower.tolist(), "upper": boot_upper.tolist(),
                "covered": ((boot_lower <= truth) & (truth <= boot_upper)).astype(float).tolist(),
                "width": (boot_upper - boot_lower).tolist(),
                "failure": not boot["fit"].success,
            }
        else:
            kwargs = dict(task["kwargs"])
            data = generate_dataset(task["scenario"], rng, **kwargs)
            result = _apply_test(data, "proposed_gaussian", method_rng, task["backend"])
            record = {
                "scenario": task["scenario"], "axis": task["axis"], "value": task["value"],
                "rejected": result.rejected, "estimated_change": result.estimated_change,
                "true_change": data.true_change,
                "location_error": None if data.true_change is None else abs(result.estimated_change - data.true_change),
                "failure": not result.null_fit.success,
            }
        record.update({
            "key": task["key"], "experiment": task["experiment"], "mode": task["mode"],
            "cell": task["cell"], "replication": task["replication"], "backend": task["backend"],
            "runtime_seconds": time.perf_counter() - started, "status": "ok",
        })
        return record
    except Exception as exc:
        return {
            "key": task["key"], "experiment": task["experiment"], "mode": task["mode"],
            "cell": task["cell"], "replication": task["replication"], "backend": task["backend"],
            "runtime_seconds": time.perf_counter() - started, "status": "error",
            "error": repr(exc), "traceback": traceback.format_exc(),
        }


def _cells(experiment: str, mode: Mode) -> list[dict]:
    if experiment == "testing":
        nulls = [
            {"scenario": s, "days": d, "death_scale": q, "magnitude": 0.0,
             "change_fraction": 0.50, "method": m}
            for s in ("N0", "N1", "N2", "N3") for d in (90, 180, 365)
            for q in (0.35, 1.0, 1.5) for m in CONFIG.simulation.methods
        ]
        alternatives = [
            {"scenario": s, "days": d, "death_scale": 1.0, "magnitude": mag,
             "change_fraction": frac, "method": m}
            for s in ("A1", "A2") for d in (90, 180, 365)
            for mag in (0.015, 0.025, 0.035, 0.050)
            for frac in (0.25, 0.50, 0.75) for m in CONFIG.simulation.methods
        ]
        # Keep the original early-change cells first so that --resume preserves
        # their existing cell identifiers, then add the central and late cases.
        sparse_early = [
            {"scenario": "A3", "days": d, "death_scale": 1.0,
             "magnitude": 0.025, "change_fraction": 0.25, "method": m}
            for d in (90, 180, 365) for m in CONFIG.simulation.methods
        ]
        sparse_other = [
            {"scenario": "A3", "days": d, "death_scale": 1.0,
             "magnitude": 0.025, "change_fraction": frac, "method": m}
            for frac in (0.50, 0.75) for d in (90, 180, 365)
            for m in CONFIG.simulation.methods
        ]
        return nulls + alternatives + sparse_early + sparse_other
    if experiment == "intervals":
        regular = [
            {"scenario": s, "days": 180, "magnitude": mag, "change_fraction": 0.50}
            for s in ("A1", "A2") for mag in (0.025, 0.050)
        ]
        sparse = [
            {"scenario": "A3", "days": 180, "magnitude": mag, "change_fraction": 0.25}
            for mag in (0.025, 0.050)
        ]
        return regular + sparse
    if experiment == "sensitivity":
        cells = []
        for value in CONFIG.simulation.delay_multiplier_grid:
            cells += [
                {"scenario": s, "axis": "delay_multiplier", "value": value,
                 "kwargs": {"change_magnitude": 0.0 if s == "N0" else 0.025,
                            "analysis_delay_multiplier": value}}
                for s in ("N0", "A1")
            ]
        for value in CONFIG.simulation.transition_width_grid:
            cells.append({"scenario": "A1", "axis": "transition_width", "value": value,
                          "kwargs": {"change_magnitude": 0.025, "transition_width": value}})
        for value in CONFIG.simulation.overdispersion_grid:
            cells += [{"scenario": s, "axis": "overdispersion", "value": value,
                       "kwargs": {"change_magnitude": 0.0 if s == "N0" else 0.025,
                                  "overdispersion": value}}
                      for s in ("N0", "A1")]
        for value in CONFIG.simulation.age_group_grid:
            cells += [{"scenario": s, "axis": "age_groups", "value": value,
                       "kwargs": {"change_magnitude": 0.0 if s == "N0" else 0.025,
                                  "age_groups": value}}
                      for s in ("N0", "A1")]
        for value in (0.0, 0.10, 0.20, 0.30):
            cells += [{"scenario": s, "axis": "day_of_week", "value": value,
                       "kwargs": {"change_magnitude": 0.0 if s == "N0" else 0.025,
                                  "day_of_week": value}}
                      for s in ("N0", "A1")]
        for value in (0.0, 0.10, 0.20, 0.30):
            cells += [{"scenario": s, "axis": "age_delay_mismatch", "value": value,
                       "kwargs": {"change_magnitude": 0.0 if s == "N0" else 0.025,
                                  "age_specific_delay": value,
                                  "analyze_common_delay": True}}
                      for s in ("N0", "A1")]
        return cells
    raise ValueError(experiment)


def make_tasks(experiment: str, mode: Mode, backend: Backend) -> list[dict]:
    tasks = []
    for cell, values in enumerate(_cells(experiment, mode)):
        for replication in range(CONFIG.simulation.replications(mode)):
            tasks.append({
                "key": f"{experiment}:{cell}:{replication}", "experiment": experiment,
                "cell": cell, "replication": replication, "mode": mode,
                "backend": backend, **values,
            })
    return tasks


def _completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    output = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            output.add(json.loads(line)["key"])
        except (json.JSONDecodeError, KeyError):
            pass
    return output


def run_experiment(experiment: str, mode: Mode, backend: Backend, workers: int, resume: bool):
    path = RAW_DIR / f"{mode}_{experiment}.jsonl"
    done = _completed(path) if resume else set()
    tasks = [x for x in make_tasks(experiment, mode, backend) if x["key"] not in done]
    if not resume:
        path.write_text("", encoding="utf-8")
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        if workers == 1:
            for record in tqdm(map(_run_task, tasks), total=len(tasks), desc=experiment):
                stream.write(json.dumps(record, allow_nan=True) + "\n")
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                # Chunked map avoids constructing hundreds of thousands of
                # Future objects and sharply reduces inter-process overhead.
                records = pool.map(_run_task, tasks, chunksize=32)
                for record in tqdm(records, total=len(tasks), desc=experiment):
                    stream.write(json.dumps(record, allow_nan=True) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("demo", "full"), required=True)
    parser.add_argument("--experiments", nargs="+", choices=("all",) + EXPERIMENTS, default=["all"])
    parser.add_argument("--backend", choices=("c", "python"), default=CONFIG.method.backend)
    parser.add_argument("--workers", type=int, default=max(1, min(20, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    ensure_directories()
    if args.backend == "c":
        build_c_backend()
    experiments = EXPERIMENTS if "all" in args.experiments else tuple(args.experiments)
    for experiment in experiments:
        run_experiment(experiment, args.mode, args.backend, args.workers, args.resume)


if __name__ == "__main__":
    main()
