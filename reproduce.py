"""Master entry point for the Biometrical Journal reproduction workflow.

Run from any working directory; project paths are resolved from this file.
smoke/spot verify saved replications, cached redraws manuscript displays,
demo/full execute the study scripts, and environment records software versions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "reproducibility" / "reference"
STEPS = ("simulations", "analysis", "application", "sensitivity", "benchmark")


def environment() -> dict:
    """Return OS, Python, dependency, compiler and revision metadata.

    No arguments are required. Compiler discovery does not compile or install
    software. The return value is JSON-compatible and omits host directory paths.
    """
    from methods import _compiler

    packages = ("numpy", "scipy", "pandas", "matplotlib", "pyarrow", "tqdm", "pytest")
    versions = {name: importlib.metadata.version(name) for name in packages}
    compiler = None
    try:
        executable = _compiler()
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        compiler = {"name": Path(executable).name, "version": result.stdout.splitlines()[0]}
    except (RuntimeError, OSError, subprocess.SubprocessError):
        pass
    revision = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if (ROOT / ".git").exists() and shutil.which("git")
        else None
    )
    status = (
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if revision is not None
        else None
    )
    sources = [
        "config.py",
        "dgp.py",
        "methods.py",
        "simulation.py",
        "analysis.py",
        "realdata.py",
        "reproduce.py",
        "csrc/cfr_core.c",
        "scripts/realdata_sensitivity.py",
        "scripts/benchmark.py",
    ]
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": versions,
        "compiler": compiler,
        "revision": revision.stdout.strip() if revision and revision.returncode == 0 else None,
        "working_tree_modified": bool(status.stdout.strip()) if status is not None else None,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources
        },
    }


def verify_reference(directory: Path = REFERENCE) -> dict:
    """Check all frozen reference files in directory against its SHA-256 manifest.

    Returns the parsed manifest. Missing or changed files raise ValueError.
    """
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        path = directory / name
        if path.parent != directory or not path.is_file():
            raise ValueError(f"Missing reference file: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Reference checksum mismatch: {name}")
    return manifest


def spot_check(experiment: str, cell: int, replication: int, backend: str) -> dict:
    """Rerun one selected task and compare available saved statistical fields.

    Args:
        experiment: testing, intervals, or sensitivity.
        cell: Zero-based design index; defaults are documented in README.txt.
        replication: Zero-based replicate index, retaining its original RNG seeds.
        backend: c or python numerical implementation.

    Returns:
        Task, result and comparison status; timings are excluded from comparison.
        A saved-reference discrepancy or task failure raises ValueError.
    """
    from simulation import _run_task, make_task

    verify_reference()
    task = make_task(experiment, "full", backend, cell, replication)
    result = _run_task(task)
    if result["status"] != "ok":
        raise ValueError(result.get("error", "Spot-check task failed"))
    saved = json.loads((REFERENCE / "spotchecks.json").read_text(encoding="utf-8"))
    reference = next((row for row in saved if row["key"] == task["key"]), None)
    fields = ("statistic", "p_value", "estimated_change", "rejected", "true_change")
    if reference is not None:
        for name in fields:
            actual, expected = result[name], reference[name]
            equal = (
                actual == expected
                if expected is None
                else np.isclose(actual, expected, atol=1e-9, rtol=1e-9)
            )
            if not equal:
                raise ValueError(f"Spot-check discrepancy in {task['key']}: {name}")
    return {
        "task": task,
        "result": result,
        "saved_reference_match": True if reference is not None else None,
    }


def scenario_table() -> str:
    """Return Table 1's scenario definitions as a LaTeX table fragment.

    No arguments are required. Rows describe the fixed N0-N3/A1-A3 study designs;
    numerical estimates are generated separately by the simulation functions.
    """
    rows = [
        ("N0", "Stable age effect, case composition, and common delay."),
        ("N1", r"Shared death intensity falls by 45\% halfway through the window."),
        ("N2", "Case-age proportions shift halfway through the window."),
        ("N3", r"Correctly specified age-specific delay scales $4.5\exp(0.22q_j)$."),
        ("A1", r"Flattening, $\Delta\in\{-0.015,-0.025,-0.035,-0.050\}$."),
        ("A2", r"Steepening, $\Delta\in\{0.015,0.025,0.035,0.050\}$."),
        (
            "A3",
            r"Sparse flattening at 35\% baseline deaths. Testing uses $\Delta=-0.025$;"
            r" interval experiments use $\Delta=-0.025,-0.050$ at the early change.",
        ),
    ]
    return "\n".join(
        [
            r"\begin{tabular}{lp{0.72\textwidth}}\toprule",
            r"Scenario & Design feature\\\midrule",
            *[name + " & " + description + r"\\" for name, description in rows],
            r"\bottomrule\end{tabular}",
            "",
        ]
    )


def render_cached(output: Path) -> dict:
    """Render Tables 1-3 and Figures 1-4 from verified saved aggregate results.

    Args:
        output: Destination for figures, tables and a source-hash manifest.

    Returns:
        Output-file mapping. No simulation or patient-level analysis is rerun;
        saved estimates and intervals are used without recalibration.
    """
    import pandas as pd

    import analysis
    from realdata import _plot as application_plot
    from scripts.realdata_sensitivity import _plot as sensitivity_plot

    manifest = verify_reference()
    figures, tables = output / "figures", output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    testing = pd.read_csv(REFERENCE / "full_testing.csv")
    estimation = pd.read_csv(REFERENCE / "full_estimation.csv")
    sensitivity = pd.read_csv(REFERENCE / "full_sensitivity.csv")
    mcse = testing.copy()
    mcse["monte_carlo_se"] = np.sqrt(
        mcse["rejection"] * (1 - mcse["rejection"]) / mcse["replications"]
    )
    mcse.to_csv(output / "testing_monte_carlo_se.csv", index=False)
    # Manuscript Tables 1, 2 and 3, respectively.
    (tables / "table_1_scenarios.tex").write_text(scenario_table(), encoding="utf-8")
    (tables / "table_2_testing.tex").write_text(analysis.testing_table(testing), encoding="utf-8")
    (tables / "table_3_estimation.tex").write_text(
        analysis.estimation_table(estimation), encoding="utf-8"
    )
    # Manuscript Figures 1 and 2 use simulation summaries, not raw patient records.
    analysis.power_figure(testing, figures)
    analysis.sensitivity_figures(sensitivity, figures, require_complete=True)
    application = json.loads((REFERENCE / "application_results.json").read_text())
    with np.load(REFERENCE / "application_arrays.npz", allow_pickle=False) as arrays:
        result = SimpleNamespace(
            process=arrays["process"],
            information_time=arrays["information_time"],
            critical_value=application["critical_value"],
            estimated_change=application["estimated_change_index"],
        )
        intervals = {key: application["interval_" + key] for key in ("estimate", "lower", "upper")}
        # Manuscript Figure 3: the same plotting function used by the full analysis.
        application_plot(
            arrays["dates"],
            arrays["exposure"],
            arrays["deaths"],
            result,
            intervals,
            output_dir=figures,
        )
    metadata = json.loads((REFERENCE / "application_sensitivity_metadata.json").read_text())
    diagnostics = {int(key): value for key, value in metadata["delay_diagnostics"].items()}
    # Manuscript Figure 4: cutoff-specific estimates and training-delay diagnostics.
    sensitivity_plot(
        pd.read_csv(REFERENCE / "application_sensitivity.csv"), diagnostics, output_dir=figures
    )
    rendered = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.suffix in {".tex", ".pdf", ".png"}
    }
    return {"source_sha256": manifest["sha256"], "rendered_sha256": rendered}


def study_commands(mode: str, backend: str, workers: int, steps: list[str]) -> list[list[str]]:
    """Return ordered commands for selected demo/full study steps.

    Args:
        mode: demo or full simulation scale.
        backend: c or python for simulations (application scripts use c).
        workers: Number of simulation worker processes.
        steps: Ordered subset of STEPS; no source edits are required.

    Returns:
        Argument lists executed with the project root as working directory.
    """
    commands = {
        "simulations": [
            "simulation.py",
            "--mode",
            mode,
            "--backend",
            backend,
            "--workers",
            str(workers),
            "--resume",
        ],
        "analysis": ["analysis.py", "--mode", mode],
        "application": ["realdata.py", "--bootstrap-samples", "2000", "--seed", "20260809"],
        "sensitivity": [
            "-m",
            "scripts.realdata_sensitivity",
            "--gaussian-samples",
            "5000",
            "--seed",
            "20260809",
        ],
        "benchmark": ["-m", "scripts.benchmark", "--datasets", "100", "--samples", "1000"],
    }
    return [[sys.executable, "-B", *commands[step]] for step in steps]


def main() -> None:
    """Parse workflow options, execute selected steps and save a JSON run record.

    No Python arguments or return value. CLI flags select the workflow;
    generated run records include environment details and explicit commands.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "spot", "cached", "environment", "demo", "full"),
        default="smoke",
    )
    parser.add_argument("--backend", choices=("c", "python"), default="c")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--steps", nargs="+", choices=STEPS)
    parser.add_argument(
        "--experiment", choices=("testing", "intervals", "sensitivity"), default="testing"
    )
    parser.add_argument("--cell", type=int, default=4)
    parser.add_argument("--replication", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for run records/cached displays; study outputs keep their normal paths",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = args.output_dir or ROOT / "results_conditional_delay" / "reproduction" / args.mode
    output.mkdir(parents=True, exist_ok=True)
    report = {"mode": args.mode, "environment": environment()}
    if args.mode in ("demo", "full"):
        steps = args.steps or (list(STEPS) if args.mode == "full" else ["simulations", "analysis"])
        commands = study_commands(args.mode, args.backend, args.workers, steps)
        report["commands"] = [command[1:] for command in commands]
        report["executed"] = not args.dry_run
        if not args.dry_run:
            for command in commands:
                subprocess.run(command, cwd=ROOT, check=True)
    elif args.mode == "cached":
        report["cached"] = render_cached(output)
    elif args.mode == "smoke":
        report["spotchecks"] = [
            spot_check("testing", cell, 0, args.backend) for cell in (0, 4, 144)
        ]
    elif args.mode == "spot":
        report["spotcheck"] = spot_check(args.experiment, args.cell, args.replication, args.backend)
    (output / "run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mode": args.mode, "run_record": str(output / "run.json")}, indent=2))


if __name__ == "__main__":
    main()
