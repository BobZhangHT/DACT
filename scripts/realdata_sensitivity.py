"""Sensitivity of the SIVEP-Gripe analysis to delay truncation.

The prespecified full-year analysis is repeated with maximum retained
onset-to-death delays of 60, 90, and 120 days. Gaussian-bridge calibration is
used here; the B=2000 conditional-bootstrap primary analysis remains unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CONFIG, FIGURE_DIR
from methods import build_c_backend, estimate_change, gradient_change_test
from realdata import AGE_LABELS, AGE_SCORES, FILES, _daily_arrays, _delay, _read


ROOT = Path(__file__).resolve().parents[1]
DATA_MANIFEST = ROOT / "data" / "sivep" / "DATA_MANIFEST.json"
RESULT_DIR = ROOT / "results_conditional_delay" / "application"
OUTPUT_CSV = RESULT_DIR / "application_sensitivity.csv"
OUTPUT_JSON = RESULT_DIR / "application_sensitivity_metadata.json"
OUTPUT_PDF = RESULT_DIR / "application_sensitivity.pdf"
PAPER_PDF = FIGURE_DIR / "figure_application_sensitivity.pdf"

ANALYSIS_START = "2022-01-01"
ANALYSIS_END = "2022-12-31"
DELAY_MAXIMA = (60, 90, 120)


def _input_hashes(paths) -> dict[str, str]:
    """Fingerprint the actual input files without requiring a local manifest."""
    hashes = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[path.name] = digest.hexdigest()
    return hashes


def _plot(results: pd.DataFrame, delay_diagnostics: dict[int, dict]) -> None:
    delay_order = list(DELAY_MAXIMA)
    ordered = results.set_index("delay_maximum").reindex(delay_order)

    plt.rcParams.update({
        "font.size": 11, "legend.fontsize": 11, "axes.labelsize": 11,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(6.5, 5.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, :])]

    panel = axes[0]
    colors = {60: "#0072B2", 90: "#A64B00", 120: "#333333"}
    markers = {60: "o", 90: "s", 120: "^"}
    for maximum in delay_order:
        diagnostic = delay_diagnostics[maximum]
        fractions = [
            diagnostic["tail_by_age"][label]["above_maximum_fraction"]
            for label in AGE_LABELS
        ]
        panel.plot([label.replace("--", "-") for label in AGE_LABELS],
                   np.asarray(fractions) * 100.0, marker=markers[maximum],
                   color=colors[maximum], lw=1.4, ms=5, ls={60: "-", 90: "--", 120: "-."}[maximum],
                   label=f"{maximum} days")
    panel.set_ylabel("Observed delays\nomitted (%)")
    panel.set_xlabel("Age group (years)")
    panel.tick_params(axis="x", rotation=35)
    panel.legend(frameon=False, title="Maximum delay", fontsize=11,
                 title_fontsize=11)
    panel.grid(axis="y", alpha=0.2)

    panel = axes[1]
    panel.plot(delay_order, ordered["statistic"], color="#0072B2", marker="o",
               lw=1.35, label="DACT statistic")
    panel.plot(delay_order, ordered["critical_value"], color="#D55E00",
               marker="s", ls="--", lw=1.25, label="5% critical value")
    panel.set_xticks(delay_order)
    panel.set_xlabel("Maximum retained delay\n(days)")
    panel.set_ylabel("Value")
    panel.legend(frameon=False, fontsize=11)
    panel.grid(axis="y", alpha=0.2)

    panel = axes[2]
    before = np.exp(ordered["beta_before"].to_numpy(dtype=float))
    after = np.exp(ordered["beta_after"].to_numpy(dtype=float))
    panel.plot(delay_order, before, color="#0072B2", marker="o", lw=1.35,
               label="Before change")
    panel.plot(delay_order, after, color="#D55E00", marker="s", ls="--", lw=1.35,
               label="After change")
    panel.set_xticks(delay_order)
    panel.set_xlabel("Maximum retained delay\n(days)")
    panel.set_ylabel("Per-decade multiplier")
    panel.legend(frameon=False, fontsize=11)
    panel.grid(axis="y", alpha=0.2)

    for index, panel in enumerate(axes):
        panel.set_title(f"({chr(97 + index)})", loc="left",
                        fontweight="bold", fontsize=11)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_PDF, bbox_inches=None)
    OUTPUT_PDF.write_bytes(PAPER_PDF.read_bytes())
    plt.close(fig)


def run(seed: int = 20260809, gaussian_samples: int = 5000) -> pd.DataFrame:
    """Repeat the full-year analysis at three delay cutoffs."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    build_c_backend()
    data = _read()
    delay_by_maximum: dict[int, np.ndarray] = {}
    diagnostics_by_maximum: dict[int, dict] = {}
    for maximum in DELAY_MAXIMA:
        delay, diagnostics = _delay(data, maximum=maximum)
        delay_by_maximum[maximum] = delay
        diagnostics_by_maximum[maximum] = diagnostics

    rows: list[dict] = []
    for delay_index, maximum in enumerate(DELAY_MAXIMA):
        dates, _, exposure, deaths, cohort, fatal = _daily_arrays(
            data, delay_by_maximum[maximum], ANALYSIS_START, ANALYSIS_END,
            prehistory_days=max(DELAY_MAXIMA),
        )
        cell_seed = seed + delay_index
        result = gradient_change_test(
            exposure, deaths, AGE_SCORES, method="proposed_gaussian",
            rng=np.random.default_rng(cell_seed), backend="c",
            gaussian_samples=gaussian_samples,
        )
        estimate = estimate_change(exposure, deaths, AGE_SCORES, backend="c")
        delta = float(estimate.beta_after - estimate.beta_before)
        diagnostics = diagnostics_by_maximum[maximum]
        rows.append({
            "analysis_start": ANALYSIS_START,
            "analysis_end": ANALYSIS_END,
            "analysis_days": int(len(dates)),
            "delay_maximum": maximum,
            "eligible_history_cases": int(len(cohort)),
            "recorded_deaths": int(len(fatal)),
            "daily_deaths": int(deaths.sum()),
            "delay_training_usable": diagnostics["valid_training_delays"],
            "delay_training_above_maximum": diagnostics["delays_over_maximum"],
            "statistic": result.statistic,
            "critical_value": result.critical_value,
            "statistic_to_critical": result.statistic / result.critical_value,
            "gaussian_p_value": result.p_value,
            "rejected_at_0.05": result.rejected,
            "estimated_change_index": result.estimated_change,
            "estimated_change_date": str(dates[result.estimated_change - 1].date()),
            "null_beta": estimate.null_beta,
            "beta_before": estimate.beta_before,
            "beta_after": estimate.beta_after,
            "delta": delta,
            "gradient_ratio": float(np.exp(delta)),
            "gaussian_samples": gaussian_samples,
            "seed": cell_seed,
            "calibration": "Gaussian bridge on observed information grid",
            "analysis_role": "delay-truncation sensitivity",
        })

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_CSV, index=False)
    input_hashes = _input_hashes(FILES)
    metadata = {
        "description": (
            "Sensitivity of the prespecified full-year analysis to maximum "
            "retained onset-to-death delays of 60, 90, and 120 days."
        ),
        "seed": seed,
        "gaussian_samples": gaussian_samples,
        "minimum_information_fraction": CONFIG.method.minimum_information_fraction,
        "analysis_window": {"start": ANALYSIS_START, "end": ANALYSIS_END},
        "delay_maxima": list(DELAY_MAXIMA),
        "delay_diagnostics": diagnostics_by_maximum,
        "input_manifest": str(DATA_MANIFEST) if DATA_MANIFEST.exists() else None,
        "input_sha256": input_hashes,
        "output_csv": str(OUTPUT_CSV.relative_to(ROOT)),
        "figure_pdf": str(OUTPUT_PDF.relative_to(ROOT)),
    }
    OUTPUT_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _plot(results, diagnostics_by_maximum)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--gaussian-samples", type=int, default=5000)
    args = parser.parse_args()
    results = run(seed=args.seed, gaussian_samples=args.gaussian_samples)
    columns = [
        "delay_maximum", "statistic", "critical_value",
        "gaussian_p_value", "estimated_change_date", "gradient_ratio",
    ]
    print(results[columns].to_string(index=False))


if __name__ == "__main__":
    main()
