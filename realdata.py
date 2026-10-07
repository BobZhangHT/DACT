"""Reproduce the SIVEP-Gripe delay-adjusted age-effect change analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CONFIG, FIGURE_DIR, TABLE_DIR
from methods import (
    bootstrap_change_intervals, build_c_backend, delay_adjusted_exposure,
    gradient_change_test,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "sivep"
RESULT = ROOT / "results_conditional_delay" / "application"
FILES = [DATA / "INFLUD21-23-03-2026.parquet", DATA / "INFLUD22-23-03-2026.parquet"]
COLS = [
    "NU_NOTIFIC", "DT_NOTIFIC", "DT_SIN_PRI", "DT_EVOLUCA",
    "NU_IDADE_N", "TP_IDADE", "CLASSI_FIN", "CRITERIO", "HOSPITAL", "EVOLUCAO",
]
AGE_LABELS = ("0--39", "40--49", "50--59", "60--69", "70--79", "80+")
AGE_BINS = (-1, 39, 49, 59, 69, 79, 110)
AGE_SCORES = np.asarray((-3.0, -1.5, -0.5, 0.5, 1.5, 2.5))


def _read() -> pd.DataFrame:
    frames = [pd.read_parquet(path, columns=COLS) for path in FILES]
    data = pd.concat(frames, ignore_index=True)
    for name in ("DT_NOTIFIC", "DT_SIN_PRI", "DT_EVOLUCA"):
        data[name] = pd.to_datetime(data[name], errors="coerce", dayfirst=True)
    for name in ("NU_IDADE_N", "TP_IDADE", "CLASSI_FIN", "CRITERIO", "HOSPITAL", "EVOLUCAO"):
        data[name] = pd.to_numeric(data[name], errors="coerce")
    data = data[
        (data["CLASSI_FIN"] == 5) & (data["CRITERIO"] == 1)
        & (data["HOSPITAL"] == 1) & (data["TP_IDADE"] == 3)
        & data["NU_IDADE_N"].between(0, 110) & data["DT_SIN_PRI"].notna()
    ].copy()
    data = data.sort_values("DT_NOTIFIC").drop_duplicates("NU_NOTIFIC", keep="last")
    data["age_group"] = pd.cut(data["NU_IDADE_N"], AGE_BINS, labels=False)
    return data[data["age_group"].notna()].copy()


def _delay(data: pd.DataFrame, maximum: int = 90) -> tuple[np.ndarray, dict]:
    training = data[
        data["DT_SIN_PRI"].between("2021-01-01", "2021-09-30")
        & (data["EVOLUCAO"] == 2) & data["DT_EVOLUCA"].notna()
    ].copy()
    training["lag"] = (training["DT_EVOLUCA"] - training["DT_SIN_PRI"]).dt.days
    raw = training["lag"]
    training_all = training.copy()
    training = training[training["lag"].between(0, maximum)]
    f = np.empty((len(AGE_LABELS), maximum + 1), dtype=float)
    counts = {}
    tail_by_age = {}
    for j in range(len(AGE_LABELS)):
        values = training.loc[training["age_group"] == j, "lag"].to_numpy(dtype=int)
        mass = np.bincount(values, minlength=maximum + 1).astype(float) + 0.5
        f[j] = mass / mass.sum()
        counts[AGE_LABELS[j]] = int(values.size)
        # Use the pre-truncation training frame for transparent age-specific tails.
        raw_values_j = training_all.loc[training_all["age_group"] == j, "lag"]
        nonnegative_j = raw_values_j[raw_values_j >= 0]
        tail_n = int((nonnegative_j > maximum).sum())
        denominator = int(nonnegative_j.size)
        tail_by_age[AGE_LABELS[j]] = {
            "nonnegative_training_deaths": denominator,
            "usable_at_maximum": int(values.size),
            "above_maximum": tail_n,
            "above_maximum_fraction": float(tail_n / denominator) if denominator else None,
        }
    diagnostics = {
        "training_deaths": int(len(raw)), "valid_training_delays": int(len(training)),
        "negative_delays": int((raw < 0).sum()),
        "delay_maximum": int(maximum),
        "delays_over_maximum": int((raw > maximum).sum()),
        "delays_over_90": int((raw > 90).sum()),
        "median_delay": float(training["lag"].median()),
        "p90_delay": float(training["lag"].quantile(0.90)), "age_counts": counts,
        "tail_by_age": tail_by_age,
    }
    return f, diagnostics


def _daily_arrays(data: pd.DataFrame, delay: np.ndarray, analysis_start, analysis_end,
                  prehistory_days: int = 120):
    analysis_start = pd.Timestamp(analysis_start)
    analysis_end = pd.Timestamp(analysis_end)
    history_start = analysis_start - pd.Timedelta(days=prehistory_days)
    history_dates = pd.date_range(history_start, analysis_end, freq="D")
    analysis_dates = pd.date_range(analysis_start, analysis_end, freq="D")
    cases = np.zeros((len(history_dates), len(AGE_LABELS)), dtype=int)
    cohort = data[data["DT_SIN_PRI"].between(history_start, analysis_end)]
    grouped = cohort.groupby(["DT_SIN_PRI", "age_group"], observed=True).size()
    for (date, group), value in grouped.items():
        cases[(date - history_start).days, int(group)] = int(value)
    full_exposure = delay_adjusted_exposure(cases, delay, len(history_dates), backend="c")
    offset = (analysis_start - history_start).days
    exposure = full_exposure[offset:offset + len(analysis_dates)]
    deaths = np.zeros_like(exposure, dtype=int)
    fatal = data[(data["EVOLUCAO"] == 2) & data["DT_EVOLUCA"].between(analysis_start, analysis_end)]
    grouped = fatal.groupby(["DT_EVOLUCA", "age_group"], observed=True).size()
    for (date, group), value in grouped.items():
        deaths[(date - analysis_start).days, int(group)] = int(value)
    return analysis_dates, cases, exposure, deaths, cohort, fatal


def _plot(dates, exposure, deaths, result, intervals):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(np.asarray(dates))
    plt.rcParams.update({"font.size": 9, "legend.fontsize": 9, "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.6), constrained_layout=True)
    styles = (("o", "-", "#0072B2"), ("s", "--", "#A64B00"),
              ("^", "-.", "#333333"), ("D", ":", "#7755AA"),
              ("v", (0, (5, 2, 1, 2)), "#555555"),
              ("P", (0, (3, 1, 1, 1, 1, 1)), "#AA3377"))
    for j, label in enumerate(AGE_LABELS):
        marker, line, color = styles[j]
        axes[0, 0].plot(dates, exposure[:, j], lw=0.9, marker=marker, markevery=35, ms=3,
                        ls=line, color=color, label=label.replace("--", "-"))
        axes[0, 1].plot(dates, deaths[:, j], lw=0.9, marker=marker, markevery=35, ms=3,
                        ls=line, color=color, label=label.replace("--", "-"))
    axes[0, 0].set_ylabel("Delay-aligned case exposure")
    axes[0, 1].set_ylabel("Recorded deaths")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.90))
    # Use the saved observed-information clock, not a calendar-fraction trim.
    information = np.asarray(result.information_time)
    trim = CONFIG.method.minimum_information_fraction
    admissible = np.flatnonzero((information[:-1] >= trim) &
                               (information[:-1] <= 1.0 - trim))
    if not len(admissible):
        raise ValueError("No information-trimmed candidate dates")
    score_ax = axes[1, 0]
    score_ax.axvspan(dates[0], dates[admissible[0]], facecolor="0.90",
                    edgecolor="none", lw=0, zorder=0)
    score_ax.axvspan(dates[admissible[-1]], dates[-1], facecolor="0.90",
                    edgecolor="none", lw=0, zorder=0)
    absolute_score = np.abs(np.asarray(result.process))
    searched = np.zeros(len(dates), dtype=bool)
    searched[admissible] = True
    score_ax.plot(dates, np.where(~searched, absolute_score, np.nan),
                  color="0.50", ls="--", lw=1.0)
    score_ax.plot(dates, np.where(searched, absolute_score, np.nan),
                  color="#0072B2", lw=1.5)
    score_ax.set_ylim(0, 4.8)
    score_ax.set_yticks([0, 1, 2, 3, 4])
    selected = dates[result.estimated_change - 1]
    selected_score = absolute_score[result.estimated_change - 1]
    score_ax.annotate(selected.strftime("%d %b %Y") + " split\n"
                      + f"Maximum = {selected_score:.3f}",
                      xy=(selected, selected_score),
                      xytext=(0.40, 0.95), textcoords="axes fraction", fontsize=9,
                      va="top", arrowprops={"arrowstyle": "->", "color": "0.25"})
    score_ax.axhline(result.critical_value, color="#A64B00", ls="--", lw=1.5)
    score_ax.annotate(f"5% critical value = {result.critical_value:.3f}\n(model-based)",
                      xy=(dates[-50], result.critical_value),
                      xytext=(0.98, 0.54), textcoords="axes fraction",
                      ha="right", va="bottom", fontsize=9,
                      arrowprops={"arrowstyle": "->", "color": "#A64B00"})
    axes[1, 0].axvline(dates[result.estimated_change - 1], color="#D55E00", ls="--", lw=1)
    axes[1, 0].set_ylabel("Absolute DACT score")
    axes[1, 0].set_xlabel("Death-calendar date\nSearch: 15-85% cumulative information")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlim(dates[0], dates[-1])
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 5, 9)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.tick_params(axis="x", labelrotation=0)
    reference = 2
    contrasts = AGE_SCORES - AGE_SCORES[reference]
    x = np.arange(len(AGE_LABELS))
    for index, (name, color) in enumerate(
            zip(("Before change", "After change"), ("#0072B2", "#D55E00")), start=1):
        beta = intervals["estimate"][index]
        beta_lo = intervals["lower"][index]
        beta_hi = intervals["upper"][index]
        profile = np.exp(beta * contrasts)
        bound_a = np.exp(beta_lo * contrasts)
        bound_b = np.exp(beta_hi * contrasts)
        axes[1, 1].plot(x, profile, marker="o" if index == 1 else "s",
                        ls="-" if index == 1 else "--", ms=4, lw=1.25,
                        color=color, label=name)
        axes[1, 1].fill_between(
            x, np.minimum(bound_a, bound_b), np.maximum(bound_a, bound_b),
            color=color, alpha=0.14, linewidth=0,
        )
    axes[1, 1].axhline(1.0, color="0.45", ls=":", lw=0.8)
    axes[1, 1].set_xticks(x, AGE_LABELS, rotation=30, ha="right")
    axes[1, 1].set_ylabel("Relative death intensity\n(ref.: 50-59 years)")
    axes[1, 1].set_xlabel("Age group")
    axes[1, 1].legend(frameon=False, fontsize=9)
    for panel, ax in enumerate(axes.ravel()):
        ax.grid(alpha=0.15)
        ax.set_title(f"({chr(97 + panel)})", loc="left",
                     fontweight="bold", fontsize=9)
    fig.savefig(FIGURE_DIR / "figure_application.pdf", bbox_inches=None)
    fig.savefig(FIGURE_DIR / "figure_application.png", dpi=400, bbox_inches=None)
    plt.close(fig)


def run(seed: int = 20260809, bootstrap_samples: int = 2000,
        analysis_start: str = "2022-01-01", analysis_end: str = "2022-12-31") -> dict:
    RESULT.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    build_c_backend()
    data = _read()
    delay, delay_diagnostics = _delay(data)
    dates, cases, exposure, deaths, cohort, fatal = _daily_arrays(
        data, delay, analysis_start, analysis_end
    )
    result = gradient_change_test(
        exposure, deaths, AGE_SCORES, "proposed_gaussian", np.random.default_rng(seed),
        backend="c", gaussian_samples=CONFIG.method.gaussian_samples,
    )
    validation = gradient_change_test(
        exposure, deaths, AGE_SCORES, "proposed_bootstrap", np.random.default_rng(seed + 1),
        backend="c", bootstrap_samples=bootstrap_samples,
    )
    intervals = bootstrap_change_intervals(
        exposure, deaths, AGE_SCORES, np.random.default_rng(seed + 2), backend="c",
        bootstrap_samples=bootstrap_samples,
    )
    delta = float(intervals["estimate"][2] - intervals["estimate"][1])
    output = {
        "analysis_start": str(dates[0].date()), "analysis_end": str(dates[-1].date()),
        "eligible_history_cases": int(len(cohort)), "recorded_deaths_in_window": int(len(fatal)),
        "daily_deaths": int(deaths.sum()), "delay": delay_diagnostics,
        "statistic": result.statistic, "critical_value": result.critical_value,
        "gaussian_p_value": result.p_value, "bootstrap_p_value": validation.p_value,
        "bootstrap_diagnostics": {
            "null_calibration": {k: validation.bootstrap_diagnostics[k] for k in
                ("n_requested", "n_successful", "n_failed", "n_retried", "failure_counts")},
            "intervals": {k: intervals["bootstrap_diagnostics"][k] for k in
                ("n_requested", "n_successful", "n_failed", "n_retried", "failure_counts")},
        },
        "estimated_change_index": int(result.estimated_change),
        "estimated_change_date": str(dates[result.estimated_change - 1].date()),
        "beta_before": float(intervals["estimate"][1]),
        "beta_after": float(intervals["estimate"][2]), "delta": delta,
        "gradient_ratio": float(np.exp(delta)),
        "interval_labels": list(intervals["labels"]),
        "interval_estimate": intervals["estimate"].tolist(),
        "interval_lower": intervals["lower"].tolist(),
        "interval_upper": intervals["upper"].tolist(),
    }
    (RESULT / "application_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    np.savez_compressed(RESULT / "application_arrays.npz", dates=dates.astype(str), cases=cases,
                        exposure=exposure, deaths=deaths, delay=delay,
                        process=result.process, information_time=result.information_time)
    _plot(dates, exposure, deaths, result, intervals)
    table = (
        "\\begin{tabular}{lrrrr}\\toprule\n"
        "Analysis & Change date & $p$ (Gaussian) & $p$ (bootstrap) & $\\exp(\\widehat\\Delta)$\\\\\n"
        "\\midrule\nPrimary & %s & %.4f & %.4f & %.3f\\\\\n\\bottomrule\\end{tabular}\n"
        % (output["estimated_change_date"], output["gaussian_p_value"],
           output["bootstrap_p_value"], output["gradient_ratio"])
    )
    (TABLE_DIR / "table_application.tex").write_text(table, encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--analysis-start", default="2022-01-01")
    parser.add_argument("--analysis-end", default="2022-12-31")
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.bootstrap_samples, args.analysis_start, args.analysis_end), indent=2))


if __name__ == "__main__":
    main()
