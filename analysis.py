"""Aggregate simulations and create publication tables and vector figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import AGGREGATED_DIR, FIGURE_DIR, RAW_DIR, SCENARIOS, TABLE_DIR, ensure_directories

METHOD_LABEL = {
    "proposed_gaussian": "DACT", "sup_lm": "supLM",
    "cvm": "CvM", "no_delay": "No-delay DACT",
}


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    return frame[frame.get("status", "ok") == "ok"].copy()


def aggregate_testing(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario", "method", "days", "death_scale", "magnitude", "change_fraction"]
    out = frame.groupby(keys, dropna=False).agg(
        rejection=("rejected", "mean"), replications=("rejected", "size"),
        failure_rate=("failure", "mean"), runtime=("runtime_seconds", "mean"),
        median_location_error=("location_error", "median"),
    ).reset_index()
    return out


def aggregate_bootstrap_estimation(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the search-repeated conditional-bootstrap experiment.

    The current record format uses generic ``se/lower/upper`` fields.  Legacy
    full runs used ``bootstrap_*`` names; accepting both keeps the frozen
    1000-replication output reproducible without rerunning the experiment.
    """
    rows = []
    for record in frame.itertuples():
        for k, label in enumerate(record.labels):
            se_values = getattr(record, "se", None)
            covered_values = getattr(record, "covered", None)
            width_values = getattr(record, "width", None)
            if not isinstance(se_values, (list, tuple, np.ndarray)):
                se_values = record.bootstrap_se
                covered_values = record.bootstrap_covered
                width_values = record.bootstrap_width
            rows.append({
                "scenario": record.scenario, "days": record.days, "magnitude": record.magnitude,
                "change_fraction": record.change_fraction, "replication": record.replication,
                "parameter": label, "truth": record.truth[k], "estimate": record.estimate[k],
                "se": se_values[k], "covered": covered_values[k], "width": width_values[k],
                "rejected": True,
            })
    long = pd.DataFrame(rows)
    if long.empty:
        return long
    grouped = long.groupby(["scenario", "days", "magnitude", "change_fraction", "parameter"])
    return grouped.agg(
        bias=("estimate", lambda x: float(np.mean(x - long.loc[x.index, "truth"]))),
        empirical_sd=("estimate", "std"), mean_se=("se", "mean"), coverage=("covered", "mean"),
        width=("width", "mean"), successful=("estimate", "size"),
    ).reset_index()


def testing_table(df: pd.DataFrame) -> str:
    """Full-width grouped layout with one numerical result per table cell."""
    shown_null = df[(df.magnitude == 0) & (df.death_scale == 1.0)]
    shown_alt = df[(df.scenario.isin(["A1", "A2", "A3"])) &
                   (df.magnitude == 0.025) & (df.death_scale == 1.0)]
    methods = ("proposed_gaussian", "sup_lm", "cvm", "no_delay")

    def value(part: pd.DataFrame, method: str, field: str = "rejection") -> str:
        row = part[part.method == method]
        if row.empty or pd.isna(row.iloc[0][field]):
            return "--"
        if field == "median_location_error":
            return f"{row.iloc[0][field]:.0f}"
        return f"{row.iloc[0][field]:.3f}"

    lines = [
        r"\begin{table}[p]", r"\centering",
        r"\caption{Testing performance for the daily age-gradient change test. Panel A reports Type-I error at the baseline death-information scale $s_n=1$. Panel B reports rejection probability under $|\Delta|=0.025$. MedAE is DACT's median absolute change-day error in days. Each entry uses 1000 replications.}",
        r"\label{tab:testing-v4}", r"\small",
        r"\renewcommand{\arraystretch}{0.98}", r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrccccc@{}}", r"\toprule",
        r"DGP & $T$ & $\tau_0/T$ & DACT & supLM & CvM & No-delay & MedAE\\", r"\midrule",
        r"\multicolumn{8}{l}{\textit{Panel A: Type-I error at $s_n=1$ under the four null mechanisms}}\\",
    ]
    for scenario in ("N0", "N1", "N2", "N3"):
        for day_index, days in enumerate((90, 180, 365)):
            part = shown_null[(shown_null.scenario == scenario) & (shown_null.days == days)]
            entries = [value(part, method) for method in methods]
            dgp = scenario if day_index == 0 else ""
            lines.append(f"{dgp} & {days} & -- & " + " & ".join(entries) + r" & --\\")
    lines += [r"\midrule",
              r"\multicolumn{8}{l}{\textit{Panel B: rejection probability for $|\Delta|=0.025$}}\\"]
    for scenario in ("A1", "A2", "A3"):
        scenario_rows = shown_alt[shown_alt.scenario == scenario]
        row_index = 0
        for days in (90, 180, 365):
            day_rows = scenario_rows[scenario_rows.days == days]
            for fraction_index, fraction in enumerate(sorted(day_rows.change_fraction.dropna().unique())):
                part = day_rows[np.isclose(day_rows.change_fraction, fraction)]
                entries = [value(part, method) for method in methods]
                medae = value(part, "proposed_gaussian", "median_location_error")
                dgp = scenario if row_index == 0 else ""
                day_text = str(days) if fraction_index == 0 else ""
                lines.append(f"{dgp} & {day_text} & {fraction:.2f} & " +
                             " & ".join(entries) + f" & {medae}\\\\")
                row_index += 1
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def estimation_table(df: pd.DataFrame) -> str:
    """Bootstrap estimation table for regular and sparse alternatives."""
    shown = df[(df.days == 180) &
               (((df.scenario.isin(["A1", "A2"])) & (df.change_fraction == 0.50)) |
                ((df.scenario == "A3") & (df.change_fraction == 0.25)))].copy()
    labels = {"tau_over_T": r"$\tau/T$", "beta_before": r"$\beta_1$",
              "beta_after": r"$\beta_2$", "delta": r"$\Delta$",
              "exp_delta": r"$\exp(\Delta)$"}
    order = {name: index for index, name in enumerate(labels)}
    shown["parameter_order"] = shown.parameter.map(order)
    lines = [
        r"\begin{table}[p]", r"\centering",
        r"\caption{Post-change estimation for $T=180$. A1 and A2 have a central change; the A3 panel focuses on the early sparse case at $\tau_0/T=0.25$ and 35\% of the baseline death information. SD is the empirical standard deviation across 1000 datasets. Mean SE, coverage, and width use the search-repeated conditional bootstrap with $B=2000$.}",
        r"\label{tab:parameter-estimation-v4}", r"\small",
        r"\renewcommand{\arraystretch}{0.98}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrr@{}}", r"\toprule",
        r"$|\Delta|$ & Parameter & Bias & SD & Mean SE & Coverage & Width\\", r"\midrule",
    ]
    scenario_labels = {"A1": "abrupt flattening", "A2": "abrupt steepening",
                       "A3": r"sparse flattening, $\tau_0/T=0.25$"}
    available = set(shown.scenario.unique())
    for scenario in ("A1", "A2", "A3"):
        if scenario not in available:
            continue
        lines.append(r"\multicolumn{7}{l}{\textit{" + scenario + ": " +
                     scenario_labels[scenario] + r"}}\\")
        part = shown[shown.scenario == scenario]
        previous_magnitude = None
        for row in part.sort_values(["magnitude", "parameter_order"]).itertuples():
            magnitude = f"{row.magnitude:.3f}" if row.magnitude != previous_magnitude else ""
            lines.append(
                f"{magnitude} & {labels[row.parameter]} & {row.bias:.3f} & "
                f"{row.empirical_sd:.3f} & {row.mean_se:.3f} & {row.coverage:.3f} & "
                f"{row.width:.3f}\\\\"
            )
            previous_magnitude = row.magnitude
        if scenario != "A3":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def sensitivity_figures(frame: pd.DataFrame, output_dir: Path, require_complete: bool = False):
    if frame.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = frame.groupby(["scenario", "axis", "value"]).agg(
        rejection=("rejected", "mean"), location_error=("location_error", "median")
    ).reset_index()
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    axes_to_plot = ("delay_multiplier", "age_delay_mismatch", "age_groups",
                    "overdispersion", "transition_width", "day_of_week")
    missing = set(axes_to_plot) - set(summary.axis.unique())
    if missing:
        raise ValueError(f"sensitivity results are missing axes: {sorted(missing)}")
    if require_complete:
        expected_values = {
            "delay_multiplier": 5,
            "age_delay_mismatch": 4,
            "age_groups": 4,
            "overdispersion": 4,
            "transition_width": 5,
            "day_of_week": 4,
        }
        observed = summary.groupby("axis")["value"].nunique().to_dict()
        incomplete = {
            axis: (observed.get(axis, 0), expected)
            for axis, expected in expected_values.items()
            if observed.get(axis, 0) < expected
        }
        if incomplete:
            raise ValueError(
                "full sensitivity results are incomplete; observed/expected "
                f"value counts: {incomplete}"
            )
    axis_labels = {
        "delay_multiplier": "Analyzed/true delay scale",
        "age_delay_mismatch": "Ignored age-delay slope",
        "age_groups": "Number of age groups",
        "overdispersion": "Overdispersion",
        "transition_width": "Transition width (days)",
        "day_of_week": "Day-of-week amplitude",
    }
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), constrained_layout=True)
    for panel, (ax, axis) in enumerate(zip(axes.ravel(), axes_to_plot)):
        active = summary[summary.axis == axis]
        for scenario, part in active.groupby("scenario"):
            part = part.sort_values("value")
            ax.plot(part.value, part.rejection, marker="o", label=scenario)
        ax.axhline(0.05, color="black", ls="--", lw=0.8)
        ax.set_xlabel(axis_labels[axis])
        ax.set_ylabel("Rejection probability")
        ax.grid(alpha=0.15)
        if not active.empty:
            ax.legend(frameon=False)
        ax.text(0.02, 0.96, f"({chr(97 + panel)})", transform=ax.transAxes,
                va="top", fontweight="bold")
        if axis == "transition_width":
            ax.text(0.98, 0.04, "A1 only", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7)
    fig.savefig(output_dir / "figure_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "figure_sensitivity.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def power_figure(df: pd.DataFrame, output_dir: Path):
    """Power surfaces not repeated in the central-setting testing table."""
    if df.empty:
        return
    active = df[(df.method == "proposed_gaussian") & (df.magnitude > 0)
                & (df.death_scale == 1.0)]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), sharex=True, sharey=True,
                             constrained_layout=True)
    for row, scenario in enumerate(("A1", "A2")):
        for col, days in enumerate((90, 180, 365)):
            ax = axes[row, col]
            cell = active[(active.scenario == scenario) & (active.days == days)]
            for fraction, part in cell.groupby("change_fraction"):
                part = part.sort_values("magnitude")
                ax.plot(part.magnitude, part.rejection, marker="o",
                        label=f"$\\tau/T={fraction:.2f}$")
            ax.set_title(f"{scenario}, $T={days}$")
            ax.set_xlabel("Gradient-change magnitude")
            ax.set_ylabel("Power")
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.15)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=1)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "figure_power_surface.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "figure_power_surface.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def run(mode: str):
    ensure_directories()
    testing_raw = _load(RAW_DIR / f"{mode}_testing.jsonl")
    interval_raw = _load(RAW_DIR / f"{mode}_intervals.jsonl")
    sensitivity_raw = _load(RAW_DIR / f"{mode}_sensitivity.jsonl")
    testing = aggregate_testing(testing_raw) if not testing_raw.empty else pd.DataFrame()
    estimation = aggregate_bootstrap_estimation(interval_raw) if not interval_raw.empty else pd.DataFrame()
    output_dir = FIGURE_DIR if mode == "full" else AGGREGATED_DIR / "demo_figures"
    table_dir = TABLE_DIR if mode == "full" else AGGREGATED_DIR / "demo_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    AGGREGATED_DIR.mkdir(parents=True, exist_ok=True)
    if not testing.empty:
        testing.to_csv(AGGREGATED_DIR / f"{mode}_testing.csv", index=False)
        (table_dir / "table_testing.tex").write_text(testing_table(testing), encoding="utf-8")
        power_figure(testing, output_dir)
    if not estimation.empty:
        estimation.to_csv(AGGREGATED_DIR / f"{mode}_estimation.csv", index=False)
        (table_dir / "table_estimation.tex").write_text(estimation_table(estimation), encoding="utf-8")
    if not sensitivity_raw.empty:
        sensitivity_summary = sensitivity_raw.groupby(
            ["scenario", "axis", "value"]
        ).agg(
            rejection=("rejected", "mean"),
            location_error=("location_error", "median"),
            replications=("rejected", "size"),
        ).reset_index()
        sensitivity_summary.to_csv(
            AGGREGATED_DIR / f"{mode}_sensitivity.csv", index=False
        )
    sensitivity_figures(sensitivity_raw, output_dir, require_complete=(mode == "full"))
    return {"testing_rows": len(testing), "estimation_rows": len(estimation),
            "sensitivity_rows": len(sensitivity_raw)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("demo", "full"), required=True)
    args = parser.parse_args()
    print(run(args.mode))


if __name__ == "__main__":
    main()
