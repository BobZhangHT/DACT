"""Aggregate simulations and create publication tables and vector figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import AGGREGATED_DIR, FIGURE_DIR, RAW_DIR, TABLE_DIR, ensure_directories

METHOD_LABEL = {
    "proposed_gaussian": "DACT",
    "sup_lm": "supLM",
    "cvm": "CvM",
    "no_delay": "No-delay DACT",
}


def _load(path: Path) -> pd.DataFrame:
    """Read successful simulation records from a JSON-lines file.

    Args:
        path: UTF-8 JSON-lines input path.

    Returns:
        A copy containing records with status ``ok``, or an empty DataFrame
        when the file is absent. Reads the file without modifying it.
    """
    if not path.exists():
        return pd.DataFrame()
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    frame = pd.DataFrame(rows)
    return frame[frame.get("status", "ok") == "ok"].copy()


def aggregate_testing(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize testing records within each simulation design cell.

    Args:
        frame: Replication records with design keys, rejection indicators,
            failures, runtimes, and location errors.

    Returns:
        A DataFrame of rejection and failure rates, replication counts, mean
        runtime, and median location error. Does not write files.
    """
    keys = ["scenario", "method", "days", "death_scale", "magnitude", "change_fraction"]
    out = (
        frame.groupby(keys, dropna=False)
        .agg(
            rejection=("rejected", "mean"),
            replications=("rejected", "size"),
            failure_rate=("failure", "mean"),
            runtime=("runtime_seconds", "mean"),
            median_location_error=("location_error", "median"),
        )
        .reset_index()
    )
    return out


def aggregate_bootstrap_estimation(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize conditional-bootstrap estimation records by parameter.

    Args:
        frame: Records containing design keys, parameter labels, truths,
            estimates, and interval summaries. Both generic ``se/covered/width``
            and legacy ``bootstrap_*`` fields are accepted.

    Returns:
        A DataFrame with bias, empirical SD, mean SE, coverage, width, and
        successful counts; empty when there are no parameter records.
        Does not write files or rerun the bootstrap.
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
            rows.append(
                {
                    "scenario": record.scenario,
                    "days": record.days,
                    "magnitude": record.magnitude,
                    "change_fraction": record.change_fraction,
                    "replication": record.replication,
                    "parameter": label,
                    "truth": record.truth[k],
                    "estimate": record.estimate[k],
                    "se": se_values[k],
                    "covered": covered_values[k],
                    "width": width_values[k],
                    "rejected": True,
                }
            )
    long = pd.DataFrame(rows)
    if long.empty:
        return long
    grouped = long.groupby(["scenario", "days", "magnitude", "change_fraction", "parameter"])
    return grouped.agg(
        bias=("estimate", lambda x: float(np.mean(x - long.loc[x.index, "truth"]))),
        empirical_sd=("estimate", "std"),
        mean_se=("se", "mean"),
        coverage=("covered", "mean"),
        width=("width", "mean"),
        successful=("estimate", "size"),
    ).reset_index()


def testing_table(df: pd.DataFrame) -> str:
    """Render the selected testing summaries as LaTeX for Table 2.

    Args:
        df: Aggregated testing results, including design keys and method names.

    Returns:
        LaTeX source with baseline-scale null results and alternatives of
        magnitude 0.025, including DACT median location errors. Writes no files.
    """
    shown_null = df[(df.magnitude == 0) & (df.death_scale == 1.0)]
    shown_alt = df[
        (df.scenario.isin(["A1", "A2", "A3"])) & (df.magnitude == 0.025) & (df.death_scale == 1.0)
    ]
    methods = ("proposed_gaussian", "sup_lm", "cvm", "no_delay")

    def value(part: pd.DataFrame, method: str, field: str = "rejection") -> str:
        """Format the first matching method result in a selected design cell.

        Args:
            part: Rows for the selected simulation design cell.
            method: Method identifier to select.
            field: Numeric result column; defaults to rejection probability.

        Returns:
            A formatted value, or ``--`` for a missing row or value. Location
            errors use zero decimals; other fields use three. Writes no files.
        """
        row = part[part.method == method]
        if row.empty or pd.isna(row.iloc[0][field]):
            return "--"
        if field == "median_location_error":
            return f"{row.iloc[0][field]:.0f}"
        return f"{row.iloc[0][field]:.3f}"

    lines = [
        r"\begin{table}[p]",
        r"\centering",
        (
            "\\caption{Testing performance for the daily age-effect change "
            "test. Panel A reports Type-I error at the baseline "
            "death-information scale $s_n=1$. Panel B reports rejection "
            "probability under $|\\Delta|=0.025$. MedAE is DACT's median "
            "absolute change-day error in days. Each entry uses 1000 "
            "replications.}"
        ),
        r"\label{tab:testing-v4}",
        r"\small",
        r"\renewcommand{\arraystretch}{0.98}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrccccc@{}}",
        r"\toprule",
        r"DGP & $T$ & $\tau_0/T$ & DACT & supLM & CvM & No-delay & MedAE\\",
        r"\midrule",
        (
            "\\multicolumn{8}{l}{\\textit{Panel A: Type-I error at $s_n=1$ under"
            " the four null mechanisms}}\\\\"
        ),
    ]
    for scenario in ("N0", "N1", "N2", "N3"):
        for day_index, days in enumerate((90, 180, 365)):
            part = shown_null[(shown_null.scenario == scenario) & (shown_null.days == days)]
            entries = [value(part, method) for method in methods]
            dgp = scenario if day_index == 0 else ""
            lines.append(f"{dgp} & {days} & -- & " + " & ".join(entries) + r" & --\\")
    lines += [
        r"\midrule",
        r"\multicolumn{8}{l}{\textit{Panel B: rejection probability for $|\Delta|=0.025$}}\\",
    ]
    for scenario in ("A1", "A2", "A3"):
        scenario_rows = shown_alt[shown_alt.scenario == scenario]
        row_index = 0
        for days in (90, 180, 365):
            day_rows = scenario_rows[scenario_rows.days == days]
            for fraction_index, fraction in enumerate(
                sorted(day_rows.change_fraction.dropna().unique())
            ):
                part = day_rows[np.isclose(day_rows.change_fraction, fraction)]
                entries = [value(part, method) for method in methods]
                medae = value(part, "proposed_gaussian", "median_location_error")
                dgp = scenario if row_index == 0 else ""
                day_text = str(days) if fraction_index == 0 else ""
                lines.append(
                    f"{dgp} & {day_text} & {fraction:.2f} & "
                    + " & ".join(entries)
                    + f" & {medae}\\\\"
                )
                row_index += 1
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def estimation_table(df: pd.DataFrame) -> str:
    """Render the selected bootstrap summaries as LaTeX for Table 3.

    Args:
        df: Aggregated parameter-estimation results with design keys and
            bias, SD, SE, coverage, and interval-width columns.

    Returns:
        LaTeX source for T=180, central A1/A2 changes, and the early A3 change,
        in the prescribed parameter order. Writes no files.
    """
    shown = df[
        (df.days == 180)
        & (
            ((df.scenario.isin(["A1", "A2"])) & (df.change_fraction == 0.50))
            | ((df.scenario == "A3") & (df.change_fraction == 0.25))
        )
    ].copy()
    labels = {
        "tau_over_T": r"$\tau/T$",
        "beta_before": r"$\beta_1$",
        "beta_after": r"$\beta_2$",
        "delta": r"$\Delta$",
        "exp_delta": r"$\exp(\Delta)$",
    }
    order = {name: index for index, name in enumerate(labels)}
    shown["parameter_order"] = shown.parameter.map(order)
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        (
            "\\caption{Change-point and segment-parameter estimation for "
            "$T=180$. A1 and A2 have a central change; the A3 panel focuses on"
            " the early sparse case at $\\tau_0/T=0.25$ and 35\\% of the "
            "baseline death information. SD is the empirical standard "
            "deviation across 1000 datasets. Mean SE, coverage, and width use "
            "the search-repeated conditional bootstrap with $B=2000$.}"
        ),
        r"\label{tab:parameter-estimation-v4}",
        r"\small",
        r"\renewcommand{\arraystretch}{0.98}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrr@{}}",
        r"\toprule",
        r"$|\Delta|$ & Parameter & Bias & SD & Mean SE & Coverage & Width\\",
        r"\midrule",
    ]
    scenario_labels = {
        "A1": "abrupt flattening",
        "A2": "abrupt steepening",
        "A3": r"sparse flattening, $\tau_0/T=0.25$",
    }
    available = set(shown.scenario.unique())
    for scenario in ("A1", "A2", "A3"):
        if scenario not in available:
            continue
        lines.append(
            r"\multicolumn{7}{l}{\textit{" + scenario + ": " + scenario_labels[scenario] + r"}}\\"
        )
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
    """Save the simulation sensitivity panels for Figure 2.

    Args:
        frame: Raw replication records or summaries with a rejection column,
            indexed by scenario, sensitivity axis, and value.
        output_dir: Destination directory for figure_sensitivity PDF and PNG.
        require_complete: Check the full-design count of values on each axis.

    Returns:
        None. Creates the destination, changes Matplotlib defaults, and saves
        and closes the figure; an empty frame produces no output.

    Raises:
        ValueError: Required axes or requested full-design values are missing.
    """
    if frame.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = (
        frame.copy()
        if "rejection" in frame.columns
        else frame.groupby(["scenario", "axis", "value"])
        .agg(rejection=("rejected", "mean"), location_error=("location_error", "median"))
        .reset_index()
    )
    plt.rcParams.update(
        {
            "font.size": 9,
            "legend.fontsize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    axes_to_plot = (
        "delay_multiplier",
        "age_delay_mismatch",
        "age_groups",
        "overdispersion",
        "transition_width",
        "day_of_week",
    )
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
        "delay_multiplier": "Analyzed/true\ndelay scale",
        "age_delay_mismatch": "Ignored age-delay\nslope",
        "age_groups": "Number of\nage groups",
        "overdispersion": "Overdispersion",
        "transition_width": "Transition width\n(days)",
        "day_of_week": "Day-of-week\namplitude",
    }
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.9), constrained_layout=True)
    for panel, (ax, axis) in enumerate(zip(axes.ravel(), axes_to_plot)):
        active = summary[summary.axis == axis]
        for scenario, part in active.groupby("scenario"):
            part = part.sort_values("value")
            style = {"A1": ("o", "-", "#0072B2"), "N0": ("s", "--", "#A64B00")}[scenario]
            ax.plot(
                part.value,
                part.rejection,
                marker=style[0],
                ls=style[1],
                color=style[2],
                ms=4,
                lw=1.2,
                label=scenario,
            )
        ax.axhline(0.05, color="black", ls="--", lw=0.8)
        ax.set_xlabel(axis_labels[axis])
        ax.set_ylabel("Rejection probability" if panel % 3 == 0 else "")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.15)
        ax.text(
            0.02, 0.96, f"({chr(97 + panel)})", transform=ax.transAxes, va="top", fontweight="bold"
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.93))
    # Figure 2: simulation sensitivity across the six prespecified axes.
    fig.savefig(output_dir / "figure_sensitivity.pdf", bbox_inches=None)
    fig.savefig(output_dir / "figure_sensitivity.png", dpi=400, bbox_inches=None)
    plt.close(fig)


def power_figure(df: pd.DataFrame, output_dir: Path):
    """Save the DACT power panels for Figure 1.

    Args:
        df: Aggregated testing results with design keys and rejection rates.
        output_dir: Destination for figure_power_surface PDF and PNG files.

    Returns:
        None. Creates the destination and saves and closes the figure using
        positive-magnitude A1/A2 results at baseline death scale. An empty
        frame produces no output; plotting defaults are restored on exit.
    """
    if df.empty:
        return
    active = df[(df.method == "proposed_gaussian") & (df.magnitude > 0) & (df.death_scale == 1.0)]
    with plt.rc_context(
        {
            "font.size": 9,
            "legend.fontsize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        fig, axes = plt.subplots(
            2, 3, figsize=(6.5, 4.3), sharex=True, sharey=True, constrained_layout=True
        )
        styles = (("o", "-", "#0072B2"), ("s", "--", "#A64B00"), ("^", "-.", "#333333"))
        for row, scenario in enumerate(("A1", "A2")):
            for col, days in enumerate((90, 180, 365)):
                ax = axes[row, col]
                cell = active[(active.scenario == scenario) & (active.days == days)]
                for (fraction, part), (marker, line, color) in zip(
                    cell.groupby("change_fraction"), styles
                ):
                    part = part.sort_values("magnitude")
                    ax.plot(
                        part.magnitude,
                        part.rejection,
                        marker=marker,
                        ls=line,
                        color=color,
                        ms=4,
                        lw=1.2,
                        label=f"$\\tau/T={fraction:.2f}$",
                    )
                ax.text(
                    0.04,
                    0.95,
                    f"({chr(97 + row * 3 + col)})",
                    transform=ax.transAxes,
                    va="top",
                    fontweight="bold",
                )
                if row == 1:
                    ax.set_xlabel("Age-effect change\nmagnitude")
                if col == 0:
                    ax.set_ylabel("Power")
                ax.set_ylim(0, 1.05)
                ax.set_xticks([0.01, 0.025, 0.04])
                ax.grid(alpha=0.15)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.get_layout_engine().set(rect=(0, 0, 1, 0.90))
        output_dir.mkdir(parents=True, exist_ok=True)
        # Figure 1: power by change magnitude, location, and series length.
        fig.savefig(output_dir / "figure_power_surface.pdf", bbox_inches=None)
        fig.savefig(output_dir / "figure_power_surface.png", dpi=400, bbox_inches=None)
        plt.close(fig)


def run(mode: str):
    """Aggregate saved simulation records and write manuscript artifacts.

    Args:
        mode: Input/output prefix, normally ``demo`` or ``full``. Full mode
            uses manuscript directories and requires complete sensitivity axes.

    Returns:
        A dictionary of testing-summary, estimation-summary, and raw sensitivity
        row counts. Reads JSON-lines inputs and writes available CSV summaries,
        Tables 2 and 3, Figure 1 power, and Figure 2 simulation sensitivity;
        demo artifacts use separate directories. Does not run simulations.
    """
    ensure_directories()
    testing_raw = _load(RAW_DIR / f"{mode}_testing.jsonl")
    interval_raw = _load(RAW_DIR / f"{mode}_intervals.jsonl")
    sensitivity_raw = _load(RAW_DIR / f"{mode}_sensitivity.jsonl")
    testing = aggregate_testing(testing_raw) if not testing_raw.empty else pd.DataFrame()
    estimation = (
        aggregate_bootstrap_estimation(interval_raw) if not interval_raw.empty else pd.DataFrame()
    )
    output_dir = FIGURE_DIR if mode == "full" else AGGREGATED_DIR / "demo_figures"
    table_dir = TABLE_DIR if mode == "full" else AGGREGATED_DIR / "demo_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    AGGREGATED_DIR.mkdir(parents=True, exist_ok=True)
    if not testing.empty:
        testing.to_csv(AGGREGATED_DIR / f"{mode}_testing.csv", index=False)
        # Table 2: testing performance and change-location error.
        (table_dir / "table_testing.tex").write_text(testing_table(testing), encoding="utf-8")
        power_figure(testing, output_dir)
    if not estimation.empty:
        estimation.to_csv(AGGREGATED_DIR / f"{mode}_estimation.csv", index=False)
        # Table 3: bootstrap parameter-estimation performance.
        (table_dir / "table_estimation.tex").write_text(
            estimation_table(estimation), encoding="utf-8"
        )
    if not sensitivity_raw.empty:
        sensitivity_summary = (
            sensitivity_raw.groupby(["scenario", "axis", "value"])
            .agg(
                rejection=("rejected", "mean"),
                location_error=("location_error", "median"),
                replications=("rejected", "size"),
            )
            .reset_index()
        )
        sensitivity_summary.to_csv(AGGREGATED_DIR / f"{mode}_sensitivity.csv", index=False)
    sensitivity_figures(sensitivity_raw, output_dir, require_complete=(mode == "full"))
    return {
        "testing_rows": len(testing),
        "estimation_rows": len(estimation),
        "sensitivity_rows": len(sensitivity_raw),
    }


def main():
    """Parse the required --mode argument and aggregate saved simulations.

    Returns:
        None. Reads command-line arguments, writes artifacts through run, and
        prints the returned row counts to standard output.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("demo", "full"), required=True)
    args = parser.parse_args()
    print(run(args.mode))


if __name__ == "__main__":
    main()
