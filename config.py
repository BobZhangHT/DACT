"""Frozen configuration for the daily delay-adjusted age-effect study."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results_conditional_delay"
RAW_DIR = RESULTS_DIR / "raw"
AGGREGATED_DIR = RESULTS_DIR / "aggregated"
PAPER_DIR = ROOT / "paper"
FIGURE_DIR = PAPER_DIR / "figures"
TABLE_DIR = PAPER_DIR / "tables"

Mode = Literal["demo", "full"]
Backend = Literal["c", "python"]


@dataclass(frozen=True)
class DGPConfig:
    """Immutable defaults for observation horizons, age composition, epidemic intensity,
    gradient changes, and discretized gamma delay distributions.
    """

    case_days: int = 180
    followup_days: int = 70
    age_scores: tuple[float, ...] = (-3.0, -1.5, -0.5, 0.5, 1.5, 2.5)
    age_weights: tuple[float, ...] = (0.18, 0.13, 0.16, 0.19, 0.18, 0.16)
    daily_cases: int = 1200
    case_amplitude: float = 0.45
    daily_deaths: int = 145
    death_amplitude: float = 0.35
    beta_before: float = 0.72
    beta_after: float = 0.45
    change_fraction: float = 0.50
    delay_shape: float = 3.0
    delay_scale: float = 4.5
    delay_max: int = 90


@dataclass(frozen=True)
class MethodConfig:
    """Immutable backend, calibration, root-solver, trimming, and maturity settings. Consumers
    use the settings relevant to their procedure.
    """

    backend: Backend = "c"
    alpha: float = 0.05
    bootstrap_samples: int = 2000
    gaussian_samples: int = 5000
    root_tolerance: float = 1e-10
    root_max_iterations: int = 100
    beta_bounds: tuple[float, float] = (-4.0, 4.0)
    minimum_information_fraction: float = 0.15
    calendar_trim: float = 0.0
    lm_trim: float = 0.05
    maturity: float = 0.90


@dataclass(frozen=True)
class SimulationConfig:
    """Immutable random seeds, replication counts, scenario lists, method names, and
    sensitivity grids for simulation drivers.
    """

    master_seed: int = 20260809
    demo_replications: int = 100
    full_replications: int = 1000
    gaussian_calibration_samples: int = 1000
    testing_scenarios: tuple[str, ...] = ("N0", "N1", "N2", "N3", "A1", "A2", "A3")
    estimation_scenarios: tuple[str, ...] = ("A1", "A2", "A3")
    methods: tuple[str, ...] = ("proposed_gaussian", "sup_lm", "cvm", "no_delay")
    case_day_grid: tuple[int, ...] = (90, 180, 365)
    beta_change_grid: tuple[float, ...] = (0.0, 0.10, 0.18, 0.27, 0.36)
    change_fraction_grid: tuple[float, ...] = (0.20, 0.35, 0.50, 0.65, 0.80)
    death_scale_grid: tuple[float, ...] = (0.35, 0.70, 1.00, 1.50)
    age_group_grid: tuple[int, ...] = (3, 5, 6, 8)
    overdispersion_grid: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10)
    delay_multiplier_grid: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2)
    transition_width_grid: tuple[int, ...] = (0, 2, 7, 14, 28)

    def replications(self, mode: Mode) -> int:
        """Select the configured replication count for a run mode.

        Args:
            self: Instance whose stored configuration, data, or diagnostics are accessed.
            mode: Replication mode, "demo" or "full".

        Returns:
            Integer demo or full replication count; unknown modes raise ValueError.
        """
        if mode == "demo":
            return self.demo_replications
        if mode == "full":
            return self.full_replications
        raise ValueError(f"unknown mode: {mode}")


@dataclass(frozen=True)
class ProjectConfig:
    """Immutable container grouping data-generation, method, and simulation settings."""

    dgp: DGPConfig = field(default_factory=DGPConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    def as_dict(self) -> dict:
        """Convert the nested project configuration to ordinary dictionaries.

        Args:
            self: Instance whose stored configuration, data, or diagnostics are accessed.

        Returns:
            Recursively converted dictionary produced by dataclasses.asdict.
        """
        return asdict(self)


CONFIG = ProjectConfig()

SCENARIOS: dict[str, dict] = {
    "N0": {"label": "stable gradient", "change": False, "setting": "baseline"},
    "N1": {"label": "common death-level change", "change": False, "setting": "level_jump"},
    "N2": {"label": "case-age composition change", "change": False, "setting": "composition_shift"},
    "N3": {"label": "age-specific delay", "change": False, "setting": "age_delay"},
    "A1": {"label": "central flattening", "change": True, "setting": "baseline"},
    "A2": {"label": "central steepening", "change": True, "setting": "steepening"},
    "A3": {
        "label": "sparse flattening",
        "change": True,
        "setting": "sparse",
        "change_fraction": 0.25,
    },
}


def ensure_directories() -> None:
    """Create the raw, aggregated, figure, and table output directories.

    Args:
        None.

    Returns:
        None. Creates missing directories and parents; existing directories are retained.
    """
    for path in (RAW_DIR, AGGREGATED_DIR, FIGURE_DIR, TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def validate_config(config: ProjectConfig = CONFIG) -> None:
    """Check age-weight consistency and fixed manuscript simulation counts.

    Args:
        config: Project configuration to check against age-weight and manuscript-design
            constraints.

    Returns:
        None. Raises ValueError when scores/weights disagree or required
        calibration/replication counts differ.
    """
    dgp = config.dgp
    if len(dgp.age_scores) != len(dgp.age_weights):
        raise ValueError("age scores and weights must have equal length")
    if not np.isclose(sum(dgp.age_weights), 1.0):
        raise ValueError("age weights must sum to one")
    if config.method.bootstrap_samples != 2000 or config.method.gaussian_samples != 5000:
        raise ValueError("the manuscript fixes B=2000 and M=5000")
    if config.simulation.demo_replications != 100 or config.simulation.full_replications != 1000:
        raise ValueError("demo/full modes use 100/1000 replications")


validate_config()
