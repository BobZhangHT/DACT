"""Daily generators for the conditional delay-adjusted age-gradient test."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from scipy.stats import gamma

from config import CONFIG, SCENARIOS, DGPConfig
from methods import delay_adjusted_exposure_python


@dataclass
class SimulationData:
    cases: np.ndarray
    exposure: np.ndarray
    deaths: np.ndarray
    age_scores: np.ndarray
    delay_analysis: np.ndarray
    delay_truth: np.ndarray
    true_beta: np.ndarray
    true_change: int | None
    scenario: str
    setting: str

    @property
    def days(self) -> int:
        return int(self.deaths.shape[0])


def _standard_scores(groups: int) -> np.ndarray:
    if groups == len(CONFIG.dgp.age_scores):
        return np.asarray(CONFIG.dgp.age_scores, dtype=float)
    values = np.linspace(-2.5, 2.5, groups)
    return values - values.mean()


def _weights(groups: int, cfg: DGPConfig) -> np.ndarray:
    if groups == len(cfg.age_weights):
        return np.asarray(cfg.age_weights, dtype=float)
    x = np.linspace(-1.0, 1.0, groups)
    value = np.exp(-0.45 * x * x)
    return value / value.sum()


def delay_mass(groups: int, *, scale_multiplier: float = 1.0,
               age_effect: float = 0.0, heavy_tail: bool = False,
               cfg: DGPConfig = CONFIG.dgp) -> np.ndarray:
    edge = np.arange(cfg.delay_max + 2, dtype=float)
    centered = np.linspace(-1.0, 1.0, groups)
    result = np.empty((groups, cfg.delay_max + 1), dtype=float)
    for j in range(groups):
        scale = cfg.delay_scale * scale_multiplier * np.exp(age_effect * centered[j])
        if heavy_tail:
            cdf = 0.8 * gamma.cdf(edge, cfg.delay_shape, scale=scale)
            cdf += 0.2 * gamma.cdf(edge, 1.4, scale=2.4 * scale)
        else:
            cdf = gamma.cdf(edge, cfg.delay_shape, scale=scale)
        mass = np.diff(cdf)
        result[j] = mass / mass.sum()
    return result


def _beta_path(days: int, setting: str, change_fraction: float,
               change_magnitude: float | None, transition_width: int,
               cfg: DGPConfig) -> tuple[np.ndarray, int | None]:
    if setting in {"baseline", "level_jump", "composition_shift", "age_delay"}:
        # A baseline scenario becomes an alternative only when explicitly
        # supplied a positive change magnitude.
        if change_magnitude is None:
            return np.full(days, cfg.beta_before), None
        after = cfg.beta_before - float(change_magnitude)
    elif setting == "steepening":
        after = cfg.beta_before + float(change_magnitude or abs(cfg.beta_after - cfg.beta_before))
    elif setting == "sparse":
        after = cfg.beta_before - float(change_magnitude or abs(cfg.beta_after - cfg.beta_before))
    else:
        raise ValueError(setting)
    change = int(np.clip(round(change_fraction * days), 2, days - 3))
    if transition_width > 0:
        weight = expit((np.arange(days) - change) / max(transition_width / 4.0, 0.5))
        return cfg.beta_before + (after - cfg.beta_before) * weight, change
    beta = np.full(days, cfg.beta_before)
    beta[change:] = after
    return beta, change


def generate_dataset(
    scenario: str,
    rng: np.random.Generator,
    *,
    days: int | None = None,
    change_fraction: float | None = None,
    change_magnitude: float | None = None,
    death_scale: float = 1.0,
    age_groups: int | None = None,
    overdispersion: float = 0.0,
    transition_width: int = 0,
    composition_shift: float | None = None,
    delay_multiplier: float = 1.0,
    analysis_delay_multiplier: float = 1.0,
    age_specific_delay: float | None = None,
    analyze_common_delay: bool = False,
    heavy_tail: bool = False,
    day_of_week: float = 0.0,
    cfg: DGPConfig = CONFIG.dgp,
) -> SimulationData:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    spec = SCENARIOS[scenario]
    T = int(days or cfg.case_days)
    J = int(age_groups or len(cfg.age_scores))
    fraction = float(spec.get("change_fraction", cfg.change_fraction)
                     if change_fraction is None else change_fraction)
    change_for_design = int(np.clip(round(fraction * T), 2, T - 3))
    scores = _standard_scores(J)
    base_weight = _weights(J, cfg)
    shift = (0.25 if spec["setting"] == "composition_shift" else 0.0)
    if composition_shift is not None:
        shift = float(composition_shift)
    time = np.arange(T, dtype=float)
    epidemic = 1.0 + cfg.case_amplitude * np.sin(2 * np.pi * time / T - 0.8)
    epidemic += 0.18 * np.sin(4 * np.pi * time / T + 0.2)
    totals = np.maximum(80, np.rint(cfg.daily_cases * epidemic)).astype(int)
    cases = np.empty((T, J), dtype=int)
    direction = np.linspace(-1.0, 1.0, J)
    for t in range(T):
        weight = base_weight.copy()
        if shift and t >= change_for_design:
            weight *= np.exp(shift * direction)
            weight /= weight.sum()
        cases[t] = rng.multinomial(int(totals[t]), weight)

    age_delay = 0.22 if spec["setting"] == "age_delay" else 0.0
    if age_specific_delay is not None:
        age_delay = float(age_specific_delay)
    truth_delay = delay_mass(J, scale_multiplier=delay_multiplier,
                             age_effect=age_delay, heavy_tail=heavy_tail, cfg=cfg)
    analysis_delay = delay_mass(J, scale_multiplier=analysis_delay_multiplier, cfg=cfg)
    if age_delay and not analyze_common_delay and analysis_delay_multiplier == delay_multiplier:
        analysis_delay = truth_delay.copy()
    truth_exposure = delay_adjusted_exposure_python(cases, truth_delay, T)
    exposure = delay_adjusted_exposure_python(cases, analysis_delay, T)
    beta, change = _beta_path(
        T, str(spec["setting"]), fraction,
        (abs(cfg.beta_after - cfg.beta_before) if spec["change"] and change_magnitude is None
         else change_magnitude), transition_width, cfg,
    )
    eta = np.log(np.maximum(truth_exposure, 1e-12)) + beta[:, None] * scores[None, :]
    eta -= eta.max(axis=1, keepdims=True)
    probability = np.exp(eta)
    probability /= probability.sum(axis=1, keepdims=True)
    mean_total = cfg.daily_deaths * death_scale * (
        1.0 + cfg.death_amplitude * np.sin(2 * np.pi * time / T + 0.4)
    )
    if spec["setting"] == "level_jump":
        mean_total[change_for_design:] *= 0.55
    if spec["setting"] == "sparse":
        mean_total *= 0.35
    if day_of_week:
        mean_total *= np.exp(day_of_week * np.cos(2 * np.pi * time / 7.0))
    deaths = np.zeros((T, J), dtype=int)
    for t in range(T):
        n = int(rng.poisson(max(mean_total[t], 1.0)))
        active = probability[t]
        if overdispersion > 0:
            concentration = max(1.0, 1.0 / overdispersion)
            active = rng.dirichlet(concentration * active)
        deaths[t] = rng.multinomial(n, active)
    return SimulationData(cases, exposure, deaths, scores, analysis_delay,
                          truth_delay, beta, change, scenario, str(spec["setting"]))


__all__ = ["SimulationData", "delay_mass", "generate_dataset"]
