"""Delay-adjusted age-effect change tests with equivalent C/Python backends."""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.ctypeslib import ndpointer

from config import CONFIG, Backend, MethodConfig


ROOT = Path(__file__).resolve().parent
C_SOURCE = ROOT / "csrc" / "cfr_core.c"
_C_VERSION = hashlib.sha256(C_SOURCE.read_bytes()).hexdigest()[:12]
C_LIBRARY = ROOT / "csrc" / (
    f"cfr_core_{_C_VERSION}.dll" if os.name == "nt" else f"libcfr_core_{_C_VERSION}.so"
)
_C_HANDLE: ctypes.CDLL | None = None
_C_LOCK = threading.RLock()


@dataclass
class GradientFit:
    beta: float
    loglik: float
    score: np.ndarray
    information: np.ndarray
    probabilities: np.ndarray
    success: bool
    iterations: int
    backend: str
    status: str = "converged"
    boundary: bool = False


@dataclass
class ChangeTestResult:
    statistic: float
    critical_value: float
    p_value: float
    rejected: bool
    estimated_change: int
    process: np.ndarray
    information_time: np.ndarray
    null_fit: GradientFit
    calibration: str
    bootstrap_statistics: np.ndarray | None = None
    bootstrap_diagnostics: dict | None = None


@dataclass
class ChangeEstimate:
    change: int
    beta_before: float
    beta_after: float
    null_beta: float
    success: bool
    status: str = "converged"


class BootstrapFitError(RuntimeError):
    """Too few valid draws; diagnostics retain every attempted fit's status."""

    def __init__(self, diagnostics: dict):
        self.diagnostics = diagnostics
        super().__init__(f"insufficient valid bootstrap fits: {diagnostics['n_successful']}"
                         f"/{diagnostics['n_requested']}; failures were excluded, not retried")


_FIT_STATUS = {0: "converged", 1: "boundary", 2: "max_iterations",
               3: "nonfinite", 4: "no_information", 5: "no_admissible_change"}


def _compiler() -> str:
    candidates = [
        os.environ.get("CC"), r"C:\msys64\mingw64\bin\gcc.exe",
        r"C:\Program Files\LLVM\bin\clang.exe", "clang", "gcc",
    ]
    from shutil import which
    for candidate in candidates:
        if candidate and (Path(candidate).exists() or which(candidate)):
            return str(Path(candidate).resolve()) if Path(candidate).exists() else str(which(candidate))
    raise RuntimeError("No C compiler found")


def build_c_backend(force: bool = False) -> Path:
    with _C_LOCK:
        stale = not C_LIBRARY.exists() or C_LIBRARY.stat().st_mtime < C_SOURCE.stat().st_mtime
        if force or stale:
            compiler = _compiler()
            command = [compiler, "-O3", "-std=c11", "-shared", "-static-libgcc", str(C_SOURCE), "-o", str(C_LIBRARY)]
            if os.name != "nt":
                command.append("-lm")
            env = os.environ.copy()
            env["PATH"] = str(Path(compiler).parent) + os.pathsep + env.get("PATH", "")
            result = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
            if result.returncode:
                raise RuntimeError(result.stdout + "\n" + result.stderr)
    return C_LIBRARY


def _load_c_backend() -> ctypes.CDLL:
    global _C_HANDLE
    if _C_HANDLE is None:
        with _C_LOCK:
            if _C_HANDLE is None:
                build_c_backend()
                if os.name == "nt" and hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(Path(_compiler()).parent))
                lib = ctypes.CDLL(str(C_LIBRARY))
                arr = ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
                lib.cfr_conditional_components.argtypes = [
                    ctypes.c_int, ctypes.c_int, arr, arr, arr, ctypes.c_double, arr, arr,
                ]
                lib.cfr_conditional_components.restype = ctypes.c_double
                lib.cfr_delay_exposure.argtypes = [
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    arr, arr, arr,
                ]
                lib.cfr_delay_exposure.restype = ctypes.c_int
                lib.cfr_fit_beta.argtypes = [
                    ctypes.c_int, ctypes.c_int, arr, arr, arr,
                    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int,
                    arr, arr, ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
                ]
                lib.cfr_fit_beta.restype = ctypes.c_int
                lib.cfr_bootstrap_change_batch.argtypes = [
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, arr, arr, arr,
                    ctypes.c_double,
                    ctypes.c_double, ctypes.c_double, ctypes.c_double,
                    ctypes.c_int,
                    ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
                    arr, arr,
                    ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
                ]
                lib.cfr_bootstrap_change_batch.restype = ctypes.c_int
                _C_HANDLE = lib
    return _C_HANDLE


def resolve_backend(backend: Backend | None = None) -> Backend:
    active = backend or CONFIG.method.backend
    if active not in ("c", "python"):
        raise ValueError("backend must be 'c' or 'python'")
    if active == "c":
        _load_c_backend()
    return active


def _arrays(cases: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray) -> tuple[np.ndarray, ...]:
    c = np.ascontiguousarray(cases, dtype=np.float64)
    d = np.ascontiguousarray(deaths, dtype=np.float64)
    a = np.ascontiguousarray(age_scores, dtype=np.float64)
    if c.ndim != 2 or 0 in c.shape or d.shape != c.shape or a.shape != (c.shape[1],):
        raise ValueError("cases/deaths must be T by J and age_scores must have length J")
    if not all(np.all(np.isfinite(x)) for x in (c, d, a)):
        raise ValueError("exposures, deaths and age_scores must be finite")
    if np.any(c < 0) or np.any(d < 0):
        raise ValueError("exposures and deaths must be nonnegative")
    if np.any((c == 0) & (d > 0)):
        raise ValueError("positive deaths with zero exposure are incompatible")
    return c, d, a


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_fit_config(cfg: MethodConfig) -> None:
    if (len(cfg.beta_bounds) != 2 or not np.all(np.isfinite(cfg.beta_bounds))
            or cfg.beta_bounds[0] >= cfg.beta_bounds[1]):
        raise ValueError("beta_bounds must be finite and strictly increasing")
    if not np.isfinite(cfg.root_tolerance) or cfg.root_tolerance <= 0:
        raise ValueError("root_tolerance must be finite and positive")
    _positive_integer(cfg.root_max_iterations, "root_max_iterations")
    if any(not np.isfinite(x) or not 0 <= x < 0.5
           for x in (cfg.minimum_information_fraction, cfg.lm_trim)):
        raise ValueError("information trimming fractions must lie in [0, 0.5)")


def _delay_arrays(cases, delay, outcome_days):
    c = np.ascontiguousarray(cases, dtype=np.float64)
    f = np.ascontiguousarray(delay, dtype=np.float64)
    if c.ndim != 2 or f.ndim != 2 or 0 in c.shape or 0 in f.shape or c.shape[1] != f.shape[0]:
        raise ValueError("cases must be nonempty R x J and delay must be J x L")
    if (not np.all(np.isfinite(c)) or not np.all(np.isfinite(f))
            or np.any(c < 0) or np.any(f < 0) or not np.allclose(f.sum(axis=1), 1.0)):
        raise ValueError("counts and delay masses must be finite, nonnegative and normalized")
    return c, f, c.shape[0] if outcome_days is None else _positive_integer(outcome_days, "outcome_days")


def delay_adjusted_exposure_python(
    cases: np.ndarray, delay: np.ndarray, outcome_days: int | None = None,
) -> np.ndarray:
    """Convolve daily age-specific cases with the case-to-death delay mass."""
    c, f, T = _delay_arrays(cases, delay, outcome_days)
    output = np.zeros((T, c.shape[1]), dtype=float)
    for j in range(c.shape[1]):
        convolution = np.convolve(c[:, j], f[j])[:T]
        output[:len(convolution), j] = convolution
    if not np.all(np.isfinite(output)):
        raise ValueError("delay convolution produced nonfinite exposure")
    return output


def delay_adjusted_exposure_c(
    cases: np.ndarray, delay: np.ndarray, outcome_days: int | None = None,
) -> np.ndarray:
    c, f, T = _delay_arrays(cases, delay, outcome_days)
    output = np.empty((T, c.shape[1]), dtype=np.float64)
    status = _load_c_backend().cfr_delay_exposure(
        c.shape[0], T, c.shape[1], f.shape[1], c.ravel(), f.ravel(), output.ravel()
    )
    if status:
        raise RuntimeError("C delay convolution failed")
    if not np.all(np.isfinite(output)):
        raise ValueError("delay convolution produced nonfinite exposure")
    return output


def delay_adjusted_exposure(
    cases: np.ndarray, delay: np.ndarray, outcome_days: int | None = None,
    backend: Backend | None = None,
) -> np.ndarray:
    fn = delay_adjusted_exposure_c if resolve_backend(backend) == "c" else delay_adjusted_exposure_python
    return fn(cases, delay, outcome_days)


def conditional_probabilities(cases: np.ndarray, age_scores: np.ndarray, beta: float) -> np.ndarray:
    c, _, a = _arrays(cases, np.zeros_like(cases), age_scores)
    return np.exp(_log_probabilities(c, a, beta))


def _log_probabilities(c, a, beta):
    if not np.isfinite(beta):
        raise ValueError("beta must be finite")
    eta = np.full_like(c, -np.inf)
    np.log(c, out=eta, where=c > 0)
    eta += float(beta) * a[None, :]
    # An all-zero exposure row has zero deaths. Its conditional distribution
    # is irrelevant (N=0); use uniform probabilities so multinomial sampling works.
    eta[~np.any(c > 0, axis=1)] = 0.0
    eta -= eta.max(axis=1, keepdims=True)
    return eta - np.log(np.exp(eta).sum(axis=1, keepdims=True))


def conditional_components_python(
    cases: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray, beta: float
) -> tuple[float, np.ndarray, np.ndarray]:
    c, d, a = _arrays(cases, deaths, age_scores)
    log_probability = _log_probabilities(c, a, beta)
    probability = np.exp(log_probability)
    totals = d.sum(axis=1)
    mean_age = probability @ a
    observed_age = d @ a
    score = observed_age - totals * mean_age
    information = totals * np.sum(probability * (a[None, :] - mean_age[:, None]) ** 2, axis=1)
    positive = d > 0
    loglik = float(np.sum(d[positive] * log_probability[positive]))
    return loglik, score, information


def conditional_components_c(
    cases: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray, beta: float
) -> tuple[float, np.ndarray, np.ndarray]:
    c, d, a = _arrays(cases, deaths, age_scores)
    if not np.isfinite(beta):
        raise ValueError("beta must be finite")
    score = np.empty(c.shape[0], dtype=np.float64)
    information = np.empty(c.shape[0], dtype=np.float64)
    loglik = _load_c_backend().cfr_conditional_components(
        c.shape[0], c.shape[1], c.ravel(), d.ravel(), a, float(beta), score, information
    )
    return float(loglik), score, information


def conditional_components(
    cases: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray, beta: float,
    backend: Backend | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    fn = conditional_components_c if resolve_backend(backend) == "c" else conditional_components_python
    return fn(cases, deaths, age_scores, beta)


def _gradient_fit(c, d, a, beta, iterations, backend, status, cfg):
    fn = conditional_components_c if backend == "c" else conditional_components_python
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        loglik, score, information = fn(c, d, a, beta)
        probabilities = conditional_probabilities(c, a, beta)
        finite = (np.isfinite(loglik) and np.isfinite(beta)
                  and all(np.all(np.isfinite(x)) for x in (score, information, probabilities))
                  and np.isfinite(information.sum()) and np.isfinite(score.sum()))
    boundary = bool(beta == cfg.beta_bounds[0] or beta == cfg.beta_bounds[1])
    if not finite:
        status = "nonfinite"
    elif information.sum() <= 0:
        status = "no_information"
    elif boundary:
        status = "boundary"
    return GradientFit(float(beta), loglik, score, information, probabilities,
                       status == "converged", iterations, backend, status, boundary)


def fit_gradient_python(
    cases: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray,
    cfg: MethodConfig = CONFIG.method,
) -> GradientFit:
    c, d, a = _arrays(cases, deaths, age_scores)
    _validate_fit_config(cfg)
    lower, upper = cfg.beta_bounds
    _, score_lower, _ = conditional_components_python(c, d, a, lower)
    _, score_upper, _ = conditional_components_python(c, d, a, upper)
    iterations = 0
    status = "max_iterations"
    beta = 0.5 * lower + 0.5 * upper
    if not np.isfinite(score_lower.sum()) or not np.isfinite(score_upper.sum()):
        status = "nonfinite"
    elif score_lower.sum() <= 0:
        beta, status = lower, "boundary"
    elif score_upper.sum() >= 0:
        beta, status = upper, "boundary"
    else:
        for iterations in range(1, cfg.root_max_iterations + 1):
            beta = 0.5 * lower + 0.5 * upper
            _, score, _ = conditional_components_python(c, d, a, beta)
            value = score.sum()
            if not np.isfinite(value):
                status = "nonfinite"
                break
            if abs(value) <= cfg.root_tolerance or upper - lower <= cfg.root_tolerance:
                status = "converged"
                break
            if value > 0:
                lower = beta
            else:
                upper = beta
    return _gradient_fit(c, d, a, beta, iterations, "python", status, cfg)


def fit_gradient_c(
    cases: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray,
    cfg: MethodConfig = CONFIG.method,
) -> GradientFit:
    c, d, a = _arrays(cases, deaths, age_scores)
    _validate_fit_config(cfg)
    beta = np.empty(1, dtype=np.float64)
    loglik = np.empty(1, dtype=np.float64)
    iterations = np.zeros(1, dtype=np.int32)
    status = _load_c_backend().cfr_fit_beta(
        c.shape[0], c.shape[1], c.ravel(), d.ravel(), a,
        cfg.beta_bounds[0], cfg.beta_bounds[1], cfg.root_tolerance,
        cfg.root_max_iterations, beta, loglik, iterations,
    )
    if status < 0:
        raise RuntimeError(f"C fit failed with status {status}")
    return _gradient_fit(c, d, a, beta[0], int(iterations[0]), "c", _FIT_STATUS[status], cfg)


def fit_gradient(
    cases: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray,
    backend: Backend | None = None, cfg: MethodConfig = CONFIG.method,
) -> GradientFit:
    fn = fit_gradient_c if resolve_backend(backend) == "c" else fit_gradient_python
    return fn(cases, deaths, age_scores, cfg)


Functional = Literal["cusum", "sup_lm", "cvm"]


def _kolmogorov_sf(x: float) -> float:
    """Survival function from the Brownian-bridge series in the paper."""
    if x <= 0:
        return 1.0
    total = 0.0
    for r in range(1, 10000):
        term = 2.0 * ((-1.0) ** (r - 1)) * np.exp(-2.0 * r * r * x * x)
        total += term
        if abs(term) < 1.0e-15:
            break
    return float(np.clip(total, 0.0, 1.0))


def _kolmogorov_quantile(probability: float) -> float:
    """Invert the closed-form Kolmogorov survival function by bisection."""
    target = 1.0 - float(probability)
    lower, upper = 1.0e-8, 5.0
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if _kolmogorov_sf(midpoint) > target:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _process_from_fit(fit: GradientFit, kind: Functional = "cusum",
                      cfg: MethodConfig = CONFIG.method) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not fit.success:
        raise ValueError(f"unusable gradient fit: {fit.status}")
    if (not np.isfinite(fit.beta) or not np.isfinite(fit.loglik)
            or not all(np.all(np.isfinite(x)) for x in (fit.score, fit.information, fit.probabilities))):
        raise ValueError("nonfinite gradient fit")
    total_information = float(fit.information.sum())
    if not np.isfinite(total_information) or total_information <= 0:
        raise ValueError("the data contain no age-gradient information")
    cumulative_information = np.cumsum(fit.information)
    information_time = cumulative_information / total_information
    cumulative_score = np.cumsum(fit.score)
    process = cumulative_score / np.sqrt(total_information)
    # DACT searches the information-trimmed interval (15% by default).
    # supLM additionally applies lm_trim; CvM integrates the full bridge.
    fraction = (max(cfg.minimum_information_fraction, cfg.lm_trim)
                if kind == "sup_lm" else cfg.minimum_information_fraction)
    admissible = np.flatnonzero(
        (information_time[:-1] >= fraction) & (information_time[:-1] <= 1.0 - fraction)
    )
    if not len(admissible):
        raise ValueError("no admissible change location")
    return process, information_time, admissible


def _observed_statistic(
    cases: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray,
    backend: Backend | None, kind: Functional = "cusum",
    cfg: MethodConfig = CONFIG.method,
) -> tuple[float, int, np.ndarray, np.ndarray, GradientFit]:
    fit = fit_gradient(cases, deaths, age_scores, backend, cfg)
    return _statistic_from_fit(fit, kind, cfg)


def _statistic_from_fit(fit, kind, cfg):
    process, information_time, admissible = _process_from_fit(fit, kind, cfg)
    local = np.abs(process[admissible])
    index = int(admissible[int(np.argmax(local))])
    if kind == "cusum":
        statistic = float(local.max())
    elif kind == "sup_lm":
        u = information_time[admissible]
        statistic = float(np.max(process[admissible] ** 2 / (u * (1.0 - u))))
    else:
        u = np.r_[0.0, information_time[:-1], 1.0]
        values = np.r_[0.0, process[:-1] ** 2, 0.0]
        statistic = float(np.trapezoid(values, u))
    return statistic, index + 1, process, information_time, fit


def _bootstrap_diagnostics(statuses):
    statuses = np.asarray(statuses, dtype=str)
    success = statuses == "converged"
    return {
        "policy": "exclude failed/nonfinite draws; no retries; require >=2 valid draws",
        "n_requested": len(statuses), "n_successful": int(success.sum()),
        "n_failed": int((~success).sum()), "n_retried": 0,
        "success_mask": success, "fit_status": statuses,
        "failure_counts": {str(s): int(np.count_nonzero(statuses == s))
                           for s in np.unique(statuses[~success])},
    }


def _bootstrap_totals(deaths):
    totals = np.sum(deaths, axis=1)
    if (not np.all(np.isfinite(totals)) or np.any(totals != np.floor(totals))
            or np.any(totals >= np.iinfo(np.int64).max)):
        raise ValueError("bootstrap death totals must be finite integers in int64 range")
    return totals.astype(np.int64)


def _bootstrap_statistics(
    cases: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray,
    fit: GradientFit, rng: np.random.Generator, backend: Backend | None,
    kind: Functional, samples: int, cfg: MethodConfig = CONFIG.method,
    diagnostics: dict | None = None,
) -> np.ndarray:
    totals = _bootstrap_totals(deaths)
    values = np.full(samples, np.nan)
    statuses = []
    for b in range(samples):
        sampled = np.vstack([
            rng.multinomial(int(totals[t]), fit.probabilities[t]) for t in range(cases.shape[0])
        ])
        fitted = fit_gradient(cases, sampled, age_scores, backend, cfg)
        status = (fitted.status if fitted.status != "converged" else "fit_failed") if not fitted.success else "converged"
        if fitted.success:
            try:
                value = _statistic_from_fit(fitted, kind, cfg)[0]
                if np.isfinite(value):
                    values[b] = value
                else:
                    status = "nonfinite"
            except ValueError as exc:
                status = "nonfinite" if "nonfinite" in str(exc) else "no_admissible_change"
        statuses.append(status)
    report = _bootstrap_diagnostics(statuses)
    if diagnostics is not None:
        diagnostics.update(report)
    if report["n_successful"] < 2:
        raise BootstrapFitError(report)
    return values


def _rank_calibration(statistic, simulated, alpha):
    """Strict order-statistic cutoff exactly matching the conservative rank p-value."""
    values = np.asarray(simulated, dtype=float)
    values = values[np.isfinite(values)]
    count = len(values)
    k = count + 1 - int(np.floor(alpha * (count + 1)))  # one-based index
    critical = np.inf if k > count else float(np.partition(values, k - 1)[k - 1])
    p_value = float((1 + np.count_nonzero(values >= statistic)) / (count + 1))
    return critical, p_value


def _gaussian_bridge_statistics(
    information_time: np.ndarray, rng: np.random.Generator, samples: int,
    kind: Functional = "cusum", cfg: MethodConfig = CONFIG.method,
) -> np.ndarray:
    """Simulate a Brownian bridge only at the observed information times."""
    grid = np.asarray(information_time[:-1], dtype=float)
    times = np.r_[0.0, grid, 1.0]
    increments = rng.normal(size=(samples, times.size - 1)) * np.sqrt(np.diff(times))
    motion = np.cumsum(increments, axis=1)
    bridge = motion[:, :-1] - grid[None, :] * motion[:, -1, None]
    if kind == "cusum":
        fraction = cfg.minimum_information_fraction
        keep = (grid >= fraction) & (grid <= 1.0 - fraction)
        return np.max(np.abs(bridge[:, keep]), axis=1)
    if kind == "sup_lm":
        fraction = max(cfg.minimum_information_fraction, cfg.lm_trim)
        keep = (grid >= fraction) & (grid <= 1.0 - fraction)
        return np.max(bridge[:, keep] ** 2 / (grid[keep] * (1.0 - grid[keep])), axis=1)
    values = np.column_stack((np.zeros(samples), bridge ** 2, np.zeros(samples)))
    return np.trapezoid(values, times, axis=1)


def gradient_change_test(
    cases: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray,
    method: str = "proposed_asymptotic", rng: np.random.Generator | None = None,
    backend: Backend | None = None, bootstrap_samples: int | None = None,
    gaussian_samples: int | None = None,
    cfg: MethodConfig = CONFIG.method,
) -> ChangeTestResult:
    active_cases, deaths, age_scores = _arrays(cases, deaths, age_scores)
    _validate_fit_config(cfg)
    if not np.isfinite(cfg.alpha) or not 0 < cfg.alpha < 1:
        raise ValueError("alpha must be finite and in (0, 1)")
    kind: Functional = "cusum"
    calibration = "kolmogorov"
    if method == "proposed_bootstrap":
        calibration = "conditional bootstrap"
    elif method == "proposed_gaussian":
        calibration = "Gaussian bridge"
    elif method == "sup_lm":
        kind = "sup_lm"
        calibration = "Gaussian bridge"
    elif method == "cvm":
        kind = "cvm"
        calibration = "Gaussian bridge"
    elif method != "proposed_asymptotic":
        raise ValueError(f"unknown method: {method}")
    statistic, change, process, information_time, fit = _observed_statistic(
        active_cases, deaths, age_scores, backend, kind, cfg
    )
    boot = None
    diagnostics = None
    if calibration == "kolmogorov":
        critical = _kolmogorov_quantile(1.0 - cfg.alpha)
        p_value = _kolmogorov_sf(statistic)
    elif calibration == "Gaussian bridge":
        if rng is None:
            raise ValueError("rng is required for Gaussian-bridge calibration")
        count = _positive_integer(cfg.gaussian_samples if gaussian_samples is None else gaussian_samples, "gaussian_samples")
        boot = _gaussian_bridge_statistics(
            information_time, rng, count, kind, cfg
        )
        if not np.all(np.isfinite(boot)):
            raise ValueError("nonfinite Gaussian calibration statistics")
        critical, p_value = _rank_calibration(statistic, boot, cfg.alpha)
    else:
        if rng is None:
            raise ValueError("rng is required for bootstrap calibration")
        count = _positive_integer(cfg.bootstrap_samples if bootstrap_samples is None else bootstrap_samples, "bootstrap_samples")
        diagnostics = {}
        boot = _bootstrap_statistics(
            active_cases, deaths, age_scores, fit, rng, backend, kind, count, cfg, diagnostics
        )
        critical, p_value = _rank_calibration(statistic, boot, cfg.alpha)
    return ChangeTestResult(
        statistic, critical, p_value, bool(statistic > critical), change,
        process, information_time, fit, calibration, boot, diagnostics,
    )


def estimate_change(
    cases: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray,
    backend: Backend | None = None, cfg: MethodConfig = CONFIG.method,
) -> ChangeEstimate:
    cases, deaths, age_scores = _arrays(cases, deaths, age_scores)
    null_fit = fit_gradient(cases, deaths, age_scores, backend, cfg)
    if not null_fit.success:
        return ChangeEstimate(0, np.nan, np.nan, null_fit.beta, False, null_fit.status)
    try:
        _, change, _, _, _ = _statistic_from_fit(null_fit, "cusum", cfg)
    except ValueError:
        return ChangeEstimate(0, np.nan, np.nan, null_fit.beta, False, "no_admissible_change")
    before = fit_gradient(cases[:change], deaths[:change], age_scores, backend, cfg)
    after = fit_gradient(cases[change:], deaths[change:], age_scores, backend, cfg)
    return ChangeEstimate(change, before.beta, after.beta, null_fit.beta,
                          null_fit.success and before.success and after.success,
                          next((f.status for f in (before, after) if not f.success), "converged"))


def bootstrap_change_intervals(
    cases: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray,
    rng: np.random.Generator, backend: Backend | None = None,
    bootstrap_samples: int | None = None, cfg: MethodConfig = CONFIG.method,
) -> dict:
    """Fixed-B bootstrap: retain failed rows as NaN, exclude from summaries, no retries.

    The existing draws shape is preserved. Inspect bootstrap_diagnostics for the
    per-draw status and counts; fewer than two valid draws raises BootstrapFitError.
    """
    cases, deaths, age_scores = _arrays(cases, deaths, age_scores)
    _validate_fit_config(cfg)
    count = _positive_integer(cfg.bootstrap_samples if bootstrap_samples is None else bootstrap_samples, "bootstrap_samples")
    observed = estimate_change(cases, deaths, age_scores, backend, cfg)
    if not observed.success or not np.all(np.isfinite([observed.beta_before, observed.beta_after, observed.null_beta])):
        raise ValueError(f"unusable observed change fit: {observed.status}")
    totals = _bootstrap_totals(deaths)
    probability = np.empty_like(cases, dtype=float)
    probability[: observed.change] = conditional_probabilities(
        cases[: observed.change], age_scores, observed.beta_before
    )
    probability[observed.change :] = conditional_probabilities(
        cases[observed.change :], age_scores, observed.beta_after
    )
    # Generate all B multinomial samples in T vectorized calls.  This is the
    # same conditional bootstrap as drawing one B x T sample at a time, but it
    # avoids B*T round trips through Python's random-number API.
    sampled_deaths = np.empty((count, cases.shape[0], cases.shape[1]), dtype=np.int64)
    for t in range(cases.shape[0]):
        sampled_deaths[:, t, :] = rng.multinomial(
            int(totals[t]), probability[t], size=count
        )
    base_draws = np.full((count, 3), np.nan)
    statuses = np.full(count, "converged", dtype="U32")
    if resolve_backend(backend) == "c":
        active_cases = np.ascontiguousarray(cases, dtype=np.float64)
        active_deaths = np.ascontiguousarray(sampled_deaths, dtype=np.float64)
        active_scores = np.ascontiguousarray(age_scores, dtype=np.float64)
        changes = np.empty(count, dtype=np.int32)
        beta_before = np.empty(count, dtype=np.float64)
        beta_after = np.empty(count, dtype=np.float64)
        fit_status = np.empty(count, dtype=np.int32)
        status = _load_c_backend().cfr_bootstrap_change_batch(
            count, cases.shape[0], cases.shape[1], active_cases.ravel(),
            active_deaths.ravel(), active_scores,
            cfg.minimum_information_fraction, cfg.beta_bounds[0],
            cfg.beta_bounds[1], cfg.root_tolerance,
            cfg.root_max_iterations, changes, beta_before, beta_after, fit_status,
        )
        if status != 0:
            raise RuntimeError("C batch bootstrap failed")
        statuses[:] = [_FIT_STATUS[int(code)] for code in fit_status]
        base_draws[:, 0] = changes / cases.shape[0]
        base_draws[:, 1] = beta_before
        base_draws[:, 2] = beta_after
    else:
        for b in range(count):
            fitted = estimate_change(cases, sampled_deaths[b], age_scores, backend, cfg)
            if not fitted.success:
                statuses[b] = fitted.status if fitted.status != "converged" else "fit_failed"
                continue
            base_draws[b] = (fitted.change / cases.shape[0], fitted.beta_before, fitted.beta_after)
    draws = np.column_stack((
        base_draws,
        base_draws[:, 2] - base_draws[:, 1],
        np.exp(base_draws[:, 2] - base_draws[:, 1]),
    ))
    base_estimate = np.array([
        observed.change / cases.shape[0], observed.beta_before, observed.beta_after
    ])
    estimate = np.r_[
        base_estimate,
        observed.beta_after - observed.beta_before,
        np.exp(observed.beta_after - observed.beta_before),
    ]
    statuses[(statuses == "converged") & ~np.all(np.isfinite(draws), axis=1)] = "nonfinite"
    diagnostics = _bootstrap_diagnostics(statuses)
    draws[~diagnostics["success_mask"]] = np.nan
    if diagnostics["n_successful"] < 2:
        raise BootstrapFitError(diagnostics)
    valid = draws[diagnostics["success_mask"]]
    lower, upper = np.quantile(valid, [0.025, 0.975], axis=0)
    return {
        "labels": ("tau_over_T", "beta_before", "beta_after", "delta", "exp_delta"),
        "estimate": estimate, "se": valid.std(axis=0, ddof=1),
        "lower": lower, "upper": upper, "draws": draws, "fit": observed,
        "bootstrap_diagnostics": diagnostics,
    }


def dact_test(
    exposure: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray,
    rng: np.random.Generator, *, calibration: str = "gaussian",
    backend: Backend | None = None, samples: int | None = None,
) -> ChangeTestResult:
    """Run the delay-adjusted change test (DACT)."""
    method = {"gaussian": "proposed_gaussian", "bootstrap": "proposed_bootstrap",
              "asymptotic": "proposed_asymptotic"}.get(calibration)
    if method is None:
        raise ValueError("calibration must be 'gaussian', 'bootstrap', or 'asymptotic'")
    kwargs = ({"bootstrap_samples": samples} if calibration == "bootstrap"
              else {"gaussian_samples": samples})
    return gradient_change_test(
        exposure, deaths, age_scores, method, rng, backend=backend, **kwargs
    )


def sup_lm_test(
    exposure: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray,
    rng: np.random.Generator, *, backend: Backend | None = None,
    gaussian_samples: int | None = None,
) -> ChangeTestResult:
    """Run the published supLM fluctuation-process benchmark."""
    return gradient_change_test(exposure, deaths, age_scores, "sup_lm", rng,
                                backend=backend, gaussian_samples=gaussian_samples)


def cvm_test(
    exposure: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray,
    rng: np.random.Generator, *, backend: Backend | None = None,
    gaussian_samples: int | None = None,
) -> ChangeTestResult:
    """Run the Cramer--von Mises fluctuation-process benchmark."""
    return gradient_change_test(exposure, deaths, age_scores, "cvm", rng,
                                backend=backend, gaussian_samples=gaussian_samples)


def no_delay_dact_test(
    daily_cases: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray,
    rng: np.random.Generator, *, backend: Backend | None = None,
    gaussian_samples: int | None = None,
) -> ChangeTestResult:
    """Run DACT with same-day cases in place of delay-adjusted exposure."""
    return gradient_change_test(
        np.asarray(daily_cases, dtype=float), deaths, age_scores,
        "proposed_gaussian", rng, backend=backend,
        gaussian_samples=gaussian_samples,
    )


def available_tests() -> tuple[str, ...]:
    """Return the stable public names accepted by :func:`run_test`."""
    return ("dact", "sup_lm", "cvm", "no_delay_dact")


def run_test(
    name: str, exposure_or_cases: np.ndarray, deaths: np.ndarray,
    age_scores: np.ndarray, rng: np.random.Generator, *,
    backend: Backend | None = None, samples: int | None = None,
    calibration: str = "gaussian",
) -> ChangeTestResult:
    """Run any manuscript method through one public interface.

    Parameters
    ----------
    name:
        One of ``dact``, ``sup_lm``, ``cvm``, or
        ``no_delay_dact``.  The first three expect delay-adjusted exposure;
        the last expects same-day case counts.
    backend:
        ``"c"`` (the project default) or ``"python"``.
    samples:
        Number of Gaussian paths, or conditional-bootstrap samples when
        ``calibration="bootstrap"`` for ``dact``.
    """
    if name == "dact":
        return dact_test(
            exposure_or_cases, deaths, age_scores, rng,
            calibration=calibration, backend=backend, samples=samples,
        )
    if name == "sup_lm":
        return sup_lm_test(
            exposure_or_cases, deaths, age_scores, rng,
            backend=backend, gaussian_samples=samples,
        )
    if name == "cvm":
        return cvm_test(
            exposure_or_cases, deaths, age_scores, rng,
            backend=backend, gaussian_samples=samples,
        )
    if name == "no_delay_dact":
        return no_delay_dact_test(
            exposure_or_cases, deaths, age_scores, rng,
            backend=backend, gaussian_samples=samples,
        )
    raise ValueError(f"unknown test {name!r}; choose from {available_tests()}")


def verify_backend_equivalence(
    cases: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray,
    atol: float = 1.0e-9, rtol: float = 1.0e-9,
) -> dict[str, float]:
    build_c_backend()
    py = fit_gradient_python(cases, deaths, age_scores)
    cc = fit_gradient_c(cases, deaths, age_scores)
    errors = {
        "beta_error": abs(py.beta - cc.beta),
        "loglik_error": abs(py.loglik - cc.loglik),
        "score_error": float(np.max(np.abs(py.score - cc.score))),
        "information_error": float(np.max(np.abs(py.information - cc.information))),
    }
    if not all(np.isclose(value, 0.0, atol=atol, rtol=rtol) for value in errors.values()):
        raise AssertionError(f"C/Python backends differ: {errors}")
    return errors


__all__ = [
    "GradientFit", "ChangeTestResult", "ChangeEstimate", "BootstrapFitError", "build_c_backend",
    "delay_adjusted_exposure_python", "delay_adjusted_exposure_c", "delay_adjusted_exposure",
    "conditional_probabilities", "conditional_components_python",
    "conditional_components_c", "conditional_components", "fit_gradient_python",
    "fit_gradient_c", "fit_gradient", "gradient_change_test", "estimate_change",
    "bootstrap_change_intervals", "dact_test", "sup_lm_test", "cvm_test",
    "no_delay_dact_test", "available_tests", "run_test",
    "verify_backend_equivalence",
]
