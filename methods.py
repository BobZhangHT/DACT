"""Age-gradient CFR change-point methods with equivalent C/Python backends."""
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


@dataclass
class ChangeEstimate:
    change: int
    beta_before: float
    beta_after: float
    null_beta: float
    success: bool


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
                ]
                lib.cfr_bootstrap_change_batch.restype = ctypes.c_int
                _C_HANDLE = lib
    return _C_HANDLE


def resolve_backend(backend: Backend | None = None) -> Backend:
    active = backend or CONFIG.method.backend
    if active == "c":
        _load_c_backend()
    return active


def _arrays(cases: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray) -> tuple[np.ndarray, ...]:
    c = np.ascontiguousarray(cases, dtype=np.float64)
    d = np.ascontiguousarray(deaths, dtype=np.float64)
    a = np.ascontiguousarray(age_scores, dtype=np.float64)
    if c.ndim != 2 or d.shape != c.shape or a.shape != (c.shape[1],):
        raise ValueError("cases/deaths must be T by J and age_scores must have length J")
    if np.any(c < 0) or np.any(d < 0) or np.any(c.sum(axis=1) <= 0):
        raise ValueError("exposures must be nonnegative with positive row totals; deaths must be nonnegative")
    return c, d, a


def delay_adjusted_exposure_python(
    cases: np.ndarray, delay: np.ndarray, outcome_days: int | None = None,
) -> np.ndarray:
    """Convolve daily age-specific cases with the case-to-death delay mass."""
    c = np.asarray(cases, dtype=float)
    f = np.asarray(delay, dtype=float)
    if c.ndim != 2 or f.ndim != 2 or c.shape[1] != f.shape[0]:
        raise ValueError("cases must be R x J and delay must be J x L")
    if np.any(c < 0) or np.any(f < 0) or not np.allclose(f.sum(axis=1), 1.0):
        raise ValueError("counts and delay masses must be valid")
    T = int(outcome_days or c.shape[0])
    output = np.zeros((T, c.shape[1]), dtype=float)
    for j in range(c.shape[1]):
        output[:, j] = np.convolve(c[:, j], f[j])[:T]
    return output


def delay_adjusted_exposure_c(
    cases: np.ndarray, delay: np.ndarray, outcome_days: int | None = None,
) -> np.ndarray:
    c = np.ascontiguousarray(cases, dtype=np.float64)
    f = np.ascontiguousarray(delay, dtype=np.float64)
    if c.ndim != 2 or f.ndim != 2 or c.shape[1] != f.shape[0]:
        raise ValueError("cases must be R x J and delay must be J x L")
    T = int(outcome_days or c.shape[0])
    output = np.empty((T, c.shape[1]), dtype=np.float64)
    status = _load_c_backend().cfr_delay_exposure(
        c.shape[0], T, c.shape[1], f.shape[1], c.ravel(), f.ravel(), output.ravel()
    )
    if status:
        raise RuntimeError("C delay convolution failed")
    return output


def delay_adjusted_exposure(
    cases: np.ndarray, delay: np.ndarray, outcome_days: int | None = None,
    backend: Backend | None = None,
) -> np.ndarray:
    fn = delay_adjusted_exposure_c if resolve_backend(backend) == "c" else delay_adjusted_exposure_python
    return fn(cases, delay, outcome_days)


def conditional_probabilities(cases: np.ndarray, age_scores: np.ndarray, beta: float) -> np.ndarray:
    c = np.asarray(cases, dtype=float)
    a = np.asarray(age_scores, dtype=float)
    eta = np.log(c) + float(beta) * a[None, :]
    eta -= eta.max(axis=1, keepdims=True)
    probability = np.exp(eta)
    return probability / probability.sum(axis=1, keepdims=True)


def conditional_components_python(
    cases: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray, beta: float
) -> tuple[float, np.ndarray, np.ndarray]:
    c, d, a = _arrays(cases, deaths, age_scores)
    probability = conditional_probabilities(c, a, beta)
    totals = d.sum(axis=1)
    mean_age = probability @ a
    observed_age = d @ a
    score = observed_age - totals * mean_age
    information = totals * np.sum(probability * (a[None, :] - mean_age[:, None]) ** 2, axis=1)
    loglik = float(np.sum(np.where(d > 0, d * np.log(probability), 0.0)))
    return loglik, score, information


def conditional_components_c(
    cases: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray, beta: float
) -> tuple[float, np.ndarray, np.ndarray]:
    c, d, a = _arrays(cases, deaths, age_scores)
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


def fit_gradient_python(
    cases: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray,
    cfg: MethodConfig = CONFIG.method,
) -> GradientFit:
    lower, upper = cfg.beta_bounds
    _, score_lower, _ = conditional_components_python(cases, deaths, age_scores, lower)
    _, score_upper, _ = conditional_components_python(cases, deaths, age_scores, upper)
    iterations = 0
    if score_lower.sum() <= 0:
        beta = lower
    elif score_upper.sum() >= 0:
        beta = upper
    else:
        for iterations in range(cfg.root_max_iterations):
            beta = 0.5 * (lower + upper)
            _, score, _ = conditional_components_python(cases, deaths, age_scores, beta)
            if abs(score.sum()) <= cfg.root_tolerance or upper - lower <= cfg.root_tolerance:
                break
            if score.sum() > 0:
                lower = beta
            else:
                upper = beta
        beta = 0.5 * (lower + upper)
    loglik, score, information = conditional_components_python(cases, deaths, age_scores, beta)
    return GradientFit(float(beta), loglik, score, information,
                       conditional_probabilities(cases, age_scores, beta), True,
                       iterations + 1, "python")


def fit_gradient_c(
    cases: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray,
    cfg: MethodConfig = CONFIG.method,
) -> GradientFit:
    c, d, a = _arrays(cases, deaths, age_scores)
    beta = np.empty(1, dtype=np.float64)
    loglik = np.empty(1, dtype=np.float64)
    iterations = np.zeros(1, dtype=np.int32)
    status = _load_c_backend().cfr_fit_beta(
        c.shape[0], c.shape[1], c.ravel(), d.ravel(), a,
        cfg.beta_bounds[0], cfg.beta_bounds[1], cfg.root_tolerance,
        cfg.root_max_iterations, beta, loglik, iterations,
    )
    value, score, information = conditional_components_c(c, d, a, beta[0])
    return GradientFit(float(beta[0]), float(value), score, information,
                       conditional_probabilities(c, a, beta[0]), status == 0,
                       int(iterations[0]), "c")


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


def _process_from_fit(fit: GradientFit, kind: Functional = "cusum") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_information = float(fit.information.sum())
    if total_information <= 0:
        raise ValueError("the data contain no age-gradient information")
    cumulative_information = np.cumsum(fit.information)
    information_time = cumulative_information / total_information
    cumulative_score = np.cumsum(fit.score)
    process = cumulative_score / np.sqrt(total_information)
    # The maximum-score and CvM functionals use the complete interval. The supLM
    # functional divides by u(1-u) and therefore uses conventional trimming.
    fraction = (max(CONFIG.method.minimum_information_fraction, CONFIG.method.lm_trim)
                if kind == "sup_lm" else CONFIG.method.minimum_information_fraction)
    admissible = np.flatnonzero(
        (information_time[:-1] >= fraction) & (information_time[:-1] <= 1.0 - fraction)
    )
    if not len(admissible):
        raise ValueError("no admissible change location")
    return process, information_time, admissible


def _observed_statistic(
    cases: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray,
    backend: Backend | None, kind: Functional = "cusum",
) -> tuple[float, int, np.ndarray, np.ndarray, GradientFit]:
    fit = fit_gradient(cases, deaths, age_scores, backend)
    process, information_time, admissible = _process_from_fit(fit, kind)
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


def _bootstrap_statistics(
    cases: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray,
    fit: GradientFit, rng: np.random.Generator, backend: Backend | None,
    kind: Functional, samples: int,
) -> np.ndarray:
    totals = deaths.sum(axis=1).astype(int)
    values = np.empty(samples, dtype=float)
    for b in range(samples):
        sampled = np.vstack([
            rng.multinomial(int(totals[t]), fit.probabilities[t]) for t in range(cases.shape[0])
        ])
        values[b] = _observed_statistic(cases, sampled, age_scores, backend, kind)[0]
    return values


def _gaussian_bridge_statistics(
    information_time: np.ndarray, rng: np.random.Generator, samples: int,
    kind: Functional = "cusum",
) -> np.ndarray:
    """Simulate a Brownian bridge only at the observed information times."""
    grid = np.asarray(information_time[:-1], dtype=float)
    times = np.r_[0.0, grid, 1.0]
    increments = rng.normal(size=(samples, times.size - 1)) * np.sqrt(np.diff(times))
    motion = np.cumsum(increments, axis=1)
    bridge = motion[:, :-1] - grid[None, :] * motion[:, -1, None]
    if kind == "cusum":
        fraction = CONFIG.method.minimum_information_fraction
        keep = (grid >= fraction) & (grid <= 1.0 - fraction)
        return np.max(np.abs(bridge[:, keep]), axis=1)
    if kind == "sup_lm":
        fraction = max(CONFIG.method.minimum_information_fraction,
                       CONFIG.method.lm_trim)
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
    active_cases = np.asarray(cases, dtype=float)
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
        active_cases, deaths, age_scores, backend, kind
    )
    boot = None
    if calibration == "kolmogorov":
        critical = _kolmogorov_quantile(1.0 - cfg.alpha)
        p_value = _kolmogorov_sf(statistic)
    elif calibration == "Gaussian bridge":
        if rng is None:
            raise ValueError("rng is required for Gaussian-bridge calibration")
        count = cfg.gaussian_samples if gaussian_samples is None else int(gaussian_samples)
        boot = _gaussian_bridge_statistics(
            information_time, rng, count, kind
        )
        critical = float(np.quantile(boot, 1.0 - cfg.alpha, method="higher"))
        p_value = float((1 + np.count_nonzero(boot >= statistic)) /
                        (count + 1))
    else:
        if rng is None:
            raise ValueError("rng is required for bootstrap calibration")
        count = cfg.bootstrap_samples if bootstrap_samples is None else int(bootstrap_samples)
        boot = _bootstrap_statistics(
            active_cases, deaths, age_scores, fit, rng, backend, kind, count
        )
        critical = float(np.quantile(boot, 1.0 - cfg.alpha, method="higher"))
        p_value = float((1 + np.count_nonzero(boot >= statistic)) / (count + 1))
    return ChangeTestResult(
        statistic, critical, p_value, bool(statistic > critical), change,
        process, information_time, fit, calibration, boot,
    )


def estimate_change(
    cases: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray,
    backend: Backend | None = None,
) -> ChangeEstimate:
    _, change, _, _, null_fit = _observed_statistic(cases, deaths, age_scores, backend, "cusum")
    before = fit_gradient(cases[:change], deaths[:change], age_scores, backend)
    after = fit_gradient(cases[change:], deaths[change:], age_scores, backend)
    return ChangeEstimate(change, before.beta, after.beta, null_fit.beta,
                          null_fit.success and before.success and after.success)


def bootstrap_change_intervals(
    cases: np.ndarray, deaths: np.ndarray, age_scores: np.ndarray,
    rng: np.random.Generator, backend: Backend | None = None,
    bootstrap_samples: int | None = None,
) -> dict:
    count = CONFIG.method.bootstrap_samples if bootstrap_samples is None else int(bootstrap_samples)
    observed = estimate_change(cases, deaths, age_scores, backend)
    totals = deaths.sum(axis=1).astype(int)
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
    base_draws = np.empty((count, 3), dtype=float)
    if resolve_backend(backend) == "c":
        active_cases = np.ascontiguousarray(cases, dtype=np.float64)
        active_deaths = np.ascontiguousarray(sampled_deaths, dtype=np.float64)
        active_scores = np.ascontiguousarray(age_scores, dtype=np.float64)
        changes = np.empty(count, dtype=np.int32)
        beta_before = np.empty(count, dtype=np.float64)
        beta_after = np.empty(count, dtype=np.float64)
        status = _load_c_backend().cfr_bootstrap_change_batch(
            count, cases.shape[0], cases.shape[1], active_cases.ravel(),
            active_deaths.ravel(), active_scores,
            CONFIG.method.minimum_information_fraction, CONFIG.method.beta_bounds[0],
            CONFIG.method.beta_bounds[1], CONFIG.method.root_tolerance,
            CONFIG.method.root_max_iterations, changes, beta_before, beta_after,
        )
        if status != 0:
            raise RuntimeError("C batch bootstrap failed")
        base_draws[:, 0] = changes / cases.shape[0]
        base_draws[:, 1] = beta_before
        base_draws[:, 2] = beta_after
    else:
        for b in range(count):
            fitted = estimate_change(cases, sampled_deaths[b], age_scores, backend)
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
    lower, upper = np.quantile(draws, [0.025, 0.975], axis=0)
    return {
        "labels": ("tau_over_T", "beta_before", "beta_after", "delta", "exp_delta"),
        "estimate": estimate, "se": draws.std(axis=0, ddof=1),
        "lower": lower, "upper": upper, "draws": draws, "fit": observed,
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
    "GradientFit", "ChangeTestResult", "ChangeEstimate", "build_c_backend",
    "delay_adjusted_exposure_python", "delay_adjusted_exposure_c", "delay_adjusted_exposure",
    "conditional_probabilities", "conditional_components_python",
    "conditional_components_c", "conditional_components", "fit_gradient_python",
    "fit_gradient_c", "fit_gradient", "gradient_change_test", "estimate_change",
    "bootstrap_change_intervals", "dact_test", "sup_lm_test", "cvm_test",
    "no_delay_dact_test", "available_tests", "run_test",
    "verify_backend_equivalence",
]
