"""EA08 validation and conservative Monte Carlo rank decision regressions."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

import methods as m
from config import CONFIG


@pytest.fixture
def data():
    return (np.ones((6, 2)), np.array([[8, 2], [7, 3], [6, 4], [3, 7], [2, 8], [2, 8.]]),
            np.array([-1., 1.]))


@pytest.mark.parametrize("backend", ["python", "c"])
@pytest.mark.parametrize("field", [0, 1, 2])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_rejected(data, backend, field, bad):
    args = [x.copy() for x in data]
    args[field].flat[0] = bad
    with pytest.raises(ValueError, match="finite"):
        m.fit_gradient(*args, backend=backend)


@pytest.mark.parametrize("backend", ["python", "c"])
def test_incompatible_cell_and_negative_inputs(data, backend):
    c, d, a = data
    c[0, 0] = 0
    with pytest.raises(ValueError, match="incompatible"):
        m.fit_gradient(c, d, a, backend)
    c[0, 0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        m.fit_gradient(c, d, a, backend)


@pytest.mark.parametrize("fn", [m.conditional_components_python, m.conditional_components_c])
@pytest.mark.parametrize("beta", [np.nan, np.inf])
def test_nonfinite_beta(data, fn, beta):
    with pytest.raises(ValueError, match="beta must be finite"):
        fn(*data, beta)


def test_structural_zeros_and_empty_outcome_row():
    c = np.array([[3., 0, 2], [0, 0, 0], [4, 0, 5], [2, 0, 7]])
    d = np.array([[2., 0, 3], [0, 0, 0], [3, 0, 4], [2, 0, 3]])
    a = np.array([-1., 0, 1])
    with np.errstate(all="raise"):
        py = m.fit_gradient_python(c, d, a)
        cc = m.fit_gradient_c(c, d, a)
    assert py.success and cc.success
    assert np.all(py.probabilities[[0, 2, 3], 1] == 0)
    assert py.score[1] == cc.score[1] == py.information[1] == cc.information[1] == 0
    np.testing.assert_allclose(py.probabilities[1], 1 / 3)
    m.verify_backend_equivalence(c, d, a)


def test_no_floor_for_tiny_positive_exposure_and_stable_loglik():
    c = np.array([[1e-30, 2e-30], [1e-30, 2e-30]])
    d = np.array([[2., 4], [2, 4]])
    a = np.array([0., 1])
    for beta in (0., 1000.):
        py = m.conditional_components_python(c, d, a, beta)
        cc = m.conditional_components_c(c, d, a, beta)
        assert np.isfinite(py[0]) and np.isfinite(cc[0])
        for x, y in zip(py, cc):
            np.testing.assert_allclose(x, y, atol=1e-10)
    assert abs(m.fit_gradient_python(c, d, a).beta) < 1e-9
    assert abs(m.fit_gradient_c(c, d, a).beta) < 1e-9


@pytest.mark.parametrize("backend", ["python", "c"])
@pytest.mark.parametrize("side", [0, 1])
def test_boundary_is_flagged_not_success(data, backend, side):
    c, d, a = data
    d[:] = 0
    d[:, side] = 10
    fit = m.fit_gradient(c, d, a, backend)
    assert fit.beta == CONFIG.method.beta_bounds[side]
    assert fit.boundary and not fit.success and fit.status == "boundary"
    assert fit.iterations == 0
    with pytest.raises(ValueError, match="boundary"):
        m.gradient_change_test(c, d, a, backend=backend)
    assert not m.estimate_change(c, d, a, backend).success


@pytest.mark.parametrize("backend", ["python", "c"])
def test_root_convergence_exhaustion_and_cfg_propagation(data, backend):
    c, d, a = data
    cfg = replace(CONFIG.method, root_max_iterations=1)
    fit = m.fit_gradient(c, d, a, backend, cfg)
    assert not fit.success and fit.status == "max_iterations" and fit.iterations == 1
    with pytest.raises(ValueError, match="max_iterations"):
        m.gradient_change_test(c, d, a, backend=backend, cfg=cfg)
    fitted = m.fit_gradient(c, d, a, backend)
    # Analytic two-category MLE with equal exposures and scores (-1, 1).
    expected = 0.5 * np.log(d[:, 1].sum() / d[:, 0].sum())
    assert fitted.success and not fitted.boundary
    assert fitted.beta == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("backend", ["python", "c"])
@pytest.mark.parametrize("mode", ["zero_deaths", "zero_exposure", "constant_age"])
def test_no_information(data, backend, mode):
    c, d, a = data
    if mode in ("zero_deaths", "zero_exposure"):
        d[:] = 0
    if mode == "zero_exposure":
        c[:] = 0
    if mode == "constant_age":
        a[:] = 0
    fit = m.fit_gradient(c, d, a, backend)
    assert not fit.success and fit.status == "no_information"


@pytest.mark.parametrize("backend", ["python", "c"])
@pytest.mark.parametrize("options", [
    {"root_max_iterations": 0}, {"root_max_iterations": 1.5},
    {"root_tolerance": np.nan}, {"root_tolerance": 0},
    {"beta_bounds": (1, 1)}, {"beta_bounds": (1, -1)},
    {"beta_bounds": (-np.inf, 1)}, {"minimum_information_fraction": np.nan},
])
def test_bad_solver_configuration(data, backend, options):
    with pytest.raises(ValueError):
        m.fit_gradient(*data, backend=backend, cfg=replace(CONFIG.method, **options))


@pytest.mark.parametrize("fn", [m.delay_adjusted_exposure_python, m.delay_adjusted_exposure_c])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.])
def test_delay_validates_counts_and_masses(fn, bad):
    c = np.ones((3, 2))
    f = np.full((2, 2), .5)
    for arg in (0, 1):
        args = [c.copy(), f.copy()]
        args[arg].flat[0] = bad
        with pytest.raises(ValueError):
            fn(*args)


def test_delay_tail_zero_padding_and_invalid_length():
    c = np.array([[1., 2], [3, 4]])
    f = np.array([[.25, .75], [.5, .5]])
    expected = np.array([[.25, 1], [1.5, 3], [2.25, 2], [0, 0], [0, 0]])
    for fn in (m.delay_adjusted_exposure_python, m.delay_adjusted_exposure_c):
        np.testing.assert_allclose(fn(c, f, 5), expected)
        for length in (0, -1, 1.5):
            with pytest.raises(ValueError):
                fn(c, f, length)


@pytest.mark.parametrize("count", [1, 18, 19, 38, 39, 99])
@pytest.mark.parametrize("ties", [False, True])
def test_rank_cutoff_matches_p_value_including_ties(count, ties):
    values = np.arange(1, count + 1, dtype=float)
    if ties:
        values = np.ceil(values / 3)
    k = count + 1 - int(np.floor(.05 * (count + 1)))
    expected = np.inf if k > count else np.sort(values)[k - 1]
    for observed in np.r_[values, values + .1, 1000.]:
        critical, p_value = m._rank_calibration(observed, values, .05)
        assert critical == expected
        assert (observed > critical) == (p_value <= .05)


@pytest.mark.parametrize("kind", ["proposed_gaussian", "sup_lm", "cvm"])
def test_public_gaussian_rank_threshold(data, kind):
    for count in (18, 38):
        values = np.arange(1, count + 1, dtype=float)
        with patch.object(m, "_gaussian_bridge_statistics", return_value=values):
            result = m.gradient_change_test(*data, kind, np.random.default_rng(3), "python",
                                            gaussian_samples=count)
        assert result.critical_value == (np.inf if count == 18 else 38)
        assert result.rejected == (result.p_value <= CONFIG.method.alpha)
    # At M=38 ordinary quantile(higher) gives 37, which is too liberal.
    assert np.quantile(np.arange(1, 39), .95, method="higher") == 37


@pytest.mark.parametrize("backend", ["python", "c"])
def test_bootstrap_excludes_boundary_draws_and_counts(data, backend):
    c, d, a = data
    # Sparse totals generate both estimable and separated segments.
    d = np.array([[2., 1], [2, 1], [1, 2], [1, 2], [2, 1], [1, 2]])
    result = m.bootstrap_change_intervals(c, d, a, np.random.default_rng(27), backend, 80)
    report = result["bootstrap_diagnostics"]
    assert 0 < report["n_failed"] < 80
    assert report["n_retried"] == 0
    assert report["n_successful"] + report["n_failed"] == 80
    assert result["draws"].shape == (80, 5)
    assert np.isnan(result["draws"][~report["success_mask"]]).all()
    valid = result["draws"][report["success_mask"]]
    np.testing.assert_allclose(result["se"], valid.std(axis=0, ddof=1))
    np.testing.assert_allclose(result["lower"], np.quantile(valid, .025, axis=0))


def test_bootstrap_nonfinite_and_failed_fit_are_not_used(data):
    good = m.estimate_change(*data, "python")
    failed = replace(good, success=False, status="max_iterations")
    nonfinite = replace(good, beta_before=np.nan)
    with patch.object(m, "estimate_change", side_effect=[good, good, failed, nonfinite, good]):
        result = m.bootstrap_change_intervals(*data, np.random.default_rng(2), "python", 4)
    report = result["bootstrap_diagnostics"]
    assert report["failure_counts"] == {"max_iterations": 1, "nonfinite": 1}
    np.testing.assert_array_equal(report["success_mask"], [True, False, False, True])
    with patch.object(m, "estimate_change", side_effect=[good, failed, nonfinite]):
        with pytest.raises(m.BootstrapFitError) as exc:
            m.bootstrap_change_intervals(*data, np.random.default_rng(2), "python", 2)
    assert exc.value.diagnostics["n_failed"] == 2


def test_calibration_bootstrap_reports_failures_and_uses_valid_denominator(data):
    good = m.fit_gradient_python(*data)
    bad = replace(good, success=False, status="max_iterations")
    nonfinite = replace(good, loglik=np.nan)
    with patch.object(m, "fit_gradient", side_effect=[good, good, bad, nonfinite, good]):
        result = m.gradient_change_test(*data, "proposed_bootstrap", np.random.default_rng(3),
                                        "python", bootstrap_samples=4)
    assert result.bootstrap_diagnostics["failure_counts"] == {"max_iterations": 1, "nonfinite": 1}
    assert np.isnan(result.bootstrap_statistics[[1, 2]]).all()
    valid = result.bootstrap_statistics[np.isfinite(result.bootstrap_statistics)]
    assert result.p_value == (1 + np.count_nonzero(valid >= result.statistic)) / 3
    assert result.critical_value == np.inf and not result.rejected


def test_c_batch_individual_failure_statuses_and_structural_zeros():
    c = np.ones((4, 3)); c[:, 1] = 0; c[1] = 0
    a = np.array([-1., 0, 1])
    good = np.array([[3., 0, 2], [0, 0, 0], [2, 0, 3], [2, 0, 3]])
    boundary = good.copy(); boundary[:, 2] += boundary[:, 0]; boundary[:, 0] = 0
    incompatible = good.copy(); incompatible[0, 1] = 1
    draws = np.ascontiguousarray([good, boundary, incompatible, np.zeros_like(good)])
    changes = np.zeros(4, dtype=np.int32)
    before = np.empty(4); after = np.empty(4); statuses = np.empty(4, dtype=np.int32)
    lib = m._load_c_backend()
    for iterations, expected in ((100, [0, 1, 3, 4]), (1, [2, 1, 3, 4])):
        status = lib.cfr_bootstrap_change_batch(
            4, 4, 3, c.ravel(), draws.ravel(), a, .15, -4., 4., 1e-10,
            iterations, changes, before, after, statuses)
        assert status == 0
        np.testing.assert_array_equal(statuses, expected)
    result = m.estimate_change(c, good, a, "python")
    assert result.success


def test_sparse_bootstrap_backend_status_equivalence():
    c = np.ones((6, 3)); c[:, 1] = 0
    d = np.array([[2., 0, 1], [2, 0, 1], [1, 0, 2], [1, 0, 2], [2, 0, 1], [1, 0, 2]])
    a = np.array([-1., 0, 1])
    results = [m.bootstrap_change_intervals(c, d, a, np.random.default_rng(27), backend, 80)
               for backend in ("python", "c")]
    for field in ("success_mask", "fit_status"):
        np.testing.assert_array_equal(results[0]["bootstrap_diagnostics"][field],
                                      results[1]["bootstrap_diagnostics"][field])
    for field in ("draws", "se", "lower", "upper"):
        np.testing.assert_allclose(results[0][field], results[1][field], atol=1e-8, equal_nan=True)
