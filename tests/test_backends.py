import unittest

import numpy as np

from dgp import generate_dataset
from methods import (
    bootstrap_change_intervals,
    conditional_components_c,
    conditional_components_python,
    delay_adjusted_exposure_c,
    delay_adjusted_exposure_python,
    fit_gradient_c,
    fit_gradient_python,
    gradient_change_test,
    available_tests,
    run_test,
    verify_backend_equivalence,
)


class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate_dataset("A1", np.random.default_rng(19), days=90,
                                    change_magnitude=0.035)

    def test_delay_convolution(self):
        p = delay_adjusted_exposure_python(self.data.cases, self.data.delay_analysis)
        c = delay_adjusted_exposure_c(self.data.cases, self.data.delay_analysis)
        np.testing.assert_allclose(p, c, atol=1e-10, rtol=1e-12)

    def test_components(self):
        p = conditional_components_python(self.data.exposure, self.data.deaths,
                                           self.data.age_scores, 0.2)
        c = conditional_components_c(self.data.exposure, self.data.deaths,
                                      self.data.age_scores, 0.2)
        self.assertAlmostEqual(p[0], c[0], places=8)
        np.testing.assert_allclose(p[1], c[1], atol=1e-10)
        np.testing.assert_allclose(p[2], c[2], atol=1e-10)

    def test_fit(self):
        p = fit_gradient_python(self.data.exposure, self.data.deaths, self.data.age_scores)
        c = fit_gradient_c(self.data.exposure, self.data.deaths, self.data.age_scores)
        self.assertAlmostEqual(p.beta, c.beta, places=9)
        verify_backend_equivalence(self.data.exposure, self.data.deaths,
                                   self.data.age_scores)

    def test_gaussian_grid_equivalence(self):
        p = gradient_change_test(self.data.exposure, self.data.deaths,
                                 self.data.age_scores, "proposed_gaussian",
                                 np.random.default_rng(3), "python", gaussian_samples=200)
        c = gradient_change_test(self.data.exposure, self.data.deaths,
                                 self.data.age_scores, "proposed_gaussian",
                                 np.random.default_rng(3), "c", gaussian_samples=200)
        self.assertAlmostEqual(p.statistic, c.statistic, places=9)
        self.assertAlmostEqual(p.critical_value, c.critical_value, places=9)
        self.assertEqual(p.rejected, c.rejected)

    def test_interval_bootstrap_equivalence(self):
        p = bootstrap_change_intervals(self.data.exposure, self.data.deaths,
                                       self.data.age_scores, np.random.default_rng(7),
                                       "python", bootstrap_samples=8)
        c = bootstrap_change_intervals(self.data.exposure, self.data.deaths,
                                       self.data.age_scores, np.random.default_rng(7),
                                       "c", bootstrap_samples=8)
        for field in ("estimate", "se", "lower", "upper", "draws"):
            np.testing.assert_allclose(p[field], c[field], atol=1e-8)

    def test_public_method_dispatch(self):
        self.assertEqual(
            available_tests(),
            ("dact", "sup_lm", "cvm", "no_delay_dact"),
        )
        direct = run_test(
            "dact", self.data.exposure, self.data.deaths,
            self.data.age_scores, np.random.default_rng(9),
            backend="c", samples=100,
        )
        self.assertTrue(np.isfinite(direct.statistic))
