import unittest

import numpy as np

from config import CONFIG
from dgp import generate_dataset


class DGPTests(unittest.TestCase):
    def test_config(self):
        self.assertEqual(CONFIG.simulation.replications("demo"), 100)
        self.assertEqual(CONFIG.method.bootstrap_samples, 2000)

    def test_reproducible_daily_alternative(self):
        a = generate_dataset(
            "A1", np.random.default_rng(4), days=90, change_fraction=0.5, change_magnitude=0.025
        )
        b = generate_dataset(
            "A1", np.random.default_rng(4), days=90, change_fraction=0.5, change_magnitude=0.025
        )
        np.testing.assert_array_equal(a.cases, b.cases)
        np.testing.assert_array_equal(a.deaths, b.deaths)
        self.assertEqual(a.true_change, 45)

    def test_shapes_and_counts(self):
        d = generate_dataset("N0", np.random.default_rng(5), days=90)
        self.assertEqual(d.cases.shape, d.deaths.shape)
        self.assertEqual(d.exposure.shape, d.deaths.shape)
        self.assertTrue(np.all(d.cases >= 0))
        self.assertTrue(np.all(d.deaths >= 0))
        self.assertIsNone(d.true_change)

    def test_sparse_early_alternative(self):
        d = generate_dataset(
            "A3", np.random.default_rng(6), days=180, change_fraction=0.25, change_magnitude=0.025
        )
        self.assertEqual(d.true_change, 45)
        self.assertEqual(d.scenario, "A3")
