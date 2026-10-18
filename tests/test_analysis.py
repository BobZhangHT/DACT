import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis import aggregate_bootstrap_estimation, estimation_table, sensitivity_figures
from analysis import testing_table as make_testing_table


class AnalysisTests(unittest.TestCase):
    def test_tables(self):
        testing = pd.DataFrame(
            [
                {
                    "scenario": "N0",
                    "method": "proposed_gaussian",
                    "days": 180,
                    "death_scale": 1.0,
                    "magnitude": 0.0,
                    "change_fraction": 0.5,
                    "rejection": 0.05,
                    "median_location_error": float("nan"),
                    "failure_rate": 0.0,
                    "runtime": 0.01,
                }
            ]
        )
        rendered = make_testing_table(testing)
        self.assertIn("null mechanisms", rendered)
        self.assertIn("DACT", rendered)
        self.assertNotIn("CUSUM", rendered)

        raw = pd.DataFrame(
            [
                {
                    "scenario": "A1",
                    "days": 180,
                    "magnitude": 0.025,
                    "change_fraction": 0.5,
                    "replication": 0,
                    "labels": ["tau_over_T"],
                    "truth": [0.5],
                    "estimate": [0.49],
                    "bootstrap_se": [0.02],
                    "bootstrap_covered": [1.0],
                    "bootstrap_width": [0.08],
                }
            ]
        )
        rendered_estimation = estimation_table(aggregate_bootstrap_estimation(raw))
        self.assertIn("conditional bootstrap", rendered_estimation)
        self.assertNotIn("fixed-split", rendered_estimation.lower())

    def test_sensitivity_plot(self):
        frame = pd.DataFrame(
            [
                {
                    "scenario": s,
                    "axis": axis,
                    "value": value,
                    "rejected": 0.05 if s == "N0" else 0.7,
                    "location_error": 2.0,
                }
                for axis, values in {
                    "delay_multiplier": (0.8, 1.2),
                    "age_delay_mismatch": (0.0, 0.2),
                    "age_groups": (3.0, 6.0),
                    "overdispersion": (0.0, 0.05),
                    "transition_width": (0.0, 8.0),
                    "day_of_week": (0.0, 0.2),
                }.items()
                for s in (("A1",) if axis == "transition_width" else ("N0", "A1"))
                for value in values
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sensitivity_figures(frame, output)
            self.assertGreater((output / "figure_sensitivity.pdf").stat().st_size, 1000)
