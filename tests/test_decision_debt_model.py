"""Regression tests for the Decision Debt reproducibility package.

Run from repository root:

    python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from decision_debt_model import DIMS, frontier, issuance, load_capacity_matrix, reform_delta  # noqa: E402
from reproduce_section_5_2 import exhaustive_equal_weight_profiles, frontier_counts, random_profiles  # noqa: E402


class DecisionDebtRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_capacity_matrix(REPO_ROOT / "data" / "capacity_matrix.csv")

    def test_capacity_matrix_shape_and_values(self):
        self.assertEqual(self.spec.capacities.shape, (12, 9))
        self.assertEqual(self.spec.dimensions, DIMS)
        self.assertTrue(set(np.unique(self.spec.capacities).tolist()).issubset({0.0, 1.0, 2.0}))

    def test_worked_example_issuance_and_frontier(self):
        r = np.array([1, 1, 2, 0, 1, 1, 0, 0, 0.0])
        w = np.array([0.250, 0.167, 0.333, 0, 0.083, 0.167, 0, 0, 0])
        i = issuance(self.spec.capacities, r, w)
        expected = {
            "approval_voting": 0.750,
            "score_voting": 0.500,
            "quadratic_voting": 0.333,
            "condorcet_schulze": 0.750,
        }
        for method, value in expected.items():
            self.assertEqual(round(float(i[self.spec.methods.index(method)]), 3), value)
        four = [self.spec.methods.index(m) for m in expected]
        idx, u, _ = frontier(self.spec.capacities[four], r, w)
        self.assertEqual(idx.tolist(), [2])
        self.assertEqual(round(u, 3), 0.333)

    def test_reform_delta_approval_to_quadratic(self):
        r = np.array([1, 1, 2, 0, 1, 1, 0, 0, 0.0])
        w = np.array([0.250, 0.167, 0.333, 0, 0.083, 0.167, 0, 0, 0])
        delta = reform_delta(
            self.spec.capacities,
            r,
            w,
            self.spec.methods.index("approval_voting"),
            self.spec.methods.index("quadratic_voting"),
        )
        active_values = [delta[DIMS.index(d)] for d in ("A", "O", "I", "P", "M")]
        expected = [-0.250, 0.167, 0.333, 0.167, 0.000]
        for observed, value in zip(active_values, expected):
            self.assertEqual(round(float(observed), 3), value)

    def test_random_table2_regression(self):
        rs, ws = random_profiles(20000, offset=101)
        counts, tie = frontier_counts(self.spec, rs, ws)
        lookup = dict(zip(counts["method"], counts["on_frontier_pct"]))
        self.assertEqual(round(lookup["conviction_voting"], 3), 30.240)
        self.assertEqual(round(lookup["stv"], 3), 28.635)
        self.assertEqual(round(lookup["quadratic_voting"], 3), 13.120)
        self.assertEqual(round(tie, 3), 28.500)

    def test_exhaustive_table2_regression(self):
        rs, ws = exhaustive_equal_weight_profiles()
        self.assertEqual(len(rs), 19682)
        counts, _ = frontier_counts(self.spec, rs, ws)
        lookup = dict(zip(counts["method"], counts["on_frontier_pct"]))
        self.assertEqual(round(lookup["conviction_voting"], 3), 61.955)
        self.assertEqual(round(lookup["stv"], 3), 61.650)
        self.assertEqual(round(lookup["quadratic_voting"], 3), 22.980)


if __name__ == "__main__":
    unittest.main()
