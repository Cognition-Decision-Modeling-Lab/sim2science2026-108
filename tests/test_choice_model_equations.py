from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import expit, ndtr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from choice_generators import (
    M1Hyperbolic,
    M2Hyperboloid,
    M3DirectDifference,
    M4Exponential,
    M5DoubleExponential,
    M6ConstantSensitivity,
    M7AdditiveUtility,
    M8ProportionalDifference,
    M9PriorityLexicographicSemiorder,
    M10SimilarityHeuristic,
)
from choice_generators.config import (
    HEURISTIC_AMOUNT_GAP_SCALE,
    HEURISTIC_DELAY_GAP_SCALE,
)
from choice_generators.models import (
    _positive_truncated_normal_cdf,
    _similarity_judgment_probabilities,
)


TRIALS = np.array(
    [
        [10.0, 0.0, 20.0, 5.0],
        [20.0, 2.0, 35.0, 20.0],
        [40.0, 10.0, 50.0, 60.0],
    ],
    dtype=np.float64,
)

class ChoiceModelEquationTest(unittest.TestCase):
    def test_m1_probability_follows_discounted_value_direction(self):
        trials = np.array(
            [
                [10.0, 0.0, 20.0, 1.0],
                [10.0, 0.0, 11.0, 100.0],
            ]
        )

        p_ll = M1Hyperbolic()._calculate_p_ll(
            trials,
            {"k": 0.1, "m": 2.0},
        )

        self.assertGreater(p_ll[0], 0.5)
        self.assertLess(p_ll[1], 0.5)

    def test_m2_with_unit_time_sensitivity_equals_m1(self):
        k = 0.03
        m = 1.7

        p_m1 = M1Hyperbolic()._calculate_p_ll(
            TRIALS,
            {"k": k, "m": m},
        )
        p_m2 = M2Hyperboloid()._calculate_p_ll(
            TRIALS,
            {"k": k, "s": 1.0, "m": m},
        )

        np.testing.assert_allclose(p_m1, p_m2, rtol=1e-10, atol=1e-12)

    def test_m3_matches_direct_difference_equation(self):
        parameters = {"u": 0.8, "v": 0.6, "w": 0.4}
        amount_ss, delay_ss, amount_ll, delay_ll = TRIALS.T
        amount_difference = amount_ll**parameters["u"] - amount_ss**parameters["u"]
        delay_difference = delay_ll**parameters["v"] - delay_ss**parameters["v"]
        d_ll = (
            parameters["w"] * amount_difference
            - (1.0 - parameters["w"]) * delay_difference
        )
        variance = (
            parameters["w"] * amount_difference**2
            + (1.0 - parameters["w"]) * delay_difference**2
            - d_ll**2
        )
        expected = ndtr(d_ll / np.sqrt(variance))

        actual = M3DirectDifference()._calculate_p_ll(TRIALS, parameters)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_m4_matches_exponential_discounting_equation(self):
        k = 0.02
        m = 1.3
        amount_ss, delay_ss, amount_ll, delay_ll = TRIALS.T
        value_ss = amount_ss * np.exp(-k * delay_ss)
        value_ll = amount_ll * np.exp(-k * delay_ll)
        expected = expit(m * (value_ll - value_ss))

        actual = M4Exponential()._calculate_p_ll(
            TRIALS,
            {"k": k, "m": m},
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_m5_with_negligible_rate_gap_approaches_m4(self):
        delta = 0.02
        m = 1.3
        p_m4 = M4Exponential()._calculate_p_ll(
            TRIALS,
            {"k": delta, "m": m},
        )
        p_m5 = M5DoubleExponential()._calculate_p_ll(
            TRIALS,
            {
                "beta_gap": 1e-10,
                "delta": delta,
                "omega": 0.65,
                "m": m,
            },
        )

        np.testing.assert_allclose(p_m5, p_m4, rtol=1e-8, atol=1e-10)

    def test_m6_with_unit_subjective_time_sensitivity_equals_m4(self):
        k = 0.02
        m = 1.3
        p_m4 = M4Exponential()._calculate_p_ll(
            TRIALS,
            {"k": k, "m": m},
        )
        p_m6 = M6ConstantSensitivity()._calculate_p_ll(
            TRIALS,
            {"alpha": k, "beta": 1.0, "m": m},
        )

        np.testing.assert_allclose(p_m6, p_m4, rtol=1e-12, atol=1e-12)

    def test_m7_matches_additive_utility_equation(self):
        parameters = {
            "alpha": 0.7,
            "beta": 0.5,
            "lambda": 0.8,
            "m": 1.2,
        }
        amount_ss, delay_ss, amount_ll, delay_ll = TRIALS.T
        value_ss = (
            amount_ss**parameters["alpha"]
            - parameters["lambda"] * delay_ss**parameters["beta"]
        )
        value_ll = (
            amount_ll**parameters["alpha"]
            - parameters["lambda"] * delay_ll**parameters["beta"]
        )
        expected = expit(parameters["m"] * (value_ll - value_ss))

        actual = M7AdditiveUtility()._calculate_p_ll(TRIALS, parameters)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_m8_immediate_ss_removes_ll_delay_variation(self):
        trials = np.array(
            [
                [20.0, 0.0, 30.0, 7.0],
                [20.0, 0.0, 30.0, 365.0],
            ]
        )
        delay_difference = (
            trials[:, 3] - trials[:, 1]
        ) / trials[:, 3]
        parameters = {"delta": -0.2, "sigma": 1.5}
        expected = ndtr(
            parameters["sigma"]
            * (
                (trials[:, 2] - trials[:, 0]) / trials[:, 2]
                - delay_difference
                - parameters["delta"]
            )
        )

        actual = M8ProportionalDifference()._calculate_p_ll(
            trials,
            parameters,
        )

        np.testing.assert_array_equal(delay_difference, np.ones(2))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(actual[0], actual[1])

    def test_m9_priority_semiorder_equation(self):
        parameters = {
            "mu_amount": 0.30,
            "mu_delay": 0.30,
            "sigma_threshold": 0.10,
            "q_amount_first": 0.70,
        }
        trials = np.array(
            [
                [10.0, 0.0, 266.6658, 528.1612],
                [10.0, 0.0, 500.0, 1200.0],
                [400.0, 0.0, 410.0, 30.0],
                [10.0, 0.0, 255.0, 30.0],
            ],
            dtype=np.float64,
        )
        x_amount = np.clip(
            (trials[:, 2] - trials[:, 0]) / HEURISTIC_AMOUNT_GAP_SCALE,
            0.0,
            1.0,
        )
        x_delay = np.clip(
            (trials[:, 3] - trials[:, 1]) / HEURISTIC_DELAY_GAP_SCALE,
            0.0,
            1.0,
        )
        g_amount = _positive_truncated_normal_cdf(
            x_amount,
            parameters["mu_amount"],
            parameters["sigma_threshold"],
        )
        g_delay = _positive_truncated_normal_cdf(
            x_delay,
            parameters["mu_delay"],
            parameters["sigma_threshold"],
        )
        p_amount_first = (
            g_amount + 0.5 * (1.0 - g_amount) * (1.0 - g_delay)
        )
        p_delay_first = (1.0 - g_delay) * (
            g_amount + 0.5 * (1.0 - g_amount)
        )
        expected = (
            parameters["q_amount_first"] * p_amount_first
            + (1.0 - parameters["q_amount_first"]) * p_delay_first
        )
        actual = M9PriorityLexicographicSemiorder()._calculate_p_ll(
            trials,
            parameters,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertTrue(np.all(np.isfinite(actual)))
        self.assertTrue(np.all((actual >= 0.0) & (actual <= 1.0)))

    def test_m9_priority_conflict_changes_with_q(self):
        model = M9PriorityLexicographicSemiorder()
        trial = np.array([[10.0, 0.0, 500.0, 1200.0]], dtype=np.float64)
        base = {
            "mu_amount": 0.20,
            "mu_delay": 0.20,
            "sigma_threshold": 0.08,
        }
        p_high = model._calculate_p_ll(
            trial,
            {**base, "q_amount_first": 0.9},
        )
        p_low = model._calculate_p_ll(
            trial,
            {**base, "q_amount_first": 0.1},
        )
        self.assertGreater(float(p_high[0]), float(p_low[0]))

    def test_m10_similarity_tree_equation(self):
        parameters = {
            "theta_amount_diff": 0.15,
            "theta_amount_ratio": 0.80,
            "theta_delay_diff": 0.15,
            "theta_delay_ratio": 0.80,
            "tau": 0.07,
        }
        trials = np.array(
            [
                [200.0, 0.0, 210.0, 900.0],
                [50.0, 100.0, 400.0, 110.0],
                [200.0, 100.0, 210.0, 110.0],
                [50.0, 0.0, 450.0, 900.0],
            ],
            dtype=np.float64,
        )
        s_amount, s_delay = _similarity_judgment_probabilities(
            trials,
            parameters,
        )
        expected = (
            (1.0 - s_amount) * s_delay
            + 0.5
            * (
                s_amount * s_delay
                + (1.0 - s_amount) * (1.0 - s_delay)
            )
        )
        actual = M10SimilarityHeuristic()._calculate_p_ll(
            trials,
            parameters,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertTrue(np.all(np.isfinite(actual)))
        self.assertTrue(np.all((actual >= 0.0) & (actual <= 1.0)))
        self.assertLess(float(actual[0]), 0.5)
        self.assertGreater(float(actual[1]), 0.5)
        self.assertAlmostEqual(float(actual[2]), 0.5, delta=0.2)
        self.assertAlmostEqual(float(actual[3]), 0.5, delta=0.2)

    def test_m9_and_m10_have_distinct_parameterizations(self):
        m9 = M9PriorityLexicographicSemiorder()
        m10 = M10SimilarityHeuristic()
        self.assertEqual(m9.n_parameters, 4)
        self.assertEqual(m10.n_parameters, 5)
        self.assertNotEqual(m9.parameter_names, m10.parameter_names)


if __name__ == "__main__":
    unittest.main()
