from pathlib import Path
import sys
import unittest

import matplotlib.pyplot as plt
import numpy as np

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
    M11ConstantLLProbability,
)
from trial_generators.published import (
    konstantinidis_2020_exp1_intertemporal,
)


TRIALS = konstantinidis_2020_exp1_intertemporal()

MODEL_CLASSES = [
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
    M11ConstantLLProbability,
]

N_SUBJECTS = 100
SEED = 31


class ChoiceGeneratorTest(unittest.TestCase):
    def test_models(self):
        for model_class in MODEL_CLASSES:
            with self.subTest(model=model_class.__name__):
                model = model_class()

                parameter_values = model.sample_parameters(
                    seed=SEED,
                    n_subjects=N_SUBJECTS,
                )
                parameter_dict = model.get_parameters_dict(
                    parameter_values[0]
                )
                choices, trial_mask = model.generate_choices(
                    TRIALS,
                    parameter_values,
                    seed=SEED,
                )

                self.assertEqual(
                    parameter_values.shape,
                    (N_SUBJECTS, model.n_parameters),
                )
                self.assertEqual(
                    list(parameter_dict),
                    model.parameter_names,
                )
                self.assertEqual(
                    choices.shape,
                    (N_SUBJECTS, len(TRIALS), 1),
                )
                self.assertEqual(choices.dtype, np.int8)
                self.assertEqual(
                    trial_mask.shape,
                    (N_SUBJECTS, len(TRIALS)),
                )
                self.assertEqual(trial_mask.dtype, np.bool_)
                self.assertTrue(trial_mask.all())
                self.assertTrue(
                    set(np.unique(choices)).issubset({0, 1})
                )

                prior_figure, axes = model.plot_priors(show=False)
                self.assertEqual(
                    axes.shape,
                    (1, model.n_parameters),
                )

                choice_figure, axis = model.plot_choice_proportions(
                    choices,
                    show=False,
                )
                self.assertEqual(
                    axis.get_ylabel(),
                    "Proportion choosing LL",
                )
                plotted_proportions = axis.collections[0].get_offsets()[:, 1]
                expected_proportions = np.sort(
                    choices[:, :, 0].mean(axis=1)
                )
                np.testing.assert_allclose(
                    plotted_proportions,
                    expected_proportions,
                )
                plt.close(prior_figure)
                plt.close(choice_figure)

    def test_subject_specific_padded_trials_use_mask(self):
        model = M1Hyperbolic()
        parameters = model.sample_parameters(seed=SEED, n_subjects=2)
        trials = np.zeros((2, 4, 4), dtype=np.float32)
        trials[0, 0] = [10.0, 0.0, 20.0, 10.0]
        trials[0, 1] = [15.0, 0.0, 40.0, 30.0]
        trials[1, 0] = [12.0, 5.0, 30.0, 20.0]
        trial_mask = np.array(
            [
                [True, True, False, False],
                [True, False, False, False],
            ]
        )

        choices, returned_mask = model.generate_choices(
            trials,
            parameters,
            seed=SEED,
            trial_mask=trial_mask,
        )

        self.assertEqual(choices.shape, (2, 4, 1))
        np.testing.assert_array_equal(returned_mask, trial_mask)
        self.assertTrue((choices[~trial_mask] == 0).all())
        self.assertTrue(set(np.unique(choices[trial_mask])).issubset({0, 1}))

        choice_figure, axis = model.plot_choice_proportions(
            choices,
            show=False,
            trial_mask=trial_mask,
        )
        plotted_proportions = axis.collections[0].get_offsets()[:, 1]
        expected_proportions = np.sort(
            np.array(
                [
                    choices[0, trial_mask[0], 0].mean(),
                    choices[1, trial_mask[1], 0].mean(),
                ]
            )
        )
        np.testing.assert_allclose(plotted_proportions, expected_proportions)
        plt.close(choice_figure)

    def test_invalid_trials_are_rejected(self):
        trials = TRIALS.iloc[:1].copy()
        trials.loc[trials.index[0], "delay_ll"] = np.nan
        model = M1Hyperbolic()
        parameters = model.sample_parameters(n_subjects=1)

        with self.assertRaises(ValueError):
            model.generate_choices(trials, parameters)

    def test_m9_and_m10_parameter_shapes(self):
        m9 = M9PriorityLexicographicSemiorder()
        m10 = M10SimilarityHeuristic()
        m9_params = m9.sample_parameters(seed=SEED, n_subjects=N_SUBJECTS)
        m10_params = m10.sample_parameters(seed=SEED, n_subjects=N_SUBJECTS)
        self.assertEqual(m9.n_parameters, 4)
        self.assertEqual(m10.n_parameters, 5)
        self.assertEqual(m9_params.shape, (N_SUBJECTS, 4))
        self.assertEqual(m10_params.shape, (N_SUBJECTS, 5))
        self.assertEqual(
            m9.parameter_names,
            [
                "mu_amount",
                "mu_delay",
                "sigma_threshold",
                "q_amount_first",
            ],
        )
        self.assertEqual(
            m10.parameter_names,
            [
                "theta_amount_diff",
                "theta_amount_ratio",
                "theta_delay_diff",
                "theta_delay_ratio",
                "tau",
            ],
        )

    def test_m9_and_m10_padded_trials_keep_inherited_mask_behavior(self):
        trials = np.zeros((2, 4, 4), dtype=np.float32)
        trials[0, 0] = [10.0, 0.0, 20.0, 10.0]
        trials[0, 1] = [15.0, 0.0, 40.0, 30.0]
        trials[1, 0] = [12.0, 5.0, 30.0, 20.0]
        trial_mask = np.array(
            [
                [True, True, False, False],
                [True, False, False, False],
            ]
        )
        for model_class in (
            M9PriorityLexicographicSemiorder,
            M10SimilarityHeuristic,
        ):
            with self.subTest(model=model_class.__name__):
                model = model_class()
                parameters = model.sample_parameters(seed=SEED, n_subjects=2)
                choices, returned_mask = model.generate_choices(
                    trials,
                    parameters,
                    seed=SEED,
                    trial_mask=trial_mask,
                )
                self.assertEqual(choices.shape, (2, 4, 1))
                np.testing.assert_array_equal(returned_mask, trial_mask)
                self.assertTrue((choices[~trial_mask] == 0).all())
                self.assertTrue(
                    set(np.unique(choices[trial_mask])).issubset({0, 1})
                )

    def test_m9_and_m10_choices_are_reproducible_for_the_same_seed(self):
        for model_class in (
            M9PriorityLexicographicSemiorder,
            M10SimilarityHeuristic,
        ):
            with self.subTest(model=model_class.__name__):
                model = model_class()
                parameters = model.sample_parameters(seed=SEED, n_subjects=8)
                first, _ = model.generate_choices(
                    TRIALS,
                    parameters,
                    seed=SEED,
                )
                second, _ = model.generate_choices(
                    TRIALS,
                    parameters,
                    seed=SEED,
                )
                np.testing.assert_array_equal(first, second)

    def test_m9_priority_conflict_favors_amount_first(self):
        model = M9PriorityLexicographicSemiorder()
        trial = np.array([[10.0, 0.0, 500.0, 1200.0]], dtype=np.float64)
        base = {
            "mu_amount": 0.20,
            "mu_delay": 0.20,
            "sigma_threshold": 0.08,
        }
        p_high_q = model._calculate_p_ll(
            trial,
            {**base, "q_amount_first": 0.9},
        )
        p_low_q = model._calculate_p_ll(
            trial,
            {**base, "q_amount_first": 0.1},
        )
        self.assertGreater(float(p_high_q[0]), float(p_low_q[0]))
        self.assertTrue(np.all(np.isfinite(p_high_q)))
        self.assertTrue(np.all((p_high_q >= 0.0) & (p_high_q <= 1.0)))

    def test_m9_amount_and_delay_monotonicity(self):
        model = M9PriorityLexicographicSemiorder()
        parameters = {
            "mu_amount": 0.30,
            "mu_delay": 0.30,
            "sigma_threshold": 0.10,
            "q_amount_first": 0.5,
        }
        amount_trials = np.array(
            [
                [100.0, 10.0, 150.0, 40.0],
                [100.0, 10.0, 250.0, 40.0],
                [100.0, 10.0, 400.0, 40.0],
            ],
            dtype=np.float64,
        )
        delay_trials = np.array(
            [
                [100.0, 10.0, 180.0, 40.0],
                [100.0, 10.0, 180.0, 200.0],
                [100.0, 10.0, 180.0, 800.0],
            ],
            dtype=np.float64,
        )
        p_amount = model._calculate_p_ll(amount_trials, parameters)
        p_delay = model._calculate_p_ll(delay_trials, parameters)
        self.assertGreater(p_amount[2], p_amount[0])
        self.assertLess(p_delay[2], p_delay[0])

    def test_m10_similarity_four_state_tree(self):
        model = M10SimilarityHeuristic()
        soft = {
            "theta_amount_diff": 0.15,
            "theta_amount_ratio": 0.80,
            "theta_delay_diff": 0.15,
            "theta_delay_ratio": 0.80,
            "tau": 0.04,
        }

        similar_amount = np.array(
            [[200.0, 0.0, 210.0, 900.0]],
            dtype=np.float64,
        )

        similar_delay = np.array(
            [[50.0, 100.0, 400.0, 110.0]],
            dtype=np.float64,
        )

        both_similar = np.array(
            [[200.0, 100.0, 210.0, 110.0]],
            dtype=np.float64,
        )

        both_dissimilar = np.array(
            [[50.0, 0.0, 450.0, 900.0]],
            dtype=np.float64,
        )

        p_ss = float(model._calculate_p_ll(similar_amount, soft)[0])
        p_ll = float(model._calculate_p_ll(similar_delay, soft)[0])
        p_both_sim = float(model._calculate_p_ll(both_similar, soft)[0])
        p_both_dis = float(model._calculate_p_ll(both_dissimilar, soft)[0])

        self.assertLess(p_ss, 0.5)
        self.assertGreater(p_ll, 0.5)
        self.assertAlmostEqual(p_both_sim, 0.5, delta=0.15)
        self.assertAlmostEqual(p_both_dis, 0.5, delta=0.15)

    def test_m11_uses_constant_ll_probability(self):
        model = M11ConstantLLProbability()
        self.assertEqual(model.n_parameters, 1)
        self.assertEqual(model.parameter_names, ["p"])

        parameters = {"p": 0.55}
        easy_trials = np.array(
            [[10.0, 0.0, 500.0, 365.0]],
            dtype=np.float64,
        )
        hard_trials = np.array(
            [[500.0, 365.0, 10.0, 0.0]],
            dtype=np.float64,
        )
        p_easy = model._calculate_p_ll(easy_trials, parameters)
        p_hard = model._calculate_p_ll(hard_trials, parameters)
        np.testing.assert_allclose(p_easy, [0.55])
        np.testing.assert_allclose(p_hard, [0.55])

        sampled = model.sample_parameters(seed=SEED, n_subjects=1000)
        self.assertTrue(np.all(sampled[:, 0] >= 0.3))
        self.assertTrue(np.all(sampled[:, 0] <= 0.7))


if __name__ == "__main__":
    unittest.main()
