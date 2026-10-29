from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trial_generators.config import TRIAL_GENERATOR_CONFIGS
from trial_generators.generate_trials import (
    SubjectSpecificTrials,
    generate_trials,
    get_trial_source,
    is_subject_specific,
)
from trial_generators.general import (
    GENERAL_400_MAX_TRIALS,
    general_400,
)
from trial_generators.published import (
    konstantinidis_2020_exp1_intertemporal,
    thrailkill_2023_delay_discounting,
)
from trial_generators.synthetic import grid_ssll


class TrialGeneratorTest(unittest.TestCase):
    def test_konstantinidis_delays_are_in_days(self):
        trials = konstantinidis_2020_exp1_intertemporal()
        expected_delays = np.arange(2.0, 39.0, 2.0) * 365.0 / 12.0

        self.assertEqual(trials.shape, (380, 4))
        np.testing.assert_allclose(
            np.sort(trials["delay_ll"].unique()),
            expected_delays,
            rtol=1e-6,
        )

    def test_thrailkill_choice_set(self):
        trials = thrailkill_2023_delay_discounting()

        self.assertEqual(trials.shape, (27, 4))
        self.assertEqual(len(trials.drop_duplicates()), 27)
        self.assertTrue((trials["delay_ss"] == 0).all())
        self.assertEqual(trials["delay_ll"].min(), 7)
        self.assertEqual(trials["delay_ll"].max(), 186)
        self.assertTrue((trials["amount_ss"] < trials["amount_ll"]).all())
        self.assertTrue((trials["delay_ss"] < trials["delay_ll"]).all())
        self.assertTrue(all(dtype == np.float32 for dtype in trials.dtypes))

    def test_thrailkill_is_available_through_unified_interface(self):
        trials = generate_trials("pub_Thrailkill2023")

        self.assertEqual(trials.shape, (27, 4))
        self.assertEqual(
            get_trial_source("pub_Thrailkill2023"),
            "published",
        )

    def test_all_data_names_are_available_through_unified_interface(self):
        expected_shapes = {
            "syn_GridSSLL": (225, 4),
            "pub_Konstantinidis2020": (380, 4),
            "pub_Thrailkill2023": (27, 4),
        }

        for data_name, expected_shape in expected_shapes.items():
            with self.subTest(data_name=data_name):
                self.assertEqual(
                    generate_trials(data_name).shape,
                    expected_shape,
                )

        trial_data = generate_trials(
            "general_400",
            n_subjects=3,
            seed=31,
        )
        self.assertIsInstance(trial_data, SubjectSpecificTrials)
        self.assertEqual(
            trial_data.trials.shape,
            (3, GENERAL_400_MAX_TRIALS, 4),
        )
        self.assertTrue(is_subject_specific("general_400"))
        self.assertFalse(is_subject_specific("syn_GridSSLL"))

    def test_data_names_follow_source_description_format(self):
        prefix_by_source = {
            "synthetic": "syn_",
            "published": "pub_",
            "general": "general_",
        }
        for data_name, config in TRIAL_GENERATOR_CONFIGS.items():
            with self.subTest(data_name=data_name):
                expected_prefix = prefix_by_source[config["source"]]
                self.assertTrue(data_name.startswith(expected_prefix))
                description = data_name.removeprefix(expected_prefix)
                self.assertTrue(description)
                self.assertTrue(description.isalnum())

    def test_config_metadata_matches_generated_trials(self):
        for data_name, config in TRIAL_GENERATOR_CONFIGS.items():
            with self.subTest(data_name=data_name):
                if config.get("layout") == "subject_specific":
                    trial_data = generate_trials(
                        data_name,
                        n_subjects=8,
                        seed=31,
                    )
                    valid = trial_data.trials[trial_data.trial_mask]
                    money_values = valid[:, [0, 2]]
                    day_values = valid[:, [1, 3]]
                    self.assertEqual(
                        trial_data.trials.shape[1],
                        config["n_trials"],
                    )
                    self.assertGreaterEqual(
                        money_values.min(),
                        config["money_range"][0],
                    )
                    self.assertLessEqual(
                        money_values.max(),
                        config["money_range"][1],
                    )
                    self.assertGreaterEqual(
                        day_values.min(),
                        config["day_range"][0],
                    )
                    self.assertLessEqual(
                        day_values.max(),
                        config["day_range"][1],
                    )
                    self.assertTrue(config["money_unit"])
                    continue

                trials = generate_trials(data_name)
                money_values = trials[["amount_ss", "amount_ll"]].to_numpy()
                day_values = trials[["delay_ss", "delay_ll"]].to_numpy()

                self.assertEqual(len(trials), config["n_trials"])
                np.testing.assert_allclose(
                    (money_values.min(), money_values.max()),
                    config["money_range"],
                )
                np.testing.assert_allclose(
                    (day_values.min(), day_values.max()),
                    config["day_range"],
                    rtol=1e-6,
                )
                self.assertTrue(config["money_unit"])

    def test_synthetic_delays_are_in_days(self):
        trials = grid_ssll()

        np.testing.assert_allclose(
            trials["delay_ll"].max(),
            38.0 * 365.0 / 12.0,
            rtol=1e-6,
        )

    def test_general_400_returns_padded_subject_specific_trials(self):
        trials, trial_mask, n_trials_per_subject = general_400(
            n_subjects=6,
            seed=31,
        )

        self.assertEqual(trials.shape, (6, GENERAL_400_MAX_TRIALS, 4))
        self.assertEqual(trial_mask.shape, (6, GENERAL_400_MAX_TRIALS))
        self.assertEqual(n_trials_per_subject.shape, (6,))
        self.assertEqual(trials.dtype, np.float32)
        self.assertEqual(trial_mask.dtype, np.bool_)
        np.testing.assert_array_equal(
            trial_mask.sum(axis=1),
            n_trials_per_subject,
        )
        self.assertTrue(
            np.all(n_trials_per_subject >= 20)
            and np.all(n_trials_per_subject <= GENERAL_400_MAX_TRIALS)
        )

        valid = trials[trial_mask]
        self.assertTrue(np.isfinite(valid).all())
        self.assertTrue((valid[:, 0] < valid[:, 2]).all())
        self.assertTrue((valid[:, 1] < valid[:, 3]).all())
        self.assertTrue((trials[~trial_mask] == 0).all())
        self.assertEqual(get_trial_source("general_400"), "general")
        with self.assertRaises(TypeError):
            generate_trials("general_400")


if __name__ == "__main__":
    unittest.main()
