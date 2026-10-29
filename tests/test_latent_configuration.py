from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(
    0,
    str(PROJECT_ROOT / "experiments" / "step_03_latent_models"),
)

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed.")
class LatentConfigurationTest(unittest.TestCase):
    def test_real_data_are_ordered_before_simulated_models(self):
        from common import order_data_names

        self.assertEqual(
            order_data_names(("m3", "r2", "m1", "r1")),
            ("r1", "r2", "m1", "m3"),
        )
        self.assertEqual(
            order_data_names(("m10", "m9", "m2", "r1")),
            ("r1", "m2", "m9", "m10"),
        )

    def test_parameter_decoder_has_one_head_per_model(self):
        from parameter_decoder import (
            ChoiceParameterDecoder,
            ParameterDecoderConfig,
        )

        config = ParameterDecoderConfig(
            latent_dim=4,
            parameter_dims=(("m1", 2), ("m3", 3)),
        )
        model = ChoiceParameterDecoder(config)
        trials = torch.rand(5, 7, 4)
        choices = torch.randint(0, 2, (5, 7, 1)).float()
        trial_mask = torch.ones(5, 7, dtype=torch.bool)

        output = model(trials, choices, trial_mask)

        self.assertEqual(
            tuple(output["parameter_predictions"]["m1"].shape),
            (5, 2),
        )
        self.assertEqual(
            tuple(output["parameter_predictions"]["m3"].shape),
            (5, 3),
        )

    def test_parameter_decoder_default_includes_heuristic_heads(self):
        from parameter_decoder import ParameterDecoderConfig

        dims = dict(ParameterDecoderConfig().parameter_dims)
        self.assertEqual(dims["m9"], 4)
        self.assertEqual(dims["m10"], 5)

    def test_parameter_decoder_trains_all_model_heads(self):
        from neural_processes_choice import ChoiceData
        from parameter_decoder import (
            ParameterDecoderConfig,
            ParameterDecoderTrainer,
            ParameterDecoderTrainingConfig,
        )

        n_subjects = 12
        data = ChoiceData(
            trials=torch.rand(7, 4),
            choices=torch.randint(0, 2, (n_subjects, 7, 1)).float(),
            trial_mask=torch.ones(n_subjects, 7, dtype=torch.bool),
            metadata={},
        )
        model_labels = torch.tensor([0] * 6 + [1] * 6)
        subject_indices = {
            "m1": torch.arange(0, 6),
            "m3": torch.arange(6, 12),
        }
        parameters = {
            "m1": torch.rand(6, 2) + 0.1,
            "m3": torch.rand(6, 3) * 0.8 + 0.1,
        }
        splits = {
            "train": torch.tensor([0, 1, 2, 6, 7, 8]),
            "validation": torch.tensor([3, 9]),
            "test": torch.tensor([4, 5, 10, 11]),
        }
        model_config = ParameterDecoderConfig(
            embedding_dim=8,
            latent_dim=4,
            num_attention_blocks=1,
            num_attention_heads=2,
            trial_encoder_hidden_dims=(8,),
            parameter_dims=(("m1", 2), ("m3", 3)),
            decoder_hidden_dims=(8,),
        )
        trainer = ParameterDecoderTrainer(
            model_config=model_config,
            training_config=ParameterDecoderTrainingConfig(
                batch_size=4,
                max_epochs=1,
                patience=1,
                normalize_trials=False,
            ),
            device="cpu",
        )

        result = trainer.fit(
            data=data,
            model_labels=model_labels,
            parameters_by_model=parameters,
            parameter_names_by_model={
                "m1": ["k", "m"],
                "m3": ["u", "v", "w"],
            },
            subject_indices_by_model=subject_indices,
            subject_splits=splits,
        )

        self.assertEqual(
            set(result.parameter_standardizers),
            {"m1", "m3"},
        )
        self.assertGreaterEqual(result.test_metrics["loss"], 0.0)

    def test_preprocessing_persists_fixed_per_dataset_splits(self):
        import common
        from common import (
            load_experiment_data,
            load_preprocessed_experiment,
            prepare_experiment_data,
            save_preprocessed_experiment,
            select_model_labeled_data,
        )

        trial_data_name = "pub_Test"
        data_names = ("r1", "m1")
        trials = np.array(
            [
                [10.0, 0.0, 20.0, 5.0],
                [20.0, 0.0, 30.0, 10.0],
                [30.0, 0.0, 40.0, 20.0],
            ],
            dtype=np.float32,
        )
        original_project_root = common.PROJECT_ROOT
        with tempfile.TemporaryDirectory() as temporary_directory:
            common.PROJECT_ROOT = Path(temporary_directory)
            try:
                for data_index, data_name in enumerate(data_names):
                    output_dir = (
                        common.PROJECT_ROOT
                        / "output"
                        / "choices"
                        / trial_data_name
                        / data_name
                        / "n6"
                    )
                    output_dir.mkdir(parents=True)
                    choices = np.full(
                        (6, len(trials), 1),
                        data_index,
                        dtype=np.int8,
                    )
                    np.savez_compressed(
                        output_dir / "data.npz",
                        trials=trials,
                        choices=choices,
                        trial_mask=np.ones(
                            choices.shape[:2],
                            dtype=bool,
                        ),
                        trial_columns=np.array(
                            [
                                "amount_ss",
                                "delay_ss",
                                "amount_ll",
                                "delay_ll",
                            ]
                        ),
                        parameter_values=np.ones(
                            (6, 2),
                            dtype=np.float32,
                        ),
                        parameter_names=np.array(["k", "m"]),
                    )

                prepared = prepare_experiment_data(
                    trial_data_name,
                    data_names,
                )
                artifact_path = save_preprocessed_experiment(prepared)
                loaded = load_preprocessed_experiment(
                    trial_data_name,
                    data_names,
                )
                selected = select_model_labeled_data(
                    loaded,
                )
                direct = load_experiment_data(
                    trial_data_name,
                    ("m1",),
                )

                self.assertTrue(artifact_path.exists())
                self.assertEqual(loaded.metadata["model_names"], ["m1"])
                self.assertTrue(
                    torch.all(
                        loaded.model_labels[
                            loaded.subject_indices_by_data["r1"]
                        ]
                        == -1
                    )
                )
                self.assertEqual(selected.data_names, ("m1",))
                for split_name in ("train", "validation", "test"):
                    self.assertTrue(
                        torch.equal(
                            selected.subject_splits[split_name],
                            direct.subject_splits[split_name],
                        )
                    )
                all_selected_indices = torch.cat(
                    list(selected.subject_splits.values())
                )
                self.assertEqual(
                    set(all_selected_indices.tolist()),
                    set(range(6)),
                )
                np.testing.assert_allclose(
                    prepared.choice_data.trials.mean(dim=0).numpy(),
                    np.zeros(4),
                    atol=1e-6,
                )
            finally:
                common.PROJECT_ROOT = original_project_root

    def test_preprocessing_concatenates_subject_specific_trials(self):
        import common
        from common import load_experiment_data, prepare_experiment_data

        trial_data_name = "general_400"
        data_names = ("m1", "m2")
        trials = np.zeros((6, 4, 4), dtype=np.float32)
        trial_mask = np.zeros((6, 4), dtype=bool)
        trials[:, 0] = [10.0, 0.0, 20.0, 5.0]
        trials[:, 1] = [20.0, 0.0, 30.0, 10.0]
        trials[:, 2] = [30.0, 0.0, 40.0, 20.0]
        trial_mask[:, :3] = True
        original_project_root = common.PROJECT_ROOT
        with tempfile.TemporaryDirectory() as temporary_directory:
            common.PROJECT_ROOT = Path(temporary_directory)
            try:
                for data_index, data_name in enumerate(data_names):
                    output_dir = (
                        common.PROJECT_ROOT
                        / "output"
                        / "choices"
                        / trial_data_name
                        / data_name
                        / "n6"
                    )
                    output_dir.mkdir(parents=True)
                    choices = np.full(
                        (6, 4, 1),
                        data_index,
                        dtype=np.int8,
                    )
                    choices[~trial_mask] = 0
                    np.savez_compressed(
                        output_dir / "data.npz",
                        trials=trials,
                        choices=choices,
                        trial_mask=trial_mask,
                        trial_columns=np.array(
                            [
                                "amount_ss",
                                "delay_ss",
                                "amount_ll",
                                "delay_ll",
                            ]
                        ),
                        parameter_values=np.ones(
                            (6, 2),
                            dtype=np.float32,
                        ),
                        parameter_names=np.array(["k", "m"]),
                    )

                loaded = load_experiment_data(trial_data_name, data_names)
                self.assertEqual(loaded.choice_data.trials.shape, (12, 4, 4))
                self.assertEqual(loaded.choice_data.trial_mask.shape, (12, 4))
                np.testing.assert_allclose(
                    loaded.choice_data.trials[:6].numpy(),
                    trials,
                )
                np.testing.assert_allclose(
                    loaded.choice_data.trials[6:].numpy(),
                    trials,
                )

                prepared = prepare_experiment_data(
                    trial_data_name,
                    data_names,
                )
                valid = prepared.choice_data.trial_mask
                valid_trials = prepared.choice_data.trials[valid]
                np.testing.assert_allclose(
                    valid_trials.mean(dim=0).numpy(),
                    np.zeros(4),
                    atol=1e-5,
                )
            finally:
                common.PROJECT_ROOT = original_project_root


if __name__ == "__main__":
    unittest.main()
