from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import xlogy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
from trial_generators.generate_trials import (
    SubjectSpecificTrials,
    generate_trials,
    is_subject_specific,
)


N_SUBJECTS = 10000
SEED = 31
TRIAL_DATA_NAME = "pub_Thrailkill2023"
MODEL_CLASSES = {
    "m1": M1Hyperbolic,
    "m2": M2Hyperboloid,
    "m3": M3DirectDifference,
    "m4": M4Exponential,
    "m5": M5DoubleExponential,
    "m6": M6ConstantSensitivity,
    "m7": M7AdditiveUtility,
    "m8": M8ProportionalDifference,
    "m9": M9PriorityLexicographicSemiorder,
    "m10": M10SimilarityHeuristic,
    "m11": M11ConstantLLProbability,
}
TRIAL_COLUMNS = ["amount_ss", "delay_ss", "amount_ll", "delay_ll"]


def calculate_probabilities(model, trials, parameter_values, trial_mask=None):
    if isinstance(trials, pd.DataFrame):
        trial_values = trials[TRIAL_COLUMNS].to_numpy(dtype=np.float64)
        return np.array(
            [
                model._calculate_p_ll(
                    trial_values,
                    model.get_parameters_dict(parameter_row),
                )
                for parameter_row in parameter_values
            ]
        )

    trial_values = np.asarray(trials, dtype=np.float64)
    if trial_values.ndim == 2:
        return np.array(
            [
                model._calculate_p_ll(
                    trial_values,
                    model.get_parameters_dict(parameter_row),
                )
                for parameter_row in parameter_values
            ]
        )

    n_subjects, n_trials, _ = trial_values.shape
    probabilities = np.full((n_subjects, n_trials), np.nan)
    for subject_index, parameter_row in enumerate(parameter_values):
        valid = trial_mask[subject_index]
        probabilities[subject_index, valid] = model._calculate_p_ll(
            trial_values[subject_index, valid],
            model.get_parameters_dict(parameter_row),
        )
    return probabilities


if is_subject_specific(TRIAL_DATA_NAME):
    trial_data = generate_trials(
        TRIAL_DATA_NAME,
        n_subjects=N_SUBJECTS,
        seed=SEED,
    )
    assert isinstance(trial_data, SubjectSpecificTrials)
    trials = trial_data.trials
    trial_mask = trial_data.trial_mask
    choice_kwargs = {"trial_mask": trial_mask}
else:
    trials = generate_trials(TRIAL_DATA_NAME)
    trial_mask = None
    choice_kwargs = {}

summary_rows = []
participant_rows = []

n_models = len(MODEL_CLASSES)
n_columns = 4
n_rows = int(np.ceil(n_models / n_columns))
figure, axes = plt.subplots(
    n_rows,
    n_columns,
    figsize=(5 * n_columns, 3.5 * n_rows),
    sharex=True,
    sharey=True,
)
axes_flat = np.asarray(axes).reshape(-1)
histogram_bins = np.linspace(0.0, 1.0, 21)

for model_index, (model_name, model_class) in enumerate(
    MODEL_CLASSES.items()
):
    model = model_class()
    parameter_values = model.sample_parameters(
        seed=SEED,
        n_subjects=N_SUBJECTS,
    )
    probabilities = calculate_probabilities(
        model,
        trials,
        parameter_values,
        trial_mask=trial_mask,
    )
    choices, used_trial_mask = model.generate_choices(
        trials,
        parameter_values,
        seed=SEED,
        **choice_kwargs,
    )

    if trial_mask is None:
        expected_ll_proportion = probabilities.mean(axis=1)
        simulated_ll_proportion = choices[:, :, 0].mean(axis=1)
        binary_entropy = -(
            xlogy(probabilities, probabilities)
            + xlogy(1.0 - probabilities, 1.0 - probabilities)
        )
        mean_binary_entropy = binary_entropy.mean()
    else:
        n_valid = used_trial_mask.sum(axis=1)
        expected_ll_proportion = np.nanmean(probabilities, axis=1)
        simulated_ll_proportion = (
            (choices[:, :, 0] * used_trial_mask).sum(axis=1) / n_valid
        )
        binary_entropy = -(
            xlogy(probabilities, probabilities)
            + xlogy(1.0 - probabilities, 1.0 - probabilities)
        )
        mean_binary_entropy = np.nanmean(binary_entropy)

    summary_rows.append(
        {
            "model_name": model_name,
            "mean_p_ll": np.nanmean(probabilities),
            "mean_simulated_ll_proportion": simulated_ll_proportion.mean(),
            "sd_simulated_ll_proportion": simulated_ll_proportion.std(),
            "percent_almost_all_ss": (
                100.0 * np.mean(expected_ll_proportion < 0.05)
            ),
            "percent_almost_all_ll": (
                100.0 * np.mean(expected_ll_proportion > 0.95)
            ),
            "mean_binary_entropy_nats": mean_binary_entropy,
        }
    )
    participant_rows.extend(
        {
            "model_name": model_name,
            "subject_index": subject_index,
            "expected_ll_proportion": expected_ll_proportion[subject_index],
            "simulated_ll_proportion": simulated_ll_proportion[subject_index],
        }
        for subject_index in range(N_SUBJECTS)
    )

    axis = axes_flat[model_index]
    axis.hist(
        simulated_ll_proportion,
        bins=histogram_bins,
        color="tab:blue",
        alpha=0.8,
    )
    axis.axvline(
        simulated_ll_proportion.mean(),
        color="tab:red",
        linestyle="--",
        label="Mean",
    )
    axis.set_title(model_name.upper())
    axis.set_xlabel("Participant LL proportion")
    axis.set_ylabel("Number of participants")
    axis.legend()

for axis in axes_flat[n_models:]:
    axis.set_visible(False)
summary = pd.DataFrame(summary_rows)
participant_proportions = pd.DataFrame(participant_rows)

output_dir = (
    PROJECT_ROOT
    / "output"
    / "prior_predictive"
    / TRIAL_DATA_NAME
    / f"n{N_SUBJECTS}_s{SEED}"
)
output_dir.mkdir(parents=True, exist_ok=True)
summary_path = output_dir / "summary.csv"
participant_path = output_dir / "participant_ll_proportions.csv"
figure_path = output_dir / "participant_ll_proportion_distributions.png"

summary.to_csv(summary_path, index=False)
participant_proportions.to_csv(participant_path, index=False)
figure.suptitle(
    f"Prior-predictive LL proportions: {TRIAL_DATA_NAME}",
    fontsize=14,
)
figure.tight_layout()
figure.savefig(figure_path, dpi=200, bbox_inches="tight")
plt.close(figure)

print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print(f"Summary saved to: {summary_path}")
print(f"Participant proportions saved to: {participant_path}")
print(f"Distribution figure saved to: {figure_path}")
