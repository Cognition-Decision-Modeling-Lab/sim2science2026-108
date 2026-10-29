from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


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
    load_real_choice_data,
)
from trial_generators.generate_trials import (
    SubjectSpecificTrials,
)
from trial_generators.general import TRIAL_COLUMNS
from trial_generators.load_trials import load_trials


TRIAL_DATA_NAME = ["general_400", "syn_GridSSLL", "pub_Konstantinidis2020", "pub_Thrailkill2023"][3]
CHOICE_DATA_NAME = "m9"


SIMULATED_N_SUBJECTS = 50000
SEED = 31
PONGRATZ_PATH = (
    PROJECT_ROOT
    / "data"
    / "Pongratz2026"
    / "data"
    / "all_data.csv"
)

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
loaded_trials = load_trials(TRIAL_DATA_NAME)
subject_specific = isinstance(loaded_trials, SubjectSpecificTrials)
model = None
extra_metadata = {}

if CHOICE_DATA_NAME in MODEL_CLASSES:
    model = MODEL_CLASSES[CHOICE_DATA_NAME]()
    if subject_specific:
        n_subjects = int(loaded_trials.trials.shape[0])
        parameter_values = model.sample_parameters(
            seed=SEED,
            n_subjects=n_subjects,
        )
        choices, trial_mask = model.generate_choices(
            loaded_trials.trials,
            parameter_values,
            seed=SEED,
            trial_mask=loaded_trials.trial_mask,
        )
        trials_to_save = loaded_trials.trials
        trial_columns = np.array(TRIAL_COLUMNS, dtype=str)
        extra_metadata["n_trials_per_subject"] = (
            loaded_trials.n_trials_per_subject
        )
    else:
        n_subjects = SIMULATED_N_SUBJECTS
        parameter_values = model.sample_parameters(
            seed=SEED,
            n_subjects=n_subjects,
        )
        choices, trial_mask = model.generate_choices(
            loaded_trials,
            parameter_values,
            seed=SEED,
        )
        trials_to_save = loaded_trials.to_numpy(dtype=np.float32)
        trial_columns = np.array(loaded_trials.columns, dtype=str)
    parameter_names = np.array(model.parameter_names, dtype=str)
    parameter_descriptions = np.array(
        model.parameter_descriptions,
        dtype=str,
    )
    extra_metadata["seed"] = np.array(SEED)
else:
    if subject_specific:
        raise ValueError(
            "Observed choice import is not defined for subject-specific "
            f"trial designs such as {TRIAL_DATA_NAME}."
        )
    real_data = load_real_choice_data(
        trial_data_name=TRIAL_DATA_NAME,
        real_data_name=CHOICE_DATA_NAME,
        pongratz_path=PONGRATZ_PATH,
    )
    np.testing.assert_allclose(
        real_data.trials.to_numpy(dtype=np.float32),
        loaded_trials.to_numpy(dtype=np.float32),
    )
    choices = real_data.choices
    trial_mask = real_data.trial_mask
    n_subjects = real_data.n_subjects
    parameter_values = np.empty((n_subjects, 0), dtype=np.float32)
    parameter_names = np.array([], dtype=str)
    parameter_descriptions = np.array([], dtype=str)
    trials_to_save = loaded_trials.to_numpy(dtype=np.float32)
    trial_columns = np.array(loaded_trials.columns, dtype=str)
    extra_metadata.update(
        {
            "subject_ids": real_data.subject_ids,
            "source_study": np.array(real_data.source_study),
        }
    )

output_dir = (
    PROJECT_ROOT
    / "output"
    / "choices"
    / TRIAL_DATA_NAME
    / CHOICE_DATA_NAME
    / f"n{n_subjects}"
)
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "data.npz"

np.savez_compressed(
    output_path,
    trials=trials_to_save,
    trial_columns=trial_columns,
    choices=choices,
    trial_mask=trial_mask,
    parameter_values=parameter_values,
    parameter_names=parameter_names,
    parameter_descriptions=parameter_descriptions,
    trial_data_name=np.array(TRIAL_DATA_NAME),
    model_name=np.array(CHOICE_DATA_NAME),
    n_subjects=np.array(n_subjects),
    **extra_metadata,
)

print(f"Trial data: {TRIAL_DATA_NAME}")
print(f"Choice data: {CHOICE_DATA_NAME}")
print(f"Subjects: {n_subjects}")
if subject_specific:
    n_valid = loaded_trials.n_trials_per_subject
    print(f"Padded trial length: {loaded_trials.trials.shape[1]}")
    print(f"Valid trials per subject: {n_valid.min()}–{n_valid.max()}")
else:
    print(f"Trials per subject: {len(loaded_trials)}")
print(f"Parameter shape: {parameter_values.shape}")
print(f"Choice shape: {choices.shape}")
print(f"Trial mask shape: {trial_mask.shape}")
print(f"Saved to: {output_path}")

if model is not None:
    model.plot_priors(show=False)
    model.plot_choice_proportions(
        choices,
        show=False,
        trial_mask=trial_mask,
    )
    plt.show()
