import torch

from common import (
    prepare_experiment_data,
    save_preprocessed_experiment,
)

FULL_REAL = 1
TRIAL_DATA_NAME = ["general_400", "syn_GridSSLL", "pub_Konstantinidis2020", "pub_Thrailkill2023"][2]
DATA_NAMES = ("m1", "m2", "m3")
DATA_NAMES = ("m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9")
TRAIN_FRACTION = 0.75
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.10
SEED = 31


if abs(TRAIN_FRACTION + VALIDATION_FRACTION + TEST_FRACTION - 1.0) > 1e-8:
    raise ValueError("Train, validation, and test fractions must sum to one.")
experiment = prepare_experiment_data(
    trial_data_name=TRIAL_DATA_NAME,
    data_names=DATA_NAMES,
    train_fraction=TRAIN_FRACTION,
    validation_fraction=VALIDATION_FRACTION,
    seed=SEED,
    full_real=FULL_REAL,
)
output_path = save_preprocessed_experiment(experiment)

print(f"Trial data: {TRIAL_DATA_NAME}")
print(f"Full real: {FULL_REAL}")
print(f"Datasets: {experiment.data_names}")
for data_name in experiment.data_names:
    subject_indices = experiment.subject_indices_by_data[data_name]
    split_counts = {
        split_name: int(
            torch.isin(subject_indices, split_indices).sum()
        )
        for split_name, split_indices in experiment.subject_splits.items()
    }
    print(
        f"{data_name}: total={len(subject_indices)}, "
        f"train={split_counts['train']}, "
        f"validation={split_counts['validation']}, "
        f"test={split_counts['test']}"
    )
print(f"Saved to: {output_path}")
