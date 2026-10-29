from pathlib import Path
import sys

import numpy as np


TRIAL_DATA_NAME = "general_400"
N_SUBJECTS = 50000
SEED = 31

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trial_generators.generate_trials import (
    SubjectSpecificTrials,
    generate_trials,
    get_trial_source,
    is_subject_specific,
)
from trial_generators.general import TRIAL_COLUMNS


source = get_trial_source(TRIAL_DATA_NAME)
output_dir = PROJECT_ROOT / "output" / "trials" / source
output_dir.mkdir(parents=True, exist_ok=True)

if is_subject_specific(TRIAL_DATA_NAME):
    trial_data = generate_trials(
        TRIAL_DATA_NAME,
        n_subjects=N_SUBJECTS,
        seed=SEED,
    )
    assert isinstance(trial_data, SubjectSpecificTrials)
    valid = trial_data.trials[trial_data.trial_mask]
    assert valid.shape[1] == len(TRIAL_COLUMNS)
    assert np.isfinite(valid).all()
    assert (valid[:, 0] > 0).all()
    assert (valid[:, 2] > 0).all()
    assert (valid[:, 1] >= 0).all()
    assert (valid[:, 3] >= 0).all()
    assert (valid[:, 0] < valid[:, 2]).all()
    assert (valid[:, 1] < valid[:, 3]).all()
    assert trial_data.trials.shape == (
        N_SUBJECTS,
        trial_data.trials.shape[1],
        len(TRIAL_COLUMNS),
    )
    assert np.array_equal(
        trial_data.trial_mask.sum(axis=1),
        trial_data.n_trials_per_subject,
    )

    output_path = output_dir / f"{TRIAL_DATA_NAME}.npz"
    np.savez_compressed(
        output_path,
        trials=trial_data.trials,
        trial_mask=trial_data.trial_mask,
        n_trials_per_subject=trial_data.n_trials_per_subject,
        trial_columns=np.array(TRIAL_COLUMNS, dtype=str),
        seed=np.array(SEED),
        n_subjects=np.array(N_SUBJECTS),
    )

    print(f"Trial data: {TRIAL_DATA_NAME}")
    print(f"Source: {source}")
    print(f"Subjects: {N_SUBJECTS}")
    print(f"Padded trial length: {trial_data.trials.shape[1]}")
    print(
        "Valid trials per subject: "
        f"{trial_data.n_trials_per_subject.min()}–"
        f"{trial_data.n_trials_per_subject.max()}"
    )
    print(f"Saved to: {output_path}")
else:
    trials = generate_trials(TRIAL_DATA_NAME)
    expected_columns = list(TRIAL_COLUMNS)
    assert list(trials.columns) == expected_columns
    assert np.isfinite(trials.to_numpy()).all()
    assert (trials["amount_ss"] > 0).all()
    assert (trials["amount_ll"] > 0).all()
    assert (trials["delay_ss"] >= 0).all()
    assert (trials["delay_ll"] >= 0).all()
    assert (trials["amount_ss"] < trials["amount_ll"]).all()
    assert (trials["delay_ss"] < trials["delay_ll"]).all()
    assert len(trials.drop_duplicates()) == len(trials)

    output_path = output_dir / f"{TRIAL_DATA_NAME}.csv"
    trials.to_csv(output_path, index=False)

    print(f"Trial data: {TRIAL_DATA_NAME}")
    print(f"Source: {source}")
    print(f"Number of trials: {len(trials)}")
    print(f"Unique trials: {len(trials.drop_duplicates())}")
    print(f"Saved to: {output_path}")
