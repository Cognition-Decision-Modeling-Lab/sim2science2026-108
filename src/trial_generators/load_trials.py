from pathlib import Path

import numpy as np
import pandas as pd

from .generate_trials import SubjectSpecificTrials


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRIAL_DATA_DIR = PROJECT_ROOT / "output" / "trials"


def load_trials(data_name):
    matching_files = [
        path
        for path in TRIAL_DATA_DIR.rglob(f"{data_name}.*")
        if path.suffix in {".csv", ".npz"}
    ]

    if not matching_files:
        raise FileNotFoundError(f"Unknown trial data name: {data_name}")
    if len(matching_files) > 1:
        raise ValueError(f"Trial data name is not unique: {data_name}")

    path = matching_files[0]
    if path.suffix == ".npz":
        with np.load(path) as archive:
            return SubjectSpecificTrials(
                trials=np.array(archive["trials"], copy=True),
                trial_mask=np.array(archive["trial_mask"], copy=True),
                n_trials_per_subject=np.array(
                    archive["n_trials_per_subject"],
                    copy=True,
                ),
            )
    return pd.read_csv(path)
