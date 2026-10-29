from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import TRIAL_GENERATOR_CONFIGS
from .general import general_400
from .published import (
    konstantinidis_2020_exp1_intertemporal,
    thrailkill_2023_delay_discounting,
)
from .synthetic import grid_ssll


TRIAL_GENERATORS = {
    "grid_ssll": grid_ssll,
    "konstantinidis_2020_exp1_intertemporal": (
        konstantinidis_2020_exp1_intertemporal
    ),
    "thrailkill_2023_delay_discounting": (
        thrailkill_2023_delay_discounting
    ),
    "general_400": general_400,
}


@dataclass(frozen=True, slots=True)
class SubjectSpecificTrials:

    trials: np.ndarray
    trial_mask: np.ndarray
    n_trials_per_subject: np.ndarray


def _get_trial_config(data_name: str) -> dict:
    try:
        return TRIAL_GENERATOR_CONFIGS[data_name]
    except KeyError:
        available_names = ", ".join(TRIAL_GENERATOR_CONFIGS)
        raise ValueError(
            f"Unknown trial data name: {data_name}. "
            f"Available names: {available_names}"
        ) from None


def get_trial_layout(data_name: str) -> str:
    return _get_trial_config(data_name).get("layout", "shared")


def is_subject_specific(data_name: str) -> bool:
    return get_trial_layout(data_name) == "subject_specific"


def generate_trials(
    data_name: str,
    **kwargs: Any,
) -> pd.DataFrame | SubjectSpecificTrials:
    config = _get_trial_config(data_name)
    if is_subject_specific(data_name) and "n_subjects" not in kwargs:
        raise TypeError(f"{data_name} requires n_subjects.")
    generator = TRIAL_GENERATORS[config["generator"]]
    generator_kwargs = config["generator_kwargs"] | kwargs
    result = generator(**generator_kwargs)
    if is_subject_specific(data_name):
        trials, trial_mask, n_trials_per_subject = result
        return SubjectSpecificTrials(
            trials=trials,
            trial_mask=trial_mask,
            n_trials_per_subject=n_trials_per_subject,
        )
    return result


def get_trial_source(data_name: str) -> str:
    return _get_trial_config(data_name)["source"]
