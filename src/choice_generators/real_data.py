from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trial_generators.published import (
    konstantinidis_2020_exp1_intertemporal,
    thrailkill_2023_delay_discounting,
)


TRIAL_COLUMNS = ["amount_ss", "delay_ss", "amount_ll", "delay_ll"]
TRIAL_MATCH_DECIMALS = 3
REAL_CHOICE_LOADERS = {
    ("pub_Thrailkill2023", "r1"): (
        "Thrailkill_2023",
        thrailkill_2023_delay_discounting,
    ),
    ("pub_Konstantinidis2020", "r1"): (
        "Konstantinidis_2020",
        konstantinidis_2020_exp1_intertemporal,
    ),
}
PONGRATZ_COLUMNS = [
    "paper",
    "subj_ident",
    "ss_value",
    "ss_time",
    "ll_value",
    "ll_time",
    "choice",
    "subj_excl",
    "trial_excl",
]


@dataclass(frozen=True, slots=True)
class RealChoiceData:

    trials: pd.DataFrame
    choices: np.ndarray
    trial_mask: np.ndarray
    subject_ids: np.ndarray
    source_study: str

    @property
    def n_subjects(self) -> int:
        return self.choices.shape[0]


def _load_pongratz_study(
    source_path: str | Path,
    paper: str,
) -> pd.DataFrame:
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing Pongratz data file: {source_path}")

    chunks = []
    for chunk in pd.read_csv(
        source_path,
        usecols=PONGRATZ_COLUMNS,
        dtype={
            "paper": "string",
            "subj_ident": "string",
            "subj_excl": "boolean",
            "trial_excl": "boolean",
        },
        chunksize=100_000,
    ):
        selected = chunk.loc[chunk["paper"] == paper]
        if not selected.empty:
            chunks.append(selected)

    if not chunks:
        raise ValueError(f"Study {paper!r} was not found in {source_path}.")
    return pd.concat(chunks, ignore_index=True)


def _trial_match_keys(frame: pd.DataFrame, columns: list[str]) -> pd.MultiIndex:
    rounded = np.round(
        frame[columns].to_numpy(dtype=np.float64),
        TRIAL_MATCH_DECIMALS,
    )
    return pd.MultiIndex.from_arrays(rounded.T, names=columns)


def _align_observed_choices(
    rows: pd.DataFrame,
    trials: pd.DataFrame,
    source_study: str,
) -> RealChoiceData:
    subject_excluded = rows["subj_excl"].fillna(True)
    rows = rows.loc[~subject_excluded].copy()
    if rows.empty:
        raise ValueError("No non-excluded subjects remain.")

    source_trial_columns = [
        "ss_value",
        "ss_time",
        "ll_value",
        "ll_time",
    ]
    canonical_keys = _trial_match_keys(trials, TRIAL_COLUMNS)
    observed_keys = _trial_match_keys(rows, source_trial_columns)
    trial_positions = canonical_keys.get_indexer(observed_keys)
    if np.any(trial_positions < 0):
        unknown = rows.loc[
            trial_positions < 0,
            source_trial_columns,
        ].drop_duplicates()
        raise ValueError(
            "Observed data contain trials outside the canonical design:\n"
            f"{unknown}"
        )

    subject_ids = np.asarray(
        sorted(rows["subj_ident"].astype(str).unique()),
        dtype=str,
    )
    subject_lookup = {
        subject_id: index
        for index, subject_id in enumerate(subject_ids)
    }
    subject_positions = rows["subj_ident"].astype(str).map(
        subject_lookup,
    ).to_numpy()

    valid = (
        ~rows["trial_excl"].fillna(True).to_numpy(dtype=bool)
        & rows["choice"].notna().to_numpy()
    )
    valid_choices = rows.loc[valid, "choice"].to_numpy()
    if not np.isin(valid_choices, [0, 1]).all():
        raise ValueError("Valid observed choices must be coded as 0 or 1.")

    occupied = pd.DataFrame(
        {
            "subject": subject_positions[valid],
            "trial": trial_positions[valid],
        }
    )
    if occupied.duplicated().any():
        raise ValueError("A subject has duplicate observations for one trial.")

    choices = np.zeros(
        (len(subject_ids), len(trials), 1),
        dtype=np.int8,
    )
    trial_mask = np.zeros(
        (len(subject_ids), len(trials)),
        dtype=bool,
    )
    choices[
        subject_positions[valid],
        trial_positions[valid],
        0,
    ] = valid_choices.astype(np.int8)
    trial_mask[
        subject_positions[valid],
        trial_positions[valid],
    ] = True

    if not trial_mask.any(axis=1).all():
        raise ValueError("Every retained subject must have a valid trial.")

    return RealChoiceData(
        trials=trials,
        choices=choices,
        trial_mask=trial_mask,
        subject_ids=subject_ids,
        source_study=source_study,
    )


def load_real_choice_data(
    trial_data_name: str,
    real_data_name: str,
    pongratz_path: str | Path,
) -> RealChoiceData:
    try:
        source_study, trial_loader = REAL_CHOICE_LOADERS[
            (trial_data_name, real_data_name)
        ]
    except KeyError:
        raise ValueError(
            "Unknown real choice dataset: "
            f"{trial_data_name}/{real_data_name}. "
            "Registered observed-choice imports are "
            + ", ".join(
                f"{trial_name}/{real_name}"
                for trial_name, real_name in REAL_CHOICE_LOADERS
            )
            + "."
        ) from None

    rows = _load_pongratz_study(pongratz_path, source_study)
    return _align_observed_choices(
        rows=rows,
        trials=trial_loader(),
        source_study=source_study,
    )
