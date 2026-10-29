from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class ChoiceData:

    trials: Tensor
    choices: Tensor
    trial_mask: Tensor
    metadata: dict

    @property
    def n_subjects(self) -> int:
        return self.choices.shape[0]

    @property
    def n_trials(self) -> int:
        return self.choices.shape[1]

    @property
    def trial_dim(self) -> int:
        return self.trials.shape[-1]

    def validate(self) -> None:
        if self.trials.ndim not in (2, 3):
            raise ValueError("trials must have shape (T, D) or (S, T, D).")
        if self.choices.ndim != 3:
            raise ValueError("choices must have shape (S, T, C).")
        if self.trial_mask.shape != self.choices.shape[:2]:
            raise ValueError("trial_mask must have shape (S, T).")
        if self.trials.shape[-2] != self.n_trials:
            raise ValueError("trials and choices must have the same T.")
        if self.trials.ndim == 3 and self.trials.shape[0] != self.n_subjects:
            raise ValueError("Subject-specific trials must have shape (S,T,D).")
        if not torch.all(
            (self.choices[self.trial_mask] == 0)
            | (self.choices[self.trial_mask] == 1)
        ):
            raise ValueError("Valid choices must be binary.")
        if not torch.all(self.trial_mask.any(dim=1)):
            raise ValueError("Every subject must have at least one valid trial.")


def load_choice_data(path: str | Path) -> ChoiceData:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        trials = torch.as_tensor(
            archive["trials"],
            dtype=torch.float32,
        )
        choices = torch.as_tensor(
            archive["choices"],
            dtype=torch.float32,
        )
        if "trial_mask" in archive.files:
            trial_mask = torch.as_tensor(
                archive["trial_mask"],
                dtype=torch.bool,
            )
        else:
            trial_mask = torch.ones(
                choices.shape[:2],
                dtype=torch.bool,
            )

        excluded = {"trials", "choices", "trial_mask"}
        metadata = {
            key: archive[key].copy()
            for key in archive.files
            if key not in excluded
        }

    data = ChoiceData(
        trials=trials,
        choices=choices,
        trial_mask=trial_mask,
        metadata=metadata,
    )
    data.validate()
    return data


def split_subject_indices(
    n_subjects: int,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> dict[str, Tensor]:
    if n_subjects < 3:
        raise ValueError("At least three subjects are required.")

    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(n_subjects, generator=generator)
    n_train = max(1, round(n_subjects * train_fraction))
    n_validation = max(1, round(n_subjects * validation_fraction))
    if n_train + n_validation >= n_subjects:
        n_validation = 1
        n_train = n_subjects - 2

    return {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train : n_train + n_validation],
        "test": shuffled[n_train + n_validation :],
    }


class ChoiceSubjectDataset(Dataset):

    def __init__(self, data: ChoiceData, subject_indices: Tensor):
        self.data = data
        self.subject_indices = torch.as_tensor(
            subject_indices,
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.subject_indices)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        subject_index = self.subject_indices[index]
        if self.data.trials.ndim == 2:
            trials = self.data.trials
        else:
            trials = self.data.trials[subject_index]
        return {
            "subject_index": subject_index,
            "trials": trials,
            "choices": self.data.choices[subject_index],
            "trial_mask": self.data.trial_mask[subject_index],
        }


@dataclass(frozen=True, slots=True)
class TrialStandardizer:

    mean: Tensor
    scale: Tensor

    @classmethod
    def fit(
        cls,
        data: ChoiceData,
        subject_indices: Tensor,
        epsilon: float = 1e-6,
    ) -> "TrialStandardizer":
        indices = torch.as_tensor(subject_indices, dtype=torch.long)
        masks = data.trial_mask[indices]

        if data.trials.ndim == 2:
            weights = masks.sum(dim=0).to(data.trials.dtype)
            denominator = weights.sum().clamp_min(1.0)
            mean = (data.trials * weights[:, None]).sum(dim=0) / denominator
            centered = data.trials - mean
            variance = (
                centered.square() * weights[:, None]
            ).sum(dim=0) / denominator
        else:
            trials = data.trials[indices]
            valid_trials = trials[masks]
            mean = valid_trials.mean(dim=0)
            variance = valid_trials.var(dim=0, unbiased=False)

        scale = variance.sqrt().clamp_min(epsilon)
        return cls(mean=mean, scale=scale)

    def transform(self, data: ChoiceData) -> ChoiceData:
        trials = (data.trials - self.mean) / self.scale
        return replace(data, trials=trials)
