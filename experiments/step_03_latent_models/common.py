from dataclasses import dataclass, replace
from pathlib import Path
import os
import sys

import numpy as np
import torch
from torch import Tensor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neural_processes_choice import (
    ChoiceData,
    TrialStandardizer,
    load_choice_data,
)


DEFAULT_DATA_NAMES = (
    "r1",
    "m1",
    "m2",
    "m3",
    "m4",
    "m5",
    "m6",
    "m7",
    "m8",
    "m9",
    "m10",
)
LATENT_DIM = 8
SEED = 31
TRAIN_FRACTION = 0.75
VALIDATION_FRACTION = 0.15


@dataclass(frozen=True, slots=True)
class LatentExperimentData:
    choice_data: ChoiceData
    data_names: tuple[str, ...]
    data_labels: Tensor
    model_labels: Tensor
    subject_indices_by_data: dict[str, Tensor]
    parameters_by_model: dict[str, Tensor]
    parameter_names_by_model: dict[str, list[str]]
    metadata: dict
    subject_splits: dict[str, Tensor]
    trial_standardizer: TrialStandardizer | None = None


def order_data_names(data_names) -> tuple[str, ...]:
    names = tuple(data_names)
    if not names:
        raise ValueError("At least one data name is required.")
    if len(names) != len(set(names)):
        raise ValueError("Data names must be unique.")
    for name in names:
        if (
            len(name) < 2
            or name[0] not in {"r", "m"}
            or not name[1:].isdigit()
        ):
            raise ValueError(f"Invalid data name: {name}")
    return tuple(
        sorted(
            names,
            key=lambda name: (
                0 if name.startswith("r") else 1,
                int(name[1:]),
            ),
        )
    )


def choice_data_path(
    trial_data_name: str,
    data_name: str,
) -> Path:
    root = (
        PROJECT_ROOT
        / "output"
        / "choices"
        / trial_data_name
        / data_name
    )
    paths = sorted(root.glob("n*/data.npz"))
    if not paths:
        raise FileNotFoundError(f"Missing choice data under: {root}")
    if len(paths) > 1:
        raise ValueError(
            f"Multiple subject-count variants found under: {root}"
        )
    return paths[0]


def make_stratified_subject_splits(
    data_names: tuple[str, ...],
    subject_indices_by_data: dict[str, Tensor],
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
    seed: int = SEED,
    full_real: bool = False,
) -> dict[str, Tensor]:
    if (
        train_fraction <= 0.0
        or validation_fraction <= 0.0
        or train_fraction + validation_fraction >= 1.0
    ):
        raise ValueError(
            "Split fractions must be positive and leave a positive test set."
        )
    split_parts = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for data_name in data_names:
        subject_indices = subject_indices_by_data[data_name]
        n_subjects = len(subject_indices)
        if full_real and data_name.startswith("r"):
            if n_subjects < 1:
                raise ValueError(f"{data_name} needs at least one subject.")
            split_parts["test"].append(subject_indices)
            continue
        if n_subjects < 3:
            raise ValueError(f"{data_name} needs at least three subjects.")
        n_train = max(1, round(n_subjects * train_fraction))
        n_validation = max(
            1,
            round(n_subjects * validation_fraction),
        )
        if n_train + n_validation >= n_subjects:
            n_validation = 1
            n_train = n_subjects - 2
        source_offset = 0 if data_name.startswith("r") else 10_000
        data_number = int(data_name[1:])
        generator = torch.Generator().manual_seed(
            seed + source_offset + data_number
        )
        local_order = torch.randperm(
            n_subjects,
            generator=generator,
        )
        shuffled = subject_indices[local_order]
        split_parts["train"].append(shuffled[:n_train])
        split_parts["validation"].append(
            shuffled[n_train : n_train + n_validation]
        )
        split_parts["test"].append(
            shuffled[n_train + n_validation :]
        )
    return {
        name: (
            torch.cat(parts)
            if parts
            else torch.empty(0, dtype=torch.long)
        )
        for name, parts in split_parts.items()
    }


def load_experiment_data(
    trial_data_name: str,
    data_names=DEFAULT_DATA_NAMES,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
    seed: int = SEED,
    full_real: bool = False,
) -> LatentExperimentData:
    data_names = order_data_names(data_names)
    model_names = tuple(
        name for name in data_names if name.startswith("m")
    )
    model_index_by_name = {
        name: index for index, name in enumerate(model_names)
    }
    datasets = []
    data_labels = []
    model_labels = []
    source_subject_indices = []
    subject_indices_by_data = {}
    parameters_by_model = {}
    parameter_names_by_model = {}
    offset = 0

    for data_index, data_name in enumerate(data_names):
        path = choice_data_path(trial_data_name, data_name)
        dataset = load_choice_data(path)
        if dataset.trials.ndim not in (2, 3):
            raise ValueError("trials must have shape (T, D) or (S, T, D).")
        if datasets:
            reference_trials = datasets[0].trials
            if dataset.trials.ndim != reference_trials.ndim:
                raise ValueError(
                    "Cannot mix shared and subject-specific trial layouts."
                )
            if not torch.allclose(dataset.trials, reference_trials):
                raise ValueError("All datasets must use the same trials.")
        datasets.append(dataset)
        data_labels.append(
            torch.full(
                (dataset.n_subjects,),
                data_index,
                dtype=torch.long,
            )
        )
        model_labels.append(
            torch.full(
                (dataset.n_subjects,),
                model_index_by_name.get(data_name, -1),
                dtype=torch.long,
            )
        )
        source_subject_indices.append(
            torch.arange(dataset.n_subjects)
        )
        subject_indices = torch.arange(
            offset,
            offset + dataset.n_subjects,
        )
        subject_indices_by_data[data_name] = subject_indices
        offset += dataset.n_subjects
        if data_name.startswith("m"):
            parameters_by_model[data_name] = torch.as_tensor(
                dataset.metadata["parameter_values"]
            )
            parameter_names_by_model[data_name] = np.asarray(
                dataset.metadata["parameter_names"]
            ).tolist()

    reference_trials = datasets[0].trials
    if reference_trials.ndim == 2:
        combined_trials = reference_trials
    else:
        combined_trials = torch.cat(
            [item.trials for item in datasets],
            dim=0,
        )
    choice_data = ChoiceData(
        trials=combined_trials,
        choices=torch.cat([item.choices for item in datasets]),
        trial_mask=torch.cat([item.trial_mask for item in datasets]),
        metadata={},
    )
    data_labels = torch.cat(data_labels)
    model_labels = torch.cat(model_labels)
    metadata = {
        "trial_data_name": trial_data_name,
        "trial_columns": np.asarray(
            datasets[0].metadata["trial_columns"]
        ).tolist(),
        "data_names": list(data_names),
        "model_names": list(model_names),
        "subject_data_index": data_labels,
        "subject_model_index": model_labels,
        "source_subject_index": torch.cat(source_subject_indices),
        "parameter_names_by_model": parameter_names_by_model,
        "split_seed": seed,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": 1.0 - train_fraction - validation_fraction,
        "full_real": full_real,
    }
    return LatentExperimentData(
        choice_data=choice_data,
        data_names=data_names,
        data_labels=data_labels,
        model_labels=model_labels,
        subject_indices_by_data=subject_indices_by_data,
        parameters_by_model=parameters_by_model,
        parameter_names_by_model=parameter_names_by_model,
        metadata=metadata,
        subject_splits=make_stratified_subject_splits(
            data_names,
            subject_indices_by_data,
            train_fraction,
            validation_fraction,
            seed,
            full_real=full_real,
        ),
    )


def prepare_experiment_data(
    trial_data_name: str,
    data_names=DEFAULT_DATA_NAMES,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
    seed: int = SEED,
    full_real: bool = False,
) -> LatentExperimentData:
    experiment = load_experiment_data(
        trial_data_name,
        data_names,
        train_fraction,
        validation_fraction,
        seed,
        full_real=full_real,
    )
    if len(experiment.subject_splits["train"]) == 0:
        raise ValueError(
            "The train split is empty. FULL_REAL still needs simulated "
            "subjects to fit the trial standardizer."
        )
    standardizer = TrialStandardizer.fit(
        experiment.choice_data,
        experiment.subject_splits["train"],
    )
    return replace(
        experiment,
        choice_data=standardizer.transform(experiment.choice_data),
        trial_standardizer=standardizer,
    )


def preprocessed_filename(full_real: bool = False) -> str:
    return "dataFL.pt" if full_real else "data.pt"


def preprocessed_data_path(
    trial_data_name: str,
    data_names=DEFAULT_DATA_NAMES,
    seed: int = SEED,
    full_real: bool = False,
) -> Path:
    data_names = order_data_names(data_names)
    return (
        PROJECT_ROOT
        / "output"
        / "latent_models"
        / "_preprocessed"
        / trial_data_name
        / compact_data_group(data_names)
        / f"n{subject_count_group(trial_data_name, data_names)}"
        / f"s{seed}"
        / preprocessed_filename(full_real)
    )


def save_preprocessed_experiment(
    experiment: LatentExperimentData,
) -> Path:
    trial_data_name = experiment.metadata["trial_data_name"]
    full_real = bool(experiment.metadata.get("full_real", False))
    path = preprocessed_data_path(
        trial_data_name,
        experiment.data_names,
        experiment.metadata["split_seed"],
        full_real=full_real,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    standardizer = experiment.trial_standardizer
    if standardizer is None:
        raise ValueError("Prepared data must include a trial standardizer.")
    torch.save(
        {
            "version": 1,
            "trial_data_name": trial_data_name,
            "data_names": list(experiment.data_names),
            "seed": experiment.metadata["split_seed"],
            "train_fraction": experiment.metadata["train_fraction"],
            "validation_fraction": (
                experiment.metadata["validation_fraction"]
            ),
            "trials": experiment.choice_data.trials,
            "choices": experiment.choice_data.choices,
            "trial_mask": experiment.choice_data.trial_mask,
            "data_labels": experiment.data_labels,
            "model_labels": experiment.model_labels,
            "subject_indices_by_data": experiment.subject_indices_by_data,
            "parameters_by_model": experiment.parameters_by_model,
            "parameter_names_by_model": (
                experiment.parameter_names_by_model
            ),
            "metadata": experiment.metadata,
            "subject_splits": experiment.subject_splits,
            "trial_standardizer": {
                "mean": standardizer.mean,
                "scale": standardizer.scale,
            },
        },
        path,
    )
    return path


def _load_preprocessed_state(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _experiment_from_state(state: dict) -> LatentExperimentData:
    standardizer_state = state["trial_standardizer"]
    return LatentExperimentData(
        choice_data=ChoiceData(
            trials=state["trials"],
            choices=state["choices"],
            trial_mask=state["trial_mask"],
            metadata={},
        ),
        data_names=tuple(state["data_names"]),
        data_labels=state["data_labels"],
        model_labels=state["model_labels"],
        subject_indices_by_data=state["subject_indices_by_data"],
        parameters_by_model=state["parameters_by_model"],
        parameter_names_by_model=state["parameter_names_by_model"],
        metadata=state["metadata"],
        subject_splits=state["subject_splits"],
        trial_standardizer=TrialStandardizer(
            mean=standardizer_state["mean"],
            scale=standardizer_state["scale"],
        ),
    )


def _select_data_names(
    experiment: LatentExperimentData,
    data_names: tuple[str, ...],
) -> LatentExperimentData:
    if data_names == experiment.data_names:
        return experiment
    selected_old_indices = torch.cat(
        [
            experiment.subject_indices_by_data[name]
            for name in data_names
        ]
    )
    old_to_new = torch.full(
        (experiment.choice_data.n_subjects,),
        -1,
        dtype=torch.long,
    )
    old_to_new[selected_old_indices] = torch.arange(
        len(selected_old_indices)
    )
    subject_indices_by_data = {}
    model_names = tuple(
        name
        for name in data_names
        if name in experiment.parameters_by_model
    )
    model_index_by_name = {
        name: index for index, name in enumerate(model_names)
    }
    data_labels = []
    model_labels = []
    offset = 0
    for data_index, data_name in enumerate(data_names):
        n_subjects = len(
            experiment.subject_indices_by_data[data_name]
        )
        subject_indices_by_data[data_name] = torch.arange(
            offset,
            offset + n_subjects,
        )
        data_labels.append(
            torch.full((n_subjects,), data_index, dtype=torch.long)
        )
        model_labels.append(
            torch.full(
                (n_subjects,),
                model_index_by_name.get(data_name, -1),
                dtype=torch.long,
            )
        )
        offset += n_subjects
    subject_splits = {}
    for split_name, old_indices in experiment.subject_splits.items():
        selected = old_indices[
            torch.isin(old_indices, selected_old_indices)
        ]
        subject_splits[split_name] = old_to_new[selected]
    data_labels = torch.cat(data_labels)
    model_labels = torch.cat(model_labels)
    metadata = dict(experiment.metadata)
    metadata.update(
        {
            "data_names": list(data_names),
            "model_names": list(model_names),
            "subject_data_index": data_labels,
            "subject_model_index": model_labels,
            "source_subject_index": experiment.metadata[
                "source_subject_index"
            ][selected_old_indices],
        }
    )
    source_trials = experiment.choice_data.trials
    if source_trials.ndim == 2:
        selected_trials = source_trials
    else:
        selected_trials = source_trials[selected_old_indices]
    return LatentExperimentData(
        choice_data=ChoiceData(
            trials=selected_trials,
            choices=experiment.choice_data.choices[selected_old_indices],
            trial_mask=(
                experiment.choice_data.trial_mask[selected_old_indices]
            ),
            metadata={},
        ),
        data_names=data_names,
        data_labels=data_labels,
        model_labels=model_labels,
        subject_indices_by_data=subject_indices_by_data,
        parameters_by_model={
            name: experiment.parameters_by_model[name]
            for name in data_names
            if name.startswith("m")
        },
        parameter_names_by_model={
            name: experiment.parameter_names_by_model[name]
            for name in data_names
            if name.startswith("m")
        },
        metadata=metadata,
        subject_splits=subject_splits,
        trial_standardizer=experiment.trial_standardizer,
    )


def load_preprocessed_experiment(
    trial_data_name: str,
    data_names=DEFAULT_DATA_NAMES,
    seed: int = SEED,
    full_real: bool = False,
) -> LatentExperimentData:
    data_names = order_data_names(data_names)
    exact_path = preprocessed_data_path(
        trial_data_name,
        data_names,
        seed,
        full_real=full_real,
    )
    filename = preprocessed_filename(full_real)
    if exact_path.exists():
        experiment = _experiment_from_state(
            _load_preprocessed_state(exact_path)
        )
    else:
        root = (
            PROJECT_ROOT
            / "output"
            / "latent_models"
            / "_preprocessed"
            / trial_data_name
        )
        candidates = []
        requested = set(data_names)
        for path in root.glob(f"*/n*/s{seed}/{filename}"):
            state = _load_preprocessed_state(path)
            available = set(state["data_names"])
            if requested <= available:
                candidates.append((len(available), path, state))
        if not candidates:
            raise FileNotFoundError(
                "Missing preprocessed data for "
                f"{trial_data_name}, {data_names}"
                f"{', FULL_REAL' if full_real else ''}. Run step_00 first."
            )
        candidates.sort(key=lambda item: (item[0], str(item[1])))
        if (
            len(candidates) > 1
            and candidates[0][0] == candidates[1][0]
        ):
            raise ValueError(
                "Multiple equally specific preprocessed artifacts contain "
                f"{data_names}."
            )
        experiment = _experiment_from_state(candidates[0][2])
        experiment = _select_data_names(experiment, data_names)
    return experiment


def unstandardize_experiment(
    experiment: LatentExperimentData,
) -> LatentExperimentData:
    standardizer = experiment.trial_standardizer
    if standardizer is None:
        return experiment
    raw_trials = (
        experiment.choice_data.trials * standardizer.scale
        + standardizer.mean
    )
    return replace(
        experiment,
        choice_data=replace(
            experiment.choice_data,
            trials=raw_trials,
        ),
        trial_standardizer=None,
    )


def select_model_labeled_data(
    experiment: LatentExperimentData,
) -> LatentExperimentData:
    model_names = tuple(experiment.metadata["model_names"])
    if not model_names:
        raise ValueError("The experiment contains no model-labeled data.")
    return _select_data_names(experiment, model_names)


def compact_data_group(data_names=DEFAULT_DATA_NAMES) -> str:
    return "_".join(order_data_names(data_names))


def as_long_path(path: Path) -> Path:
    path = Path(path)
    if os.name != "nt":
        return path
    resolved = os.path.abspath(path)
    if resolved.startswith("\\\\?\\"):
        return Path(resolved)
    if resolved.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + resolved.removeprefix("\\\\"))
    return Path("\\\\?\\" + resolved)


def subject_count_group(
    trial_data_name: str,
    data_names=DEFAULT_DATA_NAMES,
) -> str:
    counts = [
        choice_data_path(
            trial_data_name,
            data_name,
        ).parent.name.removeprefix("n")
        for data_name in order_data_names(data_names)
    ]
    return "_".join(counts)


def output_root(
    model_type: str,
    trial_data_name: str,
    data_names=DEFAULT_DATA_NAMES,
) -> Path:
    data_names = order_data_names(data_names)
    return (
        PROJECT_ROOT
        / "output"
        / "latent_models"
        / model_type
        / trial_data_name
        / compact_data_group(data_names)
        / (
            f"n{subject_count_group(trial_data_name, data_names)}"
            f"_z{LATENT_DIM}"
        )
        / f"s{SEED}"
    )


def load_checkpoint(path: str | Path) -> dict:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        return torch.load(path, map_location="cpu")
