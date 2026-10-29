import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from neural_processes_choice.data import (
    ChoiceData,
    ChoiceSubjectDataset,
    TrialStandardizer,
)

from .config import (
    ParameterDecoderConfig,
    ParameterDecoderTrainingConfig,
)
from .model import ChoiceParameterDecoder
from .targets import ParameterStandardizer


class ParameterDataset(Dataset):
    def __init__(
        self,
        data: ChoiceData,
        subject_indices: Tensor,
        targets: Tensor,
        model_labels: Tensor,
    ):
        self.base = ChoiceSubjectDataset(data, subject_indices)
        self.targets = targets
        self.model_labels = model_labels

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = self.base[index]
        subject_index = item["subject_index"]
        item["parameter_target"] = self.targets[subject_index]
        item["model_label"] = self.model_labels[subject_index]
        return item


def masked_parameter_mse(
    predictions: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> Tensor:
    if not mask.any():
        raise ValueError("At least one parameter target is required.")
    return (predictions[mask] - targets[mask]).square().mean()


@dataclass(frozen=True, slots=True)
class ParameterTrainingResult:
    model: ChoiceParameterDecoder
    history: dict[str, list[float]]
    subject_splits: dict[str, Tensor]
    trial_standardizer: TrialStandardizer | None
    parameter_standardizers: dict[str, ParameterStandardizer]
    test_metrics: dict[str, float]
    best_epoch: int


class ParameterDecoderTrainer:
    def __init__(
        self,
        model_config: ParameterDecoderConfig,
        training_config: ParameterDecoderTrainingConfig,
        device: str | torch.device | None = None,
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.model_names = tuple(
            name for name, _ in model_config.parameter_dims
        )
        self.parameter_dims = dict(model_config.parameter_dims)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def fit(
        self,
        data: ChoiceData,
        model_labels: Tensor,
        parameters_by_model: dict[str, Tensor],
        parameter_names_by_model: dict[str, list[str]],
        subject_indices_by_model: dict[str, Tensor],
        subject_splits: dict[str, Tensor],
        trial_standardizer: TrialStandardizer | None = None,
        epoch_callback: Callable[
            [int, dict[str, float], dict[str, float]],
            None,
        ]
        | None = None,
    ) -> ParameterTrainingResult:
        self._seed()
        if self.training_config.normalize_trials:
            if trial_standardizer is not None:
                raise ValueError(
                    "Pass either preprocessed data or normalize_trials=True, "
                    "not both."
                )
            trial_standardizer = TrialStandardizer.fit(
                data,
                subject_splits["train"],
            )
            data = trial_standardizer.transform(data)
        targets, parameter_standardizers = self._prepare_targets(
            n_subjects=data.n_subjects,
            parameters_by_model=parameters_by_model,
            parameter_names_by_model=parameter_names_by_model,
            subject_indices_by_model=subject_indices_by_model,
            train_indices=subject_splits["train"],
        )
        loaders = self._make_loaders(
            data,
            targets,
            model_labels,
            subject_splits,
        )
        model = ChoiceParameterDecoder(self.model_config).to(self.device)
        optimizer = Adam(
            model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        history = {
            "train_loss": [],
            "validation_loss": [],
        }
        best_state = copy.deepcopy(model.state_dict())
        best_loss = float("inf")
        best_epoch = -1
        stale_epochs = 0

        for epoch in range(self.training_config.max_epochs):
            train_metrics = self._run_epoch(
                model,
                loaders["train"],
                optimizer,
            )
            validation_metrics = self._run_epoch(
                model,
                loaders["validation"],
            )
            history["train_loss"].append(train_metrics["loss"])
            history["validation_loss"].append(
                validation_metrics["loss"]
            )
            if epoch_callback is not None:
                epoch_callback(epoch, train_metrics, validation_metrics)
            if (
                validation_metrics["loss"]
                < best_loss - self.training_config.min_delta
            ):
                best_loss = validation_metrics["loss"]
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.training_config.patience:
                    break

        model.load_state_dict(best_state)
        test_metrics = self._run_epoch(model, loaders["test"])
        return ParameterTrainingResult(
            model=model,
            history=history,
            subject_splits=subject_splits,
            trial_standardizer=trial_standardizer,
            parameter_standardizers=parameter_standardizers,
            test_metrics=test_metrics,
            best_epoch=best_epoch,
        )

    def _prepare_targets(
        self,
        n_subjects: int,
        parameters_by_model: dict[str, Tensor],
        parameter_names_by_model: dict[str, list[str]],
        subject_indices_by_model: dict[str, Tensor],
        train_indices: Tensor,
    ) -> tuple[Tensor, dict[str, ParameterStandardizer]]:
        max_parameters = max(self.parameter_dims.values())
        targets = torch.zeros((n_subjects, max_parameters))
        standardizers = {}
        for model_name in self.model_names:
            values = parameters_by_model[model_name].float()
            names = parameter_names_by_model[model_name]
            subject_indices = subject_indices_by_model[model_name]
            expected_dim = self.parameter_dims[model_name]
            if values.shape != (len(subject_indices), expected_dim):
                raise ValueError(
                    f"Parameter values do not match head {model_name}."
                )
            local_train_indices = torch.nonzero(
                torch.isin(subject_indices, train_indices),
                as_tuple=False,
            ).flatten()
            standardizer = ParameterStandardizer.fit(
                values,
                model_name,
                names,
                local_train_indices,
            )
            targets[
                subject_indices,
                :expected_dim,
            ] = standardizer.transform(values)
            standardizers[model_name] = standardizer
        return targets, standardizers

    def _make_loaders(
        self,
        data: ChoiceData,
        targets: Tensor,
        model_labels: Tensor,
        splits: dict[str, Tensor],
    ) -> dict[str, DataLoader]:
        loaders = {}
        for offset, name in enumerate(("train", "validation", "test")):
            loaders[name] = DataLoader(
                ParameterDataset(
                    data,
                    splits[name],
                    targets,
                    model_labels,
                ),
                batch_size=self.training_config.batch_size,
                shuffle=name == "train",
                num_workers=self.training_config.num_workers,
                generator=torch.Generator().manual_seed(
                    self.training_config.seed + offset
                ),
            )
        return loaders

    def _run_epoch(
        self,
        model: ChoiceParameterDecoder,
        loader: DataLoader,
        optimizer: Adam | None = None,
    ) -> dict[str, float]:
        training = optimizer is not None
        model.train(training)
        total_squared_error = 0.0
        total_parameter_count = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                batch = {
                    key: value.to(self.device)
                    for key, value in batch.items()
                }
                if training:
                    optimizer.zero_grad(set_to_none=True)
                output = model(
                    batch["trials"],
                    batch["choices"],
                    batch["trial_mask"],
                )
                squared_error = torch.zeros((), device=self.device)
                parameter_count = 0
                for model_index, model_name in enumerate(self.model_names):
                    selected = batch["model_label"] == model_index
                    if not selected.any():
                        continue
                    n_parameters = self.parameter_dims[model_name]
                    predictions = output["parameter_predictions"][
                        model_name
                    ][selected]
                    targets = batch["parameter_target"][
                        selected,
                        :n_parameters,
                    ]
                    squared_error = squared_error + (
                        predictions - targets
                    ).square().sum()
                    parameter_count += targets.numel()
                if parameter_count == 0:
                    raise ValueError("Batch has no parameter targets.")
                loss = squared_error / parameter_count
                if training:
                    loss.backward()
                    if self.training_config.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            self.training_config.gradient_clip_norm,
                        )
                    optimizer.step()
                total_squared_error += squared_error.item()
                total_parameter_count += parameter_count
        return {"loss": total_squared_error / total_parameter_count}

    def _seed(self) -> None:
        seed = self.training_config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(
            self.training_config.deterministic,
            warn_only=True,
        )


def save_parameter_training_result(
    result: ParameterTrainingResult,
    path: str | Path,
    model_config: ParameterDecoderConfig,
    training_config: ParameterDecoderTrainingConfig,
    extra_state: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trial_state = None
    if result.trial_standardizer is not None:
        trial_state = {
            "mean": result.trial_standardizer.mean,
            "scale": result.trial_standardizer.scale,
        }
    torch.save(
        {
            "model_type": "parameter_decoder",
            "model_state_dict": result.model.state_dict(),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "subject_splits": result.subject_splits,
            "standardizer": trial_state,
            "parameter_standardizers": {
                model_name: {
                    "parameter_names": standardizer.parameter_names,
                    "transforms": standardizer.transforms,
                    "mean": standardizer.mean,
                    "scale": standardizer.scale,
                }
                for model_name, standardizer
                in result.parameter_standardizers.items()
            },
            "history": result.history,
            "test_metrics": result.test_metrics,
            "best_epoch": result.best_epoch,
            "extra_state": extra_state,
        },
        path,
    )
