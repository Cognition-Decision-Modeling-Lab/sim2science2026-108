import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from neural_processes_choice.data import (
    ChoiceData,
    ChoiceSubjectDataset,
    TrialStandardizer,
)

from .config import (
    ModelClassifierConfig,
    ModelClassifierTrainingConfig,
)
from .model import ChoiceModelClassifier


class ClassifierDataset(Dataset):
    def __init__(
        self,
        data: ChoiceData,
        subject_indices: Tensor,
        labels: Tensor,
    ):
        self.base = ChoiceSubjectDataset(data, subject_indices)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = self.base[index]
        item["model_label"] = self.labels[item["subject_index"]]
        return item


@dataclass(frozen=True, slots=True)
class ClassifierTrainingResult:
    model: ChoiceModelClassifier
    history: dict[str, list[float]]
    subject_splits: dict[str, Tensor]
    standardizer: TrialStandardizer | None
    test_metrics: dict[str, float]
    best_epoch: int


class ModelClassifierTrainer:
    def __init__(
        self,
        model_config: ModelClassifierConfig,
        training_config: ModelClassifierTrainingConfig,
        device: str | torch.device | None = None,
    ):
        self.model_config = model_config
        self.training_config = training_config
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def fit(
        self,
        data: ChoiceData,
        labels: Tensor,
        subject_splits: dict[str, Tensor],
        trial_standardizer: TrialStandardizer | None = None,
        epoch_callback: Callable[
            [int, dict[str, float], dict[str, float]],
            None,
        ]
        | None = None,
    ) -> ClassifierTrainingResult:
        self._seed()
        standardizer = trial_standardizer
        if self.training_config.normalize_trials:
            if standardizer is not None:
                raise ValueError(
                    "Pass either preprocessed data or normalize_trials=True, "
                    "not both."
                )
            standardizer = TrialStandardizer.fit(
                data,
                subject_splits["train"],
            )
            data = standardizer.transform(data)
        loaders = self._make_loaders(data, labels, subject_splits)
        model = ChoiceModelClassifier(self.model_config).to(self.device)
        optimizer = Adam(
            model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "validation_loss": [],
            "validation_accuracy": [],
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
            for name, value in train_metrics.items():
                history[f"train_{name}"].append(value)
            for name, value in validation_metrics.items():
                history[f"validation_{name}"].append(value)
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
        return ClassifierTrainingResult(
            model=model,
            history=history,
            subject_splits=subject_splits,
            standardizer=standardizer,
            test_metrics=test_metrics,
            best_epoch=best_epoch,
        )

    def _make_loaders(
        self,
        data: ChoiceData,
        labels: Tensor,
        splits: dict[str, Tensor],
    ) -> dict[str, DataLoader]:
        loaders = {}
        for offset, name in enumerate(("train", "validation", "test")):
            loaders[name] = DataLoader(
                ClassifierDataset(data, splits[name], labels),
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
        model: ChoiceModelClassifier,
        loader: DataLoader,
        optimizer: Adam | None = None,
    ) -> dict[str, float]:
        training = optimizer is not None
        model.train(training)
        total_loss = 0.0
        total_correct = 0
        total_count = 0
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
                loss = functional.cross_entropy(
                    output["model_logits"],
                    batch["model_label"],
                )
                if training:
                    loss.backward()
                    if self.training_config.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            self.training_config.gradient_clip_norm,
                        )
                    optimizer.step()
                total_loss += loss.item()
                total_correct += (
                    output["model_logits"].argmax(dim=1)
                    == batch["model_label"]
                ).sum().item()
                total_count += len(batch["model_label"])
        return {
            "loss": total_loss / len(loader),
            "accuracy": total_correct / total_count,
        }

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


def save_classifier_training_result(
    result: ClassifierTrainingResult,
    path: str | Path,
    model_config: ModelClassifierConfig,
    training_config: ModelClassifierTrainingConfig,
    extra_state: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    standardizer_state = None
    if result.standardizer is not None:
        standardizer_state = {
            "mean": result.standardizer.mean,
            "scale": result.standardizer.scale,
        }
    torch.save(
        {
            "model_type": "model_classifier",
            "model_state_dict": result.model.state_dict(),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "subject_splits": result.subject_splits,
            "standardizer": standardizer_state,
            "history": result.history,
            "test_metrics": result.test_metrics,
            "best_epoch": result.best_epoch,
            "extra_state": extra_state,
        },
        path,
    )
