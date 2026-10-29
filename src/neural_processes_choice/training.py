import copy
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

from .config import ChoiceNPConfig, ChoiceNPTrainingConfig
from .data import (
    ChoiceData,
    ChoiceSubjectDataset,
    TrialStandardizer,
    load_choice_data,
    split_subject_indices,
)
from .model import ChoiceNeuralProcess


def sample_context_mask(
    valid_trial_mask: Tensor,
    minimum_fraction: float,
    maximum_fraction: float,
    generator: torch.Generator,
) -> Tensor:
    if valid_trial_mask.ndim != 2:
        raise ValueError("valid_trial_mask must have shape (B, T).")

    context_mask = torch.zeros_like(valid_trial_mask, dtype=torch.bool)
    for row in range(valid_trial_mask.shape[0]):
        valid_indices = torch.nonzero(
            valid_trial_mask[row],
            as_tuple=False,
        ).flatten()
        if len(valid_indices) == 0:
            raise ValueError("Every subject needs at least one valid trial.")

        minimum_count = math.ceil(
            len(valid_indices) * minimum_fraction
        )
        maximum_count = math.floor(
            len(valid_indices) * maximum_fraction
        )
        if minimum_count > maximum_count:
            raise ValueError(
                "The context-fraction interval contains no integer count "
                f"for {len(valid_indices)} valid trials."
            )
        n_context = torch.randint(
            minimum_count,
            maximum_count + 1,
            (),
            generator=generator,
        ).item()
        permutation = torch.randperm(
            len(valid_indices),
            generator=generator,
        )
        selected = valid_indices[permutation[:n_context]]
        context_mask[row, selected] = True
    return context_mask


def masked_beta_vae_loss(
    logits: Tensor,
    choices: Tensor,
    target_mask: Tensor,
    mean: Tensor,
    log_variance: Tensor,
    beta: float,
) -> dict[str, Tensor]:
    if logits.shape != choices.shape:
        raise ValueError("logits and choices must have the same shape.")
    if target_mask.shape != choices.shape[:2]:
        raise ValueError("target_mask must have shape (B, T).")

    expanded_mask = target_mask.unsqueeze(-1).expand_as(logits)
    if not expanded_mask.any():
        raise ValueError("At least one valid target is required.")
    reconstruction = functional.binary_cross_entropy_with_logits(
        logits[expanded_mask],
        choices[expanded_mask],
    )
    kl_divergence = -0.5 * torch.mean(
        1.0
        + log_variance
        - mean.square()
        - log_variance.exp()
    )
    total = reconstruction + beta * kl_divergence
    return {
        "loss": total,
        "reconstruction": reconstruction,
        "kl": kl_divergence,
    }


@dataclass(frozen=True, slots=True)
class TrainingResult:

    model: ChoiceNeuralProcess
    history: dict[str, list[float]]
    subject_splits: dict[str, Tensor]
    fixed_context_masks: dict[str, Tensor]
    standardizer: TrialStandardizer | None
    test_metrics: dict[str, float]
    best_epoch: int


class ChoiceNPTrainer:

    def __init__(
        self,
        model_config: ChoiceNPConfig | None = None,
        training_config: ChoiceNPTrainingConfig | None = None,
        device: str | torch.device | None = None,
    ):
        self.model_config = model_config or ChoiceNPConfig()
        self.training_config = (
            training_config or ChoiceNPTrainingConfig()
        )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def fit(
        self,
        data: ChoiceData | str | Path,
        subject_splits: dict[str, Tensor] | None = None,
        trial_standardizer: TrialStandardizer | None = None,
        epoch_callback: Callable[
            [int, dict[str, float], dict[str, float]],
            None,
        ]
        | None = None,
    ) -> TrainingResult:
        if not isinstance(data, ChoiceData):
            data = load_choice_data(data)
        data.validate()
        if data.trial_dim != self.model_config.trial_dim:
            raise ValueError(
                "model_config.trial_dim does not match the data."
            )

        self._seed_everything(
            self.training_config.seed,
            self.training_config.deterministic,
        )
        splits = subject_splits
        if splits is None:
            splits = split_subject_indices(
                n_subjects=data.n_subjects,
                train_fraction=self.training_config.train_fraction,
                validation_fraction=(
                    self.training_config.validation_fraction
                ),
                seed=self.training_config.seed,
            )

        standardizer = trial_standardizer
        if self.training_config.normalize_trials:
            if standardizer is not None:
                raise ValueError(
                    "Pass either preprocessed data or normalize_trials=True, "
                    "not both."
                )
            standardizer = TrialStandardizer.fit(
                data,
                splits["train"],
            )
            data = standardizer.transform(data)

        loaders = self._make_loaders(data, splits)
        fixed_context_masks = self._make_fixed_context_masks(
            data,
            splits,
        )
        model = self._build_model()
        optimizer = Adam(
            model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        train_context_generator = torch.Generator().manual_seed(
            self.training_config.seed + 1000
        )

        history = {
            "train_loss": [],
            "train_reconstruction": [],
            "train_kl": [],
            "train_accuracy": [],
            "validation_loss": [],
            "validation_reconstruction": [],
            "validation_kl": [],
            "validation_accuracy": [],
        }
        best_state = copy.deepcopy(model.state_dict())
        best_validation_loss = float("inf")
        best_epoch = -1
        epochs_without_improvement = 0

        for epoch in range(self.training_config.max_epochs):
            train_metrics = self._run_training_epoch(
                model,
                loaders["train"],
                optimizer,
                train_context_generator,
            )
            validation_metrics = self._evaluate(
                model,
                loaders["validation"],
                fixed_context_masks["validation"],
            )
            self._append_history(
                history,
                "train",
                train_metrics,
            )
            self._append_history(
                history,
                "validation",
                validation_metrics,
            )
            if epoch_callback is not None:
                epoch_callback(
                    epoch,
                    train_metrics,
                    validation_metrics,
                )

            validation_loss = validation_metrics["loss"]
            improved = (
                validation_loss
                < best_validation_loss - self.training_config.min_delta
            )
            if improved:
                best_validation_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if (
                    epochs_without_improvement
                    >= self.training_config.patience
                ):
                    break

        model.load_state_dict(best_state)
        test_metrics = self._evaluate(
            model,
            loaders["test"],
            fixed_context_masks["test"],
            evaluate_on_target_only=False,
        )
        return TrainingResult(
            model=model,
            history=history,
            subject_splits=splits,
            fixed_context_masks=fixed_context_masks,
            standardizer=standardizer,
            test_metrics=test_metrics,
            best_epoch=best_epoch,
        )

    def _build_model(self) -> ChoiceNeuralProcess:
        return ChoiceNeuralProcess(self.model_config).to(self.device)

    def _make_loaders(
        self,
        data: ChoiceData,
        splits: dict[str, Tensor],
    ) -> dict[str, DataLoader]:
        loaders = {}
        for offset, split_name in enumerate(
            ("train", "validation", "test")
        ):
            generator = torch.Generator().manual_seed(
                self.training_config.seed + offset
            )
            loaders[split_name] = DataLoader(
                ChoiceSubjectDataset(data, splits[split_name]),
                batch_size=self.training_config.batch_size,
                shuffle=split_name == "train",
                num_workers=self.training_config.num_workers,
                generator=generator,
            )
        return loaders

    def _make_fixed_context_masks(
        self,
        data: ChoiceData,
        splits: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        fixed_masks = {}
        for offset, split_name in enumerate(("validation", "test")):
            split_indices = splits[split_name]
            if split_name == "test":
                sampled = data.trial_mask[split_indices].clone()
            else:
                generator = torch.Generator().manual_seed(
                    self.training_config.seed + 10_000 + offset
                )
                sampled = sample_context_mask(
                    data.trial_mask[split_indices],
                    self.training_config.context_min_fraction,
                    self.training_config.context_max_fraction,
                    generator,
                )
            full_mask = torch.zeros_like(data.trial_mask)
            full_mask[split_indices] = sampled
            fixed_masks[split_name] = full_mask
        return fixed_masks

    def _run_training_epoch(
        self,
        model: ChoiceNeuralProcess,
        loader: DataLoader,
        optimizer: Adam,
        context_generator: torch.Generator,
    ) -> dict[str, float]:
        model.train()
        accumulated = self._empty_metrics()

        for batch in loader:
            valid_mask = batch["trial_mask"]
            context_mask = sample_context_mask(
                valid_mask,
                self.training_config.context_min_fraction,
                self.training_config.context_max_fraction,
                context_generator,
            )
            batch = self._move_batch(batch)
            context_mask = context_mask.to(self.device)

            optimizer.zero_grad(set_to_none=True)
            output = model(
                context_trials=batch["trials"],
                context_choices=batch["choices"],
                context_mask=context_mask,
                target_trials=batch["trials"],
                target_mask=batch["trial_mask"],
                sample_latent=True,
            )
            loss_terms = masked_beta_vae_loss(
                logits=output["logits"],
                choices=batch["choices"],
                target_mask=batch["trial_mask"],
                mean=output["mean"],
                log_variance=output["log_variance"],
                beta=self.training_config.beta,
            )
            loss_terms["loss"].backward()
            if self.training_config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.training_config.gradient_clip_norm,
                )
            optimizer.step()
            self._accumulate(
                accumulated,
                loss_terms,
                output["logits"],
                batch["choices"],
                batch["trial_mask"],
            )

        return self._average_metrics(accumulated, len(loader))

    @torch.no_grad()
    def _evaluate(
        self,
        model: ChoiceNeuralProcess,
        loader: DataLoader,
        fixed_context_mask: Tensor,
        evaluate_on_target_only: bool | None = None,
    ) -> dict[str, float]:
        model.eval()
        accumulated = self._empty_metrics()

        for batch in loader:
            subject_indices = batch["subject_index"]
            context_mask = fixed_context_mask[subject_indices].to(
                self.device
            )
            batch = self._move_batch(batch)
            target_mask = batch["trial_mask"]
            if evaluate_on_target_only is None:
                evaluate_on_target_only = (
                    self.training_config.evaluate_on_target_only
                )
            if evaluate_on_target_only:
                target_mask = target_mask & ~context_mask
            output = model(
                context_trials=batch["trials"],
                context_choices=batch["choices"],
                context_mask=context_mask,
                target_trials=batch["trials"],
                target_mask=target_mask,
                sample_latent=False,
            )
            loss_terms = masked_beta_vae_loss(
                logits=output["logits"],
                choices=batch["choices"],
                target_mask=target_mask,
                mean=output["mean"],
                log_variance=output["log_variance"],
                beta=self.training_config.beta,
            )
            self._accumulate(
                accumulated,
                loss_terms,
                output["logits"],
                batch["choices"],
                target_mask,
            )

        return self._average_metrics(accumulated, len(loader))

    def _move_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {
            key: value.to(self.device)
            for key, value in batch.items()
        }

    @staticmethod
    def _empty_metrics() -> dict[str, float]:
        return {
            "loss": 0.0,
            "reconstruction": 0.0,
            "kl": 0.0,
            "accuracy": 0.0,
        }

    @staticmethod
    def _accumulate(
        accumulated: dict[str, float],
        loss_terms: dict[str, Tensor],
        logits: Tensor,
        choices: Tensor,
        target_mask: Tensor,
    ) -> None:
        for name in ("loss", "reconstruction", "kl"):
            accumulated[name] += loss_terms[name].item()
        predictions = logits > 0.0
        targets = choices > 0.5
        expanded_mask = target_mask.unsqueeze(-1).expand_as(predictions)
        correct = (predictions == targets) & expanded_mask
        accumulated["accuracy"] += (
            correct.sum().float()
            / expanded_mask.sum().clamp_min(1)
        ).item()

    @staticmethod
    def _average_metrics(
        accumulated: dict[str, float],
        n_batches: int,
    ) -> dict[str, float]:
        if n_batches == 0:
            raise ValueError("Cannot evaluate an empty data loader.")
        return {
            name: value / n_batches
            for name, value in accumulated.items()
        }

    @staticmethod
    def _append_history(
        history: dict[str, list[float]],
        prefix: str,
        metrics: dict[str, float],
    ) -> None:
        for name, value in metrics.items():
            history[f"{prefix}_{name}"].append(value)

    @staticmethod
    def _seed_everything(seed: int, deterministic: bool) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(
            deterministic,
            warn_only=True,
        )
        torch.backends.cudnn.deterministic = deterministic
        if deterministic:
            torch.backends.cudnn.benchmark = False


def train_choice_neural_process(
    data: ChoiceData | str | Path,
    model_config: ChoiceNPConfig | None = None,
    training_config: ChoiceNPTrainingConfig | None = None,
    device: str | torch.device | None = None,
    subject_splits: dict[str, Tensor] | None = None,
    trial_standardizer: TrialStandardizer | None = None,
    epoch_callback: Callable[
        [int, dict[str, float], dict[str, float]],
        None,
    ]
    | None = None,
) -> TrainingResult:
    trainer = ChoiceNPTrainer(
        model_config=model_config,
        training_config=training_config,
        device=device,
    )
    return trainer.fit(
        data,
        subject_splits=subject_splits,
        trial_standardizer=trial_standardizer,
        epoch_callback=epoch_callback,
    )


def save_training_result(
    result: TrainingResult,
    path: str | Path,
    model_config: ChoiceNPConfig,
    training_config: ChoiceNPTrainingConfig,
    extra_state: dict | None = None,
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
            "model_state_dict": result.model.state_dict(),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "history": result.history,
            "subject_splits": result.subject_splits,
            "fixed_context_masks": result.fixed_context_masks,
            "standardizer": standardizer_state,
            "test_metrics": result.test_metrics,
            "best_epoch": result.best_epoch,
            "extra_state": extra_state,
        },
        path,
    )
