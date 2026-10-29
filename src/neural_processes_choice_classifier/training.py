from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from neural_processes_choice import (
    ChoiceData,
    ChoiceNPTrainer,
    ChoiceNPTrainingConfig,
    ChoiceNeuralProcess,
    ChoiceSubjectDataset,
    TrainingResult,
    TrialStandardizer,
    masked_beta_vae_loss,
    sample_context_mask,
)

from .config import ChoiceNPClassifierConfig
from .model import ChoiceNeuralProcessClassifier


class LabeledChoiceSubjectDataset(Dataset):
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


class ClassifierChoiceNPTrainer(ChoiceNPTrainer):

    def __init__(
        self,
        model_config: ChoiceNPClassifierConfig,
        training_config: ChoiceNPTrainingConfig,
        labels: Tensor,
        classification_weight: float,
        device=None,
    ):
        if classification_weight < 0.0:
            raise ValueError("classification_weight must be non-negative.")
        super().__init__(model_config, training_config, device)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.classification_weight = classification_weight
        if int(self.labels.max()) >= self.model_config.n_models:
            raise ValueError(
                "labels contain a class index outside n_models."
            )
        if int(self.labels.min()) < 0:
            raise ValueError("All training labels must be non-negative.")

    def _build_model(self) -> ChoiceNeuralProcess:
        return ChoiceNeuralProcessClassifier(self.model_config).to(
            self.device
        )

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
                LabeledChoiceSubjectDataset(
                    data,
                    splits[split_name],
                    self.labels,
                ),
                batch_size=self.training_config.batch_size,
                shuffle=split_name == "train",
                num_workers=self.training_config.num_workers,
                generator=generator,
            )
        return loaders

    def _run_training_epoch(
        self,
        model: ChoiceNeuralProcessClassifier,
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
            losses = masked_beta_vae_loss(
                logits=output["logits"],
                choices=batch["choices"],
                target_mask=batch["trial_mask"],
                mean=output["mean"],
                log_variance=output["log_variance"],
                beta=self.training_config.beta,
            )
            classification = functional.cross_entropy(
                output["model_logits"],
                batch["model_label"],
            )
            losses["classification"] = classification
            losses["loss"] = (
                losses["loss"]
                + self.classification_weight * classification
            )
            losses["loss"].backward()
            if self.training_config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.training_config.gradient_clip_norm,
                )
            optimizer.step()
            self._accumulate(
                accumulated,
                losses,
                output["logits"],
                batch["choices"],
                batch["trial_mask"],
            )
            self._accumulate_classification(
                accumulated,
                losses,
                output["model_logits"],
                batch["model_label"],
            )

        return self._average_metrics(accumulated, len(loader))

    @torch.no_grad()
    def _evaluate(
        self,
        model: ChoiceNeuralProcessClassifier,
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
            losses = masked_beta_vae_loss(
                logits=output["logits"],
                choices=batch["choices"],
                target_mask=target_mask,
                mean=output["mean"],
                log_variance=output["log_variance"],
                beta=self.training_config.beta,
            )
            classification = functional.cross_entropy(
                output["model_logits"],
                batch["model_label"],
            )
            losses["classification"] = classification
            losses["loss"] = (
                losses["loss"]
                + self.classification_weight * classification
            )
            self._accumulate(
                accumulated,
                losses,
                output["logits"],
                batch["choices"],
                target_mask,
            )
            self._accumulate_classification(
                accumulated,
                losses,
                output["model_logits"],
                batch["model_label"],
            )

        return self._average_metrics(accumulated, len(loader))

    @staticmethod
    def _empty_metrics() -> dict[str, float]:
        metrics = ChoiceNPTrainer._empty_metrics()
        metrics["classification"] = 0.0
        metrics["model_accuracy"] = 0.0
        return metrics

    @staticmethod
    def _accumulate_classification(
        accumulated: dict[str, float],
        losses: dict[str, Tensor],
        model_logits: Tensor,
        model_label: Tensor,
    ) -> None:
        accumulated["classification"] += losses["classification"].item()
        accumulated["model_accuracy"] += (
            (model_logits.argmax(dim=1) == model_label).float().mean().item()
        )

    @staticmethod
    def _append_history(history, prefix, metrics):
        for name, value in metrics.items():
            history.setdefault(f"{prefix}_{name}", []).append(value)


def train_choice_neural_process_classifier(
    data: ChoiceData | str | Path,
    labels: Tensor,
    model_config: ChoiceNPClassifierConfig,
    training_config: ChoiceNPTrainingConfig,
    classification_weight: float,
    device: str | torch.device | None = None,
    subject_splits: dict[str, torch.Tensor] | None = None,
    trial_standardizer: TrialStandardizer | None = None,
    epoch_callback: Callable | None = None,
) -> TrainingResult:
    trainer = ClassifierChoiceNPTrainer(
        model_config,
        training_config,
        labels,
        classification_weight,
        device,
    )
    return trainer.fit(
        data,
        subject_splits=subject_splits,
        trial_standardizer=trial_standardizer,
        epoch_callback=epoch_callback,
    )
