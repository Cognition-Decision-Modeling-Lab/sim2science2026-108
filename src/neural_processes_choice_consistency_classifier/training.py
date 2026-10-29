from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

from neural_processes_choice import (
    ChoiceData,
    ChoiceNPTrainingConfig,
    TrainingResult,
    TrialStandardizer,
    masked_beta_vae_loss,
    sample_context_mask,
)
from neural_processes_choice_classifier import (
    ChoiceNPClassifierConfig,
    ChoiceNeuralProcessClassifier,
    ClassifierChoiceNPTrainer,
)


class ConsistencyClassifierChoiceNPTrainer(ClassifierChoiceNPTrainer):

    def __init__(
        self,
        model_config: ChoiceNPClassifierConfig,
        training_config: ChoiceNPTrainingConfig,
        labels: Tensor,
        classification_weight: float,
        consistency_weight: float,
        consistency_min_fraction: float,
        consistency_max_fraction: float,
        device=None,
    ):
        super().__init__(
            model_config,
            training_config,
            labels,
            classification_weight,
            device,
        )
        if consistency_weight < 0.0:
            raise ValueError("consistency_weight must be non-negative.")
        self.consistency_weight = consistency_weight
        self.consistency_min_fraction = consistency_min_fraction
        self.consistency_max_fraction = consistency_max_fraction

    def _run_training_epoch(
        self,
        model: ChoiceNeuralProcessClassifier,
        loader: DataLoader,
        optimizer: Adam,
        context_generator: torch.Generator,
    ) -> dict[str, float]:
        model.train()
        accumulated = self._empty_metrics()
        accumulated["consistency"] = 0.0

        for batch in loader:
            valid_mask = batch["trial_mask"]
            context_a = sample_context_mask(
                valid_mask,
                self.training_config.context_min_fraction,
                self.training_config.context_max_fraction,
                context_generator,
            )
            context_b = sample_context_mask(
                valid_mask,
                self.consistency_min_fraction,
                self.consistency_max_fraction,
                context_generator,
            )
            batch = self._move_batch(batch)
            context_a = context_a.to(self.device)
            context_b = context_b.to(self.device)

            optimizer.zero_grad(set_to_none=True)
            output = model(
                context_trials=batch["trials"],
                context_choices=batch["choices"],
                context_mask=context_a,
                target_trials=batch["trials"],
                target_mask=batch["trial_mask"],
                sample_latent=True,
            )
            mean_b, _ = model.encode_context(
                batch["trials"],
                batch["choices"],
                context_b,
            )
            losses = masked_beta_vae_loss(
                logits=output["logits"],
                choices=batch["choices"],
                target_mask=batch["trial_mask"],
                mean=output["mean"],
                log_variance=output["log_variance"],
                beta=self.training_config.beta,
            )
            mean_a = output["mean"]
            consistency_loss = F.mse_loss(mean_a, mean_b)
            mean_z = 0.5 * (mean_a + mean_b)
            total_variance = mean_z.var(
                dim=0,
                unbiased=False,
            ).sum()
            variance_loss = F.relu(
                mean_z.new_tensor(0.1) - total_variance
            )
            classification = F.cross_entropy(
                output["model_logits"],
                batch["model_label"],
            )
            losses["consistency"] = consistency_loss
            losses["classification"] = classification
            losses["loss"] = (
                losses["loss"]
                + self.consistency_weight
                * (consistency_loss + variance_loss)
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
            accumulated["consistency"] += losses["consistency"].item()

        return self._average_metrics(accumulated, len(loader))

    @torch.no_grad()
    def _evaluate(
        self,
        model: ChoiceNeuralProcessClassifier,
        loader: DataLoader,
        fixed_context_mask: Tensor,
        evaluate_on_target_only: bool | None = None,
    ) -> dict[str, float]:
        metrics = super()._evaluate(
            model,
            loader,
            fixed_context_mask,
            evaluate_on_target_only,
        )
        metrics["consistency"] = self._evaluate_consistency(model, loader)
        return metrics

    @torch.no_grad()
    def _evaluate_consistency(self, model, loader):
        model.eval()
        generator = torch.Generator().manual_seed(
            self.training_config.seed + 20_000
        )
        all_means_a = []
        all_means_b = []
        for batch in loader:
            valid_mask = batch["trial_mask"]
            context_a = sample_context_mask(
                valid_mask,
                self.training_config.context_min_fraction,
                self.training_config.context_max_fraction,
                generator,
            )
            context_b = sample_context_mask(
                valid_mask,
                self.consistency_min_fraction,
                self.consistency_max_fraction,
                generator,
            )
            batch = self._move_batch(batch)
            mean_a, _ = model.encode_context(
                batch["trials"],
                batch["choices"],
                context_a.to(self.device),
            )
            mean_b, _ = model.encode_context(
                batch["trials"],
                batch["choices"],
                context_b.to(self.device),
            )
            all_means_a.append(mean_a)
            all_means_b.append(mean_b)
        return F.mse_loss(
            torch.cat(all_means_a),
            torch.cat(all_means_b),
        ).item()

    @staticmethod
    def _empty_metrics() -> dict[str, float]:
        metrics = ClassifierChoiceNPTrainer._empty_metrics()
        metrics["consistency"] = 0.0
        return metrics


def train_choice_neural_process_consistency_classifier(
    data: ChoiceData | str | Path,
    labels: Tensor,
    model_config: ChoiceNPClassifierConfig,
    training_config: ChoiceNPTrainingConfig,
    classification_weight: float,
    consistency_weight: float,
    consistency_min_fraction: float,
    consistency_max_fraction: float,
    device: str | torch.device | None = None,
    subject_splits: dict[str, torch.Tensor] | None = None,
    trial_standardizer: TrialStandardizer | None = None,
    epoch_callback: Callable | None = None,
) -> TrainingResult:
    trainer = ConsistencyClassifierChoiceNPTrainer(
        model_config,
        training_config,
        labels,
        classification_weight,
        consistency_weight,
        consistency_min_fraction,
        consistency_max_fraction,
        device,
    )
    return trainer.fit(
        data,
        subject_splits=subject_splits,
        trial_standardizer=trial_standardizer,
        epoch_callback=epoch_callback,
    )
