from dataclasses import dataclass

from choice_set_encoder import ChoiceSetEncoderConfig


@dataclass(frozen=True, slots=True)
class ChoiceNPConfig(ChoiceSetEncoderConfig):

    decoder_hidden_dims: tuple[int, ...] = (128, 256, 128)
    log_var_min: float = -20.0
    log_var_max: float = 20.0

    def __post_init__(self):
        ChoiceSetEncoderConfig.__post_init__(self)
        if self.log_var_min >= self.log_var_max:
            raise ValueError("log_var_min must be smaller than log_var_max.")


@dataclass(frozen=True, slots=True)
class ChoiceNPTrainingConfig:

    beta: float = 0.2
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    batch_size: int = 64
    max_epochs: int = 1000
    patience: int = 50
    min_delta: float = 0.0
    gradient_clip_norm: float | None = 1.0
    train_fraction: float = 0.75
    validation_fraction: float = 0.15
    test_fraction: float = 0.10
    context_min_fraction: float = 0.60
    context_max_fraction: float = 0.90
    evaluate_on_target_only: bool = True
    normalize_trials: bool = True
    deterministic: bool = True
    seed: int = 31
    num_workers: int = 0

    def __post_init__(self):
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.batch_size <= 0 or self.max_epochs <= 0:
            raise ValueError("batch_size and max_epochs must be positive.")
        if self.patience < 0:
            raise ValueError("patience must be non-negative.")
        split_total = (
            self.train_fraction
            + self.validation_fraction
            + self.test_fraction
        )
        if abs(split_total - 1.0) > 1e-8:
            raise ValueError("Subject split fractions must sum to one.")
        if min(
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        ) <= 0.0:
            raise ValueError("Subject split fractions must be positive.")
        if not (
            0.0
            < self.context_min_fraction
            <= self.context_max_fraction
            <= 1.0
        ):
            raise ValueError(
                "Context fractions must satisfy 0 < min <= max <= 1."
            )
        if (
            self.evaluate_on_target_only
            and self.context_max_fraction >= 1.0
        ):
            raise ValueError(
                "context_max_fraction must be below one when evaluating "
                "only trials outside the context."
            )
