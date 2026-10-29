from dataclasses import dataclass

from choice_set_encoder import ChoiceSetEncoderConfig


@dataclass(frozen=True, slots=True)
class ModelClassifierConfig(ChoiceSetEncoderConfig):
    n_models: int = 3
    classifier_hidden_dims: tuple[int, ...] = (64, 32)

    def __post_init__(self):
        ChoiceSetEncoderConfig.__post_init__(self)
        if self.n_models < 2:
            raise ValueError("n_models must be at least two.")


@dataclass(frozen=True, slots=True)
class ModelClassifierTrainingConfig:
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    batch_size: int = 64
    max_epochs: int = 1000
    patience: int = 50
    min_delta: float = 0.0
    gradient_clip_norm: float | None = 1.0
    normalize_trials: bool = True
    deterministic: bool = True
    seed: int = 31
    num_workers: int = 0
