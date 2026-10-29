from dataclasses import dataclass

from choice_set_encoder import ChoiceSetEncoderConfig


@dataclass(frozen=True, slots=True)
class ParameterDecoderConfig(ChoiceSetEncoderConfig):
    parameter_dims: tuple[tuple[str, int], ...] = (
        ("m1", 2),
        ("m2", 3),
        ("m3", 3),
        ("m4", 2),
        ("m5", 4),
        ("m6", 3),
        ("m7", 4),
        ("m8", 2),
        ("m9", 4),
        ("m10", 5),
        ("m11", 1),
    )
    decoder_hidden_dims: tuple[int, ...] = (128, 64)

    def __post_init__(self):
        ChoiceSetEncoderConfig.__post_init__(self)
        names = [name for name, _ in self.parameter_dims]
        if not names or len(names) != len(set(names)):
            raise ValueError("parameter_dims must have unique model names.")
        if any(size <= 0 for _, size in self.parameter_dims):
            raise ValueError("Every parameter head must have positive size.")


@dataclass(frozen=True, slots=True)
class ParameterDecoderTrainingConfig:
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
