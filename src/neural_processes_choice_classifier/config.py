from dataclasses import dataclass

from neural_processes_choice import ChoiceNPConfig


@dataclass(frozen=True, slots=True)
class ChoiceNPClassifierConfig(ChoiceNPConfig):
    n_models: int = 3
    classifier_hidden_dims: tuple[int, ...] = (64, 32)

    def __post_init__(self):
        ChoiceNPConfig.__post_init__(self)
        if self.n_models < 2:
            raise ValueError("n_models must be at least two.")
