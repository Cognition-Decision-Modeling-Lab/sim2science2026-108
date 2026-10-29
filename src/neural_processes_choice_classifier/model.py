from torch import Tensor

from choice_set_encoder import make_mlp
from neural_processes_choice import ChoiceNeuralProcess

from .config import ChoiceNPClassifierConfig


class ChoiceNeuralProcessClassifier(ChoiceNeuralProcess):

    def __init__(self, config: ChoiceNPClassifierConfig | None = None):
        config = config or ChoiceNPClassifierConfig()
        super().__init__(config)
        self.classifier_head = make_mlp(
            [
                self.config.latent_dim,
                *self.config.classifier_hidden_dims,
                self.config.n_models,
            ]
        )

    def classify(self, latent: Tensor) -> Tensor:
        return self.classifier_head(latent)

    def forward(
        self,
        context_trials: Tensor,
        context_choices: Tensor,
        context_mask: Tensor,
        target_trials: Tensor | None = None,
        target_mask: Tensor | None = None,
        *,
        sample_latent: bool = True,
    ) -> dict[str, Tensor]:
        output = super().forward(
            context_trials,
            context_choices,
            context_mask,
            target_trials,
            target_mask,
            sample_latent=sample_latent,
        )
        output["model_logits"] = self.classify(output["latent"])
        return output
