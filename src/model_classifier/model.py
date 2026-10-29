from torch import Tensor, nn

from choice_set_encoder import (
    DeterministicChoiceSetEncoder,
    make_mlp,
)

from .config import ModelClassifierConfig


class ChoiceModelClassifier(nn.Module):
    def __init__(self, config: ModelClassifierConfig | None = None):
        super().__init__()
        self.config = config or ModelClassifierConfig()
        self.encoder = DeterministicChoiceSetEncoder(self.config)
        self.classifier_head = make_mlp(
            [
                self.config.latent_dim,
                *self.config.classifier_hidden_dims,
                self.config.n_models,
            ]
        )

    def encode(
        self,
        trials: Tensor,
        choices: Tensor,
        trial_mask: Tensor,
    ) -> Tensor:
        return self.encoder(trials, choices, trial_mask)

    def forward(
        self,
        trials: Tensor,
        choices: Tensor,
        trial_mask: Tensor,
    ) -> dict[str, Tensor]:
        latent = self.encode(trials, choices, trial_mask)
        return {
            "latent": latent,
            "model_logits": self.classifier_head(latent),
        }
