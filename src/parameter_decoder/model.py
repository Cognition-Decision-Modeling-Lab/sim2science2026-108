from torch import Tensor, nn

from choice_set_encoder import (
    DeterministicChoiceSetEncoder,
    make_mlp,
)

from .config import ParameterDecoderConfig


class ChoiceParameterDecoder(nn.Module):
    def __init__(self, config: ParameterDecoderConfig | None = None):
        super().__init__()
        self.config = config or ParameterDecoderConfig()
        self.encoder = DeterministicChoiceSetEncoder(self.config)
        self.parameter_heads = nn.ModuleDict(
            {
                model_name: make_mlp(
                    [
                        self.config.latent_dim,
                        *self.config.decoder_hidden_dims,
                        n_parameters,
                    ]
                )
                for model_name, n_parameters in self.config.parameter_dims
            }
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
            "parameter_predictions": {
                model_name: head(latent)
                for model_name, head in self.parameter_heads.items()
            },
        }
