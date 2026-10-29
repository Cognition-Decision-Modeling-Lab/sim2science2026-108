import torch
from torch import Tensor, nn

from choice_set_encoder import GaussianChoiceSetEncoder, make_mlp

from .config import ChoiceNPConfig


class ChoiceNeuralProcess(nn.Module):

    def __init__(self, config: ChoiceNPConfig | None = None):
        super().__init__()
        self.config = config or ChoiceNPConfig()
        self.encoder = GaussianChoiceSetEncoder(
            self.config,
            log_var_min=self.config.log_var_min,
            log_var_max=self.config.log_var_max,
        )

        decoder_dims = [
            self.config.trial_dim + self.config.latent_dim,
            *self.config.decoder_hidden_dims,
            self.config.choice_dim,
        ]
        self.decoder = make_mlp(decoder_dims)

    def encode_context(
        self,
        context_trials: Tensor,
        context_choices: Tensor,
        context_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.encoder(
            context_trials,
            context_choices,
            context_mask,
        )

    @staticmethod
    def reparameterize(mean: Tensor, log_variance: Tensor) -> Tensor:
        standard_deviation = torch.exp(0.5 * log_variance)
        return mean + torch.randn_like(standard_deviation) * standard_deviation

    def decode(
        self,
        target_trials: Tensor,
        latent: Tensor,
        target_mask: Tensor | None = None,
    ) -> Tensor:
        if target_trials.ndim != 3:
            raise ValueError("target_trials must have shape (B, T, D).")
        if target_trials.shape[-1] != self.config.trial_dim:
            raise ValueError("Unexpected target trial feature dimension.")
        if latent.shape != (
            target_trials.shape[0],
            self.config.latent_dim,
        ):
            raise ValueError("latent must have shape (B, latent_dim).")
        if (
            target_mask is not None
            and target_mask.shape != target_trials.shape[:2]
        ):
            raise ValueError("target_mask must have shape (B, T).")

        repeated_latent = latent.unsqueeze(1).expand(
            -1,
            target_trials.shape[1],
            -1,
        )
        if target_mask is not None:
            target_trials = target_trials.masked_fill(
                ~target_mask.bool().unsqueeze(-1),
                0.0,
            )
        logits = self.decoder(
            torch.cat([target_trials, repeated_latent], dim=-1)
        )
        if target_mask is not None:
            logits = logits.masked_fill(
                ~target_mask.bool().unsqueeze(-1),
                0.0,
            )
        return logits

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
        if target_trials is None:
            target_trials = context_trials
        if target_mask is None:
            target_mask = torch.ones(
                target_trials.shape[:2],
                dtype=torch.bool,
                device=target_trials.device,
            )

        mean, log_variance = self.encode_context(
            context_trials,
            context_choices,
            context_mask,
        )
        latent = (
            self.reparameterize(mean, log_variance)
            if sample_latent
            else mean
        )
        logits = self.decode(
            target_trials,
            latent,
            target_mask,
        )
        return {
            "logits": logits,
            "mean": mean,
            "log_variance": log_variance,
            "latent": latent,
        }
