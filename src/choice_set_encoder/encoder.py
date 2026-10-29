import torch
from torch import Tensor, nn

from .config import ChoiceSetEncoderConfig
from .set_transformer import SetTransformer


def make_mlp(
    dimensions: list[int],
    *,
    final_activation: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (input_dim, output_dim) in enumerate(
        zip(dimensions[:-1], dimensions[1:], strict=True)
    ):
        layers.append(nn.Linear(input_dim, output_dim))
        is_last = index == len(dimensions) - 2
        if final_activation or not is_last:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class ChoiceSetEncoderBackbone(nn.Module):

    def __init__(self, config: ChoiceSetEncoderConfig):
        super().__init__()
        self.config = config
        context_dim = config.trial_dim + config.choice_dim
        self.trial_encoder = make_mlp(
            [
                context_dim,
                *config.trial_encoder_hidden_dims,
                config.embedding_dim,
            ]
        )
        self.context_aggregator = SetTransformer(
            embedding_dim=config.embedding_dim,
            num_heads=config.num_attention_heads,
            num_blocks=config.num_attention_blocks,
            dropout=config.attention_dropout,
        )

    def forward(
        self,
        trials: Tensor,
        choices: Tensor,
        trial_mask: Tensor,
    ) -> Tensor:
        self._validate_inputs(trials, choices, trial_mask)
        invalid = ~trial_mask.bool().unsqueeze(-1)
        safe_trials = trials.masked_fill(invalid, 0.0)
        safe_choices = choices.masked_fill(invalid, 0.0)
        pairs = torch.cat([safe_trials, safe_choices], dim=-1)
        embeddings = self.trial_encoder(pairs)
        return self.context_aggregator(embeddings, trial_mask)

    def _validate_inputs(
        self,
        trials: Tensor,
        choices: Tensor,
        trial_mask: Tensor,
    ) -> None:
        if trials.ndim != 3 or choices.ndim != 3:
            raise ValueError("trials and choices must be three-dimensional.")
        if trials.shape[:2] != choices.shape[:2]:
            raise ValueError("trials and choices must agree on B and T.")
        if trial_mask.shape != trials.shape[:2]:
            raise ValueError("trial_mask must have shape (B, T).")
        if trials.shape[-1] != self.config.trial_dim:
            raise ValueError("Unexpected trial feature dimension.")
        if choices.shape[-1] != self.config.choice_dim:
            raise ValueError("Unexpected choice dimension.")


class DeterministicChoiceSetEncoder(nn.Module):

    def __init__(self, config: ChoiceSetEncoderConfig):
        super().__init__()
        self.backbone = ChoiceSetEncoderBackbone(config)
        self.latent_head = nn.Linear(
            config.embedding_dim,
            config.latent_dim,
        )

    def forward(
        self,
        trials: Tensor,
        choices: Tensor,
        trial_mask: Tensor,
    ) -> Tensor:
        summary = self.backbone(trials, choices, trial_mask)
        return self.latent_head(summary)


class GaussianChoiceSetEncoder(nn.Module):

    def __init__(
        self,
        config: ChoiceSetEncoderConfig,
        log_var_min: float = -20.0,
        log_var_max: float = 20.0,
    ):
        super().__init__()
        self.backbone = ChoiceSetEncoderBackbone(config)
        self.mean_head = nn.Linear(
            config.embedding_dim,
            config.latent_dim,
        )
        self.log_variance_head = nn.Linear(
            config.embedding_dim,
            config.latent_dim,
        )
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

    def forward(
        self,
        trials: Tensor,
        choices: Tensor,
        trial_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        summary = self.backbone(trials, choices, trial_mask)
        mean = self.mean_head(summary)
        log_variance = self.log_variance_head(summary).clamp(
            min=self.log_var_min,
            max=self.log_var_max,
        )
        return mean, log_variance
