from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChoiceSetEncoderConfig:

    trial_dim: int = 4
    choice_dim: int = 1
    embedding_dim: int = 64
    latent_dim: int = 8
    num_attention_blocks: int = 2
    num_attention_heads: int = 4
    attention_dropout: float = 0.05
    trial_encoder_hidden_dims: tuple[int, ...] = (128, 64)

    def __post_init__(self):
        if self.trial_dim <= 0 or self.choice_dim <= 0:
            raise ValueError("Input dimensions must be positive.")
        if self.embedding_dim <= 0 or self.latent_dim <= 0:
            raise ValueError("Embedding and latent dimensions must be positive.")
        if self.num_attention_blocks <= 0:
            raise ValueError("num_attention_blocks must be positive.")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive.")
        if self.embedding_dim % self.num_attention_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_attention_heads."
            )
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1).")
