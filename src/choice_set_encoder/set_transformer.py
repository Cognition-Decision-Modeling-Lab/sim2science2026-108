import torch
from torch import Tensor, nn


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Dropout(dropout),
        )
        self.feed_forward_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        attended, _ = self.attention(
            query,
            key_value,
            key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        output = self.attention_norm(query + attended)
        return self.feed_forward_norm(output + self.feed_forward(output))


class SetAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.block = MultiheadAttentionBlock(
            embedding_dim,
            num_heads,
            dropout,
        )

    def forward(
        self,
        values: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        output = self.block(values, values, key_padding_mask)
        if key_padding_mask is not None:
            output = output.masked_fill(
                key_padding_mask.unsqueeze(-1),
                0.0,
            )
        return output


class PoolingByMultiheadAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.seed = nn.Parameter(torch.empty(1, 1, embedding_dim))
        nn.init.xavier_uniform_(self.seed)
        self.block = MultiheadAttentionBlock(
            embedding_dim,
            num_heads,
            dropout,
        )

    def forward(
        self,
        values: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        seed = self.seed.expand(values.shape[0], -1, -1)
        return self.block(
            seed,
            values,
            key_padding_mask,
        ).squeeze(1)


class SetTransformer(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 4,
        num_blocks: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.attention_blocks = nn.ModuleList(
            SetAttentionBlock(
                embedding_dim,
                num_heads,
                dropout,
            )
            for _ in range(num_blocks)
        )
        self.pooling = PoolingByMultiheadAttention(
            embedding_dim,
            num_heads,
            dropout,
        )

    def forward(self, values: Tensor, trial_mask: Tensor) -> Tensor:
        if values.ndim != 3:
            raise ValueError("values must have shape (B, T, D).")
        if trial_mask.shape != values.shape[:2]:
            raise ValueError("trial_mask must have shape (B, T).")
        if not torch.all(trial_mask.any(dim=1)):
            raise ValueError("Every set must contain at least one valid trial.")
        key_padding_mask = ~trial_mask.bool()
        encoded = values
        for block in self.attention_blocks:
            encoded = block(encoded, key_padding_mask)
        return self.pooling(encoded, key_padding_mask)
