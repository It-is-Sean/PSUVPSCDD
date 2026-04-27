import torch
from torch import nn


class SmallAdapter(nn.Module):
    """Intentionally small probe adapter from frozen features to latent space."""

    def __init__(self, input_dim: int, latent_dim: int = 512, depth: int = 1):
        super().__init__()
        if depth not in (0, 1, 2):
            raise ValueError(f"depth must be 0, 1, or 2; got {depth}")

        if depth == 0:
            self.net = nn.Linear(input_dim, latent_dim)
        else:
            layers = []
            dim = input_dim
            for _ in range(depth):
                layers.extend([nn.Linear(dim, latent_dim), nn.GELU()])
                dim = latent_dim
            self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class VGGTToNovaAdapter(nn.Module):
    """
    Small token-wise MLP probe from frozen VGGT tokens to NOVA3R scene tokens.

    The adapter keeps the learning capacity deliberately limited: it projects
    channel width with 2-8 linear layers and uses adaptive pooling only to
    match NOVA3R's fixed number of scene tokens.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        output_tokens: int,
        hidden_dim: int = 1024,
        adapter_layers: int = 2,
    ) -> None:
        super().__init__()
        if adapter_layers < 2 or adapter_layers > 8:
            raise ValueError(f"adapter_layers must be between 2 and 8, got {adapter_layers}")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.output_tokens = int(output_tokens)
        self.hidden_dim = int(hidden_dim)
        self.adapter_layers = int(adapter_layers)

        layers: list[nn.Module] = [
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        ]
        for _ in range(self.adapter_layers - 2):
            layers.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim),
            ])
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 4:
            b, s, p, c = tokens.shape
            tokens = tokens.reshape(b, s * p, c)
        if tokens.ndim != 3:
            raise ValueError(f"Expected VGGT tokens [B,L,C] or [B,S,P,C], got {tuple(tokens.shape)}")

        x = self.net(tokens.float())
        if x.shape[1] != self.output_tokens:
            x = torch.nn.functional.adaptive_avg_pool1d(
                x.transpose(1, 2), self.output_tokens
            ).transpose(1, 2)
        return x.contiguous()



class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.query_norm(queries),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        queries = queries + attn_out
        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries


class VGGTToNovaAttentionAdapter(nn.Module):
    """
    Cross-attention probe from frozen VGGT tokens to NOVA3R scene tokens.

    This replaces the token-wise MLP adapter with learnable NOVA-sized query
    tokens that cross-attend to the frozen VGGT layer-23 token sequence.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        output_tokens: int,
        hidden_dim: int = 512,
        adapter_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        if adapter_layers < 1 or adapter_layers > 8:
            raise ValueError(f"adapter_layers must be between 1 and 8, got {adapter_layers}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim must be divisible by num_heads, got {hidden_dim} and {num_heads}")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.output_tokens = int(output_tokens)
        self.hidden_dim = int(hidden_dim)
        self.adapter_layers = int(adapter_layers)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)

        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.query_tokens = nn.Parameter(torch.randn(1, self.output_tokens, self.hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(self.hidden_dim, self.num_heads, self.mlp_ratio) for _ in range(self.adapter_layers)]
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 4:
            b, s, p, c = tokens.shape
            tokens = tokens.reshape(b, s * p, c)
        if tokens.ndim != 3:
            raise ValueError(f"Expected VGGT tokens [B,L,C] or [B,S,P,C], got {tuple(tokens.shape)}")

        context = self.input_proj(tokens.float())
        queries = self.query_tokens.expand(context.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, context)
        return self.output_proj(self.output_norm(queries)).contiguous()
