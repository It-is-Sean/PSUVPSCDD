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
        tokens = _flatten_tokens(tokens)

        x = self.net(tokens.float())
        if x.shape[1] != self.output_tokens:
            x = torch.nn.functional.adaptive_avg_pool1d(
                x.transpose(1, 2), self.output_tokens
            ).transpose(1, 2)
        return x.contiguous()


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, gated: bool = True) -> None:
        super().__init__()
        self.gated = bool(gated)
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
        if self.gated:
            # Zero-init gates stabilize training by starting near identity mapping.
            self.attn_gate = nn.Parameter(torch.zeros(1))
            self.ffn_gate = nn.Parameter(torch.zeros(1))

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        context_norm = self.context_norm(context)
        attn_out, _ = self.attn(
            self.query_norm(queries),
            context_norm,
            context_norm,
            need_weights=False,
        )
        if self.gated:
            queries = queries + torch.tanh(self.attn_gate) * attn_out
        else:
            queries = queries + attn_out
        ffn_out = self.ffn(self.ffn_norm(queries))
        if self.gated:
            queries = queries + torch.tanh(self.ffn_gate) * ffn_out
        else:
            queries = queries + ffn_out
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
        gated: bool = True,
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
        self.gated = bool(gated)

        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.query_tokens = nn.Parameter(torch.randn(1, self.output_tokens, self.hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(self.hidden_dim, self.num_heads, self.mlp_ratio, gated=self.gated) for _ in range(self.adapter_layers)]
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = _flatten_tokens(tokens)

        context = self.input_proj(tokens.float())
        queries = self.query_tokens.expand(context.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, context)
        return self.output_proj(self.output_norm(queries)).contiguous()


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.norm1(tokens),
            self.norm1(tokens),
            self.norm1(tokens),
            need_weights=False,
        )
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens


class VGGTToNovaCrossAttentionAdapter(VGGTToNovaAttentionAdapter):
    """Explicit ungated cross-attention adapter for the reusable CA branch."""

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
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            output_tokens=output_tokens,
            hidden_dim=hidden_dim,
            adapter_layers=adapter_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            gated=False,
        )


class VGGTToNovaSelfAttentionAdapter(nn.Module):
    """
    Self-attention probe from frozen VGGT tokens to NOVA3R scene tokens.

    Compared with the cross-attention variant, this path first projects the
    frozen VGGT token sequence, compresses it to NOVA3R's target token count,
    and then refines the resulting scene tokens using self-attention only.
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
        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(self.hidden_dim, self.num_heads, self.mlp_ratio) for _ in range(self.adapter_layers)]
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = _flatten_tokens(tokens)

        scene_tokens = self.input_proj(tokens.float())
        if scene_tokens.shape[1] != self.output_tokens:
            scene_tokens = torch.nn.functional.adaptive_avg_pool1d(
                scene_tokens.transpose(1, 2), self.output_tokens
            ).transpose(1, 2)
        for block in self.blocks:
            scene_tokens = block(scene_tokens)
        return self.output_proj(self.output_norm(scene_tokens)).contiguous()


def _flatten_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim == 4:
        b, s, p, c = tokens.shape
        tokens = tokens.reshape(b, s * p, c)
    if tokens.ndim != 3:
        raise ValueError(f"Expected VGGT tokens [B,L,C] or [B,S,P,C], got {tuple(tokens.shape)}")
    return tokens
