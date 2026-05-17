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


class WanPredX0LatentConvAdapter(nn.Module):
    """
    Lightweight Conv2d readout for WAN pred-x0 latent pair features.

    The pred-x0 cache stores two temporal latent slices flattened from
    [2, 60, 104, 16]. This adapter restores that grid, applies a local
    stride-2 convolutional readout, and then matches NOVA's scene-token count.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        output_tokens: int,
        temporal_tokens: int = 2,
        latent_height: int = 60,
        latent_width: int = 104,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.output_tokens = int(output_tokens)
        self.temporal_tokens = int(temporal_tokens)
        self.latent_height = int(latent_height)
        self.latent_width = int(latent_width)
        self.expected_tokens = self.temporal_tokens * self.latent_height * self.latent_width
        if self.input_dim != 16:
            raise ValueError(f"WAN pred-x0 latent conv adapter expects input_dim=16, got {self.input_dim}")
        if self.output_dim % int(norm_groups) != 0:
            raise ValueError(f"output_dim must be divisible by norm_groups, got {self.output_dim} and {norm_groups}")

        self.stem = nn.Sequential(
            nn.Conv2d(self.input_dim, self.output_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(int(norm_groups), self.output_dim),
            nn.GELU(),
            nn.Conv2d(self.output_dim, self.output_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(int(norm_groups), self.output_dim),
            nn.GELU(),
        )
        self.readout_height = (self.latent_height + 1) // 2
        self.readout_width = (self.latent_width + 1) // 2
        self.readout_tokens = self.temporal_tokens * self.readout_height * self.readout_width
        self.conv_readout_shape = (self.readout_tokens, self.output_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = _flatten_tokens(tokens)
        if tokens.shape[1] != self.expected_tokens or tokens.shape[2] != self.input_dim:
            raise ValueError(
                "Expected WAN pred-x0 latent tokens "
                f"[B,{self.expected_tokens},{self.input_dim}], got {tuple(tokens.shape)}"
            )

        b = tokens.shape[0]
        x = tokens.float().reshape(
            b,
            self.temporal_tokens,
            self.latent_height,
            self.latent_width,
            self.input_dim,
        )
        x = x.permute(0, 1, 4, 2, 3).reshape(
            b * self.temporal_tokens,
            self.input_dim,
            self.latent_height,
            self.latent_width,
        )
        x = self.stem(x)
        x = x.reshape(
            b,
            self.temporal_tokens,
            self.output_dim,
            self.readout_height,
            self.readout_width,
        )
        x = x.permute(0, 1, 3, 4, 2).reshape(b, self.readout_tokens, self.output_dim)
        if x.shape[1] != self.output_tokens:
            x = torch.nn.functional.adaptive_avg_pool1d(
                x.transpose(1, 2), self.output_tokens
            ).transpose(1, 2)
        return x.contiguous()


class WanHiddenGrid2DPoolAdapter(nn.Module):
    """
    Structured 2D pooling readout for WAN hidden pair features.

    The WAN hidden cache stores two temporal feature slices flattened from
    [2, 30, 52, 1536]. This adapter keeps that grid structure, applies the
    same token-wise MLP style as the historical adapter, and only then pools
    the spatial grid to NOVA's fixed [2, 24, 16] token layout.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        output_tokens: int,
        hidden_dim: int = 1024,
        adapter_layers: int = 4,
        temporal_tokens: int = 2,
        grid_height: int = 30,
        grid_width: int = 52,
        output_height: int = 24,
        output_width: int = 16,
        use_conv_readout: bool = False,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        if adapter_layers < 2 or adapter_layers > 8:
            raise ValueError(f"adapter_layers must be between 2 and 8, got {adapter_layers}")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.output_tokens = int(output_tokens)
        self.hidden_dim = int(hidden_dim)
        self.adapter_layers = int(adapter_layers)
        self.temporal_tokens = int(temporal_tokens)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.output_height = int(output_height)
        self.output_width = int(output_width)
        self.use_conv_readout = bool(use_conv_readout)
        self.expected_tokens = self.temporal_tokens * self.grid_height * self.grid_width
        self.grid_output_tokens = self.temporal_tokens * self.output_height * self.output_width
        if self.input_dim != 1536:
            raise ValueError(f"WAN hidden grid2d adapter expects input_dim=1536, got {self.input_dim}")
        if self.grid_output_tokens != self.output_tokens:
            raise ValueError(
                "WAN hidden grid2d adapter output grid must match output_tokens: "
                f"{self.temporal_tokens}*{self.output_height}*{self.output_width}="
                f"{self.grid_output_tokens}, output_tokens={self.output_tokens}"
            )
        if self.output_dim % int(norm_groups) != 0:
            raise ValueError(f"output_dim must be divisible by norm_groups, got {self.output_dim} and {norm_groups}")

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
        if self.use_conv_readout:
            self.conv_readout = nn.Sequential(
                nn.Conv2d(self.output_dim, self.output_dim, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(int(norm_groups), self.output_dim),
                nn.GELU(),
            )
        else:
            self.conv_readout = nn.Identity()
        self.grid_input_shape = (self.temporal_tokens, self.grid_height, self.grid_width, self.input_dim)
        self.grid_readout_shape = (self.temporal_tokens, self.output_height, self.output_width, self.output_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = _flatten_tokens(tokens)
        if tokens.shape[1] != self.expected_tokens or tokens.shape[2] != self.input_dim:
            raise ValueError(
                "Expected WAN hidden grid tokens "
                f"[B,{self.expected_tokens},{self.input_dim}], got {tuple(tokens.shape)}"
            )

        b = tokens.shape[0]
        x = self.net(tokens.float())
        x = x.reshape(
            b,
            self.temporal_tokens,
            self.grid_height,
            self.grid_width,
            self.output_dim,
        )
        x = x.permute(0, 1, 4, 2, 3).reshape(
            b * self.temporal_tokens,
            self.output_dim,
            self.grid_height,
            self.grid_width,
        )
        x = self.conv_readout(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (self.output_height, self.output_width))
        x = x.reshape(
            b,
            self.temporal_tokens,
            self.output_dim,
            self.output_height,
            self.output_width,
        )
        x = x.permute(0, 1, 3, 4, 2).reshape(b, self.output_tokens, self.output_dim)
        return x.contiguous()


class WanHiddenGrid2DConvAdapter(WanHiddenGrid2DPoolAdapter):
    """WAN hidden grid adapter with a tiny same-resolution 3x3 Conv2d readout."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["use_conv_readout"] = True
        super().__init__(*args, **kwargs)


class WanHiddenCrossAttentionResamplerAdapter(nn.Module):
    """
    Learned resampler from WAN hidden pair features to NOVA scene tokens.

    The WAN hidden cache stores two temporal slices flattened from
    [2, 30, 52, 1536]. This adapter keeps explicit temporal/row/column
    position embeddings, then lets NOVA-sized query tokens cross-attend to
    the full WAN token grid. It is intentionally still a probe adapter: the
    generator, loss, and target semantics stay unchanged.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        output_tokens: int,
        hidden_dim: int = 512,
        adapter_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        temporal_tokens: int = 2,
        grid_height: int = 30,
        grid_width: int = 52,
        gated: bool = False,
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
        self.temporal_tokens = int(temporal_tokens)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.expected_tokens = self.temporal_tokens * self.grid_height * self.grid_width
        if self.input_dim != 1536:
            raise ValueError(f"WAN hidden cross-attention adapter expects input_dim=1536, got {self.input_dim}")

        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.temporal_pos = nn.Parameter(torch.randn(1, self.temporal_tokens, 1, 1, self.hidden_dim) * 0.02)
        self.row_pos = nn.Parameter(torch.randn(1, 1, self.grid_height, 1, self.hidden_dim) * 0.02)
        self.col_pos = nn.Parameter(torch.randn(1, 1, 1, self.grid_width, self.hidden_dim) * 0.02)
        self.query_tokens = nn.Parameter(torch.randn(1, self.output_tokens, self.hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    self.hidden_dim,
                    self.num_heads,
                    self.mlp_ratio,
                    gated=bool(gated),
                )
                for _ in range(self.adapter_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim)
        self.grid_input_shape = (self.temporal_tokens, self.grid_height, self.grid_width, self.input_dim)
        self.resampler_shape = (self.output_tokens, self.output_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = _flatten_tokens(tokens)
        if tokens.shape[1] != self.expected_tokens or tokens.shape[2] != self.input_dim:
            raise ValueError(
                "Expected WAN hidden cross-attention tokens "
                f"[B,{self.expected_tokens},{self.input_dim}], got {tuple(tokens.shape)}"
            )

        b = tokens.shape[0]
        context = self.input_proj(tokens.float()).reshape(
            b,
            self.temporal_tokens,
            self.grid_height,
            self.grid_width,
            self.hidden_dim,
        )
        context = context + self.temporal_pos + self.row_pos + self.col_pos
        context = context.reshape(b, self.expected_tokens, self.hidden_dim)
        queries = self.query_tokens.expand(b, -1, -1)
        for block in self.blocks:
            queries = block(queries, context)
        return self.output_proj(self.output_norm(queries)).contiguous()


class WanMultiTimestepAggregatorAdapter(nn.Module):
    """
    Multi-timestep WAN probe with tiny per-timestep branches and learned aggregation.

    Input is expected as [B, T, L, C], where T enumerates diffusion timesteps for
    the same cached WAN hidden feature layout. Each timestep passes through a small
    branch MLP, then a learned aggregator fuses the timestep axis before the
    historical token adapter maps the fused sequence to NOVA scene tokens.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        output_tokens: int,
        num_timesteps: int,
        branch_hidden_dim: int = 512,
        branch_layers: int = 2,
        adapter_hidden_dim: int = 1024,
        adapter_layers: int = 4,
        aggregation_mode: str = "softmax_gate",
        share_branch_mlp: bool = True,
    ) -> None:
        super().__init__()
        if num_timesteps < 2:
            raise ValueError(f"num_timesteps must be >= 2, got {num_timesteps}")
        if branch_layers < 1 or branch_layers > 4:
            raise ValueError(f"branch_layers must be between 1 and 4, got {branch_layers}")
        if aggregation_mode not in {"softmax_gate", "token_attention"}:
            raise ValueError(
                f"aggregation_mode must be 'softmax_gate' or 'token_attention', got {aggregation_mode!r}"
            )
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.output_tokens = int(output_tokens)
        self.num_timesteps = int(num_timesteps)
        self.branch_hidden_dim = int(branch_hidden_dim)
        self.branch_layers = int(branch_layers)
        self.adapter_hidden_dim = int(adapter_hidden_dim)
        self.adapter_layers = int(adapter_layers)
        self.aggregation_mode = str(aggregation_mode)
        self.share_branch_mlp = bool(share_branch_mlp)

        if self.share_branch_mlp:
            self.branch_mlps = nn.ModuleList([self._build_branch_mlp()])  # shared across T
        else:
            self.branch_mlps = nn.ModuleList([self._build_branch_mlp() for _ in range(self.num_timesteps)])

        if self.aggregation_mode == "softmax_gate":
            self.timestep_logits = nn.Parameter(torch.zeros(self.num_timesteps))
            self.token_gate = None
        else:
            self.timestep_logits = None
            self.token_gate = nn.Linear(self.branch_hidden_dim, 1)

        self.downstream_adapter = VGGTToNovaAdapter(
            input_dim=self.branch_hidden_dim,
            output_dim=self.output_dim,
            output_tokens=self.output_tokens,
            hidden_dim=self.adapter_hidden_dim,
            adapter_layers=self.adapter_layers,
        )

    def _build_branch_mlp(self) -> nn.Sequential:
        layers: list[nn.Module] = []
        dim = self.input_dim
        for _ in range(self.branch_layers):
            layers.extend([
                nn.Linear(dim, self.branch_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.branch_hidden_dim),
            ])
            dim = self.branch_hidden_dim
        return nn.Sequential(*layers)

    def _project_timesteps(self, tokens: torch.Tensor) -> torch.Tensor:
        projected = []
        for idx in range(self.num_timesteps):
            branch = self.branch_mlps[0] if self.share_branch_mlp else self.branch_mlps[idx]
            projected.append(branch(tokens[:, idx].float()))
        return torch.stack(projected, dim=1)

    def _aggregate_timesteps(self, projected: torch.Tensor) -> torch.Tensor:
        if self.aggregation_mode == "softmax_gate":
            weights = torch.softmax(self.timestep_logits, dim=0).view(1, self.num_timesteps, 1, 1)
            return (projected * weights).sum(dim=1)
        scores = self.token_gate(projected).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (projected * weights).sum(dim=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError(f"Expected multi-timestep WAN tokens [B,T,L,C], got {tuple(tokens.shape)}")
        if tokens.shape[1] != self.num_timesteps or tokens.shape[-1] != self.input_dim:
            raise ValueError(
                "Unexpected multi-timestep WAN token shape: "
                f"expected [B,{self.num_timesteps},L,{self.input_dim}], got {tuple(tokens.shape)}"
            )
        projected = self._project_timesteps(tokens)
        aggregated = self._aggregate_timesteps(projected)
        return self.downstream_adapter(aggregated)


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
