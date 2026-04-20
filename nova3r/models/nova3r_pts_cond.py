# Copyright (c) 2026 Weirong Chen
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from torch_cluster import fps

from nova3r.heads.dpt_head import DPTHead
from nova3r.heads.pts3d_decoder import *
from nova3r.heads.triposg_model.autoencoder_kl_triposg import (
    FrequencyPositionalEmbedding,
    TripoSGEncoder,
    TripoSGEncoderCS,
)
from nova3r.heads.pts3d_encoder.transformer_encoder import TransformerEncoder


class Nova3rPtsCond(nn.Module, PyTorchModelHubMixin):
    """Point-conditioned autoencoder model for Stage 1 point-to-point reconstruction.

    Takes raw point clouds as input and encodes them into a compact set of
    3D tokens using a configurable aggregator (TransformerEncoder, TripoSG
    variants). Supports classifier-free guidance dropout and KL regularisation.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        patch_3d_size=256,
        num_3d_tokens=512,
        cfg=None,
        classifier_free_guidance_drop_prob=0.0,
    ):
        """3D Shape Tokenization Implementation."""
        super().__init__()
        self.cfg = cfg

        # Classifier-free guidance settings
        self.cfg_drop_prob = classifier_free_guidance_drop_prob

        self.embedder = FrequencyPositionalEmbedding(
            num_freqs=8,
            logspace=True,
            input_dim=3,
            include_pi=False,
        )

        self.num_3d_tokens = num_3d_tokens
        aggregator_type = (
            self.cfg.aggregator.type
            if "type" in self.cfg.aggregator
            else "TripoSGEncoder"
        )

        self.use_token_ln = self.cfg.aggregator.params.get("use_token_ln", False)
        self.token_noise_prob = self.cfg.aggregator.params.get("token_noise_prob", 0.0)
        self.token_noise_sigma = self.cfg.aggregator.params.get(
            "token_noise_sigma", 0.0
        )

        if self.use_token_ln:
            self.token_norm = nn.LayerNorm(self.cfg.aggregator.params.token_dim)
        else:
            self.token_norm = None

        if self.cfg.aggregator.name == "TransformerEncoder":
            self.aggregator = TransformerEncoder(**self.cfg.aggregator.params)
        elif self.cfg.aggregator.name == "triposg_point":
            self.cfg.aggregator.params.in_channels = self.embedder.out_dim
            self.aggregator = eval(aggregator_type)(**self.cfg.aggregator.params)
            self.token_proj = nn.Linear(
                self.cfg.aggregator.params.dim,
                self.cfg.aggregator.params.token_dim,
            )
        elif self.cfg.aggregator.name == "triposg_learn":
            self.tokens = nn.Parameter(
                torch.randn(num_3d_tokens, self.embedder.out_dim),
            )
            self.cfg.aggregator.params.in_channels = self.embedder.out_dim
            self.aggregator = eval(aggregator_type)(**self.cfg.aggregator.params)
            self.token_proj = nn.Linear(
                self.cfg.aggregator.params.dim,
                self.cfg.aggregator.params.token_dim,
            )
        elif self.cfg.aggregator.name == "triposg_hybrid":
            self.tokens = nn.Parameter(
                torch.randn(num_3d_tokens, self.embedder.out_dim),
            )
            self.cfg.aggregator.params.in_channels = self.embedder.out_dim
            self.aggregator = eval(aggregator_type)(**self.cfg.aggregator.params)
            self.token_proj = nn.Linear(
                self.cfg.aggregator.params.dim,
                self.cfg.aggregator.params.token_dim,
            )
            self.token_merge = nn.Linear(
                self.embedder.out_dim * 2,
                self.embedder.out_dim,
            )
        else:
            raise NotImplementedError(
                f"Aggregator {self.cfg.aggregator.name} not implemented."
            )

        self.use_kl = self.cfg.aggregator.params.get("use_kl", False)
        if self.use_kl:
            self.mean_fc = nn.Linear(
                self.cfg.aggregator.params.dim, self.cfg.aggregator.params.token_dim
            )
            self.logvar_fc = nn.Linear(
                self.cfg.aggregator.params.dim, self.cfg.aggregator.params.token_dim
            )

        self.camera_head = None
        if "point_head" in cfg:
            self.point_head = DPTHead(
                dim_in=2 * embed_dim,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
                **cfg.point_head.params,
            )
        else:
            self.point_head = DPTHead(
                dim_in=2 * embed_dim,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
            )
        self.depth_head = None
        self.rgb_head = None
        self.track_head = None

        if "pts3d_head" in cfg:
            self.pts3d_head = eval(cfg.pts3d_head.name)(**cfg.pts3d_head.params)

    def _sample_features(
        self, x: torch.Tensor, num_tokens: int = 2048, seed: Optional[int] = None
    ):
        """Sample points from features of the input point cloud.

        Args:
            x (torch.Tensor): The input point cloud with shape (B, N, C).
            num_tokens (int, optional): The number of points to sample. Defaults to 2048.
            seed (Optional[int], optional): The random seed. Defaults to None.

        Returns:
            torch.Tensor: Sampled points with shape (B, num_tokens, C).
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(
            x.shape[1], num_tokens * 4, replace=num_tokens * 4 > x.shape[1]
        )
        selected_points = x[:, indices]

        batch_size, num_points, num_channels = selected_points.shape
        flattened_points = selected_points.view(batch_size * num_points, num_channels)
        batch_indices = (
            torch.arange(batch_size).to(x.device).repeat_interleave(num_points)
        )

        sampling_ratio = 1.0 / 4
        sampled_indices = fps(
            flattened_points[:, :3],
            batch_indices,
            ratio=sampling_ratio,
            random_start=False,  # deterministic sampling
        )
        sampled_points = flattened_points[sampled_indices].view(
            batch_size, -1, num_channels
        )

        return sampled_points

    def load_state_dict(self, ckpt, **kw):
        # Duplicate weights for pts3d_blocks from frame_blocks if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith("aggregator.pts3d_blocks") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("aggregator.frame_blocks"):
                    new_ckpt[
                        key.replace("aggregator.frame_blocks", "aggregator.pts3d_blocks")
                    ] = value

        return super().load_state_dict(new_ckpt, **kw)

    def _encode(self, pointmaps, cfg_scale=1.0, test=False, **kwargs):
        """Encode a point cloud into compact 3D tokens.

        Args:
            pointmaps (torch.Tensor): Input point cloud with shape [B, N, C].
            cfg_scale (float): CFG scale used at test time. Default: 1.0.
            test (bool): Whether in inference mode. Default: False.

        Returns:
            dict: Dictionary with key 'tokens' of shape [B, num_3d_tokens, token_dim].
        """
        B, N, C = pointmaps.shape

        if self.cfg.aggregator.name == "TransformerEncoder":
            tokens = self.aggregator(pointmaps)

        elif self.cfg.aggregator.name == "triposg_point":
            x_kv = self.embedder(pointmaps)
            sample_x = self._sample_features(pointmaps, self.num_3d_tokens)
            x_q = self.embedder(sample_x)
            x = self.aggregator(x_q, x_kv)
            tokens = self.token_proj(x)

        elif self.cfg.aggregator.name == "triposg_learn":
            x_kv = self.embedder(pointmaps)
            x_q = self.tokens.unsqueeze(0).expand(B, -1, -1)
            x = self.aggregator(x_q, x_kv)
            tokens = self.token_proj(x)

        elif self.cfg.aggregator.name == "triposg_hybrid":
            x_kv = self.embedder(pointmaps)
            sample_x = self._sample_features(pointmaps, self.num_3d_tokens)
            x_q_point = self.embedder(sample_x)
            x_q_learn = self.tokens.unsqueeze(0).expand(B, -1, -1)

            x_q = torch.cat([x_q_point, x_q_learn], dim=-1)
            x_q = self.token_merge(x_q)

            x = self.aggregator(x_q, x_kv)
            tokens = self.token_proj(x)

        else:
            raise NotImplementedError(
                f"Aggregator {self.cfg.aggregator.name} not implemented for encoding."
            )

        # Apply classifier-free guidance dropout or interpolation
        if test:
            if cfg_scale != 1.0:
                tokens_cond = tokens
                tokens_uncond = torch.zeros_like(tokens_cond)
                tokens = tokens_uncond + cfg_scale * (tokens_cond - tokens_uncond)
        else:
            tokens = self._apply_cfg_dropout(tokens)

        if self.use_token_ln and self.token_norm is not None:
            tokens = self.token_norm(tokens)

        if not test and self.token_noise_prob > 0:
            if torch.rand(()) < self.token_noise_prob:
                noise = torch.randn_like(tokens) * self.token_noise_sigma
                tokens = tokens + noise

        data = {"tokens": tokens}
        return data

    def _apply_cfg_dropout(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply classifier-free guidance dropout to tokens during training.

        Randomly zeros out all tokens for a sample with probability cfg_drop_prob,
        enabling classifier-free guidance at inference time.

        Args:
            tokens (torch.Tensor): Condition tokens with shape [B, N, C].

        Returns:
            torch.Tensor: Tokens with some potentially zeroed out (during training).
        """
        if not self.training or self.cfg_drop_prob <= 0.0:
            return tokens

        B = tokens.shape[0]
        drop_mask = torch.rand(B, device=tokens.device) < self.cfg_drop_prob
        drop_mask = drop_mask.view(B, 1, 1).expand_as(tokens)
        tokens = tokens.masked_fill(drop_mask, 0.0)

        return tokens

    def _decode(self, tokens, images, token_mask=None, query_points=None, timestep=None):
        """Decode 3D tokens into point cloud coordinates.

        Args:
            tokens (torch.Tensor): Encoded 3D tokens with shape [B, num_tokens, C].
            images (torch.Tensor): Reference images with shape [B, S, C, H, W].
            token_mask (torch.Tensor, optional): Boolean mask for tokens. Default: None.
            query_points (torch.Tensor, optional): Query point coordinates. Default: None.
            timestep (torch.Tensor, optional): Diffusion timestep. Default: None.

        Returns:
            dict: Predictions including 'pts3d_xyz', 'images', and 'S'.
        """
        B, S = images.shape[:2]
        predictions = {}

        num_views = torch.ones(B, device=images.device) * S

        with torch.cuda.amp.autocast(enabled=False):
            tokens = tokens.float()

            if self.pts3d_head is not None:
                aggregated_tokens_3d_list = [tokens]

                pts3d_xyz = self.pts3d_head(
                    aggregated_tokens_3d_list,
                    mask=token_mask,
                    query_points=query_points,
                    timestep=timestep,
                    num_views=num_views,
                )

                predictions["pts3d_xyz"] = pts3d_xyz
                predictions["pts3d_xyz_rel"] = pts3d_xyz
                predictions["pts3d_rgb"] = pts3d_xyz
                predictions["pts3d_conf"] = torch.ones_like(pts3d_xyz[..., [0]])
                predictions["center_xyz"] = pts3d_xyz
                predictions["query_points"] = query_points
                predictions["timestep"] = timestep

        predictions["images"] = images
        predictions["S"] = S

        return predictions

    def forward(
        self,
        images: torch.Tensor,
        pointmaps: torch.Tensor,
        token_mask: torch.Tensor = None,
        query_points: torch.Tensor = None,
        timestep: torch.Tensor = None,
        **kwargs,
    ):
        """Full forward pass: encode point cloud, then decode to 3D predictions.

        Args:
            images (torch.Tensor): Reference images with shape [B, S, C, H, W].
            pointmaps (torch.Tensor): Input point cloud with shape [B, N, 3].
            token_mask (torch.Tensor, optional): Mask for tokens. Default: None.
            query_points (torch.Tensor, optional): Query points for decoding.
                Shape: [N, 2] or [B, N, 2]. Default: None.
            timestep (torch.Tensor, optional): Timestep for flow matching /
                diffusion. Default: None.

        Returns:
            dict: Predictions including 'pts3d_xyz', 'images', and 'S'.
        """
        B, N, C = pointmaps.shape

        encoder_data = self._encode(pointmaps)
        tokens = encoder_data["tokens"]

        predictions = self._decode(
            tokens,
            images,
            token_mask=token_mask,
            query_points=query_points,
            timestep=timestep,
        )

        return predictions
