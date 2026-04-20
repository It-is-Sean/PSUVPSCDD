# Copyright (c) 2026 Weirong Chen
from typing import Dict, List, Optional, Tuple, Union


import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from nova3r.models.aggregator_pts3d import AggregatorPts3D
from nova3r.heads.pts3d_decoder import *
from nova3r.heads.triposg_model.autoencoder_kl_triposg import FrequencyPositionalEmbedding
from nova3r.layers.hunyuan_block import FourierEmbedder

import numpy as np
import math
from torch_cluster import fps

from einops import rearrange

class Nova3rImgCond(nn.Module, PyTorchModelHubMixin):
    """Image-conditioned flow matching model for amodal 3D reconstruction.

    Takes images as input and generates complete 3D point clouds using a VGGT
    backbone encoder and flow matching decoder.
    """

    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, patch_3d_size=256, num_3d_tokens=512, cfg=None, classifier_free_guidance_drop_pro=0.0):
        """3D Shape Tokenization Implementation
        """
        super().__init__()
        self.cfg = cfg

        # Classifier-free guidance settings
        self.cfg_drop_prob = classifier_free_guidance_drop_pro

        self.vggt_aggregator = AggregatorPts3D(**self.cfg.aggregator.params)


        token_dim = self.cfg.aggregator.params.token_dim

        self.detach_vit_token = self.cfg.aggregator.params.detach_vit_token if 'detach_vit_token' in self.cfg.aggregator.params else False
        self.detach_vggt_token = self.cfg.aggregator.params.detach_vggt_token if 'detach_vggt_token' in self.cfg.aggregator.params else False
        

        self.embedder = FrequencyPositionalEmbedding(
            num_freqs=8,
            logspace=True,
            input_dim=3,
            include_pi=False,
        ) 
        # self.embedder.out_dim
        # self.embed_3d_proj = nn.Linear(self.embedder.out_dim, token_dim)

        self.img_token_proj = nn.Linear(embed_dim * 2, token_dim)
        # Initialize img_token_proj with small values to prevent early NaN
        nn.init.trunc_normal_(self.img_token_proj.weight, std=0.02)
        nn.init.zeros_(self.img_token_proj.bias)

        self.num_3d_tokens = num_3d_tokens

        self.use_token_ln = self.cfg.aggregator.params.get('use_token_ln', False)
        self.token_noise_prob = self.cfg.aggregator.params.get('token_noise_prob', 0.0)
        self.token_noise_sigma = self.cfg.aggregator.params.get('token_noise_sigma', 0.0)

        if self.use_token_ln:
            self.token_norm = nn.LayerNorm(self.cfg.aggregator.params.token_dim)
        else:
            self.token_norm = None


        if 'pts3d_head' in cfg:
            self.pts3d_head = eval(cfg.pts3d_head.name)(
            **cfg.pts3d_head.params)
    

    def _embed_3d(self, pts3d: torch.Tensor):
        x = self.embedder(pts3d)  # [B, N, d_point]
        x = self.embed_3d_proj(x)  # [B, N, d]
        return x

    def _sample_features(
        self, x: torch.Tensor, num_tokens: int = 2048, oversample_factor: int = 4, seed: Optional[int] = None
    ):
        """
        Sample points from features of the input point cloud.

        Args:
            x (torch.Tensor): The input point cloud. shape: (B, N, C)
            num_tokens (int, optional): The number of points to sample. Defaults to 2048.
            oversample_factor (int, optional): Factor to oversample before FPS. Defaults to 4.
            seed (Optional[int], optional): The random seed. Defaults to None.
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(
            x.shape[1], num_tokens * oversample_factor, replace=num_tokens * oversample_factor > x.shape[1]
        )
        selected_points = x[:, indices]

        batch_size, num_points, num_channels = selected_points.shape
        flattened_points = selected_points.view(batch_size * num_points, num_channels)
        batch_indices = (
            torch.arange(batch_size).to(x.device).repeat_interleave(num_points)
        )

        # fps sampling
        sampling_ratio = 1.0 / oversample_factor
        sampled_indices = fps(
            flattened_points[:, :3],
            batch_indices,
            ratio=sampling_ratio,
            # random_start=self.training,
            random_start=False,  # for deterministic sampling
        )
        sampled_points = flattened_points[sampled_indices].view(
            batch_size, -1, num_channels
        )

        return sampled_points


    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the pts3d_blocks if not present
        new_ckpt = dict(ckpt)

        return super().load_state_dict(new_ckpt, **kw)
    
    def _encode_scene(self, img_tokens, input_pts3d):
        """Encode input images through VGGT backbone to get visual tokens."""
        B = img_tokens.shape[0]

        if self.cfg.aggregator.name == 'triposg_point':
            x_kv = self.img_token_proj(img_tokens)  # [B, N, d_point]
            sample_x = self._sample_features(input_pts3d, self.num_3d_tokens)
            x_q = self._embed_3d(sample_x)  # [B, N, d_point]
            x = self.aggregator(x_q, x_kv)

            tokens = self.token_proj(x)
            return tokens

        elif self.cfg.aggregator.name == 'triposg_learn':
            # x_kv = self._embed_3d(input_pts3d)  # [B, N, d_point]
            x_kv = self.img_token_proj(img_tokens)

            x_q = self.tokens.unsqueeze(0).expand(B, -1, -1)  # [B, num_tokens, d_point]
            x = self.aggregator(x_q, x_kv)
            tokens = self.token_proj(x)
            return tokens
        elif self.cfg.aggregator.name == 'triposg_hybrid':
            # x_kv = self._embed_3d(input_pts3d)
            x_kv = self.img_token_proj(img_tokens)
            
            # FIX: set input_pts3d to 0
            # input_pts3d = torch.zeros_like(input_pts3d)
            sample_x = self._sample_features(input_pts3d, self.num_3d_tokens)
            
            x_q_point = self._embed_3d(sample_x) 
            x_q_learn = self.tokens.unsqueeze(0).expand(B, -1, -1)  # [B, num_tokens, d_point]

            x_q = torch.cat([x_q_point, x_q_learn], dim=-1)  # [B, num_tokens, d_point * 2]
            x_q = self.token_merge(x_q)  # [B, num_tokens, d_point]

            x = self.aggregator(x_q, x_kv)
            tokens = self.token_proj(x)
            return tokens
        else:
            raise NotImplementedError(f"Aggregator {self.cfg.aggregator.name} not implemented for encoding.")


    def forward_vggt(
        self, images: torch.Tensor,
    ):
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        # aggregated_tokens_list, patch_start_idx = self.vggt_aggregator(images, detach_vit_token=self.detach_vit_token)

        aggregated_tokens_list, aggregated_tokens_3d_list, patch_start_idx, patch_tokens = self.vggt_aggregator(images, detach_vit_token=self.detach_vit_token)

        pts3d = None
        pts3d_conf = None

        return aggregated_tokens_3d_list, pts3d, pts3d_conf


    def _encode(self, images, test=False, **kwargs):
        """Project visual tokens to 3D token space with optional learned token initialization."""
        aggregated_tokens_list, pts3d, pts3d_conf  = self.forward_vggt(images)
        scene_tokens = aggregated_tokens_list[-1]  # [B, S, K, C]
        B, K, C = scene_tokens.shape

        tokens = self.img_token_proj(scene_tokens)

        pointmaps = None

        if self.use_token_ln and self.token_norm is not None:
            tokens = self.token_norm(tokens)

        encode_data = {'pointmaps': pointmaps, 'tokens': tokens}
        return encode_data

    def _apply_cfg_dropout(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Apply classifier-free guidance dropout to tokens during training.
        
        Randomly drops (zeros out) tokens with probability cfg_drop_prob.
        This allows the model to learn both conditional and unconditional generation,
        enabling classifier-free guidance at inference time.
        
        Args:
            tokens (torch.Tensor): Condition tokens with shape [B, N, C]
            
        Returns:
            torch.Tensor: Tokens with some potentially zeroed out (during training)
        """
        if not self.training or self.cfg_drop_prob <= 0.0:
            return tokens
        
        B = tokens.shape[0]
        # Create a mask for each sample in the batch
        # With probability cfg_drop_prob, drop all tokens for that sample
        drop_mask = torch.rand(B, device=tokens.device) < self.cfg_drop_prob

        # Expand mask to match token dimensions [B, 1, 1] for broadcasting
        drop_mask = drop_mask.view(B, 1, 1).expand_as(tokens)
        # Zero out tokens where drop_mask is True
        tokens = tokens.masked_fill(drop_mask, 0.0)
        
        return tokens

    def _encode_with_cfg(self, images: torch.Tensor, cfg_scale: float = 1.0, **kwargs) -> dict:
        """
        Encode images with classifier-free guidance for inference.
        
        This method runs the encoder and applies CFG interpolation using
        the cfg_scale parameter.
        
        Args:
            images (torch.Tensor): Input images with shape [B, S, C, H, W]
            cfg_scale (float): Classifier-free guidance scale. 
                               1.0 = no guidance (conditional only)
                               >1.0 = stronger guidance toward condition
                               0.0 = unconditional only
            **kwargs: Additional arguments
            
        Returns:
            dict: Dictionary containing 'tokens' with CFG-interpolated tokens
        """
        # Get conditional tokens (normal encoding, without CFG dropout)
        # Temporarily ensure we're in eval mode
        original_training = self.training
        self.eval()
        
        aggregated_tokens_list, pts3d, pts3d_conf = self.forward_vggt(images)
        scene_tokens = aggregated_tokens_list[-1]  # [B, S, K, C]
        tokens_cond = self.img_token_proj(scene_tokens)
        
        # Restore training mode
        if original_training:
            self.train()
        
        # If cfg_scale is 1.0, just return conditional tokens (no CFG)
        if cfg_scale == 1.0:
            return {'pointmaps': None, 'tokens': tokens_cond}
        
        # Unconditional tokens are zeros
        tokens_uncond = torch.zeros_like(tokens_cond)
        
        # Apply CFG interpolation: uncond + cfg_scale * (cond - uncond)
        tokens = tokens_uncond + cfg_scale * (tokens_cond - tokens_uncond)
        
        return {'pointmaps': None, 'tokens': tokens}

    def _decode(self, tokens, images, token_mask=None, query_points=None, timestep=None):
        """Decode 3D tokens into point cloud coordinates using flow matching ODE solver."""
        B, S = images.shape[:2]

        predictions = {}

        num_views = torch.ones(B, device=images.device) * S  # [B], assuming each image in the sequence is a different view
        # with torch.cuda.amp.autocast(enabled=False):
            # tokens = tokens.float()
      

        if self.pts3d_head is not None:
            aggregated_tokens_3d_list = [tokens]
            cond_tokens = tokens


            pts3d_xyz = self.pts3d_head(
                aggregated_tokens_3d_list, mask=token_mask,
                query_points=query_points, timestep=timestep, num_views=num_views
            )

            predictions["pts3d_xyz"] = pts3d_xyz
            predictions['pts3d_xyz_rel'] = pts3d_xyz
            predictions["pts3d_rgb"] = pts3d_xyz
            predictions["pts3d_conf"] = torch.ones_like(pts3d_xyz[...,[0]]) 
            predictions["center_xyz"] = pts3d_xyz
            predictions['query_points'] = query_points
            predictions['timestep'] = timestep

        predictions["images"] = images
        predictions["S"] = S

        return predictions


    def forward(
        self,
        images: torch.Tensor,
        token_mask: torch.Tensor = None,
        query_points: torch.Tensor = None,
        timestep: torch.Tensor = None,
        **kwargs
    ):
        """Full forward pass: encode images, then decode to 3D point clouds.

        Args:
            images (torch.Tensor): Input images with shape [B, S, C, H, W].
            token_mask (torch.Tensor, optional): Mask for tokens. Default: None.
            query_points (torch.Tensor, optional): Query points for decoding.
            timestep (torch.Tensor, optional): Timestep for flow matching.

        Returns:
            dict: Predictions including 'pts3d_xyz', 'images', and 'S'.
        """
        encode_data = self._encode(images)

        tokens = encode_data['tokens']

        predictions = self._decode(tokens, images, token_mask=token_mask, query_points=query_points, timestep=timestep)

        return predictions
