# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# Main encoder/decoder blocks
# --------------------------------------------------------
# References: 
# timm
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/helpers.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/mlp.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/patch_embed.py


import torch
import torch.nn as nn 

from itertools import repeat
import collections.abc


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):

    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope 

    def forward(self, x, xpos):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1,3)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        # q,k,v = qkv.unbind(2)  # make torchscript happy (cannot use tensor as tuple)
               
        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)
               
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, rope=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, xpos):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

class TimeEmbedBlock(Block):
    def __init__(self, dim, num_heads, emb_channels=None, use_adaLN=False, skip_t_emb=False, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, rope=None):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, drop,     attn_drop, drop_path, act_layer, norm_layer, rope)

        self.skip_t_emb = skip_t_emb
        self.use_adaLN = use_adaLN
        self.emb_out_channels = (
            6 * dim if use_adaLN else dim
        )

        if emb_channels is None:
            emb_channels = dim

        if self.skip_t_emb:
            self.emb_layers = None
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    emb_channels,
                    self.emb_out_channels,
                    bias=True,
                ),
            )
        
    def forward(self, x, xpos, emb):
        """
        Args:
        x: tensor, shape (B, N, C)
        xpos: tensor, shape (B, N, 2)
        emb: tensor, shape (B, N, emb_channels)
        """
        if self.skip_t_emb:
            emb_out = torch.zeros_like(x)
        else:
            emb_out = self.emb_layers(emb).type(x.dtype)    # B, C
  
        if self.use_adaLN:
            # scale, shift = emb_out.chunk(2, dim=-1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb_out.chunk(6, dim=1)
            # h = h * (1 + scale) + shift
            # h = self.norm1(x)
            # h = h * (1 + scale_msa) + shift_msa
            # h = self.attn(h, xpos)
            # h = h * gate_msa
            # x = x + self.drop_path(h)

            # h = self.norm2(x)
            # h = h * (1 + scale_mlp) + shift_mlp
            # h = self.mlp(h)
            # h = h * gate_mlp
            # x = x + self.drop_path(h)
            x = x + gate_msa.unsqueeze(1) * self.drop_path(self.attn(modulate(self.norm1(x), shift_msa, scale_msa), xpos))

            x = x + gate_mlp.unsqueeze(1) * self.drop_path(self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
            
        else:
            h = self.norm1(x)
            h = h + emb_out.unsqueeze(1)

            x = x + self.drop_path(self.attn(h, xpos))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CrossAttention(nn.Module):
    
    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.rope = rope
        
    def forward(self, query, key, value, qpos, kpos):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]
        
        q = self.projq(query).reshape(B,Nq,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        k = self.projk(key).reshape(B,Nk,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        v = self.projv(value).reshape(B,Nv,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        
        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)
            
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DecoderBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x, y, xpos, ypos):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        y_ = self.norm_y(y)
        x = x + self.drop_path(self.cross_attn(self.norm2(x), y_, y_, xpos, ypos))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x, y
    
class TimeEmbedDecoderBlock(DecoderBlock):
    def __init__(self, dim, num_heads, emb_channels=None, use_adaLN=False, skip_t_emb=False, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=None):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer, norm_mem, rope)

        self.skip_t_emb = skip_t_emb
        self.use_adaLN = use_adaLN
        self.emb_out_channels = (
            6 * dim if use_adaLN else dim
        )

        if emb_channels is None:
            emb_channels = dim

        if self.skip_t_emb:
            self.emb_layers = None
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    emb_channels,
                    self.emb_out_channels,
                    bias=True,
                ),
            )
        
    def forward(self, x, y, xpos, ypos, emb):
        """
        Args:
        x: tensor, shape (B, N, C)
        y: tensor, shape (B, N, C)
        xpos: tensor, shape (B, N, 2)
        ypos: tensor, shape (B, N, 2)
        emb: tensor, shape (B, N, emb_channels)
        """
        if self.skip_t_emb:
            emb_out = torch.zeros_like(x)
        else:
            emb_out = self.emb_layers(emb).type(x.dtype)    # B, C


        if self.use_adaLN:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb_out.chunk(6, dim=-1)
            # h = self.norm1(x)
            # h = h * (1 + scale_msa) + shift_msa
            # h = self.attn(h, xpos)
            # h = h * gate_msa
            # x = x + self.drop_path(h)
            x = x + gate_msa.unsqueeze(1) * self.drop_path(self.attn(modulate(self.norm1(x), shift_msa, scale_msa), xpos))

            y_ = self.norm_y(y)
            x = x + self.drop_path(self.cross_attn(self.norm2(x), y_, y_, xpos, ypos))

            x = x + gate_mlp.unsqueeze(1) * self.drop_path(self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp)))
        else:
            h = self.norm1(x)
            h = h + emb_out.unsqueeze(1)

            x = x + self.drop_path(self.attn(h, xpos))
            y_ = self.norm_y(y)
            x = x + self.drop_path(self.cross_attn(self.norm2(x), y_, y_, xpos, ypos))
            x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x, y

        
# patch embedding
class PositionGetter(object):
    """ return positions of patches """

    def __init__(self):
        self.cache_positions = {}
        
    def __call__(self, b, h, w, device):
        if not (h,w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h,w] = torch.cartesian_prod(y, x) # (h, w, 2)
        pos = self.cache_positions[h,w].view(1, h*w, 2).expand(b, -1, 2).clone()
        return pos

class PatchEmbed(nn.Module):
    """ just adding _init_weights + position getter compared to timm.models.layers.patch_embed.PatchEmbed"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
        self.position_getter = PositionGetter()
        
    def forward(self, x):
        B, C, H, W = x.shape
        torch._assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        torch._assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, pos
        
    def _init_weights(self):
        w = self.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1])) 

