"""
Author: Yawei Li
Email: yawei.li@vision.ee.ethz.ch

Introducing locality mechanism to "Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions".
"""

import torch
import torch.nn as nn
from functools import partial

from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from models.localvit import LocalityFeedForward
from models.pvt import Attention, PyramidVisionTransformer
import math

__all__ = [
    'localvit_pvt_tiny'
]


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.sr_ratio = sr_ratio
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.conv = LocalityFeedForward(dim, dim, 1, mlp_ratio, reduction=dim)

    def forward(self, x, H, W):
        batch_size, num_token, embed_dim = x.shape  # (B, 197, dim)
        patch_size = int(math.sqrt(num_token))

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.sr_ratio == 1:
            cls_token, x = torch.split(x, [1, num_token - 1], dim=1)  # (B, 1, dim), (B, 196, dim)
            # print(cls_token.shape, x.shape)
            x = x.transpose(1, 2).view(batch_size, embed_dim, patch_size, patch_size)  # (B, dim, 14, 14)
            x = self.conv(x).flatten(2).transpose(1, 2)  # (B, 196, dim)
            x = torch.cat([cls_token, x], dim=1)
        else:
            x = x.transpose(1, 2).view(batch_size, embed_dim, patch_size, patch_size)  # (B, dim, 14, 14)
            x = self.conv(x).flatten(2).transpose(1, 2)  # (B, 196, dim)
        return x


class LocalViT_PVT(PyramidVisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dims,
                 num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate,
                 attn_drop_rate, drop_path_rate, norm_layer,
                 depths, sr_ratios)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])

        # init weights
        self.apply(self._init_weights)


@register_model
def localvit_pvt_tiny(pretrained=False, **kwargs):
    model = LocalViT_PVT(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model
