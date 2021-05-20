"""
Author: Yawei Li
Email: yawei.li@vision.ee.ethz.ch

Introducing locality mechanism to "Tokens-to-token vit: Training vision transformers from scratch on imagenet".
"""
import torch
import math
import torch.nn as nn
import numpy as np
from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import DropPath
from models.token_transformer import Token_transformer_local
from models.token_performer import Token_performer_local
from models.t2t_vit import T2T_module, T2T_ViT
from models.t2t_vit_block import Attention
from models.localvit import LocalityFeedForward

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'localvit_T2t_conv7': _cfg(),
}


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_patches=196, reduction=4):
        super().__init__()
        self.num_patches = num_patches
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.conv = LocalityFeedForward(dim, dim, 1, mlp_ratio, reduction=reduction)

    def forward(self, x):
        # print(x.shape)
        batch_size, num_token, embed_dim = x.shape                                  # (B, 197, dim)
        patch_size = int(math.sqrt(num_token))

        x = x + self.drop_path(self.attn(self.norm1(x)))                            # (B, 197, dim)
        cls_token, x = torch.split(x, [1, self.num_patches], dim=1)                 # (B, 1, dim), (B, 196, dim)
        # print(cls_token.shape, x.shape)
        x = x.transpose(1, 2).view(batch_size, embed_dim, patch_size, patch_size)   # (B, dim, 14, 14)
        x = self.conv(x).flatten(2).transpose(1, 2)                                 # (B, 196, dim)
        x = torch.cat([cls_token, x], dim=1)
        return x


class T2T_module_local(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_transformer_local(dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1)
            self.attention2 = Token_transformer_local(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'performer':
            print('adopt performer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_performer_local(dim=in_chans * 7 * 7, in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = Token_performer_local(dim=token_dim * 3 * 3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'convolution':  # just for comparison with conolution, not our model
            # for this tokens type, you need change forward as three convolution operation
            print('adopt convolution layers for tokens-to-token')
            self.soft_split0 = nn.Conv2d(3, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))  # the 1st convolution
            self.soft_split1 = nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 2nd convolution
            self.project = nn.Conv2d(token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 3rd convolution

        self.num_patches = (img_size // (4 * 2 * 2)) * (img_size // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)

        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)
        B, new_HW, C = x.shape
        x = x.transpose(1,2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)

        return x

class LocalViT_T2T(T2T_ViT):
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, token_dim=64,  reduction=4):
        super().__init__(img_size, tokens_type, in_chans, num_classes, embed_dim, depth, num_heads, mlp_ratio,
                         qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, token_dim)

        self.tokens_to_token = T2T_module_local(
                img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim, token_dim=token_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                reduction=reduction
            )
            for i in range(depth)])

        self.apply(self._init_weights)


@register_model
def localvit_T2t_conv7(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = LocalViT_T2T(tokens_type='performer', embed_dim=256, depth=7, num_heads=4, mlp_ratio=2., reduction=128,
                    **kwargs)
    model.default_cfg = default_cfgs['localvit_T2t_conv7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model




