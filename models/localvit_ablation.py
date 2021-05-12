import torch
import torch.nn as nn
import math
from functools import partial
from timm.models.vision_transformer import _cfg
from timm.models.layers import DropPath
from timm.models.registry import register_model
from models.localvit import LocalVisionTransformer, Attention, InvertedResidual, LocalityFeedForward, InvertedResidualV4
from timm.models.vision_transformer import Block as Block_vit
from models.localvit import Block as Block_conv_vit
import copy


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, qk_reduce=1, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, conv_expand_ratio=6, num_patches=196,
                 wo_depthwise=False, residual_version='mbv2', use_se='', reduction=4):
        super().__init__()
        self.num_patches = num_patches
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, qk_reduce=qk_reduce,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        # self.conv_down = nn.Sequential(nn.Conv1d(num_patches + 1, num_patches, 1), nn.ReLU6())
        if residual_version == 'mbv2':
            self.conv = InvertedResidual(dim, dim, 1, conv_expand_ratio, wo_depthwise=wo_depthwise)
        elif residual_version == 'mbv3':
            self.conv = LocalityFeedForward(dim, dim, 1, conv_expand_ratio, wo_dp_conv=wo_depthwise,
                                            use_se=use_se, reduction=reduction)
        elif residual_version == 'mbv4':
            self.conv = InvertedResidualV4(dim, dim, 1, conv_expand_ratio, wo_depthwise=wo_depthwise,
                                           use_se=use_se, reduction=reduction)
        else:
            raise NotImplementedError('{} is not implemented.'.format(residual_version))
        # self.conv_up = nn.Sequential(nn.Conv1d(num_patches, num_patches + 1, 1), nn.ReLU6())
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # print(x.shape)
        batch_size, num_token, embed_dim = x.shape                                  # (B, 197, dim)
        patch_size = int(math.sqrt(num_token))

        x = x + self.drop_path(self.attn(self.norm1(x)))                            # (B, 197, dim)
        # x = self.norm2(x)
        cls_token, x = torch.split(x, [1, self.num_patches], dim=1)                 # (B, 1, dim), (B, 196, dim)
        # print(cls_token.shape, x.shape)
        x = x.transpose(1, 2).view(batch_size, embed_dim, patch_size, patch_size)   # (B, dim, 14, 14)
        x = self.conv(x).flatten(2).transpose(1, 2)                                 # (B, 196, dim)
        x = self.drop_path(torch.cat([cls_token, x], dim=1))
        return x


class LocalVisionTransformer_drop_path(LocalVisionTransformer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, conv_expand_ratio=6,
                 wo_depthwise=False, residual_version='mbv2', use_se='', reduction=4):
        super(LocalVisionTransformer_drop_path, self).__init__(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            hybrid_backbone=hybrid_backbone, norm_layer=norm_layer, conv_expand_ratio=conv_expand_ratio,
            wo_depthwise=wo_depthwise, residual_version=residual_version, use_se=use_se, reduction=reduction
        )

        num_patches = self.patch_embed.num_patches

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        kwargs = {
            'conv_expand_ratio': conv_expand_ratio, 'num_patches': num_patches, 'wo_depthwise': wo_depthwise,
            'residual_version': residual_version, 'use_se': use_se, 'reduction': reduction,
            'dim': embed_dim, 'num_heads': num_heads, 'mlp_ratio': mlp_ratio, 'qkv_bias': qkv_bias,
            'qk_scale': qk_scale, 'drop': drop_rate, 'attn_drop': attn_drop_rate, 'norm_layer': norm_layer
        }

        blocks = []
        for i in range(depth):
            blocks.append(Block(drop_path=dpr[i], **kwargs))
        self.blocks = nn.ModuleList(blocks)

        self.apply(self._init_weights)


class LocalVisionTransformer_ablation(LocalVisionTransformer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, conv_expand_ratio=6,
                 wo_depthwise=False, residual_version='mbv2', use_se='', reduction=4,
                 block_ratio=1/3, block_place_low=True):
        super(LocalVisionTransformer_ablation, self).__init__(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            hybrid_backbone=hybrid_backbone, norm_layer=norm_layer, conv_expand_ratio=conv_expand_ratio,
            wo_depthwise=wo_depthwise, residual_version=residual_version, use_se=use_se, reduction=reduction
        )

        num_patches = self.patch_embed.num_patches

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        kwargs = {
            'conv_expand_ratio': conv_expand_ratio, 'num_patches': num_patches, 'wo_depthwise': wo_depthwise,
            'residual_version': residual_version, 'use_se': use_se, 'reduction': reduction
        }
        kwargs1 = {
            'dim': embed_dim, 'num_heads': num_heads, 'mlp_ratio': mlp_ratio, 'qkv_bias': qkv_bias,
            'qk_scale': qk_scale, 'drop': drop_rate, 'attn_drop': attn_drop_rate, 'norm_layer': norm_layer
        }
        kwargs2 = copy.copy(kwargs1)

        if block_place_low:
            kwargs1.update(kwargs)
            block1 = Block_conv_vit
            block2 = Block_vit
        else:
            block_ratio = 1 - block_ratio
            block1 = Block_vit
            block2 = Block_conv_vit
            kwargs2.update(kwargs)

        blocks = []
        for i in range(depth):
            if i <= int(depth * block_ratio) - 1:
                blocks.append(block1(drop_path=dpr[i], **kwargs1))
            else:
                blocks.append(block2(drop_path=dpr[i], **kwargs2))
        self.blocks = nn.ModuleList(blocks)

        self.apply(self._init_weights)


class LocalVisionTransformer_ablation_middle(LocalVisionTransformer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, conv_expand_ratio=6,
                 wo_depthwise=False, residual_version='mbv2', use_se='', reduction=4,
                 block_ratio=1/3, block_place_low=True):
        super(LocalVisionTransformer_ablation_middle, self).__init__(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            hybrid_backbone=hybrid_backbone, norm_layer=norm_layer, conv_expand_ratio=conv_expand_ratio,
            wo_depthwise=wo_depthwise, residual_version=residual_version, use_se=use_se, reduction=reduction
        )

        num_patches = self.patch_embed.num_patches

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        kwargs = {
            'conv_expand_ratio': conv_expand_ratio, 'num_patches': num_patches, 'wo_depthwise': wo_depthwise,
            'residual_version': residual_version, 'use_se': use_se, 'reduction': reduction
        }
        kwargs1 = {
            'dim': embed_dim, 'num_heads': num_heads, 'mlp_ratio': mlp_ratio, 'qkv_bias': qkv_bias,
            'qk_scale': qk_scale, 'drop': drop_rate, 'attn_drop': attn_drop_rate, 'norm_layer': norm_layer
        }
        kwargs2 = copy.copy(kwargs1)

        block1 = Block_vit
        block2 = Block_conv_vit
        kwargs2.update(kwargs)

        blocks = []
        for i in range(depth):
            if i <= int(depth * block_ratio) - 1 or i > int(depth * (1 - block_ratio)) - 1:
                blocks.append(block1(drop_path=dpr[i], **kwargs1))
            else:
                blocks.append(block2(drop_path=dpr[i], **kwargs2))
        self.blocks = nn.ModuleList(blocks)

        self.apply(self._init_weights)


@register_model
def conv_vit_tiny_patch16_ex4_v3_r192_low13(pretrained=False, **kwargs):
    model = LocalVisionTransformer_ablation(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True, conv_expand_ratio=4,
        residual_version='mbv3', use_se='se', reduction=192, block_ratio=1/3, block_place_low=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    # if pretrained:
    #     checkpoint = torch.hub.load_state_dict_from_url(
    #         url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
    #         map_location="cpu", check_hash=True
    #     )
    #     model.load_state_dict(checkpoint["model"])
    return model

@register_model
def conv_vit_tiny_patch16_ex4_v3_r192_low23(pretrained=False, **kwargs):
    model = LocalVisionTransformer_ablation(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True, conv_expand_ratio=4,
        residual_version='mbv3', use_se='se', reduction=192, block_ratio=2/3, block_place_low=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    # if pretrained:
    #     checkpoint = torch.hub.load_state_dict_from_url(
    #         url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
    #         map_location="cpu", check_hash=True
    #     )
    #     model.load_state_dict(checkpoint["model"])
    return model

@register_model
def conv_vit_tiny_patch16_ex4_v3_r192_high13(pretrained=False, **kwargs):
    model = LocalVisionTransformer_ablation(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True, conv_expand_ratio=4,
        residual_version='mbv3', use_se='se', reduction=192, block_ratio=1/3, block_place_low=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    # if pretrained:
    #     checkpoint = torch.hub.load_state_dict_from_url(
    #         url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
    #         map_location="cpu", check_hash=True
    #     )
    #     model.load_state_dict(checkpoint["model"])
    return model

@register_model
def conv_vit_tiny_patch16_ex4_v3_r192_mid13(pretrained=False, **kwargs):
    model = LocalVisionTransformer_ablation_middle(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True, conv_expand_ratio=4,
        residual_version='mbv3', use_se='se', reduction=192, block_ratio=1/3, block_place_low=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    # if pretrained:
    #     checkpoint = torch.hub.load_state_dict_from_url(
    #         url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
    #         map_location="cpu", check_hash=True
    #     )
    #     model.load_state_dict(checkpoint["model"])
    return model


# reduction = 192
@register_model
def conv_vit_tiny_patch16_ex4_v3_r192_drop_path(pretrained=False, **kwargs):
    model = LocalVisionTransformer_drop_path(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True, conv_expand_ratio=4,
        residual_version='mbv3', use_se='se', reduction=192,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    # if pretrained:
    #     checkpoint = torch.hub.load_state_dict_from_url(
    #         url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
    #         map_location="cpu", check_hash=True
    #     )
    #     model.load_state_dict(checkpoint["model"])
    return model