"""
Author: Yawei Li
Email: yawei.li@vision.ee.ethz.ch

Introducing locality mechanism to "DeiT: Data-efficient Image Transformers".
"""
import torch
import torch.nn as nn
import math
from functools import partial
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import DropPath
from timm.models.registry import register_model


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1, sigmoid=True):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        if sigmoid:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = h_sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class LocalityFeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, stride, expand_ratio=4., act='hs+se', reduction=4,
                 wo_dp_conv=False, dp_first=False):
        """
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
                    hs+se: h_swish and SE module
                    hs+eca: h_swish and ECA module
                    hs+ecah: h_swish and ECA module. Compared with eca, h_sigmoid is used.
        :param reduction: reduction rate in SE module.
        :param wo_dp_conv: without depth-wise convolution.
        :param dp_first: place depth-wise convolution as the first layer.
        """
        super(LocalityFeedForward, self).__init__()
        hidden_dim = int(in_dim * expand_ratio)
        kernel_size = 3

        layers = []
        # the first linear layer is replaced by 1x1 convolution.
        layers.extend([
            nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)])

        # the depth-wise convolution between the two linear layers
        if not wo_dp_conv:
            dp = [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)
            ]
            if dp_first:
                layers = dp + layers
            else:
                layers.extend(dp)

        if act.find('+') >= 0:
            attn = act.split('+')[1]
            if attn == 'se':
                layers.append(SELayer(hidden_dim, reduction=reduction))
            elif attn.find('eca') >= 0:
                layers.append(ECALayer(hidden_dim, sigmoid=attn == 'eca'))
            else:
                raise NotImplementedError('Activation type {} is not implemented'.format(act))

        # the second linear layer is replaced by 1x1 convolution.
        layers.extend([
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_dim)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = x + self.conv(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, qk_reduce=1, attn_drop=0., proj_drop=0.):
        """
        :param dim:
        :param num_heads:
        :param qkv_bias:
        :param qk_scale:
        :param qk_reduce: reduce the output dimension for QK projection
        :param attn_drop:
        :param proj_drop:
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qk_reduce = qk_reduce
        self.dim = dim
        self.qk_dim = int(dim / self.qk_reduce)

        self.qkv = nn.Linear(dim, int(dim * (1 + 1 / qk_reduce * 2)), bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.qk_reduce == 1:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        else:
            q, k, v = torch.split(self.qkv(x), [self.qk_dim, self.qk_dim, self.dim], dim=-1)
            q = q.reshape(B, N, self.num_heads, -1).transpose(1, 2)
            k = k.reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = v.reshape(B, N, self.num_heads, -1).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, qk_reduce=1, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, act='hs+se', reduction=4, wo_dp_conv=False, dp_first=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, qk_reduce=qk_reduce,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # The MLP is replaced by the conv layers.
        self.conv = LocalityFeedForward(dim, dim, 1, mlp_ratio, act, reduction, wo_dp_conv, dp_first)

    def forward(self, x):
        batch_size, num_token, embed_dim = x.shape                                  # (B, 197, dim)
        patch_size = int(math.sqrt(num_token))

        x = x + self.drop_path(self.attn(self.norm1(x)))                            # (B, 197, dim)
        # Split the class token and the image token.
        cls_token, x = torch.split(x, [1, num_token - 1], dim=1)                    # (B, 1, dim), (B, 196, dim)
        # Reshape and update the image token.
        x = x.transpose(1, 2).view(batch_size, embed_dim, patch_size, patch_size)   # (B, dim, 14, 14)
        x = self.conv(x).flatten(2).transpose(1, 2)                                 # (B, 196, dim)
        # Concatenate the class token and the newly computed image token.
        x = torch.cat([cls_token, x], dim=1)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        #########################################
        # Origianl implementation
        # self.norm2 = norm_layer(dim)
        #         mlp_hidden_dim = int(dim * mlp_ratio)
        #         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        #########################################

        # Replace the MLP layer by LocalityFeedForward.
        self.conv = LocalityFeedForward(dim, dim, 1, mlp_ratio, act='hs+se', reduction=dim//4)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        #########################################
        # Origianl implementation
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        #########################################

        # Change the computation accordingly in three steps.
        batch_size, num_token, embed_dim = x.shape
        patch_size = int(math.sqrt(num_token))
        # 1. Split the class token and the image token.
        cls_token, x = torch.split(x, [1, embed_dim - 1], dim=1)
        # 2. Reshape and update the image token.
        x = x.transpose(1, 2).view(batch_size, embed_dim, patch_size, patch_size)
        x = self.conv(x).flatten(2).transpose(1, 2)
        # 3. Concatenate the class token and the newly computed image token.
        x = torch.cat([cls_token, x], dim=1)
        return x


class LocalVisionTransformer(VisionTransformer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm,
                 act=3, reduction=4, wo_dp_conv=False, dp_first=False):
        # print(hybrid_backbone is None)
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
                         num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                         drop_path_rate, hybrid_backbone, norm_layer)

        # parse act
        if act == 1:
            act = 'relu6'
        elif act == 2:
            act = 'hs'
        elif act == 3:
            act = 'hs+se'
        elif act == 4:
            act = 'hs+eca'
        else:
            act = 'hs+ecah'

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act=act, reduction=reduction, wo_dp_conv=wo_dp_conv, dp_first=dp_first
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)



@register_model
def localvit_tiny_mlp6_act1(pretrained=False, **kwargs):
    model = LocalVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=6, qkv_bias=True, act=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# reduction = 4
@register_model
def localvit_tiny_mlp4_act3_r4(pretrained=False, **kwargs):
    model = LocalVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True, act=3, reduction=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# reduction = 192
@register_model
def localvit_tiny_mlp4_act3_r192(pretrained=False, **kwargs):
    model = LocalVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True, act=3, reduction=192,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def localvit_small_mlp4_act3_r384(pretrained=False, **kwargs):
    model = LocalVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True, act=3, reduction=384,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
