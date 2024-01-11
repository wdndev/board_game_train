
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools

from nn_utils import DropPath

def to_2tuple(x):
    return (x, x)

class Attention4D(nn.Module):
    def __init__(self, dim=384, key_dim=32, num_heads=8, attn_ratio=4, resolution=7, act_layer=nn.ReLU, stride=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = key_dim * num_heads

        if stride is not None:
            self.resolution = math.ceil(resolution / stride)
            self.stride_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim),
                                             nn.BatchNorm2d(dim), )
            self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
        else:
            self.resolution = resolution
            self.stride_conv = None
            self.upsample = None

        self.N = self.resolution ** 2
        self.N2 = self.N
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + self.nh_kd * 2

        self.q = nn.Sequential(
            nn.Conv2d(dim, self.nh_kd, 1),
            nn.BatchNorm2d(self.nh_kd)
        )
        self.k = nn.Sequential(
            nn.Conv2d(dim, self.nh_kd, 1),
            nn.BatchNorm2d(self.nh_kd)
        )
        self.v = nn.Sequential(
            nn.Conv2d(dim, self.dh, 1),
            nn.BatchNorm2d(self.dh)
        )
        self.local_v = nn.Sequential(
            nn.Conv2d(self.dh, self.dh, 3, padding=1, groups=self.dh),
            nn.BatchNorm2d(self.dh)
        )
        self.talking_head1 = nn.Conv2d(self.num_heads, self.num_heads, 1)
        self.talking_head2 = nn.Conv2d(self.num_heads, self.num_heads, 1)

        self.proj = nn.Sequential(
            act_layer(),
            nn.Conv2d(self.dh, dim, 1),
            nn.BatchNorm2d(dim)
        )

        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        N = len(points)
        self.N = N
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        # self.attention_biases = self.create_parameter((len(attention_offsets), num_heads), default_initializer=nn.initializer.Constant(0.0))
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.attention_bias_idxs = idxs

    def forward(self, x):
        B, C, H, W = x.shape

        if self.stride_conv is not None:
            x = self.stride_conv(x)

        q = self.q(x).flatten(2).reshape((B, self.num_heads, -1, self.N)).transpose((0, 1, 3, 2))
        k = self.k(x).flatten(2).reshape((B, self.num_heads, -1, self.N)).transpose((0, 1, 3, 2))
        v = self.v(x)
        v_local = self.local_v(v)
        v = v.flatten(2).reshape((B, self.num_heads, -1, self.N)).transpose((0, 1, 3, 2))

        attn = q @ k.transpose((0, 1, 3, 2)) * self.scale

        attn = attn + self.attention_biases[self.attention_bias_idxs].transpose((1, 0)).reshape((1, self.num_heads, self.N, self.N))

        attn = self.talking_head1(attn)
        attn = F.softmax(attn, axis=-1)
        attn = self.talking_head2(attn)

        x = attn @ v

        out = x.transpose((0, 1, 3, 2)).reshape((B, self.dh, self.resolution, self.resolution)) + v_local
        if self.upsample is not None:
            out = self.upsample(out)

        out = self.proj(out)
        return out

def stem(in_chs, out_chs, act_layer=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs // 2),
        act_layer(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs),
        act_layer())


class LGQuery(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.pool = nn.AvgPool2d(1, 2, 0)
        self.local = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2, padding=1, groups=in_dim)
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, x):
        local_q = self.local(x)
        pool_q = self.pool(x)
        q = local_q + pool_q
        q = self.proj(q)
        return q

class Attention4DDownsample(nn.Module):
    def __init__(self, dim=384, key_dim=16, num_heads=8, attn_ratio=4,
                 resolution=7, out_dim=None, act_layer=nn.GELU):
        super().__init__()

        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads

        self.resolution = resolution

        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        if out_dim is not None:
            self.out_dim = out_dim
        else:
            self.out_dim = dim
        self.resolution2 = math.ceil(self.resolution / 2)
        self.q = LGQuery(dim, self.nh_kd)

        self.N = self.resolution ** 2
        self.N2 = self.resolution2 ** 2

        self.k = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.key_dim, 1),
            nn.BatchNorm2d(self.num_heads * self.key_dim)
        )
        self.v = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.d, 1),
            nn.BatchNorm2d(self.num_heads * self.d),
        )
        self.v_local = nn.Sequential(
            nn.Conv2d(self.dh, self.dh, kernel_size=3, stride=2, padding=1, groups=self.num_heads * self.d),
            nn.BatchNorm2d(self.num_heads * self.d)
        )

        self.proj = nn.Sequential(
            act_layer(),
            nn.Conv2d(self.dh, self.out_dim, 1),
            nn.BatchNorm2d(self.out_dim)
        )

        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        points_ = list(itertools.product(range(self.resolution2), range(self.resolution2)))
        N = len(points)
        N_ = len(points_)
        self.N = N
        self.N_ = N_
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * math.ceil(self.resolution / self.resolution2) - p2[0] + (size - 1) / 2),
                    abs(p1[1] * math.ceil(self.resolution / self.resolution2) - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        # self.attention_biases = self.create_parameter((len(attention_offsets), num_heads), default_initializer=nn.initializer.Constant(0.0))
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.attention_bias_idxs = idxs

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).flatten(2).reshape((B, self.num_heads, -1, self.N2)).transpose((0, 1, 3, 2))
        k = self.k(x).flatten(2).reshape((B, self.num_heads, -1, self.N))
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape((B, self.num_heads, -1, self.N)).transpose((0, 1, 3, 2))

        attn = q @ k * self.scale

        attn = attn + self.attention_biases[self.attention_bias_idxs].transpose((1, 0)).reshape((1, self.num_heads, self.N_, self.N))

        attn = F.softmax(attn, axis=-1)
        x = (attn @ v).transpose((0, 1, 3, 2))

        out = x.reshape((B, self.dh, self.resolution2, self.resolution2)) + v_local

        out = self.proj(out)
        return out


class Embedding(nn.Module):
    def __init__(self, patch_size=3, stride=2, padding=1,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d,
                 light=False, asub=False, resolution=None, act_layer=nn.ReLU, attn_block=Attention4DDownsample):
        super().__init__()
        self.light = light
        self.asub = asub

        if self.light:
            self.new_proj = nn.Sequential(
                nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=2, padding=1, groups=in_chans),
                nn.BatchNorm2d(in_chans),
                nn.Hardswish(),
                nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(embed_dim),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=2, padding=0),
                nn.BatchNorm2d(embed_dim)
            )
        elif self.asub:
            self.attn = attn_block(dim=in_chans, out_dim=embed_dim,
                                   resolution=resolution, act_layer=act_layer)
            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                                  stride=stride, padding=padding)
            self.bn = norm_layer(embed_dim) if norm_layer else nn.Identity()
        else:
            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                                  stride=stride, padding=padding)
            self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if self.light:
            out = self.new_proj(x) + self.skip(x)
        elif self.asub:
            out_conv = self.conv(x)
            out_conv = self.bn(out_conv)
            out = self.attn(x) + out_conv
        else:
            x = self.proj(x)
            out = self.norm(x)
        return out
    
class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., mid_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mid_conv = mid_conv
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

        if self.mid_conv:
            self.mid = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features)
            self.mid_norm = nn.BatchNorm2d(hidden_features)

        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.norm2 = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)

        if self.mid_conv:
            x_mid = self.mid(x)
            x_mid = self.mid_norm(x_mid)
            x = self.act(x_mid)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.norm2(x)

        x = self.drop(x)
        return x

class AttnFFN(nn.Module):
    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 resolution=7, stride=None):

        super().__init__()

        self.token_mixer = Attention4D(dim, resolution=resolution, act_layer=act_layer, stride=stride)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, mid_conv=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            # self.layer_scale_1 = self.create_parameter((1, dim, 1, 1), default_initializer=nn.initializer.Constant(layer_scale_init_value))
            # self.layer_scale_2 = self.create_parameter((1, dim, 1, 1), default_initializer=nn.initializer.Constant(layer_scale_init_value))

            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
            self.layer_scale_2 = nn.Parameter( layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(x))
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))

        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x

class FFN(nn.Module):
    def __init__(self, dim, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, mid_conv=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            # self.layer_scale_2 = self.create_parameter((1, dim, 1, 1), default_initializer=nn.initializer.Constant(layer_scale_init_value))
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.mlp(x))
        return x
    

def eformer_block(dim, index, layers,
                  pool_size=3, mlp_ratio=4.,
                  act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                  drop_rate=.0, drop_path_rate=0.,
                  use_layer_scale=True, layer_scale_init_value=1e-5, vit_num=1, resolution=7, e_ratios=None):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
                block_idx + sum(layers[:index])) / (sum(layers) - 1)
        mlp_ratio = e_ratios[str(index)][block_idx]
        if index >= 2 and block_idx > layers[index] - 1 - vit_num:
            if index == 2:
                stride = 2
            else:
                stride = None
            blocks.append(AttnFFN(
                dim, mlp_ratio=mlp_ratio,
                act_layer=act_layer, norm_layer=norm_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                resolution=resolution,
                stride=stride,
            ))
        else:
            blocks.append(FFN(
                dim, pool_size=pool_size, mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))
    blocks = nn.Sequential(*blocks)
    return blocks

class EfficientFormerV2(nn.Module):
    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=4, downsamples=None,
                 pool_size=3,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.GELU,
                 num_classes=1000,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 vit_num=0,
                 distillation=True,
                 resolution=224,
                 e_ratios=None):
        super().__init__()

        self.num_classes = num_classes

        self.patch_embed = stem(3, embed_dims[0], act_layer=act_layer)

        network = []
        for i in range(len(layers)):
            stage = eformer_block(embed_dims[i], i, layers,
                                  pool_size=pool_size, mlp_ratio=mlp_ratios,
                                  act_layer=act_layer, norm_layer=norm_layer,
                                  drop_rate=drop_rate,
                                  drop_path_rate=drop_path_rate,
                                  use_layer_scale=use_layer_scale,
                                  layer_scale_init_value=layer_scale_init_value,
                                  resolution=math.ceil(resolution / (2 ** (i + 2))),
                                  vit_num=vit_num,
                                  e_ratios=e_ratios)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                if i >= 2:
                    asub = True
                else:
                    asub = False
                network.append(
                    Embedding(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1],
                        resolution=math.ceil(resolution / (2 ** (i + 2))),
                        asub=asub,
                        act_layer=act_layer, norm_layer=norm_layer,
                    )
                )

        self.network = nn.ModuleList(network)

        # Classifier head
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(
            embed_dims[-1], num_classes) if num_classes > 0 \
            else nn.Identity()
        self.dist = distillation
        if self.dist:
            self.dist_head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()

    def forward_tokens(self, x):

        for idx, block in enumerate(self.network):
            x = block(x)
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_tokens(x)

        x = self.norm(x)
        if self.dist:
            cls_out = self.head(x.flatten(2).mean(-1)), self.dist_head(x.flatten(2).mean(-1))
            if not self.training:
                cls_out = (cls_out[0] + cls_out[1]) / 2
        else:
            cls_out = self.head(x.flatten(2).mean(-1))
        # for image classification
        return cls_out

expansion_ratios_S0 = {
    '0': [4, 4],
    '1': [4, 4],
    '2': [4, 3, 3, 3, 4, 4],
    '3': [4, 3, 3, 4],
}

def efficientformerv2_s0(pretrained=False):
    model = EfficientFormerV2(
        layers=[2, 2, 6, 4],
        embed_dims=[32, 48, 96, 176],
        downsamples=[True, True, True, True],
        vit_num=2,
        drop_path_rate=0.0,
        num_classes=10,
        distillation=False,
        e_ratios=expansion_ratios_S0)
    return model

def print_model_parameters(model):
    """ 打印模型各个层参数
    """
    param_sum = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_sum += param.numel()
            print(f"Layer: {name}, Parameters: {param.numel()}")
    print(f"Total of parameters: {param_sum}")

if __name__ == '__main__':
    tensor = torch.randn(5, 3, 224, 224)
    print(tensor.shape)
    model = efficientformerv2_s0()

    x = model(tensor)
    print(x.shape)

