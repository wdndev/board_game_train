# -*- coding: utf-8 -*-
#  @file        - chess_transformer.py
#  @author      - dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - 将原始transformer应用于棋类
#  @version     - 0.0
#  @date        - 2023.12.20
#  @copyright   - Copyright (c) 2023 

""" 将棋盘使用卷积打成patch后, 每个patch编码后送入transformer中
"""

from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.nn_utils import DropPath

class PolicyHead(nn.Module):
    """ 策略头
    """
    def __init__(self, n_features = 384, board_size = 19):
        super().__init__()
        self.board_size = board_size
        # self.fc = nn.Linear(n_features, board_size*board_size)
        self.fc = nn.Sequential(
            nn.Linear(n_features, 2*board_size*board_size),
            nn.ReLU(),
            nn.Linear( 2*board_size*board_size, board_size*board_size),
        )

    def forward(self, x):
        x = self.fc(x)
        return F.softmax(x, dim=1)
    

class ValueHead(nn.Module):
    """ 价值头
    """
    def __init__(self, n_features = 384, board_size = 19):
        super().__init__()
        self.board_size = board_size
        self.fc = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class PatchEmbed(nn.Module):
    """ 棋盘 to Embedding
    """
    def __init__(self, board_size=19, patch_size=3, in_c=4, embed_dim=384, norm_layer=None) -> None:
        super().__init__()
        self.board_size = board_size
        self.path_size = patch_size
        # (((board_size - patch_size) + 1 - patch_size) + 1 - patch_size) + 1
        # self.num_patches = (board_size - 3 * patch_size + 3) * (board_size - 3 * patch_size + 3)
        self.num_patches = ((board_size - patch_size) + 1) * ((board_size - patch_size) + 1)

        # self.conv_1 = nn.Conv2d(in_c, 64, kernel_size=patch_size, stride=1)
        # self.conv_2 = nn.Conv2d(64, 128, kernel_size=patch_size, stride=1)
        # self.conv_3 = nn.Conv2d(128, embed_dim, kernel_size=patch_size, stride=1)
        self.conv = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=1)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.board_size and W == self.board_size, \
            f"Input Board size ({H}*{W}) doesn't match model ({self.board_size}*{self.board_size})."
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    """ 多头注意力机制
    """
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=4,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class Mlp(nn.Module):
    """
    MLP as used in Chess Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(TransformerBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ChessTransformer(nn.Module):
    def __init__(self, board_size=19, patch_size=3, in_c=4, embed_dim=384, 
                 depth=4, num_heads=4, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., 
                 norm_layer=None, act_layer=None):
        """
        Args:
            board_size (int, tuple): input board size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(ChessTransformer, self).__init__()
        self.model_name = "chess_transformer"
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = PatchEmbed(board_size=board_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # 分类头
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
 
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        self.policy_head = PolicyHead(n_features=self.num_features, board_size=board_size)
        self.value_head = ValueHead(n_features=self.num_features, board_size=board_size)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 169, 384]
        # [1, 1, 384] -> [B, 1, 384]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_token, x), dim=1)  # [B, 170, 384]

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        # print(x.shape)

        return self.pre_logits(x[:, 0])

    def forward(self, x):
        x = self.forward_features(x)
        # print(x.shape)
        p_hat = self.policy_head(x)
        v_hat = self.value_head(x)

        return p_hat, v_hat


def chess_transformer_88_128(in_chans=4, board_size=19):
    """ Total of parameters: 1999454
    """
    model = ChessTransformer(in_c=in_chans,
                             board_size=board_size,
                             embed_dim=128,
                             depth=8,
                             num_heads=8,
                             representation_size=None)
    model.model_name = "chess_transformer_88_128"
    return model

def chess_transformer_88_112(in_chans=4, board_size=19):
    """ Total of parameters: 1610238
    """
    model = ChessTransformer(in_c=in_chans,
                             board_size=board_size,
                             embed_dim=112,
                             depth=8,
                             num_heads=8,
                             representation_size=None)
    model.model_name = "chess_transformer_88_112"
    return model

def chess_transformer_88_96(in_chans=4, board_size=19):
    """ Total of parameters: 1270174
    """
    model = ChessTransformer(in_c=in_chans,
                             board_size=board_size,
                             embed_dim=96,
                             depth=8,
                             num_heads=8,
                             representation_size=None)
    model.model_name = "chess_transformer_88_96"
    return model

def chess_transformer_66_96(in_chans=4, board_size=19):
    """ Total of parameters: 1046302
    """
    model = ChessTransformer(in_c=in_chans,
                             board_size=board_size,
                             embed_dim=96,
                             depth=6,
                             num_heads=6,
                             representation_size=None)
    model.model_name = "chess_transformer_66_96"
    return model

def chess_transformer_88_64(in_chans=4, board_size=19):
    """ Total of parameters: 737502
    """
    model = ChessTransformer(in_c=in_chans,
                             board_size=board_size,
                             embed_dim=64,
                             depth=8,
                             num_heads=8,
                             representation_size=None)
    model.model_name = "chess_transformer_88_64"
    return model

def chess_transformer_44_64(in_chans=4, board_size=19):
    """ Total of parameters: 537566
    """
    model = ChessTransformer(in_c=in_chans,
                             board_size=board_size,
                             embed_dim=64,
                             depth=4,
                             num_heads=4,
                             representation_size=None)
    model.model_name = "chess_transformer_44_64"
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

if __name__ == "__main__":
    # 随机初始化一个 2x4x19x19 的张量
    tensor = torch.randn(5, 18, 19, 19)

    patch = PatchEmbed(in_c=18, embed_dim=112)
    x = patch(tensor)
    print(x.shape)
    print(patch.num_patches)

    model = chess_transformer_88_112(in_chans=18, board_size=19)

    p_hat, v_hat = model(tensor)
    print(p_hat.shape)
    print(v_hat.shape)

    print_model_parameters(model)
