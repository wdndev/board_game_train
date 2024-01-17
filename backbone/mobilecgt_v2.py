# -*- coding: utf-8 -*-
#  @file        - mobilecgt_v2.py
#  @author      - dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - CGT v2 (computer game transformer)实现
#  @version     - 0.0
#  @date        - 2023.12.20
#  @copyright   - Copyright (c) 2023 

"""
    mobile computer game transformer with linear self attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_2d(inp, 
            oup, 
            kernel_size=3, 
            stride=1, 
            padding=0, 
            groups=1, 
            bias=False, 
            norm=True, 
            act=True
            ):
    """ 卷积层
        - 3*3卷积, padding=1 时, 保持图像大小
        - 1*1卷积, padding=0 时, 保持图像大小
    """
    conv = nn.Sequential()
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.SiLU())
    return conv

class PolicyHead(nn.Module):
    """ 策略头
    """
    def __init__(self, in_channel = 128):
        super().__init__()
        self.conv1 = conv_2d(in_channel, in_channel // 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = conv_2d(in_channel // 2, 1, kernel_size=3, stride=1, padding=1, bias=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten(1)
        return F.softmax(x, dim=1)
    
class ValueHead(nn.Module):
    """ 价值头
    """
    def __init__(self, in_channel = 128):
        super().__init__()
        self.conv1 = conv_2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = conv_2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=True)

        self.fc = nn.Linear(in_features=in_channel, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.mean((2, 3))
        x = self.fc(x)

        return F.tanh(x)

class InvertedResidual(nn.Module):
    """ 倒残差结构
    参数：
        - inp : 
        - oup : 
        - stride : 
        - expand_ratio :  在深度转换中通过这个因子扩展输入通道 

    形状：
        - input : math:`(N, C_{in}, H_{in}, W_{in})
        - output : math:`(N, C_{out}, H_{out}, W_{out})`

    注意：
        当stride=1且输入特征矩阵与输出特征矩阵shape相同时才有shortcut连接
    """
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        # hidden_dim = int(round(inp * expand_ratio))
        hidden_dim = int(round(inp * expand_ratio))
        self.block = nn.Sequential()
        if expand_ratio != 1:
            self.block.add_module('exp_1x1', conv_2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0))
        self.block.add_module('conv_3x3', conv_2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim))
        self.block.add_module('red_1x1', conv_2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, act=False))
        self.use_res_connect = self.stride == 1 and inp == oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)  

class LinearSelfAttention(nn.Module):
    def __init__(self, embed_dim, attn_dropout=0):
        super().__init__()
        self.qkv_proj = conv_2d(embed_dim, 1+2*embed_dim, kernel_size=1, bias=True, norm=False, act=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = conv_2d(embed_dim, embed_dim, kernel_size=1, bias=True, norm=False, act=False)
        self.embed_dim = embed_dim

    def forward(self, x):
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1)
        context_score = F.softmax(q, dim=-1)
        context_score = self.attn_dropout(context_score)

        context_vector = k * context_score
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        out = F.relu(v) * context_vector.expand_as(v)
        out = self.out_proj(out)
        return out

class LinearAttnFFN(nn.Module):
    def __init__(self, embed_dim, ffn_latent_dim, dropout=0, attn_dropout=0):
        super().__init__()
        self.pre_norm_attn = nn.Sequential(
            nn.GroupNorm(num_channels=embed_dim, eps=1e-5, affine=True, num_groups=1),
            LinearSelfAttention(embed_dim, attn_dropout),
            nn.Dropout(dropout)
        )
        self.pre_norm_ffn = nn.Sequential(
            nn.GroupNorm(num_channels=embed_dim, eps=1e-5, affine=True, num_groups=1),
            conv_2d(embed_dim, ffn_latent_dim, kernel_size=1, stride=1, bias=True, norm=False, act=True),
            nn.Dropout(dropout),
            conv_2d(ffn_latent_dim, embed_dim, kernel_size=1, stride=1, bias=True, norm=False, act=False),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # self attention
        x = x + self.pre_norm_attn(x)
        # Feed Forward network
        x = x + self.pre_norm_ffn(x)
        return x

class MobileCGTBlockv2(nn.Module):
    def __init__(self, 
                 inp, 
                 embed_dim,         # 词嵌入维度
                 ffn_multiplier,    # transformer ffn 隐藏层倍数
                 attn_blocks,       # transformer层数
                 ):
        super(MobileCGTBlockv2, self).__init__()

        self.patch_h = 2
        self.patch_w = 2

        # local representation
        self.local_rep = nn.Sequential()
        self.local_rep.add_module('conv_3x3', conv_2d(inp, inp, kernel_size=3, stride=1, padding=1, groups=inp))
        self.local_rep.add_module('conv_1x1', conv_2d(inp, embed_dim, kernel_size=1, stride=1, norm=False, act=False))
        
        # global representation
        self.global_rep = nn.Sequential()
        ffn_dims = [int((ffn_multiplier*embed_dim)//16*16)] * attn_blocks
        for i in range(attn_blocks):
            ffn_dim = ffn_dims[i]
            self.global_rep.add_module(f'LinearAttnFFN_{i}', LinearAttnFFN(embed_dim, ffn_dim))
        self.global_rep.add_module('LayerNorm2D', nn.GroupNorm(num_channels=embed_dim, eps=1e-5, affine=True, num_groups=1))

        self.conv_proj = conv_2d(2*embed_dim, inp, kernel_size=1, stride=1, padding=0, act=False)

    def unfolding_pytorch(self, feature_map):
        batch_size, in_channels, img_h, img_w = feature_map.shape
        # [B, C, H, W] --> [B, C, P, N]
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )
        return patches, (img_h, img_w)

    def folding_pytorch(self, patches, output_size):
        batch_size, in_dim, patch_size, n_patches = patches.shape
        # [B, C, P, N]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)
        feature_map = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        return feature_map

    def forward(self, x):
        res = x.clone()
        fm_conv = self.local_rep(x)
        # print("1: ", fm_conv.shape)
        x, output_size = self.unfolding_pytorch(fm_conv)
        # print("2: ", x.shape)
        x = self.global_rep(x)
        x = self.folding_pytorch(patches=x, output_size=output_size)
        # print("3: ", x.shape)
        x = self.conv_proj(torch.cat((x, fm_conv), dim=1))
        x = x + res
        return x
    
        # res = x.clone()
        # fm_conv = self.local_rep(x)

        # # print("1: ", fm_conv.shape)

        # x = self.global_rep(fm_conv)

        # x = self.conv_proj(torch.cat((x, fm_conv), dim=1))
        # x = x + res
        # return x

class MobileCGTv2(nn.Module):
    def __init__(self, in_c):  
        """
            mobile computer game transformer with linear self attention
        """
        super().__init__()

        self.model_name = "mobile_cgt_v2"
        # transformer ffn 隐藏层倍数
        ffn_multiplier = 2
        # 最后输出 1*1 卷积倍数
        last_layer_exp_factor = 2
        channels = [16, 32, 64, 64, 80]
        embed_dim = [64, 80, 96]

        self.conv_0 = conv_2d(in_c, channels[0], kernel_size=3, stride=1, padding=1)

        self.layer_1 = nn.Sequential(
            InvertedResidual(channels[0], channels[0], stride=1, expand_ratio=1),

            InvertedResidual(channels[0], channels[1], stride=1, expand_ratio=2),
            InvertedResidual(channels[1], channels[1], stride=1, expand_ratio=1),
            InvertedResidual(channels[1], channels[1], stride=1, expand_ratio=2),
        )
        self.layer_2 = nn.Sequential(
            InvertedResidual(channels[1], channels[2], stride=1, expand_ratio=2),
            MobileCGTBlockv2(channels[2], embed_dim[0], ffn_multiplier, attn_blocks=2)
        )

        self.layer_3 = nn.Sequential(
            InvertedResidual(channels[2], channels[3], stride=1, expand_ratio=2),
            MobileCGTBlockv2(channels[3], embed_dim[1], ffn_multiplier, attn_blocks=4)
        )

        self.layer_4 = nn.Sequential(
            InvertedResidual(channels[3], channels[4], stride=1, expand_ratio=2),
            MobileCGTBlockv2(channels[4], embed_dim[2], ffn_multiplier, attn_blocks=3)
        )

        self.conv_1x1_exp = conv_2d(channels[-1], channels[-1]*last_layer_exp_factor, kernel_size=3, stride=1, padding=1)

        self.policy_head = PolicyHead(in_channel=channels[-1]*last_layer_exp_factor)
        self.value_head = ValueHead(in_channel=channels[-1]*last_layer_exp_factor)

    def forward(self, x):
        x = self.conv_0(x)

        x = self.layer_1(x)
        x = self.layer_2(x) 
        x = self.layer_3(x)
        x = self.layer_4(x)

        x = self.conv_1x1_exp(x)

        p_hat = self.policy_head(x)
        v_hat = self.value_head(x)
        
        return p_hat, v_hat

    
def mobile_cgt_v2(in_chans=4, board_size=19):
    """ Total of parameters: 1240861
    """
    model = MobileCGTv2(in_c=in_chans)
    model.model_name = "mobile_cgt_v2"
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
    tensor = torch.randn(5, 5, 19, 19)

    print(tensor.shape)

    # conv = conv_2d(inp=4, oup=64, stride=1, padding=1)
    # x = conv(tensor)
    # print(x.shape)

    # model = InvertedResidual(inp=64, oup=64, stride=1, expand_ratio=1)

    # x = model(x)
    # print(x.shape)

    # tf_tensor = torch.randn(5, 361, 64)
    # print(tf_tensor.shape)
    # tf = TransformerEncoder(embed_dim=64, ffn_latent_dim=128)
    # x = tf(tf_tensor)
    # print(x.shape)
    

    # mcb = MobileCGTBlockv2(inp=4, embed_dim=64, ffn_multiplier=2, attn_blocks=2)

    # x = mcb(tensor)
    # print(x.shape)

    mcgt = mobile_cgt_v2(in_chans=5)

    p_hat, v_hat = mcgt(tensor)

    print("p_hat: ", p_hat.shape)
    print("v_hat: ", v_hat.shape)

    # print_model_parameters(mcgt)