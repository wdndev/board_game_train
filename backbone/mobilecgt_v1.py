# -*- coding: utf-8 -*-
#  @file        - mobilecgt_v1.py
#  @author      - dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - CGT v1 (computer game transformer)实现
#  @version     - 0.0
#  @date        - 2023.12.20
#  @copyright   - Copyright (c) 2023 

"""
    mobile computer game transformer with self attention
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
            act=True):
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

class Attention(nn.Module):
    """ 多头注意力机制
    """
    def __init__(self, embed_dim, heads=4, dim_head=8, attn_dropout=0):
        super().__init__()
        self.qkv_proj = nn.Linear(embed_dim, 3*embed_dim, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim
        self.num_heads = heads
        self.scale = dim_head ** -0.5

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        # [N, WH, C]
        b_sz, S_len, in_channels = x.shape
        # self-attention
        # [N, S, C] --> [N, S, 3C] --> [N, S, 3, h, c] where C = hc
        qkv = self.qkv_proj(x).reshape(b_sz, S_len, 3, self.num_heads, -1)
        # [N, S, 3, h, c] --> [N, h, 3, S, C]
        qkv = qkv.transpose(1, 3).contiguous()
        # [N, h, 3, S, C] --> [N, h, S, C] x 3
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        q = q * self.scale
        # [N h, T, c] --> [N, h, c, T]
        k = k.transpose(-1, -2)
        # QK^T
        # [N, h, S, c] x [N, h, c, T] --> [N, h, S, T]
        attn = torch.matmul(q, k)
        batch_size, num_heads, num_src_tokens, num_tgt_tokens = attn.shape
        attn_dtype = attn.dtype
        attn_as_float = self.softmax(attn.float())
        attn = attn_as_float.to(attn_dtype)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [N, h, S, T] x [N, h, T, c] --> [N, h, S, c]
        out = torch.matmul(attn, v)
        # [N, h, S, c] --> [N, S, h, c] --> [N, S, C]
        out = out.transpose(1, 2).reshape(b_sz, S_len, -1)
        out = self.out_proj(out)

        return out

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, ffn_latent_dim, heads=8, dim_head=8, dropout=0, attn_dropout=0):
        super().__init__()
        self.pre_norm_mha = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-5, elementwise_affine=True),
            Attention(embed_dim, heads, dim_head, attn_dropout),
            nn.Dropout(dropout)
        )
        self.pre_norm_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-5, elementwise_affine=True),
            nn.Linear(embed_dim, ffn_latent_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_latent_dim, embed_dim, bias=True),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Multi-head attention
        x = x + self.pre_norm_mha(x)
        # Feed Forward network
        x = x + self.pre_norm_ffn(x)
        return x


class MobileCGTBlockv1(nn.Module):
    def __init__(self, 
                 inp, 
                 embed_dim,         # 词嵌入维度
                 ffn_multiplier,    # transformer ffn 隐藏层倍数
                 heads,             # 注意力头数
                 dim_head,          # 每一个head的维数
                 attn_blocks,       # transformer层数
                 ):
        super(MobileCGTBlockv1, self).__init__()

        # local representation
        self.local_rep = nn.Sequential()
        # 改进4，局部表示块中的深度卷积层
        self.local_rep.add_module('conv_3x3', conv_2d(inp, inp, kernel_size=3, stride=1, padding=1, groups=inp))
        self.local_rep.add_module('conv_1x1', conv_2d(inp, embed_dim, kernel_size=1, stride=1, padding=0,norm=False, act=False))
        
        # global representation
        self.global_rep = nn.Sequential()
        ffn_dims = [int((ffn_multiplier*embed_dim)//16*16)] * attn_blocks
        for i in range(attn_blocks):
            ffn_dim = ffn_dims[i]
            self.global_rep.add_module(f'TransformerEncoder_{i}', TransformerEncoder(embed_dim, ffn_dim, heads, dim_head))
        self.global_rep.add_module('LayerNorm', nn.LayerNorm(embed_dim, eps=1e-5, elementwise_affine=True))

        self.conv_proj = conv_2d(embed_dim, inp, kernel_size=1, stride=1, padding=0)
        # 改进1，在融合块中将3×3卷积层替换为1×1卷积层
        self.fusion = conv_2d(inp+embed_dim, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        res = x.clone()
        fm_conv = self.local_rep(x)
        # print("aa: ", fm_conv.shape)

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = fm_conv.flatten(2).transpose(1, 2)

        x = self.global_rep(x)

        # transpose: [B, HW, C] -> [B, C, HW]
        # reshape : [B, C, HW] -> [B, C, H, W]
        x = x.transpose(1, 2).reshape(B, -1, H, W)

        x = self.conv_proj(x)
        # 改进2，局部和全局特征融合
        x = self.fusion(torch.cat((fm_conv, x), dim=1))
        # 改进4，融合输入特征
        x = x + res
        return x

class MobileCGTv1(nn.Module):
    def __init__(self, in_c):  
        """
            mobile computer game transformer with self attention
        """
        super().__init__()

        self.model_name = "mobile_cgt_v1"
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
            MobileCGTBlockv1(channels[2], embed_dim[0], ffn_multiplier, heads=4, dim_head=8, attn_blocks=2)
        )

        self.layer_3 = nn.Sequential(
            InvertedResidual(channels[2], channels[3], stride=1, expand_ratio=2),
            MobileCGTBlockv1(channels[3], embed_dim[1], ffn_multiplier, heads=4, dim_head=8, attn_blocks=4)
        )

        self.layer_4 = nn.Sequential(
            InvertedResidual(channels[3], channels[4], stride=1, expand_ratio=2),
            MobileCGTBlockv1(channels[4], embed_dim[2], ffn_multiplier, heads=4, dim_head=8, attn_blocks=3)
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
    

def mobile_cgt_v1(in_chans=4, board_size=19):
    """ Total of parameters: 1317156
    """
    model = MobileCGTv1(in_c=in_chans)
    model.model_name = "mobile_cgt_v1"
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
    tensor = torch.randn(5, 4, 19, 19)

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
    

    # mcb = MobileCGTBlockv1(inp=30, embed_dim=64, ffn_multiplier=2, heads=4, dim_head=8, attn_blocks=2)

    # x = mcb(tensor)
    # print(x.shape)

    mcgt = mobile_cgt_v1(in_chans=4)

    p_hat, v_hat = mcgt(tensor)

    print("p_hat: ", p_hat.shape)
    print("v_hat: ", v_hat.shape)

    # print_model_parameters(mcgt)