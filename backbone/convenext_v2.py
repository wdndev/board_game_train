# -*- coding: utf-8 -*-
#  @file        - convenext_v2.py
#  @author      - dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - convenext_v2 模块
#  @version     - 0.0
#  @date        - 2023.12.20
#  @copyright   - Copyright (c) 2023 

"""https://github.com/Jacky-Android/convnext-v2-pytorch/blob/main/model.py

"""
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

class LayerNorm(nn.Module):
    """ LayerNorm支持两种数据格式:channels_last(默认)或channels_first
        输入维度的排序
        - Channels_last对应于形状(batch_size, height, width, channels)的输入
        - channels_first对应于形状(batch_size, channels, height, width)的输入
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Block(nn.Module):
    """ ConvNeXtV2 Block
    """
    def __init__(self, dim, drop_path=0.):
        """
            - dim (int): 输入通道维度
            - drop_path (float): 随机丢弃率, 默认0
        """
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
    
class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
    """
    def __init__(self, in_chans=3, board_size=19, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0.
                 ):
        """
            - in_chans (int): 输入特征通道， Default: 3
            - board_size (int): 棋盘尺寸大小. Default: 19
            - depths (tuple(int)): 每个阶段的块数. Default: [3, 3, 9, 3]
            - dims (int): 每个阶段特征维度. Default: [96, 192, 384, 768]
            - drop_path_rate (float): 随机深度丢弃率. Default: 0.
            - head_init_scale (float): 初始化分类器的权重和偏差的缩放值. Default: 1.
        """
        super().__init__()
        self.model_name = "chess_convnextv2"
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=2),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=1),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer

        self.policy_head = PolicyHead(n_features=dims[-1], board_size=board_size)
        self.value_head = ValueHead(n_features=dims[-1], board_size=board_size)


    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)

        p_hat = self.policy_head(x)
        v_hat = self.value_head(x)

        return p_hat, v_hat

# Total of parameters: 5437390
def chess_convnextv2_2262_48(in_chans=4, board_size=19):
    """ Total of parameters: 5437390
    """
    model = ConvNeXtV2(in_chans=in_chans, 
                       board_size=board_size, 
                       depths=[2, 2, 6, 2], 
                       dims=[48, 96, 192, 384])
    model.model_name = "chess_convnextv2_2262_48"
    return model

def chess_convnextv2_2262_40(in_chans=4, board_size=19):
    """ Total of parameters: 3922022
    """
    model = ConvNeXtV2(in_chans=in_chans, 
                       board_size=board_size, 
                       depths=[2, 2, 6, 2], 
                       dims=[40, 80, 160, 320])
    model.model_name = "chess_convnextv2_2262_40"
    return model

def chess_convnextv2_2262_32(in_chans=4, board_size=19):
    """ Total of parameters: 2667774
    """
    model = ConvNeXtV2(in_chans=in_chans, 
                       board_size=board_size, 
                       depths=[2, 2, 6, 2], 
                       dims=[32, 64, 128, 256])
    model.model_name = "chess_convnextv2_2262_32"
    return model

def chess_convnextv2_2242_32(in_chans=4, board_size=19):
    """ Total of parameters: 2388990
    """
    model = ConvNeXtV2(in_chans=in_chans, 
                       board_size=board_size, 
                       depths=[2, 2, 4, 2], 
                       dims=[32, 64, 128, 256])
    model.model_name = "chess_convnextv2_2242_32"
    return model

def chess_convnextv2_1121_32(in_chans=4, board_size=19):
    """ Total of parameters: 1522078
    """
    model = ConvNeXtV2(in_chans=in_chans, 
                       board_size=board_size, 
                       depths=[1, 1, 2, 1], 
                       dims=[32, 64, 128, 256])
    model.model_name = "chess_convnextv2_1121_32"
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

    model = chess_convnextv2_2242_32(in_chans=4, board_size=19)

    p_hat, v_hat = model(tensor)
    print(p_hat.shape)
    print(v_hat.shape)

    # Total of parameters: 5437390
    print_model_parameters(model)

