# -*- coding: utf-8 -*-
#  @file        - resnet.py
#  @author      - dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - ResNet 块
#  @version     - 0.0
#  @date        - 2023.12.20
#  @copyright   - Copyright (c) 2023 

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.nn_utils import ResidualBlock

class PolicyHead(nn.Module):
    """ 策略头
    """
    def __init__(self, in_channel = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel // 2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channel // 2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channel // 2, 1, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = x.flatten(1)
        return F.softmax(x, dim=1)
    
class ValueHead(nn.Module):
    """ 价值头
    """
    def __init__(self, in_channel = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=in_channel, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = x.mean((2, 3))
        x = self.fc(x)

        return F.tanh(x)


class ChessResNet(nn.Module):
    """
    First network
    """
    def __init__(self, in_chans=4, depths=10, channel_size=128):
        """
            - in_feature_channel : 输入特征通道数
            - board_size : 棋盘大小
        """
        super(ChessResNet, self).__init__()
        self.model_name = "resnet"
        self.in_chans = in_chans

        self.conv = nn.Conv2d(self.in_chans, channel_size, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(channel_size)
        self.relu = nn.ReLU()
        self.residues = nn.Sequential(
            *[ResidualBlock(channel_size, channel_size) for _ in range(depths)]
        )
        self.policy_head = PolicyHead(in_channel=channel_size)
        self.value_head = ValueHead(in_channel=channel_size)


    def forward(self, x):
        """ 4个特征: 我方棋子位置, 对方棋子位置, 可落子位置, 第一步还是第二步
        """
        x = self.conv(x)
        self.bn(x)
        x = self.relu(x)
        
        x = self.residues(x)
        
        p_hat = self.policy_head(x)
        v_hat = self.value_head(x)

        return p_hat, v_hat

def resnet_4b64c(in_chans=4, board_size=19):
    """ Total of parameters: 391364
    """
    model = ChessResNet(in_chans=in_chans, 
                       depths=4, 
                       channel_size=64)
    model.model_name = "resnet_4b64c"
    return model

def resnet_8b96c(in_chans=4, board_size=19):
    """ Total of parameters: 1543204
    """
    model = ChessResNet(in_chans=in_chans, 
                       depths=8, 
                       channel_size=96)
    model.model_name = "resnet_8b96c"
    return model

def resnet_4b128c(in_chans=4, board_size=19):
    """ Total of parameters: 1557506
    """
    model = ChessResNet(in_chans=in_chans, 
                       depths=4)
    model.model_name = "resnet_4b128c"
    return model

def resnet_6b128c(in_chans=4, board_size=19):
    """ Total of parameters: 2148610
    """
    model = ChessResNet(in_chans=in_chans, 
                       depths=6)
    model.model_name = "resnet_6b128c"
    return model

def resnet_10b128c(in_chans=4, board_size=19):
    """ Total of parameters: 3330818
    """
    model = ChessResNet(in_chans=in_chans, 
                       depths=10)
    model.model_name = "resnet_10b128c"
    return model

def resnet_15b128c(in_chans=4, board_size=19):
    """ Total of parameters: 4808578
    """
    model = ChessResNet(in_chans=in_chans, 
                       depths=15)
    model.model_name = "resnet_15b128c"
    return model

def resnet_20b128c(in_chans=4, board_size=19):
    """ Total of parameters: 6286338
    """
    model = ChessResNet(in_chans=in_chans, 
                       depths=20)
    model.model_name = "resnet_20b128c"
    return model

def resnet_100b128c(in_chans=4, board_size=19):
    """ Total of parameters: 29942148
    """
    model = ChessResNet(in_chans=in_chans, 
                       depths=100)
    model.model_name = "resnet_100b128c"
    return model

def resnet_100b256c(in_chans=4, board_size=19):
    """ Total of parameters: 119603972
    """
    model = ChessResNet(in_chans=in_chans, 
                       depths=100,
                       channel_size=256)
    model.model_name = "resnet_100b256c"
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
    tensor = torch.randn(5, 3, 19, 19)

    model = resnet_100b256c(in_chans=3, board_size=19)

    p_hat, v_hat = model(tensor)
    print(p_hat.shape)
    print(v_hat.shape)


    print_model_parameters(model)


