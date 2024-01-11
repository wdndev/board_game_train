import os
import glob
import sys
from pickle import Pickler, Unpickler
import torch
import re
import numpy as np
import itertools


def create_directory(directory_path):
    """ 检查文件夹是否存在，如果不存在，则创建文件夹
    """
    # 
    if not os.path.exists(directory_path):
        # 如果文件夹不存在，则创建它
        os.makedirs(directory_path)

    return directory_path

def print_model_parameters(model):
    """ 打印模型各个层参数
    """
    param_sum = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_sum += param.numel()
            print(f"Layer: {name}, Parameters: {param.numel()}")
    print(f"Total of parameters: {param_sum}")

def get_latest_model_name(folder_path):
    # 拼接文件名模式
    pattern = os.path.join(folder_path, 'model_*.pth')

    # 查找所有匹配的文件路径
    file_paths = glob.glob(pattern)

    if not file_paths:
        return None

    # 从文件路径中提取序号并找到最大的序号文件
    max_file_path = max(file_paths, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    return max_file_path


