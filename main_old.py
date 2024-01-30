

import os
import torch
from torch.utils.data import Dataset, DataLoader

from datasets.data_type import BoardState, GameState
from datasets.conn6_dataset import Conn6QDataset
from train_net import TrainNet
from utils.helper import load_model_weight, print_model_parameters
from utils.file_buffer import FileBuffer

from backbone.resnet import ResNet10B128C
from backbone.chess_transformer import chess_vit_big, chess_vit, chess_vit_small
from backbone.convenext_v2 import chess_convnextv2_1121

batch_size = 128  # Define your desired batch size


# model = ResNet10B128C(in_chans=4, board_size=19)
model = chess_convnextv2_1121(in_chans=4, board_size=19)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_net = TrainNet(model, device)
# 加载权重参数
# model_name = get_latest_model_name("model")
# model_weight = load_model_weight("best")
# if model_weight is not None:
#     train_net.net_work.model.load_state_dict(model_weight)

# # 计算参数数量并打印
# num_params = print_model_parameters(model)


train_folder_path = 'datasets/chess_label_sets' 
train_file_buffer = FileBuffer(train_folder_path)
file_newest_num = 20000
file_select_num = 10000
file_batch_size = 1000
train_file_list, test_file_list = train_file_buffer.get_batched_files(file_newest_num, file_select_num, file_batch_size)

print("file total num: ", train_file_buffer.size())
print("train file num: ", len(train_file_list))
print("train batch size: ", len(train_file_list[0]))
print("test file num: ", len(test_file_list))

# print(test_file_list)

test_sets = Conn6QDataset(train_folder_path, test_file_list)
test_loader = DataLoader(test_sets, batch_size=batch_size, shuffle=True)



for i in range(len(train_file_list)):
    train_sets = Conn6QDataset(train_folder_path, train_file_list[i])
    # print("train size: ", len(train_sets))
    train_loader = DataLoader(train_sets, batch_size=batch_size, shuffle=True)
    train_net.train(train_loader, test_loader, i + 1)
    # train_net.eval(model, test_loader)



# # # 训练数据集
# train_folder_path = 'datasets/train_sets' 
# train_sets = Connect6Dataset(train_folder_path)
# print("train size: ", len(train_sets))
# train_loader = DataLoader(train_sets, batch_size=batch_size, shuffle=True)

# # 测试数据集
# test_folder_path = 'datasets/test_sets' 
# test_sets = Connect6Dataset(test_folder_path)
# print("test size: ", len(test_sets))
# test_loader = DataLoader(test_sets, batch_size=batch_size, shuffle=True)


# train_net.train(train_loader, test_loader, 1)

# model.load_state_dict(load_model_weight(2))
# train_net.eval(model, test_loader)


