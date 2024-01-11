
import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader


from datasets.data_type import BoardState, GameState
from datasets.conn6_dataset import Conn6QDataset, Conn6Dataset
from train_net import TrainNet
from utils.helper import print_model_parameters
from utils.file_buffer import FileBuffer

from backbone.resnet import resnet_10b128c, resnet_4b64c
from backbone.chess_transformer import chess_transformer_88_64, chess_transformer_88_96
from backbone.convenext_v2 import chess_convnextv2_1121_32, chess_convnextv2_2242_32
from backbone.mobilevit_v1 import mobile_vit
from backbone.mobilecgt_v1 import mobile_cgt_v1
from backbone.mobilecgt_v2 import mobile_cgt_v2
from backbone.mobilecgt_v3 import mobile_cgt_v3, mobile_cgt_v3_1_6m
from backbone.emo import emo_1m


def random_file_name(folder_path):
    file_names = os.listdir(folder_path)
    file_path = os.path.join(folder_path, random.choice(file_names))
    return file_path


if __name__ == "__main__":

    batch_size = 128
    lr = 1e-3
    
    model = mobile_cgt_v3_1_6m(in_chans=3, board_size=19)
    # model = ResNetTest(n_feature_planes=3, nun_block=10, in_channels=128, out_channels=128)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_net = TrainNet(model, device, lr=lr)

    # for name, parms in model.named_parameters():
    #     print(type(parms.grad))
    #     print("11111111111111:", parms.grad)
    #     print("2222222222222: ", parms)


    while (True):
        train_random_path = random_file_name("datasets/conn6_data/train_data")
        val_random_path = random_file_name("datasets/conn6_data/val_data")
        
        train_sets = Conn6Dataset(np.load(train_random_path))
        test_sets = Conn6Dataset(np.load(val_random_path))
        train_loader = DataLoader(train_sets, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_sets, batch_size=batch_size, shuffle=True)
        train_net.train(train_loader, test_loader)




    # for idx, file_name in enumerate(file_list):
    #     val_radom_path = random_file_name("datasets/conn6_data/val_data")
    #     file_path = os.path.join(DATASETS_DIR, file_name)
    #     train_sets = Conn6Dataset(np.load(file_path))
    #     test_sets = Conn6Dataset(np.load(val_radom_path))
    #     train_loader = DataLoader(train_sets, batch_size=batch_size, shuffle=True, drop_last=True)
    #     test_loader = DataLoader(test_sets, batch_size=batch_size, shuffle=True)
    #     train_net.train(train_loader, test_loader, idx + 1, epoches=1)

    


