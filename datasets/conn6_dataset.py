

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


from datasets.data_type import BoardState, GameState, deserialize_from_json


class Conn6QDataset(Dataset):
    def __init__(self, folder_path : str, file_list = None):
        self.board_list = []
        self.init_data(folder_path, file_list)

    def init_data(self, folder_path : str, file_list):
        if (file_list is None):
            file_list = os.listdir(folder_path)
        # 遍历文件夹中的文件
        for file_name in file_list:
            # print("folder_path: ", folder_path)
            # print("file_name: ", file_name)
            file_path = os.path.join(folder_path, file_name)
            # file_path=file_path.replace('\\', "/")
            # print("file_path: ", file_path)
            # 确认当前路径是文件而不是文件夹
            if os.path.isfile(file_path):
                game_state = deserialize_from_json(file_path)

                for board_state in game_state.game[game_state.random_step:]:
                    # 数据扩充，旋转90，180，270
                    for i in range(4):
                        rot_board = np.rot90(board_state.board, i).tolist()
                        rot_pi = np.rot90(board_state.pi, i).tolist()
                        self.board_list.append(BoardState(rot_board, rot_pi, board_state.q,
                                                          board_state.z, board_state.curr_player,
                                                          board_state.stage, board_state.board_size))
                        # 水平翻转棋盘
                        flip_board = np.fliplr(rot_board).tolist()
                        flip_pi = np.fliplr(rot_pi).tolist()
                        self.board_list.append(BoardState(flip_board, flip_pi, board_state.q,
                                                          board_state.z, board_state.curr_player,
                                                          board_state.stage, board_state.board_size))
        
    def __len__(self):
        return len(self.board_list)
    
    def __getitem__(self, idx):
        feature = torch.tensor(self.board_list[idx].get_feature()).float()
        pi = torch.tensor(self.board_list[idx].pi).float()
        z = torch.tensor(self.board_list[idx].z).float()
        return feature, pi, z


class Conn6Dataset(Dataset):
    def __init__(self, data):
        """
        初始化，定义数据内容和标签
        :param data: 传入的data
        """
        # 当前棋子分布，对方棋子分布
        self.state = torch.FloatTensor(data["bf"])
        # position target, pi
        self.pt = torch.FloatTensor(data["pt"])
        # value 胜负平
        self.vt = torch.FloatTensor(data["vt"])
        # 是否第二步 0，第一步，1，第二步
        self.gf = torch.FloatTensor(data["gf"])

    def __len__(self):
        """
        返回数据集大小
        :return: 数据集大小
        """
        return self.vt.shape[0]

    def data_strong(self, bf, gf, pt):
        k = np.random.randint(0, 8)
        if k < 4:
            bf = torch.rot90(bf, k, (1, 2))
            pt = torch.rot90(pt, k, (0, 1))
        elif k >= 4:
            k = k - 4
            bf = torch.flip(torch.rot90(bf, k, (1, 2)), (1, 2))
            pt = torch.flip(torch.rot90(pt, k, (0, 1)), (0, 1))
        else:
            assert False
        bf = torch.cat(
            (bf, gf.view((1, 1, 1)) * torch.ones(size=(1, bf.shape[1], bf.shape[2]))),
            dim=0,
        )
        pt = pt.flatten()
        return bf, pt

    def get_value_label(self, vt):
        return [vt[0] - vt[1]]

    def __getitem__(self, index):
        """
        得到数据内容和标签
        :param index: 序号
        :return: boardFeature, policyTarget, valueTarget
        """
        bf, pt = self.data_strong(self.state[index], self.gf[index], self.pt[index])
        vt = torch.Tensor(self.get_value_label(self.vt[index]))
        return bf, pt, vt

# # 指定文件夹路径
# folder_path = 'train_sets' 
# data_sets = Connect6Dataset(folder_path)

# batch_size = 32  # Define your desired batch size
# train_loader = DataLoader(data_sets, batch_size=batch_size, shuffle=True)

# print(len(train_loader))
# # Example to access a single batch from the DataLoader
# for batch_data, batch_pi, batch_z in train_loader:
#     # Use batch_data, batch_pi, batch_z for training
#     print(batch_data.shape, batch_pi.shape, batch_z.shape)  # Example usage



