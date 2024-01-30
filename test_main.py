import os
from time import strftime, localtime
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from datasets.conn6_dataset import Conn6QDataset, Conn6Dataset

from datasets.data_type import BoardState, GameState

def random_file_name(folder_path):
    file_names = os.listdir(folder_path)
    random_file = random.choice(file_names)
    file_path = os.path.join(folder_path, random_file)
    return file_path

if __name__ == "__main__":
    # val_random_path = random_file_name("datasets/conn6_all_feature/val_data")
    # train_sets = Conn6Dataset(np.load(val_random_path), is_all_feature=True)
    # train_loader = DataLoader(train_sets, batch_size=128, shuffle=True)
    # # train_loader
    # print(len(train_loader))
    # bf, pt, vt = train_sets[9]

    for i in range(0, 601, 5):
        print(i)

   