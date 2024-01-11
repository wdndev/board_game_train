# -*- coding: utf-8 -*-
#  @file        - file_buffer.py
#  @author      - dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - 文件缓存池，从文件夹中随机选择文件，用于训练和测试
#  @version     - 0.0
#  @date        - 2023.12.26
#  @copyright   - Copyright (c) 2023 

import os
import random
import re
import math

class FileBuffer:
    """ 文件缓存池
    """
    def __init__(self, folder_path, prefix= "conn6_", suffix = ".json"):
        self.folder_path = folder_path
        self.prefix = prefix
        self.suffix = suffix

    def size(self):
        """ 文件池中文件的数量
        """
        file_list = os.listdir(self.folder_path)
        return len(file_list)

    def get_latest_files(self, newest_num : int, select_num : int) -> list:
        """ 从最新的 newest_num 个文件中，随机选择 select_num 个文件名
        Arg:
            - newest_num : 最新文件个数
            - select_num : 随机选取的文件个数
        Return:
            - selected_files : 返回一个选择文件列表
        """
        if (select_num > newest_num):
            select_num = newest_num

        file_list = self.get_sorted_files()

        if (newest_num > len(file_list)):
            newest_num = len(file_list)

        # 选择最新的 newest_num 个文件，并从中随机选择 select_num 个文件名
        latest_files = file_list[:newest_num]
        selected_files = random.sample(latest_files, min(select_num, len(latest_files)))

        return selected_files
    
    def get_batched_files(self, newest_num : int, select_num : int, batch_size : int, test_ratio=0.04) -> tuple:
        """ 从最新的 newest_num 个文件中，随机选择 select_num 个文件名, 并以batch_size批次返回
        Arg:
            - newest_num : 最新文件个数
            - select_num : 随机选取的文件个数
            - batch_size : 批次大小
        Return:
            - train_sets : 二维列表, 每个batch区分
            - test_sets : 一维列表
        """
        selected_files = self.get_latest_files(newest_num, select_num)
        if len(selected_files) < select_num:
            select_num = len(selected_files)

        train_sets = []
        test_sets = []
        remaining_files = list(set(selected_files))  # 去重剩余文件列表

        # 计算测试集大小
        test_size = int(len(remaining_files) * test_ratio) + 1

        # 获取随机文件列表
        random.shuffle(remaining_files)
        # print("remaining_files: ", len(remaining_files))

        # 划分训练集和测试集
        test_sets = remaining_files[:test_size]
        train_files = remaining_files[test_size:]
        

        while len(train_sets) * batch_size < select_num:
            batch = []
            for _ in range(batch_size):
                if len(train_files) > 0:
                    file = random.choice(train_files)
                    batch.append(file)
                    train_files.remove(file)
                else:
                    break
            # print(len(batch))
            if (len(batch) == 0):
                break
            train_sets.append(batch)
        return train_sets, test_sets
    

    def get_sorted_files(self) -> list:
        """ 获取当前文件池内所有文件, 以时间顺序排列
        """
        file_list = os.listdir(self.folder_path)
        
        # 使用正则表达式匹配时间信息
        pattern = rf"{re.escape(self.prefix)}(\d{{17}}){re.escape(self.suffix)}"
        file_list = [f for f in file_list if re.match(pattern, f)]
        
        # 根据时间信息进行排序
        file_list.sort(key=lambda x: re.match(pattern, x).group(1), reverse=True)
        return file_list

    def get_shuffled_files(self) -> list:
        """ 获取当前文件池内所有文件, 打乱顺序输出
        """
        file_list = os.listdir(self.folder_path)
        
        # 使用正则表达式匹配时间信息
        pattern = rf"{re.escape(self.prefix)}(\d{{17}}){re.escape(self.suffix)}"
        file_list = [f for f in file_list if re.match(pattern, f)]
        
        # 打乱文件列表顺序
        random.shuffle(file_list)
        return file_list

if __name__ == "__main__":
    folder_path = "datasets/test/test_sets"  # 替换为您的文件夹路径
    file_buffer = FileBuffer(folder_path)
    
    # # 获取最新的 5 个文件名中的 3 个
    # latest_files = file_buffer.get_latest_files(10, 2)
    # print("Selected files:", latest_files)

    # 获取最新的 5 个文件中的 3 个，并按照 batch_size = 2 形式返回
    batched_files = file_buffer.get_batched_files(5, 2, 2)
    print("Batched files:", batched_files)

    # # 获取时间排序的文件名列表
    # sorted_files = file_buffer.get_sorted_files()
    # print("Sorted files:", sorted_files)
    
    # # 获取打乱顺序的文件名列表
    # shuffled_files = file_buffer.get_shuffled_files()
    # print("Shuffled files:", shuffled_files)


