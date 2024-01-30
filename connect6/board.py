# -*- coding: utf-8 -*-
#  @file        - board.py
#  @author      - CallMeRoot-J, dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - 六子棋棋盘类
#  @version     - 0.0
#  @date        - 2024.01.27
#  @copyright   - Copyright (c) 2023 

from collections import OrderedDict
from copy import deepcopy
import torch
import numpy as np

class Conn6Board:
    """
    棋盘逻辑类
    """
    EMPTY = 0
    BLACK = 1
    WHITE = -1
    priority_conv_weights = [[[2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2],
                            [2, 2, 1, 2, 2],
                            [2, 2, 2, 2, 2],
                            [2, 2, 2, 2, 2]],
                           [[71, 173, 281, 173, 71],
                            [173, 409, 541, 409, 173],
                            [281, 541, 1, 541, 281],
                            [173, 409, 541, 409, 173],
                            [71, 173, 281, 173, 71]],
                           [[29, 113, 229, 113, 29],
                            [113, 349, 463, 349, 113],
                            [229, 463, 1, 463, 229],
                            [113, 349, 463, 349, 113],
                            [29, 113, 229, 113, 29]]]

    def __init__(self, board_size=19, n_feature_planes=18,  first_point=180):
        '''
        初始化棋盘
        :param board_size: 棋盘大小
        :param n_feature_planes: 特征层的个数
        '''
        
        self.board_size = board_size
        self.n_feature_planes = n_feature_planes
        
        self.reset(first_point)
        
    
    def reset(self, first_point=180):
        """ 复位
        """
        self.first_point = first_point
        self.current_player = self.WHITE
        # 棋盘状态字典，key为action，value为current_play。
        self.state = OrderedDict()
        # 上一个落点
        self.pre_action = None
        # 历史棋局的矩阵形式。
        self.board = np.full(shape=(self.board_size, self.board_size), fill_value=self.EMPTY)
        self.board[self.first_point//self.board_size][self.first_point % self.board_size] = self.BLACK
        self.stage = 0
        self.stage0_board = np.full(shape=(self.board_size, self.board_size), fill_value=self.EMPTY)
        self.stage0_board[self.first_point // self.board_size][self.first_point % self.board_size] = self.BLACK
        self.stage0_move_priority_board = np.full(shape=(self.board_size, self.board_size), fill_value=self.EMPTY)
        self.stage1_point = np.full(shape=(self.board_size, self.board_size), fill_value=self.EMPTY)
        self.available_action = self.get_available()


    def copy(self):
        '''
        copy the board
        :return: board
        '''
        return deepcopy(self)

    def in_board(self, fx: int, fy: int) -> bool:
        '''
        juage whether a piece is inside the board
        '''
        return 0 <= fx < self.board_size and 0 <= fy < self.board_size

    def get_available(self):
        available = []
        if self.stage == 0:
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if self.board[i][j] == self.EMPTY:
                    # if self.board[i][j] == self.EMPTY and i < 15 and j < 15:
                        available.append(i*self.board_size+j)
                    self.stage0_move_priority_board[i][j] = self.get_move_priority(i, j)
        else:
            self.stage1_point = np.full(shape=(self.board_size, self.board_size), fill_value=self.EMPTY)
            x = self.pre_action//self.board_size
            y = self.pre_action % self.board_size
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if self.board[i][j] == self.EMPTY and \
                            self.stage0_move_priority_board[i][j] < self.stage0_move_priority_board[x][y]:
                    # if self.board[i][j] == self.EMPTY and i < 15 and j < 15 and\
                    #         self.stage0_move_priority_board[i][j] < self.stage0_move_priority_board[x][y]:
                        available.append(i*self.board_size+j)
                        self.stage1_point[i][j] = 1
        return available

    def do_action(self, action: int):
        """
        执行动作
        :param action: 动作
        :return: None
        """
        self.state[action] = self.current_player
        self.pre_action = action
        self.board[action//self.board_size][action % self.board_size] = self.current_player
        if self.stage == 0:
            self.current_player = self.current_player
            self.stage = 1
        else:
            self.current_player = -self.current_player
            self.stage = 0
            self.stage0_board = self.board
        self.available_action = self.get_available()

    def get_move_priority(self, cx, cy):
        """
        获取优先值
        :param cx:
        :param cy:
        :return:
        """
        conv_total = 0
        for i in range(-2, 3):
            x = i + cx
            if x < 0:
                continue
            if x >= self.board_size:
                break
            for j in range(-2, 3):
                y = j + cy
                if y < 0:
                    continue
                if y >= self.board_size:
                    break
                color = self.stage0_board[x][y]
                if color == self.current_player:
                    conv_total += self.priority_conv_weights[1][i+2][j+2]
                elif color == -self.current_player:
                    conv_total += self.priority_conv_weights[2][i+2][j+2]
                elif color == self.EMPTY:
                    conv_total += self.priority_conv_weights[0][i+2][j+2]
                else:
                    pass
        return conv_total

    def is_game_over(self):
        """
        判断游戏是否结束。
        :return: Tuple[bool, int]
        *bool->是否结束（分出胜负或者平局）True 否则 False
        *int ->游戏的赢家。
                **如果游戏分出胜负，Board.BLACK 或者 Board.WHITE
                **如果游戏未分出胜负，None
        """
        # 如果下的棋子不到 10 个，就直接判断游戏还没结束
        if len(self.state) < 11:
            return False, None

        n = self.board_size
        act = self.pre_action
        player = self.state[act]
        row, col = act // n, act % n

        # 搜索方向
        directions = [[(0, -1), (0, 1)],  # 水平搜索
                      [(-1, 0), (1, 0)],  # 竖直搜索
                      [(-1, -1), (1, 1)],  # 主对角线搜索
                      [(1, -1), (-1, 1)]]  # 副对角线搜索

        for i in range(4):
            count = 1
            for j in range(2):
                flag = True
                row_t, col_t = row, col
                while flag:
                    row_t = row_t + directions[i][j][0]
                    col_t = col_t + directions[i][j][1]
                    if self.in_board(row_t, col_t) and self.state.get(row_t * n + col_t, self.EMPTY) == player:
                        # 遇到相同颜色时 count+1
                        count += 1
                    else:
                        flag = False
            # 分出胜负
            if count >= 6:
                return True, player
        # 平局
        if not self.available_action:
            return True, None
        return False, None

    def get_feature_planes(self) -> torch.Tensor:
        """
        棋盘状态特征张量，维度：（n_feature_planes, board_size, board_size）
        :return: torch.Tensor,棋盘状态特征张量.
        """
        n = self.board_size
        feature_planes = torch.zeros((self.n_feature_planes, n, n))
        feature_planes[0] = 1
        # for a in range(19):
        #     for b in range(19):
        #         if a <= 14 and b <= 14:
        #             feature_planes[0][a][b] = 1
        # feature_planes[0] = 1
        # 添加历史信息
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == self.current_player:
                    feature_planes[1][i][j] = 1
                if self.board[i][j] == -self.current_player:
                    feature_planes[2][i][j] = 1
                if self.stage == 0:
                    if self.board[i][j] == self.EMPTY:
                        feature_planes[3][i][j] = 1
                    conv_total = bin(self.stage0_move_priority_board[i][j])
                    zero_num = 13 - len(conv_total) + 2
                    for k in range(13):
                        if k < zero_num:
                            feature_planes[k+4][i][j] = 0
                        else:
                            feature_planes[k+4][i][j] = int(conv_total[2+k-zero_num])
        if self.stage == 1:
            feature_planes[3] = torch.Tensor(self.stage1_point)
            feature_planes[17] = 1
        return feature_planes

    def print_board(self):
        """ 打印棋盘
        """
        print("********************************")
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                self.print_color_char(self.board[i][j])
            print()
        print("---------------------------------")

    def print_color_char(self, value):
        """ 打印字符
        """
        # ANSI转义码，用于控制终端文本颜色
        ANSI_RESET = "\x1B[0m"
        ANSI_YELLOW = "\x1B[33m"
        ANSI_GREEN = "\x1B[32m"
        ANSI_BLACK = "\x1B[30m"
        ANSI_WHITE = "\x1B[37m"

        if (value == -1):
            print(ANSI_YELLOW + " w " + ANSI_RESET, end='')
        elif (value == 1):
            print(ANSI_GREEN + " b " + ANSI_RESET, end='')
        elif (value == 0):
            print(ANSI_WHITE + " - " + ANSI_RESET, end='')