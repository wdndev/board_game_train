# -*- coding: utf-8 -*-
#  @file        - node.py
#  @author      - CallMeRoot-J, dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - node节点类
#  @version     - 0.0
#  @date        - 2024.01.27
#  @copyright   - Copyright (c) 2023

from typing import Tuple, Union
import time
import numpy as np
from .node import Node
from .board import Conn6Board
import torch

class AlphaZeroMCTS:
    """
    基于策略-价值网络的蒙特卡洛搜索树
    """

    def __init__(self, net_work, c_puct: float = 1, n_iters=100.0, device='cpu') -> None:
        """
        初始化。
        :param net_work: PolicyValueNet，策略价值网络
        :param c_puct: 搜索常数。
        :param n_iters: 搜索迭代次数。
        """
        self.c_puct = c_puct
        self.n_iters = n_iters
        self.net_work = net_work
        self.device = device
        self.net_work.to(self.device)
        self.root = Node(prior_prob=1, parent=None, stage=None)

    def get_action(self, board0: Conn6Board):
        """
        根据当前局面返回下一步动作。
        :param board: Conn6Board,棋盘。
        :return:
        """
        # board_feature_planes = board0.get_feature_planes().to(self.device)
        # board_feature_planes.unsqueeze_(0)
        # p_hat, v_hat = self.net_work(board_feature_planes)
        # p = p_hat.flatten()            
        # p += 1e-10
        # p = (p) / torch.sum(p)
        # # for a in board0.available_action:
        # #     p[a] += 1
        # # p = (p) / torch.sum(p)
        # if self.device == 'cpu':
        #     p = p.detach().numpy()
        # else:
        #     p = p.cpu().detach().numpy()
        # index = np.array([i for i in range(361)])
        # action = np.random.choice(a=index, p=p)
        # print("111111: ", action)

        start_time = time.time()
        self.root.stage = board0.stage
        s = board0.stage
        count = 0
        # while time.time() - start_time <= self.n_iters:
        for i in range(2):
            # 拷贝棋盘
            count += 1
            board = board0.copy()

            # 如果没有遇到叶节点，就一直向下搜索并更新棋盘
            node = self.root
            # d=0
            while not node.is_leaf_node():
                # d=d+1
                action, node = node.select()
                board.do_action(action)
                # print(i,d)
                # print(board.board)

            # 判断游戏是否结束，如果没结束就拓展叶节点
            is_over, winner = board.is_game_over()
            if not is_over:
                board_feature_planes = board.get_feature_planes().to(self.device)
                board_feature_planes.unsqueeze_(0)
                # print(board_feature_planes.shape)
                board_feature_planes = torch.cat((board_feature_planes[:, :3, ...], board_feature_planes[:, -1:, ...]), dim=1)
                # print(board_feature_planes.shape)
                p_hat, v_hat = self.net_work(board_feature_planes)
                p = p_hat.flatten()
                value = v_hat[0].item()
                # 只取可行落子点
                if self.device == 'cpu':
                    p = p[board.available_action].detach().numpy()
                else:
                    p = p[board.available_action].cpu().detach().numpy()


                # p, value = self.net_work.predict(board)
                # print(p)
                # print(value if board.current_player==1 else -value)
                # 添加狄利克雷噪声
                # if self.is_self_play:
                #     p = 0.75 * p + 0.25 * \
                #         np.random.dirichlet(0.03 * np.ones(len(p)))
                node.expand(zip(board.available_action, p))
            elif winner is not None:
                if winner == board.current_player:
                    value = 1
                else:
                    value = -1
            else:
                value = 0
            # 反向传播
            node.backup(value)

        # 计算 π，在自我博弈状态下：游戏的前三十步，温度系数为 1，后面的温度系数趋于无穷小
        T = 0.5 if len(board0.state) <= 30 else 1e-3
        T = 1
        visits = np.array([i.N for i in self.root.children.values()])
        # print(visits)
        pi_ = self.__getPi(visits, T)
        # 根据 π 选出动作及其对应节点
        actions = list(self.root.children.keys())
        action = int(np.random.choice(actions, p=pi_))
        rate = float(50 - (-self.root.children[action].Q / 2) * 100)
        if board0.stage == 1:
            rate = float((-self.root.children[action].Q / 2) * 100 + 50)
        # print(f'INFO: Iter Count:{count}')
        # print(f'INFO: Win  Rate:{rate: < 10.2f}%')
        self.reset_root()
        return action

    def __getPi(self, visits, T) -> np.ndarray:
        """ 根据节点的访问次数计算 π """
        # pi = visits**(1/T) / np.sum(visits**(1/T)) 会出现标量溢出问题，所以使用对数压缩
        x = 1 / T * np.log(visits + 1e-10)
        x = np.exp(x - x.max())
        pi = x / x.sum()
        return pi

    def reset_root(self):
        """ 重置根节点 """
        self.root = Node(prior_prob=1, c_puct=self.c_puct, parent=None, stage=None)
