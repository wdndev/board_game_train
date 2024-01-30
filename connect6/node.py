# -*- coding: utf-8 -*-
#  @file        - node.py
#  @author      - CallMeRoot-J, dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - node节点类
#  @version     - 0.0
#  @date        - 2024.01.27
#  @copyright   - Copyright (c) 2023


from math import sqrt
from typing import Tuple, Iterable


class Node:
    """
    蒙特卡洛树节点。
    """

    def __init__(self, prior_prob: float, stage, c_puct: float = 1, parent=None):
        """
        初始化节点。
        :param prior_prob: float，先验概率P(s,a)
        :param c_puct: float，探索系数
        :param parent: None，父节点
        """
        self.Q = 0
        self.U = 0
        self.N = 0
        self.P = prior_prob
        self.c_puct = c_puct
        self.parent = parent
        self.stage = stage
        # {action: Node}
        self.children = {}  # type -> Dict[int, Node]

    def select(self) -> tuple:
        """
        选择score最大的子节点和该节点对应的action
        :return: tuple[int, None] action: int, child: None
        """
        nextQ = self.Q - 0.1
        nextQ = -nextQ if self.stage == 1 else nextQ
        for k, v in self.children.items():
            if (v.N == 0):  # 没计算过神经网络
                v.Q = nextQ

        action, node = max(self.children.items(), key=lambda item: item[1].get_score())
        return action, node

    def expand(self, action_prob: Iterable[Tuple[int, float]]):
        """
        拓展节点。
        :param action_prob: Iterable
        每个元素都是[action, prior_probd]的元组，根据这个远组创建子节点
        action_prob的长度为当前棋盘的可用落点的总数
        :return:
        """
        # self.N = 1
        # self.Q = value
        for action, prior_prob in action_prob:
            self.children[action] = Node(prior_prob, (self.stage + 1) % 2, self.c_puct, self)

    def __update(self, value: float):
        """
        更新节点的访问次数 N(s,a) 和节点的累计平均奖励Q(s,a)
        :param value:float -> 用来更新节点的内部数据。
        :return:
        """
        self.Q = (self.N * self.Q + value) / (self.N + 1)
        self.N += 1

    def backup(self, value: float):
        """
        反向传播。
        :param value:
        :return:
        """
        if self.parent:
            if self.stage % 2 == 0:
                self.parent.backup(-value)
            elif self.stage % 2 == 1:
                self.parent.backup(value)
            else:
                assert (False, "stage not 0,1")

        self.__update(value)

    def get_score(self):
        """
        计算节点得分。
        :return:
        """
        # print( self.N,self.P,self.Q,self.parent.N)
        self.U = self.c_puct * self.P * sqrt(self.parent.N) / (1 + self.N)
        Q = -self.Q if self.stage == 0 else self.Q  # 注意self.Q是己方胜率
        score = self.U + Q
        return score

    def is_leaf_node(self):
        """
        是否为叶子节点。
        :return:
        """
        return len(self.children) == 0


