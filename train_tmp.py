import random
import time
import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim, cuda
from torch.utils.data import DataLoader
import torch.nn.init as init
import math
from datasets.conn6_dataset import Conn6Dataset
from backbone.resnet import resnet_10b128c

class ValueLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def cross_entropy_with_softlabel(
        self, input, target, reduction="mean", adjust=False, weight=None
    ):
        """
        :param input: (batch, *)
        :param target: (batch, *) same shape as input,
            each item must be a valid distribution: target[i, :].sum() == 1.
        :param adjust: subtract soft-label bias from the loss
        :param weight: (batch, *) same shape as input,
            if not none, a weight is specified for each loss item
        """
        input = input.view(input.shape[0], -1)
        target = target.view(target.shape[0], -1)
        if weight is not None:
            weight = weight.view(weight.shape[0], -1)

        logprobs = torch.log_softmax(input, dim=1)
        if weight is not None:
            logprobs = logprobs * weight
        batchloss = -torch.sum(target * logprobs, dim=1)

        if adjust:
            eps = 1e-8
            bias = target * torch.log(target + eps)
            if weight is not None:
                bias = bias * weight
            bias = torch.sum(bias, dim=1)
            batchloss += bias

        if reduction == "none":
            return batchloss
        elif reduction == "mean":
            return torch.mean(batchloss)
        elif reduction == "sum":
            return torch.sum(batchloss)
        else:
            assert 0, f"Unsupported reduction mode {reduction}."

    def forward(self, value, z):
        """
        前馈计算loss
        :param value: 神经网络的value输出
        :param z: valueTarget
        :return: loss
        """
        # 均方差
        # value_loss = F.mse_loss(value, z)

        value_loss = self.cross_entropy_with_softlabel(value, z, adjust=True)

        loss = value_loss
        return loss


def weights_init(init_type):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find("Conv") == 0 or classname.find("Linear") == 0) and hasattr(
            m, "weight"
        ):
            if init_type == "gaussian":
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == "normal":
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == "default":
                pass
            else:
                assert 0, f"Unsupported initialization: {init_type}"
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class Trainer:
    """训练网络"""

    def __init__(
        self, lr=2e-3, batch_size=128, is_use_gpu=True, model_type="4b32f", **kwargs
    ):
        self.lr = lr
        self.batch_size = batch_size
        self.is_use_gpu = is_use_gpu
        self.device = torch.device(
            "cuda:0" if is_use_gpu and cuda.is_available() else "cpu"
        )
        self.model_type = model_type
        if not os.path.exists(f"./log/{self.model_type}"):
            os.makedirs(f"./log/{self.model_type}")
        if not os.path.exists(f"./weights/{self.model_type}"):
            os.makedirs(f"./weights/{self.model_type}")
        # 创建网络和优化器
        self.policy_value_net, self.optimizer = self.get_policy_value_net()
        # 创建损失函数
        self.criterion = ValueLoss()
        self.train_loss = []
        self.test_loss = []
        self.train_policy_loss = []
        self.test_policy_loss = []
        self.train_value_loss = []
        self.test_value_loss = []

    #
    def get_policy_value_net(self):
        """
        创建策略-价值网络，如果存在历史最优模型则直接载入最优模型.
        :return: net.
        """
        path = f"./weights/{self.model_type}"
        file = os.listdir(path)
        # net = ResNet(3, 6, 128, 128).to(self.device)
        net = resnet_10b128c(3, 19).to(self.device)
        # net = FlatConv3x3NNUE(2, 32, 16).to(self.device)
        # net = conv3NNUE(2, 32, 16, 1).to(self.device)
        # net = LadderConvNNUE(2, 128, 32, 1).to(self.device)
        # net = SplitedConv5NNUE(2, 32, 16, 4).to(self.device)
        if len(file) != 0:
            # 从历史模型中选取最新模型
            best_model = path + "/" + file[-1]
            model = best_model
            print(f"💎 载入模型 {model} ..")
            check_point = torch.load(model)
            net.load_state_dict(check_point["model"])
            opt = optim.Adam(net.parameters(), lr=self.lr)
            opt.load_state_dict(check_point["optimizer"])
        else:
            opt = optim.Adam(net.parameters(), lr=self.lr)
        return net, opt

    def train(self):
        """train model."""
        print("INFO: Train Start!...")
        path = f"./weights/{self.model_type}"
        file = os.listdir(path)
        if len(file) == 0:
            self.policy_value_net.apply(weights_init("kaiming"))
        self.policy_value_net.train()
        print("INFO: Loading Data...")
        data_files = os.listdir("datasets/conn6_data/train_data")
        while True:
            for i_files in range(len(data_files)):
                # while True:
                k = np.random.randint(0, len(data_files))
                train_data_set = Conn6Dataset(np.load(f"datasets/conn6_data/train_data/{data_files[k]}"))
                # val_data_set = GameDataSet(np.load("./data/vdata/data.npz"))
                td = DataLoader(
                    train_data_set, self.batch_size, shuffle=True, drop_last=False
                )
                train_data_loader = iter(td)
                file_iter_count = 0
                start_time = time.time()
                for train_batch_id, (train_bf, train_pt, train_vt) in enumerate(train_data_loader):
                    file_iter_count += 1
                    feature_planes = train_bf.to(self.device)
                    pt = train_pt.to(self.device)
                    z = train_vt.to(self.device)
                    # 前馈
                    policy, value = self.policy_value_net(feature_planes)
                    self.optimizer.zero_grad()
                    # 计算损失
                    policy_loss = self.criterion(policy, pt)
                    value_loss = self.criterion(value, z)
                    # 误差反向传播
                    loss = policy_loss + value_loss
                    loss.backward()
                    # 更新参数
                    self.optimizer.step()
                    # 记录误差
                    self.train_policy_loss.append([policy_loss.item()])
                    self.train_value_loss.append([value_loss.item()])
                    self.train_loss.append([loss.item()])
                    end_time = time.time()

                    if (train_batch_id + 1) % 50 == 0:
                        # with torch.no_grad():
                        #     vd = DataLoader(
                        #         val_data_set, self.batch_size, shuffle=True, drop_last=False
                        #     )
                        #     val_data_loader = iter(vd)
                        #     val_bf, val_vt = next(val_data_loader)
                        #     state = val_bf.to(self.device)
                        #     v = val_vt.to(self.device)
                        #     # 前馈
                        #     target = self.policy_value_net(state)
                        #     # 计算损失
                        #     value_loss = self.criterion(target, v)
                        #     # 记录误差
                        #     self.test_value_loss.append(value_loss.item())
                        #     val_data_loader = iter(vd)
                        print()
                        print(f"INFO: Train   Step {int(file_iter_count)}")
                        print(
                            f"INFO: Train   Time {float(end_time - start_time): <10.5f}"
                        )
                        print(
                            f"INFO:Train Policy   Loss {float(np.mean(self.train_policy_loss)): <10.5f}"
                        )
                        print(
                            f"INFO:Train Value   Loss {float(np.mean(self.train_value_loss)): <10.5f}"
                        )
                        print(
                            f"INFO:Train Total   Loss {float(np.mean(self.train_loss)): <10.5f}"
                        )
                        # print(
                        #     f"INFO:Test Value   Loss {float(np.mean(self.test_value_loss)): <10.5f}"
                        # )
                        print()
                        start_time = end_time
                np.savez(
                    f"./log/{self.model_type}/loss_{time.time()}.npz",
                    value=np.array(self.train_value_loss),
                    policy=np.array(self.train_policy_loss),
                    loss=np.array(self.train_loss)
                    # test=np.array(self.test_value_loss),
                )
                self.train_value_loss = []
                self.train_policy_loss = []
                self.train_loss = []
                self.test_value_loss = []
                path = (
                    f"weights/{self.model_type}/best_policy_value_net_{time.time()}.pth"
                )
                self.policy_value_net.eval()
                torch.save(
                    {
                        "model": self.policy_value_net.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    },
                    path,
                )
                print(f"🎉 训练结束，已将当前模型保存到 {os.path.join(os.getcwd(), path)}")
                # self.policy_value_net, self.optimizer = self.get_policy_value_net()


# if '__name__' == '__main__':
train_config = {
    "lr": 1e-3,
    "batch_size": 256,
    "is_use_gpu": True,
    "model_type": "chess_convnextv2_2262",
}
train_model = Trainer(**train_config)
train_model.train()
