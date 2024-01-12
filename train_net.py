
import os
import sys
import time
import math
import random
from typing import Any
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from module.lr_scheduler import WarmupLRScheduler, Cosine, NoDecay

from utils.bar import Bar
from utils.helper import create_directory
from utils.average_meter import AverageMeter
from utils.log import Logger

class TrainNet:
    """ 神经网络训练
    """
    def __init__(self, net_work:nn.Module, device, board_size = 19, lr = 5e-3, is_lr_decay=True, is_load_model=True) -> None:
        
        self.net_work = net_work
        self.device = device
        self.lr = lr
        self.model_name = self.net_work.model_name

        self.net_work.to(self.device)
        self.optimizer = optim.Adam(self.net_work.parameters(), lr=self.lr)
        self.lr_scheduler = None
        # 学习率衰减
        if is_lr_decay:
            self.lr_scheduler = Cosine(optimizer=self.optimizer, start_lr=self.lr, 
                                        warmup_iter=2, end_iter=500, num_iter=0)
        else:
            self.lr_scheduler = NoDecay(optimizer=self.optimizer, start_lr=self.lr, 
                                        warmup_iter=2, end_iter=500, num_iter=0)

        self.model_path = create_directory("logs/" + self.model_name)
        # print(self.model_path)
        sys.stdout = Logger(str(self.model_path + "/train.log"))

        self.writer = SummaryWriter(log_dir=self.model_path + "/tensorboard")
        # 显示网络结构
        input_channel_dim = next(self.net_work.parameters()).shape[1] 
        self.writer.add_graph(self.net_work, input_to_model = torch.rand(5,input_channel_dim,board_size,board_size).to(device))
        self.train_step_count = 0

        # 加载模型、优化器和学习率参数
        if is_load_model:
            model_weight, optimizer_weight, lr_scheduler, train_step_count = self.load_model_weight(self.model_path + "/model", "best")
            if model_weight is not None and optimizer_weight is not None:
                self.net_work.load_state_dict(model_weight)
                self.optimizer.load_state_dict(optimizer_weight)
                if lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(lr_scheduler)
                self.train_step_count = train_step_count + 1
                print("Loading model parameters succeeded!")
            else:
                self.initialize_weights(self.net_work)
                print("Failed to load model parameters.")
                print("The model parameters were randomly initialized!!!")
        else:
            self.initialize_weights(self.net_work)
            print("The model parameters were randomly initialized!!!")
        
        self.train_pi_meter = AverageMeter()
        self.train_v_meter = AverageMeter()
        self.eval_pi_meter = AverageMeter()
        self.eval_v_meter = AverageMeter()

    def train(self, train_loader, test_loader):
        """ 训练
        """
        self.train_pi_meter.reset()
        self.train_v_meter.reset()
        self.eval_pi_meter.reset()
        self.eval_v_meter.reset()
        
        # 所有损失
        train_all_losses = []
        eval_all_losses = []

        batch_num = int(len(train_loader))
        bar = Bar('Training', max=batch_num)
        start_epoch = time.time()
        for batch_idx, batch_data in enumerate(train_loader):
            start = time.time()
            # 训练一次
            train_pi_loss, train_v_loss = self.train_one_step(self.net_work, self.optimizer, batch_data)
            # 测试一次
            eval_pi_loss, eval_v_loss = self.eval_one_step(self.net_work, self.get_random_batch(test_loader))
            # 记录损失
            self.train_pi_meter.update(train_pi_loss.item())
            self.train_v_meter.update(train_v_loss.item())
            self.eval_pi_meter.update(eval_pi_loss.item())
            self.eval_v_meter.update(eval_v_loss.item())
            train_all_losses.append([train_pi_loss.item(), train_v_loss.item()])
            eval_all_losses.append([eval_pi_loss.item(), eval_v_loss.item()])

            # 学习率
            curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            x_axis = self.train_step_count * batch_num + batch_idx
            self.writer.add_scalar("Learning Rate", curr_lr, global_step=x_axis)

            batch_time = (time.time() - start)
            # plot progress
            bar.suffix = 'Iter-Epoch: {iter} | Batch: {batch}/{batch_num} | Time: {bt:.3f}s/{total:} | Loss_pi: {tpi:.4f}/{epi:.4f} | Loss_v: {tv:.3f}/{ev:.4f}'.format(
                    iter = self.train_step_count,
                    batch=batch_idx,
                    batch_num=batch_num,
                    bt=batch_time,
                    total=bar.elapsed_td,  # 总时间
                    # eta=bar.eta_td,
                    tpi=self.train_pi_meter.avg,
                    epi=self.eval_pi_meter.avg,
                    tv=self.train_v_meter.avg,
                    ev=self.eval_v_meter.avg,
            )
            bar.next()
        bar.finish()
        epoch_time = (time.time() - start_epoch)

        # 保存所有损失
        self.save_csv_data("train_loss", [self.train_pi_meter.avg, self.train_v_meter.avg, self.train_pi_meter.avg + self.train_v_meter.avg])
        self.save_csv_data("eval_loss", [self.eval_pi_meter.avg, self.eval_v_meter.avg, self.eval_pi_meter.avg+self.eval_v_meter.avg])
        self.save_all_loss("all_train_loss", train_all_losses)
        self.save_all_loss("all_eval_loss", eval_all_losses)
        # 保存当前学习率
        curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.save_csv_data("lr", [curr_lr])
        # 当前轮次时间
        self.save_csv_data("epoch_time", [epoch_time])


        # 绘制曲线
        x_axis = self.train_step_count
        self.writer.add_scalars("Policy Loss", {
                                "Train": self.train_pi_meter.avg,
                                "Val": self.eval_pi_meter.avg}, global_step=x_axis)
        self.writer.add_scalars("Value Loss", {
                            "Train":self.train_v_meter.avg,
                            "Val": self.eval_v_meter.avg}, global_step=x_axis)
        self.writer.add_scalars("Total Loss", {
                            "Train": self.train_pi_meter.avg + self.train_v_meter.avg, 
                            "Val": self.eval_pi_meter.avg+self.eval_v_meter.avg}, global_step=x_axis)
        # 每个epoch，记录梯度，权值
        for name, params in self.net_work.named_parameters():
            if 'bn' not in name:
                self.writer.add_histogram(name + "_gard", params.grad, global_step=x_axis)
                self.writer.add_histogram(name + "_data", params, global_step=x_axis)

        self.save_model_weight(self.net_work, self.optimizer, self.lr_scheduler, self.train_step_count, self.model_path + "/model", num_id = str(x_axis))
        # 最新的模型，当作最好的，后续可以改
        # if self.train_step_count > 10:
        self.save_model_weight(self.net_work, self.optimizer, self.lr_scheduler, self.train_step_count ,self.model_path + "/model", "best")
        print("Successful epoches, saving the model is successful!")
        # 学习率衰减
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.train_step_count += 1


    def get_random_batch(self, dataload):
        """ 从 dataload 中随机获取一个batch
        """
        # 获取数据集的长度（总批次数）
        total_batches = len(dataload)
        # 从总批次数中随机选择一个索引
        random_index = random.randint(0, total_batches - 1)
        # 指定 DataLoader 的迭代器到随机选择的批次
        for idx, batch in enumerate(dataload):
            if idx == random_index:
                # 这里你可以使用抽取到的 batch 数据
                return batch

    def train_one_step(self, net_work, optimizer, batch_data):
        """ 训练一个batch
        """
        net_work.train()

        (batch_state, batch_pi, batch_z) = batch_data
        batch_state = batch_state.to(self.device)
        batch_pi = batch_pi.to(self.device)
        batch_z = batch_z.to(self.device)

        # forward
        pi_hat, v_hat = net_work(batch_state)

        # print("pi_hat: ", pi_hat.shape)
        # print("v_hat: ", v_hat.shape)
        # print("batch_pi: ", batch_pi.shape)
        # print("batch_z: ", batch_z.shape)

        # Calculate losses
        loss_pi = self.calculate_loss_pi(pi_hat, batch_pi)
        loss_v = self.calculate_loss_v(v_hat, batch_z)
        total_loss = loss_pi + loss_v

        # Backpropagation and optimization
        optimizer.zero_grad()
        total_loss.backward()

        optimizer.step()

        return loss_pi, loss_v


    @torch.no_grad()
    def eval_one_step(self, net_work, batch_data):
        """ 验证一步
        """
        net_work.eval()
        (batch_state, batch_pi, batch_z) = batch_data
        batch_state = batch_state.to(self.device)
        batch_pi = batch_pi.to(self.device)
        batch_z = batch_z.to(self.device)

        # forward
        pi_hat, v_hat = net_work(batch_state)

        # Calculate losses
        loss_pi = self.calculate_loss_pi(pi_hat, batch_pi)
        loss_v = self.calculate_loss_v(v_hat, batch_z)
        total_loss = loss_pi + loss_v

        return loss_pi, loss_v

    def calculate_loss_pi(self, pi_hat, pi):
        """ 策略损失
        """
        nn.CrossEntropyLoss()
        pi_hat = pi_hat.view(pi_hat.shape[0], -1)
        pi = pi.view(pi.shape[0], -1)
        return self.cross_entropy_loss(pi_hat, pi, adjust=True)

    def calculate_loss_v(self, v_hat, z):
        """ 价值损失
        """
        v_hat = v_hat.view(v_hat.shape[0], -1)
        z = z.view(z.shape[0], -1)
        return torch.mean(torch.pow((v_hat - z), 2))

    def cross_entropy_loss(self, input, target, reduction="mean", adjust=False, weight=None):
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

        logprobs = torch.log(input)
        if weight is not None:
            logprobs = logprobs * weight
        batchloss = -torch.sum(target * logprobs, dim=1)

        if adjust:
            eps = 1e-8
            bias = target * torch.log(target + eps)
            if weight is not None:
                bias = bias * weight
            bias = torch.sum(bias, dim=1)
            # print("bias: ", bias)
            batchloss += bias

        if reduction == "none":
            return batchloss
        elif reduction == "mean":
            return torch.mean(batchloss)
        elif reduction == "sum":
            return torch.sum(batchloss)
        else:
            assert 0, f"Unsupported reduction mode {reduction}."

    
    def save_all_loss(self, type_name : str, losses: list):
        """ 记录损失
        """
        file_path = self.model_path + "/record_train_data"
        file_name = os.path.join(file_path, type_name + '.csv')
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        ans = ''
        with open(file_name, "a") as file_point:
            for idx in range(len(losses)):
                for i in losses[idx]:
                    ans = ans + str(i) + ','
                file_point.write(ans[:-1] + '\n')
                ans = ''
        
    def save_csv_data(self, type_name : str, data: list, suffix:str='csv'):
        """ 记录损失
        """
        file_path = self.model_path + "/record_train_data"
        file_name = os.path.join(file_path, type_name + '.' + suffix)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        ans = ''
        with open(file_name, "a") as file_point:
            for i in data:
                ans = ans + str(i) + ','
            file_point.write(ans[:-1] + '\n')
            ans = ''

    def load_model_weight(self, model_path, num_id : str):
        """ 加载神经网络和优化器参数
            - model_path : 模型路径
            - num_id : 模型id
        """
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        file_path = os.path.join(model_path + "/", "model_" + num_id + ".pth")
        if not os.path.exists(file_path):
            print ("No model in path {}".format(file_path))
            return None, None, None, None
        params = torch.load(file_path)
        return params["state_dict"], params["optimizer"], params["lr_param"], params["train_step_count"]

    def save_model_weight(self, net : nn.Module, 
                          optimizer : optim.Optimizer, 
                          lr_scheduler : WarmupLRScheduler,
                          train_step_count : int, 
                          model_path : str, num_id : str):
        """ 保存神经网络和优化器参数
            - net : 神经网络
            - opertimizer : 优化器
            - lr_scheduler : 学习率
            - model_path : 模型路径
            - num_id : 模型id
        """
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        file_path = os.path.join(model_path + "/", "model_" + num_id + ".pth")
        torch.save({
            # 保存神经网络模型放到 checkpoint.pth 目录中
            'state_dict': net.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'lr_param' : lr_scheduler.state_dict(),
            "train_step_count" : train_step_count,
        }, file_path)
    
    def initialize_weights(self, model="kaiming_normal", init_type="kaiming_normal"):
        """ 初始化神经网络参数
        """
        assert init_type in ["kaiming_normal", "kaiming_uniform", "signed_const"]
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if init_type == "signed_const":
                    n = math.sqrt(
                        2.0 / (m.kernel_size[0] * m.kernel_size[1] * m.in_channels)
                    )
                    m.weight.data = m.weight.data.sign() * n
                elif init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                if init_type == "signed_const":
                    n = math.sqrt(2.0 / m.in_features)
                    m.weight.data = m.weight.data.sign() * n
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

