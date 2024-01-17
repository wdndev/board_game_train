import torch
from backbone.resnet import resnet_10b128c, resnet_4b64c, resnet_15b128c
from train_net import TrainNet

def to_jit_model(net_work:torch.nn.Module, jit_model:str):
    """ 转化模型为jit格式
    """ 
    net_work.eval()
    jit_model = torch.jit.script(net_work)
    torch.jit.save(jit_model, jit_model)

if __name__ == "__main__":

    batch_size = 512
    lr = 1e-3
    in_chans = 18
    board_size = 19
    is_lr_decay = True

    is_all_feature = True if in_chans == 18 else False
    
    model = resnet_10b128c(in_chans=in_chans, board_size=board_size)
    model.model_name = "resnet_10b128c_inc18"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_net = TrainNet(model, device, lr=lr, is_lr_decay=is_lr_decay)
    train_net.to_jit(train_net.net_work, "test")
