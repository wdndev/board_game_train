import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange, reduce

from .nn_utils import DropPath
from .nn_activations import *


inplace = True


# ========== For Common ==========
class LayerNorm2d(nn.Module):
    """ 层归一化
    """
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)
    
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x

def get_norm(norm_layer='in_1d'):
    eps = 1e-6
    norm_dict = {
        'none': nn.Identity,
        'in_1d': partial(nn.InstanceNorm1d, eps=eps),
        'in_2d': partial(nn.InstanceNorm2d, eps=eps),
        'in_3d': partial(nn.InstanceNorm3d, eps=eps),
        'bn_1d': partial(nn.BatchNorm1d, eps=eps),
        'bn_2d': partial(nn.BatchNorm2d, eps=eps),
        # 'bn_2d': partial(nn.SyncBatchNorm, eps=eps),
        'bn_3d': partial(nn.BatchNorm3d, eps=eps),
        'gn': partial(nn.GroupNorm, eps=eps),
        'ln_1d': partial(nn.LayerNorm, eps=eps),
        'ln_2d': partial(LayerNorm2d, eps=eps),
    }
    return norm_dict[norm_layer]


def get_act(act_layer='relu'):
	act_dict = {
		'none': nn.Identity,
		'sigmoid': Sigmoid,
		'swish': Swish,
		'mish': Mish,
		'hsigmoid': HardSigmoid,
		'hswish': HardSwish,
		'hmish': HardMish,
		'tanh': Tanh,
		'relu': nn.ReLU,
		'relu6': nn.ReLU6,
		'prelu': PReLU,
		'gelu': GELU,
		'silu': nn.SiLU
	}
	return act_dict[act_layer]


class LayerScale(nn.Module):
    """ 按元素乘以一个可学习参数 gamma, 以调整输入张量 x 的比例
    """
    def __init__(self, dim, init_values=1e-5, inplace=True):
        """ 
            inplace: 指定是否原地操作，默认为 True
        """
        super().__init__()
        self.inplace = inplace
        # 定义一个可学习参数 gamma，其形状为 (1, 1, dim)
        self.gamma = nn.Parameter(init_values * torch.ones(1, 1, dim))
    
    def forward(self, x):
        # 是否进行原地操作
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LayerScale2D(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=True):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(1, dim, 1, 1))
    
    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    
    
class ConvNormAct(nn.Module):
    """ 实现了卷积、归一化 (Normalization)、激活函数和可选的跳跃连接。
    """
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, bias=False,
                 skip=False, norm_layer='bn_2d', act_layer='relu', inplace=True, drop_path_rate=0.):
        super(ConvNormAct, self).__init__()
        # 确定是否有跳跃连接，且输入和输出维度相同
        self.has_skip = skip and dim_in == dim_out
        # 计算填充量以保持输入和输出特征图大小相同
        padding = math.ceil((kernel_size - stride) / 2)
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = get_norm(norm_layer)(dim_out)
        self.act = get_act(act_layer)(inplace=inplace)
        # 根据设置的 DropPath 率创建 DropPath 层（如果存在）
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()
    
    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x

class SqueezeExcite(nn.Module):
    """ Squeeze-and-Excitation w/ specific features for EfficientNet/MobileNet family

    Args:
        in_chs (int): input channels to layer
        rd_ratio (float): ratio of squeeze reduction
        act_layer (nn.Module): activation layer of containing block
        gate_layer (Callable): attention gate function
        force_act_layer (nn.Module): override block's activation fn if this is set/bound
        rd_round_fn (Callable): specify a fn to calculate rounding of reduced chs
    """

    def __init__(
            self, in_chs, rd_ratio=0.25, rd_channels=None, act_layer='relu',
            gate_layer='sigmoid', force_act_layer=None, rd_round_fn=None):
        super(SqueezeExcite, self).__init__()
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = rd_round_fn(in_chs * rd_ratio)
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = get_act(gate_layer)(inplace=inplace)
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = get_act(gate_layer)(inplace=inplace)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

# ========== Multi-Scale Populations, for down-sampling and inductive bias ==========
class MSPatchEmb(nn.Module):
    """ 多尺度空间嵌入
    """
    def __init__(self, dim_in, emb_dim, kernel_size=2, c_group=-1, stride=1, dilations=[1, 2, 3],
                 norm_layer='bn_2d', act_layer='silu'):
        super().__init__()
        # 记录给定的卷积空洞率的数量
        self.dilation_num = len(dilations)
        # 确保输入维度可以被给定的组数整除
        assert dim_in % c_group == 0
        # 计算输入维度和嵌入维度的最大公约数
        c_group = math.gcd(dim_in, emb_dim) if c_group == -1 else c_group
        self.convs = nn.ModuleList()
        # 根据每个给定的空洞卷积个数，构建卷积层序列
        for i in range(len(dilations)):
            # 计算当前空洞率对应的填充大小
            padding = math.ceil(((kernel_size - 1) * dilations[i] + 1 - stride) / 2)
            # print("{0} : {1}".format(i, padding))
            # 添加卷积、标准化和激活函数层到ModuleList中
            self.convs.append(nn.Sequential(
                nn.Conv2d(dim_in, emb_dim, kernel_size, stride, padding, dilations[i], groups=c_group),
                get_norm(norm_layer)(emb_dim),
                get_act(act_layer)(emb_dim)))
    
    def forward(self, x):
        # 如果只有一个卷积空洞
        if self.dilation_num == 1:
            x = self.convs[0](x)
        else:
            # 对于多个卷积空洞，通过多个卷积层生成不同尺度的特征
            x = torch.cat([self.convs[i](x).unsqueeze(dim=-1) for i in range(self.dilation_num)], dim=-1)
            x = reduce(x, 'b c h w n -> b c h w', 'mean').contiguous()
        return x
    
class PolicyHead(nn.Module):
    """ 策略头
    """
    def __init__(self, in_channel = 128):
        super().__init__()
        self.conv1 = ConvNormAct(in_channel, in_channel // 4, kernel_size=3, stride=1, bias=True, norm_layer='bn_2d', act_layer='silu')
        self.conv2 = ConvNormAct(in_channel // 4, 1, kernel_size=3, stride=1, bias=True, norm_layer='bn_2d', act_layer='silu')
        # self.conv1 = conv_2d(in_channel, in_channel // 2, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv2 = conv_2d(in_channel // 2, 1, kernel_size=3, stride=1, padding=1, bias=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten(1)
        return F.softmax(x, dim=1)
    
class ValueHead(nn.Module):
    """ 价值头
    """
    def __init__(self, in_channel = 128):
        super().__init__()
        self.conv1 = ConvNormAct(in_channel, in_channel, kernel_size=3, stride=1, bias=True, norm_layer='bn_2d', act_layer='silu')
        self.conv2 = ConvNormAct(in_channel, in_channel // 4, kernel_size=3, stride=1, bias=True, norm_layer='bn_2d', act_layer='silu')

        self.fc = nn.Linear(in_features=in_channel // 4, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.mean((2, 3))
        x = self.fc(x)

        return x


class iRMB(nn.Module):
    
    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0, norm_layer='bn_2d',
                 act_layer='relu', v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=64, window_size=3,
                 attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False):
        super().__init__()
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        # 设置是否有跳跃连接
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        # 如果是多尺度注意力机制
        self.attn_s = attn_s
        if self.attn_s:
            assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
            self.dim_head = dim_head
            self.window_size = window_size
            self.num_head = dim_in // dim_head
            self.scale = self.dim_head ** -0.5
            self.attn_pre = attn_pre
            self.qk = ConvNormAct(dim_in, int(dim_in * 2), kernel_size=1, bias=qkv_bias, norm_layer='none', act_layer='none')
            self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias, norm_layer='none', act_layer=act_layer, inplace=inplace)
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            if v_proj:
                self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, bias=qkv_bias, norm_layer='none', act_layer=act_layer, inplace=inplace)
            else:
                self.v = nn.Identity()
        self.conv_local = ConvNormAct(dim_mid, dim_mid, kernel_size=dw_ks, stride=stride, dilation=dilation, groups=dim_mid, norm_layer='bn_2d', act_layer='silu', inplace=inplace)
        self.se = SqueezeExcite(dim_mid, rd_ratio=se_ratio, act_layer=act_layer) if se_ratio > 0.0 else nn.Identity()
        
        self.proj_drop = nn.Dropout(drop)
        self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer='none', act_layer='none', inplace=inplace)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
    
    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        B, C, H, W = x.shape
        if self.attn_s:
            # padding
            if self.window_size <= 0:
                window_size_W, window_size_H = W, H
            else:
                window_size_W, window_size_H = self.window_size, self.window_size
            pad_l, pad_t = 0, 0
            pad_r = (window_size_W - W % window_size_W) % window_size_W
            pad_b = (window_size_H - H % window_size_H) % window_size_H
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
            n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
            x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()
            # attention
            b, c, h, w = x.shape
            qk = self.qk(x)
            qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head, dim_head=self.dim_head).contiguous()
            q, k = qk[0], qk[1]
            attn_spa = (q @ k.transpose(-2, -1)) * self.scale
            attn_spa = attn_spa.softmax(dim=-1)
            attn_spa = self.attn_drop(attn_spa)
            if self.attn_pre:
                x = rearrange(x, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ x
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
                x_spa = self.v(x_spa)
            else:
                v = self.v(x)
                v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ v
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
            # unpadding
            x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        else:
            x = self.v(x)

        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
        
        x = self.proj_drop(x)
        x = self.proj(x)
        
        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x


class EMO(nn.Module):
    
    def __init__(self, dim_in=3, 
                 depths=[1, 2, 4, 2], stem_dim=16, embed_dims=[64, 128, 256, 512], exp_ratios=[4., 4., 4., 4.],
                 norm_layers=['bn_2d', 'bn_2d', 'bn_2d', 'bn_2d'], act_layers=['relu', 'relu', 'relu', 'relu'],
                 dw_kss=[3, 3, 5, 5], se_ratios=[0.0, 0.0, 0.0, 0.0], dim_heads=[32, 32, 32, 32],
                 window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True], qkv_bias=True,
                 attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False, pre_dim=0):
        super().__init__()
        self.mode_name = "emo"
        dprs = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.stage0 = nn.ModuleList([
            MSPatchEmb(  # down to 112
                dim_in, stem_dim, kernel_size=dw_kss[0], c_group=1, stride=1, dilations=[1],
                norm_layer=norm_layers[0], act_layer='none'),
            iRMB(  # ds
                stem_dim, stem_dim, norm_in=False, has_skip=False, exp_ratio=1,
                norm_layer=norm_layers[0], act_layer=act_layers[0], v_proj=False, dw_ks=dw_kss[0],
                stride=1, dilation=1, se_ratio=1,
                dim_head=dim_heads[0], window_size=window_sizes[0], attn_s=False,
                qkv_bias=qkv_bias, attn_drop=attn_drop, drop=drop, drop_path=0.,
                attn_pre=attn_pre
            )
        ])
        emb_dim_pre = stem_dim
        for i in range(len(depths)):
            layers = []
            dpr = dprs[sum(depths[:i]):sum(depths[:i + 1])]
            for j in range(depths[i]):
                if j == 0:
                    stride, has_skip, attn_s, exp_ratio = 1, False, False, exp_ratios[i] * 2
                else:
                    stride, has_skip, attn_s, exp_ratio = 1, True, attn_ss[i], exp_ratios[i]
                layers.append(iRMB(
                    emb_dim_pre, embed_dims[i], norm_in=True, has_skip=has_skip, exp_ratio=exp_ratio,
                    norm_layer=norm_layers[i], act_layer=act_layers[i], v_proj=True, dw_ks=dw_kss[i],
                    stride=stride, dilation=1, se_ratio=se_ratios[i],
                    dim_head=dim_heads[i], window_size=window_sizes[i], attn_s=attn_s,
                    qkv_bias=qkv_bias, attn_drop=attn_drop, drop=drop, drop_path=dpr[j], v_group=v_group,
                    attn_pre=attn_pre
                ))
                emb_dim_pre = embed_dims[i]
            self.__setattr__(f'stage{i + 1}', nn.ModuleList(layers))
        
        self.norm = get_norm(norm_layers[-1])(embed_dims[-1])
        if pre_dim > 0:
            self.pre_head = nn.Sequential(nn.Linear(embed_dims[-1], pre_dim), get_act(act_layers[-1])(inplace=inplace))
            self.pre_dim = pre_dim
        else:
            self.pre_head = nn.Identity()
            self.pre_dim = embed_dims[-1]
        self.policy_head = PolicyHead(in_channel=self.pre_dim)
        self.value_head = ValueHead(in_channel=self.pre_dim)

    def forward_features(self, x):
        for blk in self.stage0:
            x = blk(x)
        for blk in self.stage1:
            x = blk(x)
        for blk in self.stage2:
            x = blk(x)
        for blk in self.stage3:
            x = blk(x)
        for blk in self.stage4:
            x = blk(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm(x)

        p_hat = self.policy_head(x)
        v_hat = self.value_head(x)
        
        return p_hat, v_hat
        x = self.norm(x)
        x = reduce(x, 'b c h w -> b c', 'mean').contiguous()
        x = self.pre_head(x)
        x = self.head(x)
        return {'out': x, 'out_kd': x}


def emo_1m(in_chans=4, board_size=19):
    """ Total of parameters: 143348
    """
    model = EMO(
        dim_in=in_chans,
        depths=[2, 2, 4, 3], stem_dim=24, embed_dims=[32, 32, 32, 32], exp_ratios=[2., 2.5, 3.0, 3.5],
        norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
        dw_kss=[3, 3, 5, 5], dim_heads=[4, 4, 4, 4], window_sizes=[5, 5, 5, 5], attn_ss=[False, False, True, True],
        qkv_bias=True, attn_drop=0., drop=0., drop_path=0.04036, v_group=False, attn_pre=True, pre_dim=0)
    model.model_name = "meo_1m"
    return model


# def EMO_2M(pretrained=False, **kwargs):
#     model = EMO(
#         # dim_in=3, num_classes=1000, img_size=224,
#         depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[32, 48, 120, 200], exp_ratios=[2., 2.5, 3.0, 3.5],
#         norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
#         dw_kss=[3, 3, 5, 5], dim_heads=[16, 16, 20, 20], window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True],
#         qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=True, pre_dim=0,
#         **kwargs)
#     return model


# def EMO_5M(pretrained=False, **kwargs):
#     model = EMO(
#         # dim_in=3, num_classes=1000, img_size=224,
#         depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[48, 72, 160, 288], exp_ratios=[2., 3., 4., 4.],
#         norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
#         dw_kss=[3, 3, 5, 5], dim_heads=[24, 24, 32, 32], window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True],
#         qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=True, pre_dim=0,
#         **kwargs)
#     return model


# def EMO_6M(pretrained=False, **kwargs):
#     model = EMO(
#         # dim_in=3, num_classes=1000, img_size=224,
#         depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[48, 72, 160, 320], exp_ratios=[2., 3., 4., 5.],
#         norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
#         dw_kss=[3, 3, 5, 5], dim_heads=[16, 24, 20, 32], window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True],
#         qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=True, pre_dim=0,
#         **kwargs)
#     return model

def print_model_parameters(model):
    """ 打印模型各个层参数
    """
    param_sum = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_sum += param.numel()
            print(f"Layer: {name}, Parameters: {param.numel()}")
    print(f"Total of parameters: {param_sum}")

if __name__ == '__main__':
    # 随机初始化一个 2x4x19x19 的张量
    tensor = torch.randn(5, 4, 15, 15)
    print(tensor.shape)

    # test_tensor = torch.randn(5, 3, 224, 224)
    print(tensor.shape)
    emo = emo_1m(in_chans=4)
    # emo = EMO(dim_in=3, num_classes=1000, img_size=224,
    #     depths=[2, 2, 8, 3], stem_dim=24, embed_dims=[32, 48, 80, 168], exp_ratios=[2., 2.5, 3.0, 3.5],
    #     norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
    #     dw_kss=[3, 3, 5, 5], dim_heads=[16, 16, 20, 21], window_sizes=[5, 5, 5, 5], attn_ss=[False, False, True, True],
    #     qkv_bias=True, attn_drop=0., drop=0., drop_path=0.04036, v_group=False, attn_pre=True, pre_dim=0)
    # x = emo(tensor)
    # print(x['out'].shape)
    p_hat, v_hat = emo(tensor)

    print("p_hat: ", p_hat.shape)
    print("v_hat: ", v_hat.shape)
    
    print_model_parameters(emo)

    # mspe = MSPatchEmb( 3, 64, kernel_size=3, c_group=1, stride=1, dilations=[1], norm_layer='bn_2d', act_layer='none')
    # irmb = iRMB(dim_in=64, dim_out=64)

    # x = mspe(tensor)
    # print(x.shape)

    # x = irmb(x)
    # print(x.shape)
