import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from mmcv.cnn.bricks import ConvModule, build_activation_layer, build_norm_layer
from spp import SPP,sppELAN
from einops import rearrange
import typing as t
import numpy as np
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math

class Conv(nn.Module):
    """标准的卷积模块，包含卷积、归一化和激活函数。"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, activation=True):
        super(Conv, self).__init__()
        padding = (kernel_size - 1) // 2  # 计算填充大小
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding=padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU() if activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class ELAN(nn.Module):
    """ELAN 结构的实现。"""
    def __init__(self, in_channels, out_channels):
        super(ELAN, self).__init__()
        hidden_channels = out_channels // 2

        self.branch1_conv1 = Conv(in_channels, hidden_channels, kernel_size=1)
        self.branch1_conv2 = Conv(hidden_channels, hidden_channels, kernel_size=3)

        self.branch2_conv1 = Conv(in_channels, hidden_channels, kernel_size=1)
        self.branch2_conv2 = Conv(hidden_channels, hidden_channels, kernel_size=3)

        self.concat_conv = Conv(hidden_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        b1 = self.branch1_conv1(x)
        b1 = self.branch1_conv2(b1)

        b2 = self.branch2_conv1(x)
        b2 = self.branch2_conv2(b2)

        concat = torch.cat([b1, b2], dim=1)
        out = self.concat_conv(concat)
        return out

class SPP(nn.Module):
    """空间金字塔池化（SPP）层的实现。"""
    def __init__(self, in_channels):
        super(SPP, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1)

        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

        self.conv2 = Conv(hidden_channels * 4, in_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)
        x = torch.cat([x, x1, x2, x3], dim=1)
        x = self.conv2(x)
        return x

class sppELAN(nn.Module):
    """sppELAN 模块的完整实现。"""
    def __init__(self, in_channels, out_channels):
        super(sppELAN, self).__init__()
        self.initial_conv = Conv(in_channels, in_channels, kernel_size=1)

        self.spp = SPP(in_channels)

        self.elan = ELAN(in_channels, out_channels)

    def forward(self, x):
        x1 = self.initial_conv(x)
        x2 = self.spp(x1)
        x3 = self.elan(x2) + x2
        return x3


class Downsampling(nn.Module):
    """
    Downsampling implemented by a layer of convolution.
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding),
                                  )

    def forward(self, x):
        x = self.conv(x)
        return x

class Upsampling(nn.Module):
    """
    Upsampling implemented by a layer of convTranspose2d.
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        mlp_hidden_dim = int(dim * mlp_ratio)  # 转换为整数
        self.fc1 = nn.Conv2d(dim, mlp_hidden_dim, 1)
        self.pos = nn.Conv2d(mlp_hidden_dim, mlp_hidden_dim, 3, padding=1, groups=mlp_hidden_dim)
        self.fc2 = nn.Conv2d(mlp_hidden_dim, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x

class SpatialAttention(nn.Module):
    def __init__(self, dim, kernel_size, expand_ratio=2):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.att = nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)        
        x = self.att(x) * self.v(x)
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self,  dim, kernel_size, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.attn = SpatialAttention(dim, kernel_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6           
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SCSA(nn.Module):
    def __init__(
            self,
            dim: int,
            head_num: int,
            window_size: int = 7,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
    ):
        super(SCSA, self).__init__()  # 调用 nn.Module 的构造函数
        self.dim = dim  # 特征维度
        self.head_num = head_num  # 注意力头数
        self.head_dim = dim // head_num  # 每个头的维度
        self.scaler = self.head_dim ** -0.5  # 缩放因子
        self.group_kernel_sizes = group_kernel_sizes  # 分组卷积核大小
        self.window_size = window_size  # 窗口大小
        self.qkv_bias = qkv_bias  # 是否使用偏置
        self.fuse_bn = fuse_bn  # 是否融合批归一化
        self.down_sample_mode = down_sample_mode  # 下采样模式

        assert self.dim % 4 == 0, 'The dimension of input feature should be divisible by 4.'  # 确保维度可被4整除
        self.group_chans = group_chans = self.dim // 4  # 分组通道数

        # 定义局部和全局深度卷积层
        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)

        # 注意力门控层
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)  # 水平方向的归一化
        self.norm_w = nn.GroupNorm(4, dim)  # 垂直方向的归一化

        self.conv_d = nn.Identity()  # 直接连接
        self.norm = nn.GroupNorm(1, dim)  # 通道归一化
        # 定义查询、键和值的卷积层
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)  # 注意力丢弃层
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()  # 通道注意力门控

        # 根据窗口大小和下采样模式选择下采样函数
        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans  # 重组合下采样
                # 维度降低
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)  # 平均池化
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)  # 最大池化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入张量 x 的维度为 (B, C, H, W)
        """
        # 计算空间注意力优先级
        b, c, h_, w_ = x.size()  # 获取输入的形状
        # (B, C, H)
        x_h = x.mean(dim=3)  # 沿着宽度维度求平均
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)  # 拆分通道
        # (B, C, W)
        x_w = x.mean(dim=2)  # 沿着高度维度求平均
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)  # 拆分通道

        # 计算水平注意力
        x_h_attn = self.sa_gate(self.norm_h(torch.cat((  
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)  # 调整形状

        # 计算垂直注意力
        x_w_attn = self.sa_gate(self.norm_w(torch.cat((  
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)  # 调整形状

        # 计算最终的注意力加权
        x = x * x_h_attn * x_w_attn

        # 基于自注意力的通道注意力
        # 减少计算量
        y = self.down_func(x)  # 下采样
        y = self.conv_d(y)  # 维度转换
        _, _, h_, w_ = y.size()  # 获取形状

        # 先归一化，然后重塑 -> (B, H, W, C) -> (B, C, H * W)，并生成 q, k 和 v
        y = self.norm(y)  # 归一化
        q = self.q(y)  # 计算查询
        k = self.k(y)  # 计算键
        v = self.v(y)  # 计算值
        # (B, C, H, W) -> (B, head_num, head_dim, N)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))

        # 计算注意力
        attn = q @ k.transpose(-2, -1) * self.scaler  # 点积注意力计算
        attn = self.attn_drop(attn.softmax(dim=-1))  # 应用注意力丢弃
        # (B, head_num, head_dim, N)
        attn = attn @ v  # 加权值
        # (B, C, H_, W_)
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
        # (B, C, 1, 1)
        attn = attn.mean((2, 3), keepdim=True)  # 求平均
        attn = self.ca_gate(attn)  # 应用通道注意力门控
        return attn * x  # 返回加权后的输入


class stegTransX(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=16):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=3, stride=1, padding=1),
            )

        self.encode_0 = nn.Sequential(
                                    Block(dim=dim,kernel_size=7),
                                    )
        self.downsample_0 = Downsampling(in_channels=dim, out_channels=dim * 2, kernel_size=4, stride=2, padding=1)

        self.encode_1 = nn.Sequential(
                                    Block(dim=dim*2,kernel_size=7),
                                    )
        self.downsample_1 = Downsampling(in_channels=dim * 2, out_channels=dim * 4, kernel_size=4, stride=2, padding=1)

        self.encode_2 = nn.Sequential(
                                    Block(dim=dim*4,kernel_size=7),
                                    )
        self.downsample_2 = Downsampling(in_channels=dim * 4, out_channels=dim * 8, kernel_size=4, stride=2, padding=1)

        self.encode_3 = nn.Sequential(
                                    Block(dim=dim*8,kernel_size=7),
                                    )
        self.downsample_3 = Downsampling(in_channels=dim * 8, out_channels=dim * 16, kernel_size=4, stride=2, padding=1)

        self.bottle= nn.Sequential(
                                SCSA(dim=dim*16,head_num=16),
                                sppELAN(in_channels=dim*16,out_channels=dim*16),
                                )
        self.upsample_0 = UpsampleBlock(in_channels=dim*16,out_channels=dim*8)
        self.decode_0 = nn.Sequential(
                                    Block(dim=dim*16,kernel_size=7),
                                    )
        self.upsample_1 = UpsampleBlock(in_channels=dim*16,out_channels=dim*4)
        self.decode_1 = nn.Sequential(
                                    Block(dim=dim*8,kernel_size=7),
                                    )

        self.upsample_2 = UpsampleBlock(in_channels=dim*8,out_channels=dim*2) 
        self.decode_2 = nn.Sequential(
                                    Block(dim=dim*4,kernel_size=7),
                                    )

        self.upsample_3 = UpsampleBlock(in_channels=dim*4,out_channels=dim)
        self.decode_3 = nn.Sequential(
                                    Block(dim=dim*2,kernel_size=7),
                                    )
        self.out =  nn.Sequential(
                                nn.Conv2d(in_channels=dim * 2, out_channels=out_channels, kernel_size=3,stride=1,padding=1),
                            )
 




    def forward(self, x):
        first = self.stem(x)

        encode_0 = self.encode_0(first)
        downsample_0 = self.downsample_0(encode_0)

        encode_1 = self.encode_1(downsample_0)
        downsample_1 = self.downsample_1(encode_1)

        encode_2 = self.encode_2(downsample_1)
        downsample_2 = self.downsample_2(encode_2)

        encode_3 = self.encode_3(downsample_2)
        downsample_3 = self.downsample_3(encode_3)

        # bottle neck
        bottle = self.bottle(downsample_3)

        upsample_0 = self.upsample_0(bottle)
        decode_0 = torch.cat([upsample_0, encode_3], dim=1)
        decode_0 = self.decode_0(decode_0)

        upsample_1 = self.upsample_1(decode_0)
        decode_1 = torch.cat([upsample_1, encode_2], dim=1)
        decode_1 = self.decode_1(decode_1)
        

        upsample_2 = self.upsample_2(decode_1)
        decode_2 = torch.cat([upsample_2, encode_1], dim=1)
        decode_2 = self.decode_2(decode_2)

        upsample_3 = self.upsample_3(decode_2)
        decode_3 = torch.cat([upsample_3, encode_0], dim=1)
        decode_3 = self.decode_3(decode_3)

        out = self.out(decode_3)

        return out

if __name__ == "__main__":
    import torch
    import thop
    from thop import profile
    from torch.autograd import Variable
    import numpy as np
    import os

    dc = stegTransX(in_channels=3, out_channels=3)
    net = dc.cuda()
    input = torch.randn(1, 3, 256, 256)
    input = input.cuda()
    flops, params = profile(net, (input,))
    flops, params = thop.clever_format([flops, params], "%.3f") 
    print('flops: ', flops, 'params: ', params)
