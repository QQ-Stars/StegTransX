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
from einops import rearrange
import typing as t

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

class Attention(nn.Module):  ### OSRA
    def __init__(self, dim,
                 num_heads=1,
                 qk_scale=None,
                 attn_drop=0,
                 sr_ratio=1,):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                ConvModule(dim, dim,
                           kernel_size=sr_ratio+3,
                           stride=sr_ratio,
                           padding=(sr_ratio+3)//2,
                           groups=dim,
                           bias=False,
                           norm_cfg=dict(type='BN2d'),
                           act_cfg=dict(type='GELU')),
                ConvModule(dim, dim,
                           kernel_size=1,
                           groups=dim,
                           bias=False,
                           norm_cfg=dict(type='BN2d'),
                           act_cfg=None,),)
        else:
            self.sr = nn.Identity()
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, relative_pos_enc=None):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, self.num_heads, C//self.num_heads, -1).transpose(-1, -2)
        kv = self.sr(x)
        kv = self.local_conv(kv) + kv
        k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)
        k = k.reshape(B, self.num_heads, C//self.num_heads, -1)
        v = v.reshape(B, self.num_heads, C//self.num_heads, -1).transpose(-1, -2)
        attn = (q @ k) * self.scale
        if relative_pos_enc is not None:
            if attn.shape[2:] != relative_pos_enc.shape[2:]:
                relative_pos_enc = F.interpolate(relative_pos_enc, size=attn.shape[2:], 
                                                 mode='bicubic', align_corners=False)
            attn = attn + relative_pos_enc
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-1, -2)
        return x.reshape(B, C, H, W)

class DynamicConv2d(nn.Module): ### IDConv
    def __init__(self,
                 dim,
                 kernel_size=3,
                 reduction_ratio=4,
                 num_groups=1,
                 bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        self.proj = nn.Sequential(
            ConvModule(dim, 
                       dim//reduction_ratio,
                       kernel_size=1,
                       norm_cfg=dict(type='BN2d'),
                       act_cfg=dict(type='GELU'),),
            nn.Conv2d(dim//reduction_ratio, dim*num_groups, kernel_size=1),)

        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):

        B, C, H, W = x.shape
        scale = self.proj(self.pool(x)).reshape(B, self.num_groups, C, self.K, self.K)
        scale = torch.softmax(scale, dim=1)
        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1, keepdim=False)
        weight = weight.reshape(-1, 1, self.K, self.K)

        if self.bias is not None:
            scale = self.proj(torch.mean(x, dim=[-2, -1], keepdim=True))
            scale = torch.softmax(scale.reshape(B, self.num_groups, C), dim=1)
            bias = scale * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
        else:
            bias = None

        x = F.conv2d(x.reshape(1, -1, H, W),
                     weight=weight,
                     padding=self.K//2,
                     groups=B*C,
                     bias=bias)
        
        return x.reshape(B, C, H, W)

class HybridTokenMixer(nn.Module): ### D-Mixer
    def __init__(self, 
                 dim,
                 kernel_size=3,
                 num_groups=2,
                 num_heads=1,
                 sr_ratio=1,
                 reduction_ratio=8):
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divided by 2."

        self.local_unit = DynamicConv2d(
            dim=dim//2, kernel_size=kernel_size, num_groups=num_groups)
        self.global_unit = Attention(
            dim=dim//2, num_heads=num_heads, sr_ratio=sr_ratio)
        
        inner_dim = max(16, dim//reduction_ratio)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),)

    def forward(self, x, relative_pos_enc=None):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = self.local_unit(x1)
        x2 = self.global_unit(x2, relative_pos_enc)
        x = torch.cat([x1, x2], dim=1)
        x = self.proj(x) + x ## STE
        return x

class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels,
                             kernel_size=scale[i],
                             padding=scale[i]//2,
                             groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)
            
    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x

class Mlp(nn.Module):  ### MS-FFN
    """
    Mlp implemented by with 1x1 convolutions.

    Input: Tensor with shape [B, C, H, W].
    Output: Tensor with shape [B, C, H, W].
    Args:
        in_features (int): Dimension of input features.
        hidden_features (int): Dimension of hidden features.
        out_features (int): Dimension of output features.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0,):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            build_activation_layer(act_cfg),
            nn.BatchNorm2d(hidden_features),
        )
        self.dwconv = MultiScaleDWConv(hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_features),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        
        x = self.fc1(x)

        x = self.dwconv(x) + x
        x = self.norm(self.act(x))
        
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value, 
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x

class Block(nn.Module):
    """
    Network Block.
    Args:
        dim (int): Embedding dim.
        kernel_size (int): kernel size of dynamic conv. Defaults to 3.
        num_groups (int): num_groups of dynamic conv. Defaults to 2.
        num_heads (int): num_groups of self-attention. Defaults to 1.
        mlp_ratio (float): Mlp expansion ratio. Defaults to 4.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='GN', num_groups=1)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.
        drop_path (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-5.
    """

    def __init__(self,
                 dim=64,
                 kernel_size=7,
                 sr_ratio=1,
                 num_groups=2,
                 num_heads=1,
                 mlp_ratio=4,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 drop=0,
                 drop_path=0.,
                 layer_scale_init_value=1e-5,
                 grad_checkpoint=False):

        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.token_mixer = HybridTokenMixer(dim,
                                            kernel_size=kernel_size,
                                            num_groups=num_groups,
                                            num_heads=num_heads,
                                            sr_ratio=sr_ratio)
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_cfg=act_cfg,
                       drop=drop,)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale(dim, layer_scale_init_value)
            self.layer_scale_2 = LayerScale(dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()

    def _forward_impl(self, x, relative_pos_enc=None):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.layer_scale_1(
                self.token_mixer(self.norm1(x), relative_pos_enc)))
        x = x + self.drop_path(self.layer_scale_2(self.mlp(self.norm2(x))))
        return x

    def forward(self, x, relative_pos_enc=None):
        if self.grad_checkpoint and x.requires_grad:
            x = checkpoint.checkpoint(self._forward_impl, x, relative_pos_enc)
        else:
            x = self._forward_impl(x, relative_pos_enc)
        return x

class basic_blocks(nn.Module):
    def __init__(self,
                 dim,
                 index,
                 layers,
                 kernel_size=7,
                 num_groups=2,
                 num_heads=1,
                 sr_ratio=1,
                 mlp_ratio=4,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 drop_rate=0,
                 drop_path_rate=0,
                 layer_scale_init_value=1e-5,
                 grad_checkpoint=False):
        super().__init__()
        blocks = nn.ModuleList()
        for block_idx in range(layers[index]):
            block_dpr = drop_path_rate * (
                block_idx + sum(layers[:index])) / (sum(layers) - 1)
            blocks.append(
                Block(
                    dim,
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                    num_heads=num_heads,
                    sr_ratio=sr_ratio,
                    mlp_ratio=mlp_ratio,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    layer_scale_init_value=layer_scale_init_value,
                    grad_checkpoint=grad_checkpoint,
                ))
    def forward(self,x):
        for blk in self.blocks:
            x = blk(x)
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
                                    Block(dim=dim,kernel_size=7,sr_ratio=8,num_groups=2,num_heads=2,mlp_ratio=4),
                                    )
        self.downsample_0 = Downsampling(in_channels=dim, out_channels=dim * 2, kernel_size=4, stride=2, padding=1)

        self.encode_1 = nn.Sequential(
                                    Block(dim=dim*2,kernel_size=7,sr_ratio=8,num_groups=2,num_heads=4,mlp_ratio=4),
                                    )
        self.downsample_1 = Downsampling(in_channels=dim * 2, out_channels=dim * 4, kernel_size=4, stride=2, padding=1)

        self.encode_2 = nn.Sequential(
                                    Block(dim=dim*4,kernel_size=7,sr_ratio=8,num_groups=4,num_heads=8,mlp_ratio=4),    
                                    )
        self.downsample_2 = Downsampling(in_channels=dim * 4, out_channels=dim * 8, kernel_size=4, stride=2, padding=1)

        self.encode_3 = nn.Sequential(
                                    Block(dim=dim*8,kernel_size=7,sr_ratio=8,num_groups=4,num_heads=16,mlp_ratio=4),
                                    )
        self.downsample_3 = Downsampling(in_channels=dim * 8, out_channels=dim * 16, kernel_size=4, stride=2, padding=1)

        self.bottle= nn.Sequential(
                                SCSA(dim=dim*16,head_num=16),
                                sppELAN(in_channels=dim*16,out_channels=dim*16),
                                )
        self.upsample_0 = UpsampleBlock(in_channels=dim*16,out_channels=dim*8)
        self.decode_0 = nn.Sequential(
                                    Block(dim=dim*16,kernel_size=7,sr_ratio=8,num_groups=4,num_heads=16,mlp_ratio=4),
                                    )
        self.upsample_1 = UpsampleBlock(in_channels=dim*16,out_channels=dim*4)
        self.decode_1 = nn.Sequential(
                                    Block(dim=dim*8,kernel_size=7,sr_ratio=8,num_groups=4,num_heads=8,mlp_ratio=4),
                                    )

        self.upsample_2 = UpsampleBlock(in_channels=dim*8,out_channels=dim*2) 
        self.decode_2 = nn.Sequential(
                                    Block(dim=dim*4,kernel_size=7,sr_ratio=8,num_groups=2,num_heads=4,mlp_ratio=4),
                                    )

        self.upsample_3 = UpsampleBlock(in_channels=dim*4,out_channels=dim)
        self.decode_3 = nn.Sequential(
                                    Block(dim=dim*2,kernel_size=7,sr_ratio=8,num_groups=2,num_heads=2,mlp_ratio=4),
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
