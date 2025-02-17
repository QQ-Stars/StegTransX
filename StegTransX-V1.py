import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from ppm import RPPM
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from mmcv.cnn.bricks import ConvModule, build_activation_layer, build_norm_layer
from spp import SPP,sppELAN
from einops import rearrange
import typing as t

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
                                # RPPM(in_channels=dim*16,branch_channels=dim*4,out_channels=dim*16,num_scales=5),
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
