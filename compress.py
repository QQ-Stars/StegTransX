import os
import torch
from torch import nn
import math
import numpy as np
import random
import torchvision.transforms.functional as TF
import warnings
import logging
import torch.nn.functional as F
from math import exp
import itertools
import kornia
import kornia.augmentation as K
import kornia.filters as KF

# 定义必要的函数
def diff_round(x):
    """Differentiable rounding function"""
    sign = torch.ones_like(x)
    sign[torch.floor(x) % 2 == 0] = -1
    y = sign * torch.cos(x * math.pi) / 2
    out = torch.round(x) + y - y.detach()
    return out

def quality_to_factor(quality):
    """Calculate factor corresponding to quality"""
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality * 2
    return quality / 100.

# 定义 JPEG 量化表
y_table_np = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32).T

c_table_np = np.empty((8, 8), dtype=np.float32)
c_table_np.fill(99)
c_table_np[:4, :4] = np.array([
    [17, 18, 24, 47],
    [18, 21, 26, 66],
    [24, 26, 56, 99],
    [47, 66, 99, 99]]).T



# 定义 DiffJPEG 类及其依赖模块
class rgb_to_ycbcr_jpeg(nn.Module):
    def __init__(self):
        super(rgb_to_ycbcr_jpeg, self).__init__()
        matrix = np.array([
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312]
        ], dtype=np.float32).T
        self.register_buffer('shift', torch.tensor([0., 128., 128.]))
        self.register_buffer('matrix', torch.from_numpy(matrix))

    def forward(self, image):
        image = image.permute(0, 2, 3, 1)
        result = torch.tensordot(image, self.matrix.to(image.device), dims=1) + self.shift.to(image.device)
        result = result.view(image.shape)
        return result

class chroma_subsampling(nn.Module):
    def __init__(self):
        super(chroma_subsampling, self).__init__()

    def forward(self, image):
        image_2 = image.permute(0, 3, 1, 2).clone()
        avg_pool = nn.AvgPool2d(kernel_size=2, stride=(2, 2),
                                count_include_pad=False)
        cb = avg_pool(image_2[:, 1, :, :].unsqueeze(1))
        cr = avg_pool(image_2[:, 2, :, :].unsqueeze(1))
        cb = cb.permute(0, 2, 3, 1)
        cr = cr.permute(0, 2, 3, 1)
        return image[:, :, :, 0], cb.squeeze(3), cr.squeeze(3)

class block_splitting(nn.Module):
    def __init__(self):
        super(block_splitting, self).__init__()
        self.k = 8

    def forward(self, image):
        height, width = image.shape[1:3]
        batch_size = image.shape[0]
        image_reshaped = image.view(batch_size, height // self.k, self.k, -1, self.k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, -1, self.k, self.k)

class dct_8x8(nn.Module):
    def __init__(self):
        super(dct_8x8, self).__init__()
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)

        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * math.pi / 16) * np.cos((2 * y + 1) * v * math.pi / 16)

        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)

        self.register_buffer('tensor', torch.from_numpy(tensor).float())
        self.register_buffer('scale', torch.from_numpy(np.outer(alpha, alpha) * 0.25).float())

    def forward(self, image):
        image = image - 128
        result = self.scale.to(image.device) * torch.tensordot(image, self.tensor.to(image.device), dims=2)
        result = result.view(image.shape)
        return result

class compress_jpeg(nn.Module):
    def __init__(self, rounding=torch.round, factor=1):
        super(compress_jpeg, self).__init__()
        self.l1 = nn.Sequential(
            rgb_to_ycbcr_jpeg(),
            chroma_subsampling()
        )
        self.l2 = nn.Sequential(
            block_splitting(),
            dct_8x8()
        )

        self.factor = factor
        self.rounding = rounding

        # 将量化表注册为缓冲区
        self.register_buffer('c_table', torch.from_numpy(c_table_np))
        self.register_buffer('y_table', torch.from_numpy(y_table_np))

    def forward(self, image):
        y, cb, cr = self.l1(image)
        components = {
            'y': y,
            'cb': cb,
            'cr': cr
        }
        for k in components.keys():
            comp = self.l2(components[k])
            if k in ('cb', 'cr'):
                comp = comp.float() / (self.c_table.to(comp.device) * self.factor)
                comp = self.rounding(comp)
            else:
                comp = comp.float() / (self.y_table.to(comp.device) * self.factor)
                comp = self.rounding(comp)
            components[k] = comp
        return components['y'], components['cb'], components['cr']

class idct_8x8(nn.Module):
    def __init__(self):
        super(idct_8x8, self).__init__()
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        self.register_buffer('alpha', torch.from_numpy(np.outer(alpha, alpha)).float())
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * u + 1) * x * math.pi / 16) * np.cos((2 * v + 1) * y * math.pi / 16)
        self.register_buffer('tensor', torch.from_numpy(tensor).float())

    def forward(self, image):
        image = image * self.alpha.to(image.device)
        result = 0.25 * torch.tensordot(image, self.tensor.to(image.device), dims=2) + 128
        result = result.view(image.shape)
        return result

class block_merging(nn.Module):
    def __init__(self):
        super(block_merging, self).__init__()

    def forward(self, patches, height, width):
        k = 8
        batch_size = patches.shape[0]
        image_reshaped = patches.view(batch_size, height // k, width // k, k, k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, height, width)

class chroma_upsampling(nn.Module):
    def __init__(self):
        super(chroma_upsampling, self).__init__()

    def forward(self, y, cb, cr):
        def repeat(x, k=2):
            height, width = x.shape[1:3]
            x = x.unsqueeze(-1)
            x = x.repeat(1, 1, k, k)
            x = x.view(-1, height * k, width * k)
            return x

        cb = repeat(cb)
        cr = repeat(cr)
        return torch.cat([y.unsqueeze(3), cb.unsqueeze(3), cr.unsqueeze(3)], dim=3)

class ycbcr_to_rgb_jpeg(nn.Module):
    def __init__(self):
        super(ycbcr_to_rgb_jpeg, self).__init__()
        matrix = np.array([
            [1., 0., 1.402],
            [1, -0.344136, -0.714136],
            [1, 1.772, 0]
        ], dtype=np.float32).T
        self.register_buffer('shift', torch.tensor([0, -128., -128.]))
        self.register_buffer('matrix', torch.from_numpy(matrix))

    def forward(self, image):
        result = torch.tensordot(image + self.shift.to(image.device), self.matrix.to(image.device), dims=1)
        result = result.view(image.shape)
        return result.permute(0, 3, 1, 2)

class decompress_jpeg(nn.Module):
    def __init__(self, height, width, rounding=torch.round, factor=1):
        super(decompress_jpeg, self).__init__()
        self.rounding = rounding
        self.factor = factor

        self.idct = idct_8x8()
        self.merging = block_merging()
        self.chroma = chroma_upsampling()
        self.colors = ycbcr_to_rgb_jpeg()
        self.height, self.width = height, width

        # 将量化表注册为缓冲区
        self.register_buffer('c_table', torch.from_numpy(c_table_np))
        self.register_buffer('y_table', torch.from_numpy(y_table_np))

    def forward(self, y, cb, cr):
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            if k in ('cb', 'cr'):
                comp = components[k]
                comp = comp * (self.c_table.to(comp.device) * self.factor)
                height, width = int(self.height / 2), int(self.width / 2)
            else:
                comp = components[k]
                comp = comp * (self.y_table.to(comp.device) * self.factor)
                height, width = self.height, self.width
            comp = self.idct(comp)
            components[k] = self.merging(comp, height, width)
        image = self.chroma(components['y'], components['cb'], components['cr'])
        image = self.colors(image)
        image = torch.min(255 * torch.ones_like(image), torch.max(torch.zeros_like(image), image))
        return image

class DiffJPEG(nn.Module):
    def __init__(self, height, width, differentiable=True, quality=80):
        '''Initialize the DiffJPEG layer'''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compress_jpeg(rounding=rounding, factor=factor)
        self.decompress = decompress_jpeg(height, width, rounding=rounding, factor=factor)

    def forward(self, x):
        y, cb, cr = self.compress(x)
        recovered = self.decompress(y, cb, cr)
        return recovered

    def set_quality(self, quality):
        factor = quality_to_factor(quality)
        self.compress.factor = factor
        self.decompress.factor = factor

# JPEG压缩
def jpeg_compress_batch(images, quality=90):
    B, C, H, W = images.size()
    device = images.device

    # 将图像从 [0,1] 变换到 [0,255]
    images_255 = images * 255.0

    # 创建可微分JPEG模拟器
    jpeg_simulator = DiffJPEG(height=H, width=W, differentiable=True, quality=quality).to(device)

    # 应用JPEG压缩模拟
    compressed_images = jpeg_simulator(images_255)

    # 将图像从 [0,255] 变换回 [0,1]
    compressed_images = compressed_images / 255.0
    compressed_images = torch.clamp(compressed_images, 0, 1)

    return compressed_images

def test_jpeg_compress_with_simulated_input():
    # 设置随机种子以确保结果可复现
    torch.manual_seed(0)

    # 模拟输入：创建一个随机图像张量，形状为 [B, 3, H, W]，数值范围为 [0, 1]
    B, C, H, W = 2, 3, 256, 256  # 您可以调整批量大小和图像尺寸
    random_image = torch.rand(B, C, H, W)

    # 将图像张量移动到设备（CPU 或 GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random_image = random_image.to(device)

    # 设置压缩质量
    quality = 80  # 您可以调整质量因子，范围是 1（最低质量）到 100（最高质量）

    # 应用 JPEG 压缩
    compressed_image = jpeg_compress_batch(random_image, quality=quality)

    # 打印原始和压缩后图像的张量统计信息
    print("原始图像张量统计信息：")
    print(f"最小值: {random_image.min().item()}, 最大值: {random_image.max().item()}, 平均值: {random_image.mean().item()}")

    print("\n压缩后图像张量统计信息：")
    print(f"最小值: {compressed_image.min().item()}, 最大值: {compressed_image.max().item()}, 平均值: {compressed_image.mean().item()}")

    # 检查张量形状是否一致
    print(f"\n原始图像张量形状：{random_image.shape}")
    print(f"压缩后图像张量形状：{compressed_image.shape}")

    # 验证是否存在差异
    difference = torch.abs(random_image - compressed_image)
    print(f"\n原始图像和压缩后图像之间的平均差异：{difference.mean().item()}")

if __name__ == '__main__':
    test_jpeg_compress_with_simulated_input()