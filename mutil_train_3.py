import os
# os.environ["CUDA_VISIBLE_DEVICES"]='1'  # 指定使用1号显卡为主卡
import torch
import torch.nn
from torch import nn
import torch.optim
import math
import numpy as np
import config as c
import random
import torchvision.transforms.functional as TF
from tensorboardX import SummaryWriter
import warnings
import logging
import dataset
import timm
import timm.scheduler
from torchsummary import summary
from tqdm import tqdm
import torch.nn.functional as F
from math import exp
# from stegtrans import stegDcnv4
from stegTransx import stegDcnv4
from compress import jpeg_compress_batch
from critic import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(2024)  # 将42替换为您喜欢的任何种子值


class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

class LaplacianPyramidLoss(nn.Module):
    def __init__(self, num_levels=3, kernel_size=5, sigma=1.0):
        """
        初始化多尺度拉普拉斯金字塔损失函数
        :param num_levels: 金字塔的层数
        :param kernel_size: 高斯模糊核的大小
        :param sigma: 高斯模糊的标准差
        """
        super(LaplacianPyramidLoss, self).__init__()
        self.num_levels = num_levels
        self.kernel = self.create_gaussian_kernel(kernel_size, sigma)
        self.padding = kernel_size // 2
        self.charbonnier_loss = L1_Charbonnier_loss()

    def create_gaussian_kernel(self, kernel_size, sigma):
        """
        创建一个高斯模糊核
        :param kernel_size: 核的大小
        :param sigma: 高斯分布的标准差
        :return: 高斯核
        """
        # 创建1D高斯核
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid([ax, ax], indexing='ij')  # 使用 'ij' 以兼容新版 PyTorch
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        return kernel

    def gaussian_blur(self, img):
        """
        对图像进行高斯模糊
        :param img: 输入图像，形状 [batch_size, channels, height, width]
        :return: 模糊后的图像
        """
        channels = img.shape[1]
        kernel = self.kernel.to(img.device)
        kernel = kernel.repeat(channels, 1, 1, 1)
        return F.conv2d(img, kernel, padding=self.padding, groups=channels)

    def build_pyramid(self, img):
        """
        构建拉普拉斯金字塔
        :param img: 输入图像，形状 [batch_size, channels, height, width]
        :return: 拉普拉斯金字塔列表
        """
        gaussian_pyramid = [img]
        for _ in range(self.num_levels):
            img = self.gaussian_blur(img)
            img = F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)
            gaussian_pyramid.append(img)
        
        laplacian_pyramid = []
        for i in range(self.num_levels):
            current = gaussian_pyramid[i]
            next_level = gaussian_pyramid[i + 1]
            # 上采样回原始尺寸
            next_level_up = F.interpolate(next_level, size=current.shape[2:], mode='bilinear', align_corners=False)
            # 计算拉普拉斯层
            laplacian = current - self.gaussian_blur(next_level_up)
            laplacian_pyramid.append(laplacian)
        return laplacian_pyramid

    def forward(self, pred, target):
        """
        计算多尺度拉普拉斯金字塔损失
        :param pred: 生成图像，形状 [batch_size, channels, height, width]
        :param target: 目标图像，形状 [batch_size, channels, height, width]
        :return: 多尺度拉普拉斯金字塔损失值
        """
        pred_pyramid = self.build_pyramid(pred)
        target_pyramid = self.build_pyramid(target)
        
        loss = 0.0
        for l_pred, l_target in zip(pred_pyramid, target_pyramid):
            loss += self.charbonnier_loss(l_pred, l_target)
        return loss

class Restrict_Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X):
        over_upper = torch.relu(X - 1)
        below_lower = torch.relu(-X)
        error = over_upper + below_lower
        loss = torch.mean(error ** 2)  # 使用平方惩罚
        return loss


def computePSNR(origin, pred):
    img_1 = np.array(origin).astype(np.float64)*255
    img_2 = np.array(pred).astype(np.float64)*255
    mse = np.mean((img_1 - img_2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def setup_logging():
    # 配置日志记录
    logging.basicConfig(level=logging.INFO, filename=c.log_path, filemode="w",
                        format="%(asctime)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

def main():
    warnings.filterwarnings("ignore")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    setup_logging()
    writer = SummaryWriter(logdir=c.t_log, comment='tt', filename_suffix="steg")

    # 初始化模型
    Hnet = stegDcnv4(in_channels=12, out_channels=3)
    Rnet = stegDcnv4(in_channels=3, out_channels=9)

    Hnet.to(device)
    Rnet.to(device)
    if c.is_load:
        Hnet.load_state_dict(torch.load(c.Hload))
        Rnet.load_state_dict(torch.load(c.Rload))

    Hoptim = torch.optim.AdamW(Hnet.parameters(), lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    Roptim = torch.optim.AdamW(Rnet.parameters(), lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

    Hscheduler = timm.scheduler.CosineLRScheduler(optimizer=Hoptim, t_initial=c.epochs, lr_min=1e-7, warmup_t=0,
                                                  warmup_lr_init=5e-6)
    Rscheduler = timm.scheduler.CosineLRScheduler(optimizer=Roptim, t_initial=c.epochs, lr_min=1e-7, warmup_t=0,
                                                  warmup_lr_init=5e-6)
    conceal_loss_function = L1_Charbonnier_loss().to(device)
    reveal_loss_function = L1_Charbonnier_loss().to(device)
    restrict_loss_funtion = Restrict_Loss().to(device)
    lp_loss=LaplacianPyramidLoss(num_levels=5)


    best_loss = float('inf')
    batch_idx = 0
    for epoch in range(c.epochs):
        loop = tqdm((zip(dataset.trainloader, dataset.trainloader1, dataset.trainloader2,dataset.trainloader3)),total = len(dataset.trainloader),leave=True)
        Hnet.train()
        Rnet.train()
        total_loss = 0.0

        for data1, data2, data3, data4 in loop:
            data1 = data1.to(device)
            data2 = data2.to(device)
            data3 = data3.to(device)
            data4 = data4.to(device)

            cover = data1
            secret1 = data2
            secret2 = data3
            secret3 = data4

            all_secret = torch.cat((secret1, secret2, secret3), 1)

            input_img=torch.cat((cover, all_secret), 1)

            Hoptim.zero_grad()
            Roptim.zero_grad()

        
            H_output = Hnet(input_img)
            if c.compress:
                R_input = jpeg_compress_batch(H_output,quality=c.quality)
            else:
                R_input=H_output
            R_output = Rnet(R_input)

            Rsecret1=R_output[:, :3, :, :]
            Rsecret2=R_output[:, 3:6, :, :]
            Rsecret3=R_output[:, 6:, :, :]

            Hloss = conceal_loss_function(H_output, cover)+ restrict_loss_funtion(H_output) + lp_loss(H_output, cover)
            Rloss1 = reveal_loss_function(Rsecret1, secret1) + restrict_loss_funtion(Rsecret1) + lp_loss(Rsecret1, secret1)
            Rloss2 = reveal_loss_function(Rsecret2, secret2) + restrict_loss_funtion(Rsecret2) + lp_loss(Rsecret2, secret2)
            Rloss3 = reveal_loss_function(Rsecret3, secret3) + restrict_loss_funtion(Rsecret3) + lp_loss(Rsecret3, secret3)
            Rloss = Rloss1 + Rloss2 +Rloss3
            All_loss = c.lamda1 * Hloss + c.lamda2 * Rloss

            # 使用scaler缩放损失并执行反向传播
            All_loss.backward()

            # 使用scaler执行优化器的更新步骤
            Hoptim.step()
            Roptim.step()

            total_loss += All_loss.item()
            loop.set_description(f'Train Epoch [{epoch}/{c.epochs}]')
            loop.set_postfix({'Hloss': Hloss.item(), 'Rloss': Rloss.item()})
            batch_idx += 1

        # 学习率调度和日志记录
        current_lr = Hoptim.param_groups[0]['lr']
        current_lr1 = Roptim.param_groups[0]['lr']
        logging.info(
            f'Train Epoch [{epoch}/{c.epochs}] All_loss: {total_loss / len(dataset.trainloader)} HCurrent_lr: {current_lr} RCurrent_lr: {current_lr1}')
        writer.add_scalars("Train", {"Train_Loss": total_loss / len(dataset.trainloader)}, epoch + 1)

        # 验证循环
        loop = tqdm((zip(dataset.testloader, dataset.testloader1, dataset.testloader2, dataset.testloader3)), total=len(dataset.testloader), leave=True)
        Hnet.eval()
        Rnet.eval()

        with torch.no_grad():
            psnr_c = []
            psnr_s1 = []
            psnr_s2 = []
            psnr_s3 = []

            y_psnr_c = []
            y_psnr_s1 = []   
            y_psnr_s2 = []
            y_psnr_s3 = []

            total_loss = 0.0
            for data1, data2, data3, data4 in loop:
                data1 = data1.to(device)
                data2 = data2.to(device)
                data3 = data3.to(device)
                data4 = data4.to(device)

                cover = data1
                secret1 = data2
                secret2 = data3
                secret3 = data4

                all_secret = torch.cat((secret1, secret2, secret3), 1)
                input_img=torch.cat((cover, all_secret), 1)
                    
                
                H_output = Hnet(input_img)
                if c.compress:
                    R_input = jpeg_compress_batch(H_output,quality=c.quality)
                else:
                    R_input=H_output
                R_output = Rnet(R_input)

                Hloss = conceal_loss_function(H_output, cover) + restrict_loss_funtion(H_output) + lp_loss(H_output, cover)
                Rloss = reveal_loss_function(R_output, all_secret)  + restrict_loss_funtion(R_output) + lp_loss(R_output, all_secret)
                total_loss += c.lamda1 * Hloss + c.lamda2 * Rloss

                cover = cover.cpu()
                secret1 = secret1.cpu()
                secret2 = secret2.cpu()
                secret3 = secret3.cpu()

                Rsecret1=R_output[:, :3, :, :].clamp(0, 1).cpu()
                Rsecret2=R_output[:, 3:6, :, :].clamp(0, 1).cpu()
                Rsecret3=R_output[:, 6:, :, :].clamp(0, 1).cpu()

                stego = H_output.clamp(0, 1).cpu()

                psnr_temp_c=computePSNR(cover,stego)
                psnr_c.append(psnr_temp_c)
                y_psnr_temp_c=calculate_psnr_skimage(cover,stego)
                y_psnr_c.append(y_psnr_temp_c)

                psnr_temp_s1=computePSNR(Rsecret1,secret1)
                psnr_s1.append(psnr_temp_s1)
                y_psnr_temp_s1=calculate_psnr_skimage(Rsecret1,secret1)
                y_psnr_s1.append(y_psnr_temp_s1)

                psnr_temp_s2=computePSNR(Rsecret2,secret2)
                psnr_s2.append(psnr_temp_s2)
                y_psnr_temp_s2=calculate_psnr_skimage(Rsecret2,secret2)
                y_psnr_s2.append(y_psnr_temp_s2)

                psnr_temp_s3=computePSNR(Rsecret3,secret3)
                psnr_s3.append(psnr_temp_s3)
                y_psnr_temp_s3=calculate_psnr_skimage(Rsecret3,secret3)
                y_psnr_s3.append(y_psnr_temp_s3)

                loop.set_description(f'Val')
                loop.set_postfix({'Hloss': Hloss.item(), 'Rloss': Rloss.item()})

            logging.info(
                f'Val Epoch [{epoch}/{c.epochs}] val_loss: {total_loss / len(dataset.testloader)} psnr_c: {np.mean(psnr_c)} psnr_s1: {np.mean(psnr_s1)} psnr_s2: {np.mean(psnr_s2)} psnr_s3: {np.mean(psnr_s3)} y_psnr_c:{np.mean(y_psnr_c)} y_psnr_s1:{np.mean(y_psnr_s1)} y_psnr_s2:{np.mean(y_psnr_s2)} y_psnr_s3:{np.mean(y_psnr_s3)}')
            writer.add_scalars("PSNR_C", {"average": np.mean(psnr_c)}, epoch + 1)
            writer.add_scalars("PSNR_S1", {"average": np.mean(psnr_s1)}, epoch + 1)
            writer.add_scalars("PSNR_S2", {"average": np.mean(psnr_s2)}, epoch + 1)
            writer.add_scalars("PSNR_S3", {"average": np.mean(psnr_s3)}, epoch + 1)
            writer.add_scalars("val", {"val_Loss": total_loss / len(dataset.testloader)}, epoch + 1)

            Hscheduler.step(epoch)
            Rscheduler.step(epoch)

            # 保存最佳模型
            if best_loss > total_loss:
                best_loss = total_loss
                torch.save(Hnet.state_dict(), f'{c.HMODEL_PATH}/Hmodel.pth')
                torch.save(Rnet.state_dict(), f'{c.RMODEL_PATH}/Rmodel.pth')
                total_loss = 0.0
                logging.info(f'model checkpoint saved!')

            # 每100个epoch保存一次模型
            if epoch % 100 == 0:
                torch.save(Hnet.state_dict(), f'{c.HMODEL_PATH_100}/Hmodel.pth')
                torch.save(Rnet.state_dict(), f'{c.RMODEL_PATH_100}/Rmodel.pth')
                logging.info(f'model checkpoint saved!')

if __name__ == '__main__':
    main()
