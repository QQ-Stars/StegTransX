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
from stegTransx import stegTransx 
from torchvision import models

# 导入autocast和GradScaler
import random
import numpy as np
import torch
from compress import jpeg_compress_batch
from critic import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(2024) 


def computePSNR(origin, pred):
    img_1 = np.array(origin).astype(np.float64)*255
    img_2 = np.array(pred).astype(np.float64)*255
    mse = np.mean((img_1 - img_2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_out_of_range_ratio(tensor):
    # 计算大于1或小于0的元素数量
    out_of_range_count = torch.sum((tensor > 1) | (tensor < 0))
    
    # 计算总元素数量
    total_count = tensor.numel()
    
    # 计算比例
    ratio = out_of_range_count.float() / total_count
    
    return ratio.item()

def main():
    warnings.filterwarnings("ignore")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    Hnet = stegTransx (in_channels=6, out_channels=3)
    Rnet = stegTransx (in_channels=3, out_channels=3)

    Hnet.to(device)
    Rnet.to(device)

    Hnet.load_state_dict(torch.load(c.Hload))
    Rnet.load_state_dict(torch.load(c.Rload))

    # 验证循环
    loop = tqdm((dataset.testloader), total=len(dataset.testloader), leave=True)
    Hnet.eval()
    Rnet.eval()

    with torch.no_grad():
        # psnr_c = []
        # psnr_s = []
        y_psnr_c = []
        y_psnr_s = []
        ssim_s=[]
        ssim_c=[]
        mae_s=[]
        mae_c=[]
        rmse_s=[]
        rmse_c=[]

        for data in loop:
            data = data.to(device)
            cover = data[data.shape[0] // 2:]
            secret = data[:data.shape[0] // 2]

            input_img = torch.cat([cover,secret], dim=1)  # [batch_size, 6, H, W]
            
            H_output = Hnet(input_img)
            r = calculate_out_of_range_ratio(H_output)
            if c.compress:
                R_input = jpeg_compress_batch(H_output,quality=c.quality)
            else:
                R_input=H_output
            R_input = R_input.clamp(0, 1) # V1需要去掉截断测试，V2需要加上截断测试
            
            R_output = Rnet(R_input)

            cover = cover.cpu()
            secret = secret.cpu()
            Rsecret=R_output.clamp(0, 1).cpu()
            stego = H_output.clamp(0, 1).cpu()

            # psnr
            # psnr_temp_S = computePSNR(Rsecret, secret)
            y_psnr_temp_S = calculate_psnr_skimage(Rsecret, secret)
            y_psnr_s.append(y_psnr_temp_S)
            # psnr_s.append(psnr_temp_S)

            y_psnr_temp_c = calculate_psnr_skimage(stego, cover)
            # psnr_temp_c = computePSNR(stego, cover)
            y_psnr_c.append(y_psnr_temp_c)
            # psnr_c.append(psnr_temp_c)

            # ssim
            ssim_temp_c=calculate_ssim_skimage(stego, cover)
            ssim_temp_s=calculate_ssim_skimage(Rsecret, secret)
            ssim_c.append(ssim_temp_c)
            ssim_s.append(ssim_temp_s)

            # mae
            mae_temp_c=calculate_mae(stego, cover)
            mae_temp_s=calculate_mae(Rsecret, secret)
            mae_s.append(mae_temp_s)
            mae_c.append(mae_temp_c)

            #RMSE
            rmse_temp_c=calculate_rmse(stego, cover)
            rmse_temp_s=calculate_rmse(Rsecret, secret)
            rmse_c.append(rmse_temp_c)
            rmse_s.append(rmse_temp_s)

        print(f'psnr_c:{np.mean(y_psnr_c)} psnr_s:{np.mean(y_psnr_s)}')
        print(f'ssim_c:{np.mean(ssim_c)} ssim_s:{np.mean(ssim_s)}')
        print(f'MAE_c: {np.mean(mae_c)} MAE_s:{np.mean(mae_s)}')
        print(f'RMSE_c:{np.mean(rmse_c)} RMSE_s:{np.mean(rmse_s)}')
        print(f'r:{r}')

if __name__ == '__main__':
    main()
