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
from stegTransx import stegTransX
from compress import jpeg_compress_batch
from critic import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(2024)  # 将42替换为您喜欢的任何种子值

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
    Hnet = stegTransX(in_channels=6, out_channels=3)
    Rnet = stegTransX(in_channels=3, out_channels=3)

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
        loop = tqdm((zip(dataset.trainloader, dataset.trainloader1)),total = len(dataset.trainloader),leave=True)
        Hnet.train()
        Rnet.train()
        total_loss = 0.0

        for data1, data2 in loop:
            data1 = data1.to(device)
            data2 = data2.to(device)
            cover = data1
            secret = data2
            input_img=torch.cat((cover, secret), 1)
            Hoptim.zero_grad()
            Roptim.zero_grad()

            # 使用autocast进行混合精度训练
        
            H_output = Hnet(input_img)
            if c.compress:
                R_input = jpeg_compress_batch(H_output,quality=c.quality)
            else:
                R_input=H_output
            R_input = R_input.clamp(0,1)
            R_output = Rnet(R_input)

            Hloss = conceal_loss_function(H_output, cover)+ restrict_loss_funtion(H_output) + lp_loss(H_output, cover)
            Rloss = reveal_loss_function(R_output, secret) + restrict_loss_funtion(R_output) + lp_loss(R_output, secret)
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
        loop = tqdm((dataset.testloader), total=len(dataset.testloader), leave=True)
        Hnet.eval()
        Rnet.eval()

        with torch.no_grad():
            psnr_c = []
            psnr_s = []
            y_psnr_c = []
            y_psnr_s = []            
            total_loss = 0.0
            for data in loop:
                data = data.to(device)
                cover = data[data.shape[0] // 2:]
                secret = data[:data.shape[0] // 2]

                input_img = torch.cat([cover,secret], dim=1)  # [batch_size, 6, H, W]
                
                
                H_output = Hnet(input_img)
                if c.compress:
                    R_input = jpeg_compress_batch(H_output,quality=c.quality)
                else:
                    R_input=H_output
                R_input = R_input.clamp(0, 1)
                R_output = Rnet(R_input)


                Hloss = conceal_loss_function(H_output, cover) + restrict_loss_funtion(H_output) + lp_loss(H_output, cover)
                Rloss = reveal_loss_function(R_output, secret)  + restrict_loss_funtion(R_output) + lp_loss(R_output, secret)
                total_loss += c.lamda1 * Hloss + c.lamda2 * Rloss

                cover = cover.cpu()
                secret = secret.cpu()
                Rsecret=R_output.clamp(0, 1).cpu()
                stego = H_output.clamp(0, 1).cpu()

                psnr_temp_S = computePSNR(Rsecret, secret)
                y_psnr_temp_S = calculate_psnr_skimage(Rsecret, secret)
                y_psnr_s.append(y_psnr_temp_S)
                psnr_s.append(psnr_temp_S)

                y_psnr_temp_c = calculate_psnr_skimage(stego, cover)
                psnr_temp_c = computePSNR(stego, cover)
                y_psnr_c.append(y_psnr_temp_c)
                psnr_c.append(psnr_temp_c)

                loop.set_description(f'Val')
                loop.set_postfix({'Hloss': Hloss.item(), 'Rloss': Rloss.item()})

            logging.info(
                f'Val Epoch [{epoch}/{c.epochs}] val_loss: {total_loss / len(dataset.testloader)} psnr_c: {np.mean(psnr_c)} psnr_s: {np.mean(psnr_s)} y_psnr_c:{np.mean(y_psnr_c)} y_psnr_s:{np.mean(y_psnr_s)}')
            writer.add_scalars("PSNR_C", {"average": np.mean(psnr_c)}, epoch + 1)
            writer.add_scalars("PSNR_S", {"average": np.mean(psnr_s)}, epoch + 1)
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
