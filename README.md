# StegTransX: A Lightweight Deep Steganography Method for High-Capacity Hiding and JPEG Compression Resistance

StegTransX是一个基于深度学习的图像隐写算法实现, "StegTransX: A Lightweight Deep Steganography Method for High-Capacity Hiding and JPEG Compression Resistance"。

## 环境要求

- Python 3.9+
- CUDA 12.0
- PyTorch 2.1.0
- NVIDIA GPU (已在RTX 4090上测试)

## 主要依赖库

```bash
torch==2.1.0
torchvision==0.16.0
numpy>=1.21.0
pillow>=9.0.0
tqdm>=4.65.0
matplotlib>=3.5.0
```

## 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/QQ-Stars/StegTransX.git
cd StegTransX
```

2. 创建并激活虚拟环境（推荐）：
```bash
conda create -n stegtransx python=3.9
conda activate stegtransx
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

训练模型：
```bash
python train.py
```

## 项目结构

```
StegTransX/
├── train.py          # 训练脚本
├── test.py           # 测试脚本
├── critic.py         # 评价指标
├── mutil_train_x.py           # 多图隐藏训练脚本
├── compress.py           # 压缩模块函数
├── dataset.py             # 数据集脚本
└── config.py          # 配置文件
```

## 数据集准备

数据集应按照以下格式组织：

```
dataset/
└── imagenet/
    ├── train/
    │   ├── images_1.jpeg
    │   ├── images_2.jpeg
    │   └── ...
    └── test/
        ├── images_1.jpeg
        ├── images_2.jpeg
        └── ...
```

### 数据集要求
- 支持的图像格式：JPEG、PNG等
- 建议使用DIV2K数据集进行训练
- 训练图像建议大小：256x256

### 数据预处理
训练时会自动对图像进行以下处理：
1. 将图像调整为统一大小
2. 标准化处理

## 注意事项

- 确保使用兼容的CUDA版本（CUDA 12.0）
- 建议使用高性能GPU进行训练

## 参考项目

本项目在实现过程中参考了以下优秀的开源项目：

1. **StegFormer**: [aoli-gei/StegFormer](https://github.com/aoli-gei/StegFormer)

2. **HiNet**: [TomTomTommi/HiNet](https://github.com/TomTomTommi/HiNet)

3. **DeepMIH**: [TomTomTommi/DeepMIH](https://github.com/TomTomTommi/DeepMIH)

4. **DAH-Net**: [zhangle408/Deep-adaptive-hiding-network](https://github.com/zhangle408/Deep-adaptive-hiding-network)

## 许可证

本项目采用 Apache License 2.0 许可证。详情请见 [LICENSE](LICENSE) 文件。 
