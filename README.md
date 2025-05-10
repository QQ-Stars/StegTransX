# StegTransX: A Lightweight Deep Steganography Method for High-Capacity Hiding and JPEG Compression Resistance

**Xintao Duan , Zhao Wang , Sen Li and Chuan Qin**

The official pytorch implementation of the paper [StegTransX: A lightweight deep steganography method for high-capacity hiding and JPEG compression resistance](https://www.sciencedirect.com/science/article/abs/pii/S0020025525003962)).

## Abstract

Image steganography refers to the concealment of multiple secret images within a cover image of the same resolution. This technique must ensure not only sufficient security but also high capacity and robustness. In this paper, we introduce StegTransX, which is a lightweight deep steganography framework that combines local and global modeling to improve data embedding performance. StegTransX can hide one or multiple secret images within a cover image of the same size while maintaining the visual quality of the stego image and the reconstruction quality of the secret images. Furthermore, considering that images typically undergo compression during channel transmission, we introduce a JPEG compression attack module to increase the robustness of secret information recovery under realistic compression scenarios in actual information transmission. Additionally, we propose an effective multiscale loss and constraint loss to preserve the quality of the stego image and improve the reconstruction quality of the secret images. The experimental results demonstrate that StegTransX outperforms existing state-of-the-art (SOTA) steganography models. In the case of single-image hiding, the PSNRs of both the cover/stego images and the secret/recovered images improved by more than 4 dB. Moreover, StegTransX also outperforms the state-of-the-art (SOTA) steganography models in multi-image hiding and JPEG compression resistance.

## Citation

If you find this work helps you, please cite:

```bibtex
@article{DUAN2025122264,
  title = {StegTransX: A lightweight deep steganography method for high-capacity hiding and JPEG compression resistance},
  journal = {Information Sciences},
  volume = {716},
  pages = {122264},
  year = {2025},
  issn = {0020-0255},
  doi = {https://doi.org/10.1016/j.ins.2025.122264},
  author = {Xintao Duan and Zhao Wang and Sen Li and Chuan Qin}
}
```

