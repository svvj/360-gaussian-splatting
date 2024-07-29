#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from math import exp
import numpy as np
import time

def save_tensor_as_image(tensor, path):
    # テンソルを[0, 255]の範囲にスケーリング
    tensor = tensor.squeeze().cpu()  # チャンネル次元を削除してCPUに移動
    tensor = tensor - tensor.min()  # 最小値を0に
    tensor = tensor / tensor.max()  # 最大値を1に
    tensor = (tensor * 255).byte()  # [0, 255]にスケーリングしてバイト型に変換

    # テンソルをPIL画像に変換
    transform_to_pil = transforms.ToPILImage()
    pil_image = transform_to_pil(tensor)

    # 画像を保存
    pil_image.save(path)
    print(f"Image saved to {path}")
    
def latitude_weight(height):
    y = torch.arange(height).float()
    latitude = (y / height - 0.5) * np.pi
    weight = torch.cos(latitude)
    return weight.unsqueeze(-1).unsqueeze(0).expand(3, -1, -1)

def l1_loss(network_output, gt, weights=None, save_path_prefix=None):
    if weights is None:
        weights = torch.ones_like(gt)

    if save_path_prefix:
        save_tensor_as_image(network_output, f"{save_path_prefix}_network_output.png")
        save_tensor_as_image(gt, f"{save_path_prefix}_gt.png")

        # 5秒スリープ
        time.sleep(5)

    return torch.abs((network_output - gt) * weights).mean()

def total_variation_loss(image):
    """
    Calculate the total variation loss for an image.
    
    Args:
        image (torch.Tensor): Input image tensor with shape (batch_size, channels, height, width) or (channels, height, width).
    
    Returns:
        torch.Tensor: Total variation loss.
    """
    if image.dim() == 4:
        # Calculate the difference between adjacent pixel values in the horizontal direction
        loss_h = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
        
        # Calculate the difference between adjacent pixel values in the vertical direction
        loss_v = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]))
    elif image.dim() == 3:
        # Calculate the difference between adjacent pixel values in the horizontal direction
        loss_h = torch.mean(torch.abs(image[:, :-1, :] - image[:, 1:, :]))
        
        # Calculate the difference between adjacent pixel values in the vertical direction
        loss_v = torch.mean(torch.abs(image[:, :, :-1] - image[:, :, 1:]))
    else:
        raise ValueError("Unsupported tensor dimension: {}".format(image.dim()))
    
    # Sum the horizontal and vertical losses
    loss = loss_h + loss_v
    
    return loss

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True, weights=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    if weights is None:
        weights = torch.ones_like(img1)
    
    img1 = img1 * weights
    img2 = img2 * weights

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

