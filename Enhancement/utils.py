# Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement
# Yuanhao Cai, Hao Bian, Jing Lin, Haoqian Wang, Radu Timofte, Yulun Zhang
# International Conference on Computer Vision (ICCV), 2023
# https://arxiv.org/abs/2303.06705
# https://github.com/caiyuanhao1998/Retinexformer

import numpy as np
import os
import cv2
import math
import torch
from pdb import set_trace as stx


def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def PSNR(img1, img2):
    mse_ = np.mean((img1 - img2) ** 2)
    if mse_ == 0:
        return 100
    return 10 * math.log10(1 / mse_)


# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_gray_img(filepath):
    return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)


def save_gray_img(filepath, img):
    cv2.imwrite(filepath, img)


def visualization(feature, save_path, type='max', colormap=cv2.COLORMAP_JET):
    '''
    :param feature: [C,H,W]
    :param save_path: saving path
    :param type: 'mean' or 'max'
    :param colormap: the type of the pseudocolor map
    '''
    feature = feature.cpu().numpy()
    if type == 'mean':
        feature = np.mean(feature, axis=0)
    else:
        feature = np.max(feature, axis=0)
    normed_feat = (feature - feature.min()) / (feature.max() - feature.min())
    normed_feat = (normed_feat * 255).astype('uint8')
    color_feat = cv2.applyColorMap(normed_feat, colormap)
    # stx()
    cv2.imwrite(save_path, color_feat)

def model_complexity(test_model, H=256, W=256, C=3, N=1):
    """Return Params(M) and THOP operations in G for one model forward pass.

    Args:
        test_model (torch.nn.Module): Model to analyse. DataParallel is
            unwrapped before profiling.
        H, W, C, N (int): Dummy input height, width, channels and batch size.

    Returns:
        dict: Parameters, THOP operations and their reporting metadata.

    Raises:
        ImportError: If the optional ``thop`` package is not installed.
    """
    try:
        from thop import profile
    except ImportError as exc:
        raise ImportError(
            'Model complexity requires thop. Install it with "pip install thop".'
        ) from exc

    model = test_model.module if hasattr(test_model, 'module') else test_model
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    inputs = torch.randn((N, C, H, W), device=device)
    with torch.no_grad():
        operations, _ = profile(model, inputs=(inputs,), verbose=False)
    if was_training:
        model.train()

    params = sum(parameter.numel() for parameter in model.parameters())
    return {
        'params_m': params / 1e6,
        # THOP's operation convention is recorded verbatim; no 2x MAC-to-FLOP
        # conversion is silently applied.
        'tflops_g': operations / 1e9,
        'input_size': f'{N}x{C}x{H}x{W}',
        'complexity_tool': 'thop.profile',
        'complexity_note': 'THOP returned operations / 1e9; no 2x MAC conversion.'
    }


def my_summary(test_model, H=256, W=256, C=3, N=1):
    """Print and return model complexity using the project-wide reporting rule."""
    summary = model_complexity(test_model, H=H, W=W, C=C, N=N)
    print(f"Params(M): {summary['params_m']:.4f}")
    print(f"TFLOPs(G): {summary['tflops_g']:.4f}")
    print(f"Input size: {summary['input_size']}")
    print(f"Tool: {summary['complexity_tool']}")
    print(summary['complexity_note'])
    return summary
