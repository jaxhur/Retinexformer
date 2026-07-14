# Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement
# Yuanhao Cai, Hao Bian, Jing Lin, Haoqian Wang, Radu Timofte, Yulun Zhang
# International Conference on Computer Vision (ICCV), 2023
# https://arxiv.org/abs/2303.06705
# https://github.com/caiyuanhao1998/Retinexformer

"""Evaluate one LOL checkpoint and write its complete reproducibility outputs."""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import utils
from basicsr.models import create_model
from basicsr.utils.options import parse


IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff'}


def self_ensemble(x, model):
    """Average the eight flip/rotation predictions used by the original test."""
    def forward_transformed(input_, hflip, vflip, rotate):
        if hflip:
            input_ = torch.flip(input_, (-2, ))
        if vflip:
            input_ = torch.flip(input_, (-1, ))
        if rotate:
            input_ = torch.rot90(input_, dims=(-2, -1))
        output = model(input_)
        if isinstance(output, list):
            output = output[-1]
        if rotate:
            output = torch.rot90(output, dims=(-2, -1), k=3)
        if vflip:
            output = torch.flip(output, (-1, ))
        if hflip:
            output = torch.flip(output, (-2, ))
        return output

    predictions = []
    for hflip in (False, True):
        for vflip in (False, True):
            for rotate in (False, True):
                predictions.append(forward_transformed(x, hflip, vflip, rotate))
    return torch.stack(predictions).mean(dim=0)


def _relative_key(path, root):
    """Return a normalized, platform-independent key relative to ``root``."""
    relative_path = os.path.relpath(path, root)
    return os.path.normcase(os.path.normpath(relative_path)).replace('\\', '/')


def _display_path(path):
    """Format persisted paths with forward slashes on every platform."""
    return os.path.abspath(path).replace('\\', '/')


def _image_path_map(root):
    """Map every image under a root directory to its normalized relative path."""
    image_paths = {}
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            path = os.path.join(directory, filename)
            if os.path.splitext(filename)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            key = _relative_key(path, root)
            if key in image_paths:
                raise ValueError(f'Duplicate normalized image key in {root}: {key}')
            image_paths[key] = path
    if not image_paths:
        raise ValueError(f'No supported images found in {root}.')
    return image_paths


def paired_image_paths(lq_root, gt_root):
    """Pair LQ and GT images by normalized relative path and validate coverage."""
    lq_paths = _image_path_map(lq_root)
    gt_paths = _image_path_map(gt_root)
    missing_lq = sorted(set(gt_paths) - set(lq_paths))
    missing_gt = sorted(set(lq_paths) - set(gt_paths))
    if missing_lq or missing_gt:
        details = []
        if missing_lq:
            details.append(f'missing LQ ({len(missing_lq)}): {missing_lq[:3]}')
        if missing_gt:
            details.append(f'missing GT ({len(missing_gt)}): {missing_gt[:3]}')
        raise ValueError('LQ/GT pairing failed; ' + '; '.join(details))
    return [(key, lq_paths[key], gt_paths[key]) for key in sorted(lq_paths)]


def create_lpips_metric(device, network):
    """Create an LPIPS evaluator that accepts RGB tensors in the [-1, 1] range."""
    try:
        import lpips
    except ImportError as exc:
        raise ImportError(
            'LPIPS evaluation requires the lpips package. Install it with '
            '"pip install lpips".'
        ) from exc
    return lpips.LPIPS(net=network).to(device).eval()


def run_model(input_, model, use_self_ensemble):
    """Run one padded input through the model and unwrap multi-stage outputs."""
    if use_self_ensemble:
        return self_ensemble(input_, model)
    output = model(input_)
    return output[-1] if isinstance(output, list) else output


def write_test_log(log_path, args, lq_root, gt_root, enhanced_root, metrics):
    """Write the command context and final averages beside the metric CSV."""
    with open(log_path, 'w', encoding='utf-8') as file:
        file.write(f'Command: {" ".join(sys.argv)}\n')
        file.write(f'Checkpoint: {_display_path(args.weights)}\n')
        file.write(f'LQ root: {_display_path(lq_root)}\n')
        file.write(f'GT root: {_display_path(gt_root)}\n')
        file.write(f'Enhanced images: {_display_path(enhanced_root)}\n')
        for name, value in metrics.items():
            file.write(f'{name}: {value}\n')


def main():
    parser = argparse.ArgumentParser(
        description='Image Enhancement evaluation for a paired LOL dataset')
    parser.add_argument('--opt', required=True, type=str,
                        help='Path to the LOL option YAML file.')
    parser.add_argument('--weights', required=True, type=str,
                        help='Explicit *_G.pth generator checkpoint for testing.')
    parser.add_argument('--dataset', required=True, type=str,
                        help='Stable output directory name, e.g. LOL-v1.')
    parser.add_argument('--result_dir', default='./test_result', type=str,
                        help='Parent directory for test results.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Single GPU device identifier.')
    parser.add_argument('--GT_mean', action='store_true',
                        help='Use GT mean to rectify output (not recommended).')
    parser.add_argument('--self_ensemble', action='store_true',
                        help='Use eight-way self-ensemble.')
    parser.add_argument('--lpips_net', choices=('alex', 'vgg'), default='alex',
                        help='LPIPS backbone recorded in metric.csv.')
    parser.add_argument('--complexity_size', type=int, default=256,
                        help='Square input size used for Params/TFLOPs statistics.')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if not args.weights.endswith('_G.pth'):
        raise ValueError('--weights must point to a generator checkpoint named *_G.pth.')
    if not torch.cuda.is_available():
        raise RuntimeError('This evaluation script requires one CUDA GPU.')
    device = torch.device('cuda')
    print(f'export CUDA_VISIBLE_DEVICES={args.gpus}')

    opt = parse(args.opt, is_train=False)
    opt['dist'] = False
    model_restoration = create_model(opt).net_g.to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    if 'params' not in checkpoint:
        raise KeyError(f"Checkpoint {args.weights} does not contain the 'params' key.")
    model_restoration.load_state_dict(checkpoint['params'], strict=True)
    model_restoration = nn.DataParallel(model_restoration).eval()
    print(f'===> Testing using weights: {args.weights}')

    lq_root = opt['datasets']['val']['dataroot_lq']
    gt_root = opt['datasets']['val']['dataroot_gt']
    pairs = paired_image_paths(lq_root, gt_root)
    print(f'Test dataset length: {len(pairs)}')

    result_root = os.path.join(args.result_dir, args.dataset)
    enhanced_root = os.path.join(result_root, 'enhanced')
    os.makedirs(enhanced_root, exist_ok=True)
    lpips_metric = create_lpips_metric(device, args.lpips_net)
    complexity = utils.model_complexity(
        model_restoration,
        H=args.complexity_size,
        W=args.complexity_size,
        C=3,
        N=1)

    psnr_values, ssim_values, lpips_values = [], [], []
    factor = 4
    with torch.inference_mode():
        for relative_path, input_path, target_path in tqdm(pairs):
            image = np.float32(utils.load_img(input_path)) / 255.
            target = np.float32(utils.load_img(target_path)) / 255.
            input_ = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
            target_tensor = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).to(device)

            # Pad to the model window and crop back before metric calculation.
            _, _, height, width = input_.shape
            padded_height = (height + factor - 1) // factor * factor
            padded_width = (width + factor - 1) // factor * factor
            input_padded = F.pad(input_, (0, padded_width - width, 0,
                                          padded_height - height), 'reflect')
            restored = run_model(input_padded, model_restoration, args.self_ensemble)
            restored = torch.clamp(restored[:, :, :height, :width], 0, 1)

            restored_image = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if args.GT_mean:
                # This setting follows KinD/LLFlow style evaluation but leaks
                # ground-truth information and should not be used by default.
                restored_mean = cv2.cvtColor(restored_image, cv2.COLOR_RGB2GRAY).mean()
                target_mean = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY).mean()
                restored_image = np.clip(restored_image * target_mean /
                                         max(restored_mean, 1e-12), 0, 1)
                restored = torch.from_numpy(restored_image).permute(2, 0, 1).unsqueeze(0).to(device)

            psnr_values.append(utils.PSNR(target, restored_image))
            ssim_values.append(utils.calculate_ssim(
                (target * 255).round().astype(np.uint8),
                (restored_image * 255).round().astype(np.uint8)))
            lpips_values.append(float(lpips_metric(
                restored * 2 - 1, target_tensor * 2 - 1).mean().item()))

            save_path = os.path.join(enhanced_root, relative_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            utils.save_img(save_path, (restored_image * 255).round().astype(np.uint8))
            torch.cuda.empty_cache()

    averages = {
        'psnr': float(np.mean(psnr_values)),
        'ssim': float(np.mean(ssim_values)),
        'lpips': float(np.mean(lpips_values)),
    }
    metric_path = os.path.join(result_root, 'metric.csv')
    with open(metric_path, 'w', newline='', encoding='utf-8') as file:
        columns = [
            'dataset', 'psnr', 'ssim', 'lpips', 'lpips_backbone', 'params_m',
            'tflops_g', 'input_size', 'checkpoint', 'enhanced_images',
            'complexity_tool', 'complexity_note'
        ]
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerow({
            'dataset': args.dataset,
            **averages,
            'lpips_backbone': args.lpips_net,
            'params_m': f"{complexity['params_m']:.6f}",
            'tflops_g': f"{complexity['tflops_g']:.6f}",
            'input_size': complexity['input_size'],
            'checkpoint': _display_path(args.weights),
            'enhanced_images': _display_path(enhanced_root),
            'complexity_tool': complexity['complexity_tool'],
            'complexity_note': complexity['complexity_note'],
        })

    write_test_log(os.path.join(result_root, 'test.log'), args, lq_root,
                   gt_root, enhanced_root, {**averages, **complexity})
    print(f"PSNR: {averages['psnr']:.6f}")
    print(f"SSIM: {averages['ssim']:.6f}")
    print(f"LPIPS ({args.lpips_net}): {averages['lpips']:.6f}")
    print(f"Params(M): {complexity['params_m']:.6f}")
    print(f"TFLOPs(G): {complexity['tflops_g']:.6f}")
    print(f'Enhanced images: {enhanced_root}')
    print(f'Metrics: {metric_path}')


if __name__ == '__main__':
    main()
