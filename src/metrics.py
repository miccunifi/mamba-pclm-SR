"""
Metrics for MambaIR: PSNR and SSIM with the standard convention of the SR literature
(paper Sec. 4.1) -> computed on the Y channel (luminance, YCbCr) with `crop_border`
pixels discarded per side before the computation (usually crop_border = scale).

Backend
-------
If BasicSR is installed (pip install basicsr) the metrics directly use the functions
from the paper:
    basicsr.metrics.psnr_ssim.calculate_psnr_pt / calculate_ssim_pt
(tensor versions, batch (B,C,H,W) in range [0,1]). For recent torchvision versions,
where the `functional_tensor` module was removed, a shim is applied at import time
(BasicSR needs it).

If BasicSR is not available, an equivalent internal implementation is used
(same convention: Y channel, 11x11 Gaussian window for SSIM).

API
---
    calc_psnr(sr, hr, crop_border, test_y_channel=True) -> float
    calc_ssim(sr, hr, crop_border, test_y_channel=True) -> float
Both accept tensors (B,C,H,W) or (C,H,W) in [0,1] and return the mean.
"""
import math

import torch
import torch.nn.functional as F


def _load_basicsr_metrics():
    """Load BasicSR's calculate_psnr_pt / calculate_ssim_pt.
    Applies a shim for recent torchvision (functional_tensor module removed).
    Returns (None, None) if the import fails."""
    import sys
    try:
        import torchvision.transforms.functional as _vF
        sys.modules.setdefault('torchvision.transforms.functional_tensor', _vF)
        from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt
        return calculate_psnr_pt, calculate_ssim_pt
    except Exception:
        return None, None


_BSR_PSNR_PT, _BSR_SSIM_PT = _load_basicsr_metrics()
USE_BASICSR = _BSR_PSNR_PT is not None


def _to_4d(t):
    """(C,H,W) -> (1,C,H,W); leaves (B,C,H,W) unchanged."""
    return t.unsqueeze(0) if t.dim() == 3 else t


def rgb_to_y(img):
    """RGB in [0,1] -> Y luminance (BT.601) on the 0-255 scale (range ~[16,235]).
    img: (B,3,H,W) -> (B,1,H,W)."""
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return 65.481 * r + 128.553 * g + 24.966 * b + 16.0


def _shave(t, border):
    if border and border > 0:
        return t[..., border:-border, border:-border]
    return t


def _psnr_builtin(sr, hr, crop_border, test_y_channel=True):
    sr, hr = sr.clamp(0, 1), hr.clamp(0, 1)
    if test_y_channel and sr.shape[1] == 3:
        sr, hr = rgb_to_y(sr), rgb_to_y(hr)
    else:
        sr, hr = sr * 255.0, hr * 255.0
    sr, hr = _shave(sr, crop_border), _shave(hr, crop_border)
    mse = ((sr - hr) ** 2).mean().item()
    if mse == 0:
        return float('inf')
    return 10.0 * math.log10(255.0 ** 2 / mse)


def _gaussian_window(window_size=11, sigma=1.5):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])                # (11,11)


def _ssim_builtin(sr, hr, crop_border, test_y_channel=True):
    sr, hr = sr.clamp(0, 1), hr.clamp(0, 1)
    if test_y_channel and sr.shape[1] == 3:
        sr, hr = rgb_to_y(sr), rgb_to_y(hr)
    else:
        sr, hr = sr * 255.0, hr * 255.0
    sr, hr = _shave(sr, crop_border), _shave(hr, crop_border)
    # Stability constants (K1=0.01, K2=0.03) on the 0-255 scale; 'valid' conv, no padding.
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    win = _gaussian_window().to(sr).view(1, 1, 11, 11)
    mu1, mu2 = F.conv2d(sr, win), F.conv2d(hr, win)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(sr * sr, win) - mu1_sq
    sigma2_sq = F.conv2d(hr * hr, win) - mu2_sq
    sigma12 = F.conv2d(sr * hr, win) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def _quantize_uint8(t):
    """Snap the [0,1] tensor to the uint8 grid: round(t*255)/255.
    This is the protocol of the official tests (SwinIR/BasicSR save the SR as uint8 and
    compute the metrics on it); without this step the PSNR differs by ~0.01 dB."""
    return t.mul(255.0).round().div(255.0)


@torch.no_grad()
def calc_psnr(sr, hr, crop_border, test_y_channel=True, quantize=True):
    """PSNR (Y channel, `crop_border` pixels discarded per side). Uses BasicSR if
    installed, otherwise the internal (equivalent) implementation.
    With quantize=True (default) SR and HR are snapped to the uint8 grid before the
    computation, as in the official test protocol (main_test_swinir.py)."""
    sr, hr = _to_4d(sr).clamp(0, 1), _to_4d(hr).clamp(0, 1)
    if quantize:
        sr, hr = _quantize_uint8(sr), _quantize_uint8(hr)
    if USE_BASICSR:
        return _BSR_PSNR_PT(sr, hr, crop_border=crop_border,
                            test_y_channel=test_y_channel).mean().item()
    return _psnr_builtin(sr, hr, crop_border, test_y_channel)


@torch.no_grad()
def calc_ssim(sr, hr, crop_border, test_y_channel=True, quantize=True):
    """SSIM (Y channel, `crop_border` pixels discarded per side). Uses BasicSR if
    installed, otherwise the internal (equivalent) implementation.
    With quantize=True (default) SR and HR are snapped to the uint8 grid before the
    computation, as in the official test protocol (main_test_swinir.py)."""
    sr, hr = _to_4d(sr).clamp(0, 1), _to_4d(hr).clamp(0, 1)
    if quantize:
        sr, hr = _quantize_uint8(sr), _quantize_uint8(hr)
    if USE_BASICSR:
        return _BSR_SSIM_PT(sr, hr, crop_border=crop_border,
                            test_y_channel=test_y_channel).mean().item()
    return _ssim_builtin(sr, hr, crop_border, test_y_channel)


if __name__ == '__main__':
    # Smoke test: identical images give the maximum PSNR (inf, or 80 dB with BasicSR) and
    # SSIM 1; noisy copies give lower, finite values.
    print(f"backend metriche: {'BasicSR' if USE_BASICSR else 'interno (equivalente a BasicSR)'}")
    sr = torch.rand(2, 3, 64, 64)
    hr = sr.clone()
    print(f"identiche : PSNR {calc_psnr(sr, hr, 2):.2f}  SSIM {calc_ssim(sr, hr, 2):.4f}")
    hr2 = (sr + 0.05 * torch.randn_like(sr)).clamp(0, 1)
    print(f"rumorose  : PSNR {calc_psnr(sr, hr2, 2):.2f}  SSIM {calc_ssim(sr, hr2, 2):.4f}")