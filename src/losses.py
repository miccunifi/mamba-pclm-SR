"""
Loss functions for MambaIR (Guo et al., ECCV 2024).

The paper (Sec. 3.6) uses:
  - L1 loss for super-resolution (eq. 8):
        L = || I_HQ - I_GT ||_1
  - Charbonnier loss for denoising (eq. 9), with eps = 1e-3:
        L = sqrt( || I_HQ - I_GT ||^2 + eps^2 )

The SR training .yml files use:
    pixel_opt:
      type: L1Loss
      loss_weight: 1.0
      reduction: mean
so for SR you need `L1Loss`. Use the `build_loss(...)` factory to build the
loss directly from the `pixel_opt` block of the .yml.
"""
import torch
import torch.nn as nn


class L1Loss(nn.Module):
    """L1 loss (eq. 8 of the paper), with weight and reduction as in the .yml files.

    Args:
        loss_weight: multiplicative factor on the loss.
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        assert reduction in ('mean', 'sum', 'none'), f"reduction non valida: {reduction}"
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        return self.loss_weight * nn.functional.l1_loss(pred, target, reduction=self.reduction)


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (eq. 9 of the paper), a "smoothed" L1. Used for denoising.

    Args:
        loss_weight: multiplicative factor on the loss.
        reduction: 'mean' | 'sum' | 'none'.
        eps: smoothing constant (paper: 1e-3).
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super().__init__()
        assert reduction in ('mean', 'sum', 'none'), f"reduction non valida: {reduction}"
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps ** 2)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return self.loss_weight * loss


class FFTLoss(nn.Module):
    """L1 in the Fourier domain ("frequency loss", SAFMN style):
        L = w * || FFT2(pred) - FFT2(target) ||_1
    computed on the real and imaginary parts of torch.fft.fft2.

    Normally use it AS AN EXTRA TERM on top of the pixel loss:
        L_tot = L1(pred, gt) + w * L_FFT(pred, gt)      # SAFMN uses w = 0.05
    (in train.py: --fft_weight 0.05). It is used for the "orthogonality probe" of the
    rebuttal: showing that the training-recipe gains of the baselines also transfer to
    PCLM-SR. Check the exact weight in the repo of the reference paper.

    Args:
        loss_weight: multiplicative factor on the loss (the "w" above).
        reduction: 'mean' | 'sum'.
    """

    def __init__(self, loss_weight=0.05, reduction='mean'):
        super().__init__()
        assert reduction in ('mean', 'sum'), f"reduction non valida: {reduction}"
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        pf = torch.fft.fft2(pred, dim=(-2, -1))
        tf = torch.fft.fft2(target, dim=(-2, -1))
        # Stack real and imaginary parts so the L1 covers both components of the spectrum.
        diff = torch.stack([pf.real - tf.real, pf.imag - tf.imag], dim=-1).abs()
        loss = diff.mean() if self.reduction == 'mean' else diff.sum()
        return self.loss_weight * loss


_LOSSES = {
    'L1Loss': L1Loss,
    'CharbonnierLoss': CharbonnierLoss,
    'FFTLoss': FFTLoss,
}


def build_loss(opt):
    """Build the loss from the `pixel_opt` dictionary of the .yml.

    Example:
        loss = build_loss({'type': 'L1Loss', 'loss_weight': 1.0, 'reduction': 'mean'})
    """
    # Copy so that pop() does not mutate the caller's config dict.
    opt = dict(opt)
    loss_type = opt.pop('type')
    if loss_type not in _LOSSES:
        raise ValueError(f"loss sconosciuta: {loss_type!r} (usa una di {list(_LOSSES)})")
    return _LOSSES[loss_type](**opt)
