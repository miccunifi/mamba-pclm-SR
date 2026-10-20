"""
visualize.py — Qualitative visualizations during MambaIR training.

Two labelled grids, one row per sample, four columns:

    [  LR (input)  |  SR (output)  |  GT (target)  |  Residual  ]

Each column has a text header and each row is numbered (#0, #1, ...),
so it is clear at a glance which panel is which.

1) save_image_grid : the images. The LR is brought to the GT/SR resolution
   with bicubic interpolation (also as a baseline). The "residual" is the
   reconstruction error |SR - GT|, amplified by `residual_gain` to make it
   visible.
2) save_fourier_grid : for each panel, the magnitude of the Fourier spectrum
   (log, fftshift, low frequencies at the center), normalized per image and
   colored with the MAGMA colormap.

The mosaic is built with torchvision.utils.make_grid; the labels and (for the
Fourier grid) the colormap are added with matplotlib.
"""
import os

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


COL_LABELS = ['LR', 'SR', 'GT', 'Residuo']
_PAD = 4
_MIN_PANEL = 120


def _upscale_lr(lr, size):
    """Bring the LR to the HR/SR resolution (bicubic) to place it next to them in the grid."""
    return F.interpolate(lr, size=size, mode='bicubic', align_corners=False).clamp(0, 1)


def _residual(sr, hr, gain):
    """Error map |SR - GT|, amplified and clamped to [0,1]."""
    return (sr - hr).abs().mul(gain).clamp(0, 1)


def _to_gray(img):
    """BT.601 luminance: (B,C,H,W) in [0,1] -> (B,1,H,W)."""
    if img.shape[1] == 3:
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b
    return img


def _fourier_logmag(img):
    """Spectrum magnitude (log, fftshift) of the luminance, normalized
    per image to [0,1]. img: (B,C,H,W) -> (B,1,H,W)."""
    g = _to_gray(img)
    f = torch.fft.fft2(g)
    f = torch.fft.fftshift(f, dim=(-2, -1))
    mag = torch.log1p(f.abs())
    b = mag.shape[0]
    flat = mag.reshape(b, -1)
    mn = flat.min(dim=1).values.view(b, 1, 1, 1)
    mx = flat.max(dim=1).values.view(b, 1, 1, 1)
    return ((mag - mn) / (mx - mn).clamp(min=1e-8))


# ----------------------------------------------------------------------------------
# colormap
# ----------------------------------------------------------------------------------
def _get_cmap(name):
    try:
        return matplotlib.colormaps[name]            # matplotlib >= 3.6
    except Exception:
        import matplotlib.cm as cm
        return cm.get_cmap(name)


def _apply_cmap(x, name='magma'):
    """Apply a colormap to a 1-channel map: (B,1,H,W) in [0,1] -> (B,3,H,W)."""
    cmap = _get_cmap(name)
    arr = x.squeeze(1).clamp(0, 1).cpu().numpy()      # (B,H,W)
    rgb = cmap(arr)[..., :3]                           # (B,H,W,3)
    return torch.from_numpy(rgb).permute(0, 3, 1, 2).float()


def _interleave(panels):
    """From K tensors (B,C,H,W) to (K*B,C,H,W) ordered by sample, so that
    make_grid(nrow=K) puts each sample on one row."""
    b = panels[0].shape[0]
    rows = []
    for i in range(b):
        for p in panels:
            rows.append(p[i:i + 1])
    return torch.cat(rows, dim=0)


def _save_annotated(panels, path, suptitle):
    """panels: list of 4 tensors (n,3,H,W) in [0,1]. Build the mosaic with
    make_grid and overlay column headers + row numbers."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    n = panels[0].shape[0]
    H, W = panels[0].shape[-2], panels[0].shape[-1]

    # Enlarge small panels by integer pixel replication so the labels stay readable.
    if min(H, W) < _MIN_PANEL:
        f = int(round(_MIN_PANEL / min(H, W)))
        if f > 1:
            panels = [p.repeat_interleave(f, -1).repeat_interleave(f, -2) for p in panels]
            H, W = H * f, W * f

    mosaic = make_grid(_interleave(panels), nrow=4, padding=_PAD, pad_value=1.0)
    img = mosaic.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    Hm, Wm = img.shape[0], img.shape[1]

    fig, ax = plt.subplots(figsize=(max(Wm / 80, 6), max(Hm / 80, 2.5)))
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # Extend the axes to the left and top to leave room for row numbers and headers.
    ax.set_xlim(-0.16 * W, Wm - 0.5)
    ax.set_ylim(Hm - 0.5, -0.22 * H)

    cell_w, cell_h = W + _PAD, H + _PAD
    for j, lab in enumerate(COL_LABELS):
        cx = j * cell_w + _PAD + W / 2
        ax.text(cx, -0.04 * H, lab, ha='center', va='bottom',
                fontsize=11, fontweight='bold', clip_on=False)
    for i in range(n):
        cy = i * cell_h + _PAD + H / 2
        ax.text(-0.02 * W, cy, f'#{i}', ha='right', va='center',
                fontsize=10, color='0.3', clip_on=False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------------
@torch.no_grad()
def save_image_grid(lr, hr, sr, path, n=4, residual_gain=5.0):
    """Labelled image grid: columns [LR(up) | SR | GT | residual]."""
    n = min(n, lr.shape[0])
    lr, hr, sr = lr[:n].float().cpu(), hr[:n].float().cpu(), sr[:n].float().clamp(0, 1).cpu()
    lr_up = _upscale_lr(lr, hr.shape[-2:])
    res = _residual(sr, hr, residual_gain)
    return _save_annotated([lr_up, sr, hr, res], path,
                           suptitle=f"LR=input  SR=output  GT=target  Residuo=|SR-GT| x{residual_gain:g}")


@torch.no_grad()
def save_fourier_grid(lr, hr, sr, path, n=4, residual_gain=5.0, cmap='magma'):
    """Fourier spectrum grid (log-magnitude, magma colormap), same layout."""
    n = min(n, lr.shape[0])
    lr, hr, sr = lr[:n].float().cpu(), hr[:n].float().cpu(), sr[:n].float().clamp(0, 1).cpu()
    lr_up = _upscale_lr(lr, hr.shape[-2:])
    res = _residual(sr, hr, residual_gain)
    panels = [_apply_cmap(_fourier_logmag(x), cmap) for x in (lr_up, sr, hr, res)]
    return _save_annotated(panels, path,
                           suptitle=f"Spettro di Fourier (log-magnitudo, '{cmap}')  -  LR/SR/GT/Residuo")


@torch.no_grad()
def save_visuals(lr, hr, sr, out_dir, tag, n=4, residual_gain=5.0):
    """Save both grids (images + Fourier) with prefix `tag`.
    Return the two paths."""
    p_img = save_image_grid(lr, hr, sr, os.path.join(out_dir, f"{tag}_img.png"),
                            n=n, residual_gain=residual_gain)
    p_fft = save_fourier_grid(lr, hr, sr, os.path.join(out_dir, f"{tag}_fft.png"),
                              n=n, residual_gain=residual_gain)
    return p_img, p_fft