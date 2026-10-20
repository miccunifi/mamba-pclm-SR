"""
plot_history.py — Plots the training curves from history.json (written by train.py).

Produces a PNG with three side-by-side panels:
  1) loss   : train_loss and val_loss vs iterations
  2) PSNR   : val_psnr vs iterations (the maximum is marked)
  3) SSIM   : val_ssim vs iterations (the maximum is marked)

history.json has this form:
  {"iter": [...], "train_loss": [...], "val_loss": [...],
   "val_psnr": [...], "val_ssim": [...]}

Usage
-----
  python plot_history.py --history ./checkpoints/light_sr_x2/history.json
  python plot_history.py --history .../history.json --output curve.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_history(history, out_path, title=None):
    """Draws loss/PSNR/SSIM from a history dict and saves the PNG."""
    it = history.get('iter', [])
    if not it:
        raise ValueError("history senza campo 'iter' o vuoto: niente da tracciare")

    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    val_psnr = history.get('val_psnr', [])
    val_ssim = history.get('val_ssim', [])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1) loss
    ax = axes[0]
    if train_loss:
        ax.plot(it, train_loss, label='train', color='tab:blue')
    # Skip a val curve when the list is empty or holds only None entries.
    if val_loss and any(v is not None for v in val_loss):
        ax.plot(it, val_loss, label='val', color='tab:orange')
    ax.set_title('Loss (L1)')
    ax.set_xlabel('iterazioni'); ax.set_ylabel('loss')
    ax.grid(True, alpha=0.3); ax.legend()

    # 2) PSNR
    ax = axes[1]
    if val_psnr and any(v is not None for v in val_psnr):
        ax.plot(it, val_psnr, color='tab:green', label='val PSNR')
        best_i = max(range(len(val_psnr)), key=lambda k: val_psnr[k])
        ax.scatter([it[best_i]], [val_psnr[best_i]], color='red', zorder=5,
                   label=f'best {val_psnr[best_i]:.2f} dB @ {it[best_i]}')
    ax.set_title('PSNR di validation')
    ax.set_xlabel('iterazioni'); ax.set_ylabel('PSNR (dB)')
    ax.grid(True, alpha=0.3); ax.legend()

    # 3) SSIM
    ax = axes[2]
    if val_ssim and any(v is not None for v in val_ssim):
        ax.plot(it, val_ssim, color='tab:purple', label='val SSIM')
        best_i = max(range(len(val_ssim)), key=lambda k: val_ssim[k])
        ax.scatter([it[best_i]], [val_ssim[best_i]], color='red', zorder=5,
                   label=f'best {val_ssim[best_i]:.4f} @ {it[best_i]}')
    ax.set_title('SSIM di validation')
    ax.set_xlabel('iterazioni'); ax.set_ylabel('SSIM')
    ax.grid(True, alpha=0.3); ax.legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Traccia le curve da history.json")
    p.add_argument('--history', required=True, help="path a history.json")
    p.add_argument('--output', default=None,
                   help="PNG di output (default: curves.png accanto a history.json)")
    args = p.parse_args()

    with open(args.history) as f:
        history = json.load(f)

    out = args.output or os.path.join(os.path.dirname(args.history) or '.', 'curves.png')
    # Use the experiment folder name (parent of history.json) as the figure title.
    title = os.path.basename(os.path.dirname(os.path.abspath(args.history)))
    plot_history(history, out, title=title)
    print(f"curve salvate in {out}")


if __name__ == '__main__':
    main()
