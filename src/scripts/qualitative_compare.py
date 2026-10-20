"""
qualitative_compare.py — Qualitative comparison: GT with the patch marked in red + a zoom of the
same patch for every model (official MambaIR-light and MambaIRv2-light with official weights,
our checkpoints, bicubic), on one image of a benchmark (default Urban100, x4).

Model specs (--models, one or more):
  mambair_light:<weights.pth>    official MambaIR-light (repos/MambaIR, mambair_arch.py)
  mambairv2_light:<weights.pth>  official MambaIRv2-light (repos/MambaIR, mambairv2light_arch.py)
  <baseline>:<weights.pth>       any name from baselines.BASELINES (SwinIR-light, SAFMN, ...)
  ours:<checkpoint.pth>          our PCLM-SR (arch rebuilt from the checkpoint's arch_cfg)
  bicubic                        bicubic upsampling (reference)
Optional label: "Label=spec" (e.g. "rho=0.5=ours:experiments/rho_050_x4/seed0/iter_500000.pth").
Both BasicSR weights ('params'/'params_ema') and train.py weights ('model') are recognised.

For every model: inference on the whole LR image, PSNR/SSIM (Y channel, crop=scale) on the full
image and PSNR on the patch (printed and reported under each zoom). Patch: --patch_size px (HR),
random position (--seed) or explicit (--patch_xy X Y, HR top-left coordinates); random image
(--seed) or --image <stem> (e.g. img062).

Usage
-----
  python scripts/qualitative_compare.py --scale 4 --image img062 \
      --models "MambaIR-light=mambair_light:repos/MambaIR/weights/MambaIR_lightSR4.pth" \
               "MambaIRv2-light=mambairv2_light:repos/MambaIR/weights/mambairv2_lightSR_x4.pth" \
               "PCLM rho=0.5=ours:experiments/rho_050_x4/seed0/iter_500000.pth" bicubic \
      --out results/figures/qual_img062_x4.png
  python scripts/qualitative_compare.py --seed 3 --patch_size 96 --models ... --out results/figures/qual_x4.png
"""
import argparse
import os
import random
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ROOT, BENCHMARK_DIR, RESULTS_DIR, benchmark_pairs, load_ours_checkpoint,  # noqa: E402
                    sr_forward, env_info)
import baselines as B  # noqa: E402
from metrics import calc_psnr, calc_ssim  # noqa: E402

ALIASES = {'mambair_light': 'MambaIR-light', 'mambairv2_light': 'MambaIRv2-light'}


def _state_dict(path):
    # Unwrap BasicSR ('params'/'params_ema') and train.py ('model') checkpoint containers.
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ck, dict):
        for k in ('params_ema', 'params', 'model', 'state_dict'):
            if k in ck:
                return ck[k]
    return ck


def build_from_spec(spec, scale, device):
    """'[label=]kind:path' -> (label, forward_fn(lr)->sr)."""
    label = None
    if '=' in spec:
        label, spec = spec.rsplit('=', 1)
    if spec == 'bicubic':
        return (label or 'Bicubic'), (lambda lr: F.interpolate(lr, scale_factor=scale, mode='bicubic',
                                                                align_corners=False).clamp(0, 1))
    kind, _, path = spec.partition(':')
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise SystemExit(f"pesi non trovati: {path}")
    if kind == 'ours':
        model, ck = load_ours_checkpoint(path, device)
        rho = ck['arch_cfg']['mamba_ratio']
        label = label or f"PCLM-SR rho={rho:.2f} (it {ck.get('iter', '?')})"
    else:
        name = ALIASES.get(kind.lower()) or B.resolve_names([kind])[0]
        model, info = B.build_baseline(name, scale, device)
        sd = _state_dict(path)
        # Some baselines are wrapped in PadToMultiple: load the weights into the inner net.
        target = model.net if isinstance(model, B.PadToMultiple) else model
        missing, unexpected = target.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise SystemExit(f"{name}: pesi non compatibili (mancanti {len(missing)}, extra {len(unexpected)}): "
                             f"{(missing + unexpected)[:5]}")
        label = label or name
    model.eval()
    return label, (lambda lr: sr_forward(model, lr, scale, tile=0))


def _crop(t, x, y, s):
    return t[..., y:y + s, x:x + s]


def _to_pil(t):
    return TF.to_pil_image(t.squeeze(0).clamp(0, 1).cpu())


def main():
    p = argparse.ArgumentParser(description="Confronto qualitativo con zoom su una patch")
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--scale', type=int, default=4)
    p.add_argument('--dataset', default='Urban100')
    p.add_argument('--benchmark_dir', default=BENCHMARK_DIR)
    p.add_argument('--image', default=None, help="stem HR (es. img062); default: casuale (--seed)")
    p.add_argument('--patch_size', type=int, default=64, help="lato della patch in pixel HR")
    p.add_argument('--patch_xy', type=int, nargs=2, default=None, help="top-left (x y) in pixel HR")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--zoom_px', type=int, default=256, help="lato di visualizzazione della patch")
    p.add_argument('--rows', type=int, default=None, help="righe di zoom (default: 1 fino a 5 pannelli, poi 2)")
    p.add_argument('--font_size', type=float, default=15, help="dimensione del testo dei titoli (pt)")
    p.add_argument('--out', default=None, help="default results/figures/qual_<image>_x<s>.png (+ .pdf)")
    p.add_argument('--no_metrics', action='store_true')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = random.Random(args.seed)
    pairs = benchmark_pairs(args.benchmark_dir, args.dataset, args.scale)
    if args.image:
        pairs = [pr for pr in pairs if os.path.splitext(os.path.basename(pr[0]))[0] == args.image]
        if not pairs:
            raise SystemExit(f"immagine {args.image} non trovata in {args.dataset}")
        hr_path, lr_path = pairs[0]
    else:
        hr_path, lr_path = rng.choice(pairs)
    stem = os.path.splitext(os.path.basename(hr_path))[0]

    hr = TF.to_tensor(Image.open(hr_path).convert('RGB')).unsqueeze(0).to(device)
    lr = TF.to_tensor(Image.open(lr_path).convert('RGB')).unsqueeze(0).to(device)
    # Crop HR to exactly LR size x scale, so SR and HR have the same shape.
    H, W = lr.shape[-2] * args.scale, lr.shape[-1] * args.scale
    hr = hr[..., :H, :W]
    ps = min(args.patch_size, H, W)
    if args.patch_xy:
        px, py = args.patch_xy
        px, py = max(0, min(px, W - ps)), max(0, min(py, H - ps))
    else:
        px, py = rng.randint(0, W - ps), rng.randint(0, H - ps)
    print(f"{args.dataset}/{stem} x{args.scale} | LR {lr.shape[-2]}x{lr.shape[-1]} -> HR {H}x{W} | "
          f"patch {ps}px @ (x={px}, y={py}) | seed {args.seed} | device {device}")

    results = []      # (label, sr, psnr_full, ssim_full, psnr_patch)
    for spec in args.models:
        label, fwd = build_from_spec(spec, args.scale, device)
        with torch.no_grad():
            sr = fwd(lr)[..., :H, :W]
        # Quantise to 8-bit values, as if the SR image were saved as PNG.
        sr = sr.mul(255).round().div(255)
        if args.no_metrics:
            m = (float('nan'),) * 3
        else:
            m = (calc_psnr(sr, hr, crop_border=args.scale), calc_ssim(sr, hr, crop_border=args.scale),
                 calc_psnr(_crop(sr, px, py, ps), _crop(hr, px, py, ps), crop_border=0))
        results.append((label, sr, *m))
        print(f"  {label:<34} PSNR {m[0]:6.2f} dB | SSIM {m[1]:.4f} | patch PSNR {m[2]:6.2f} dB")
        torch.cuda.empty_cache()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # Layout: the full GT panel spans all rows on the left, zooms fill a rows x cols grid.
    n_zoom = 1 + len(results)
    rows = args.rows or (1 if n_zoom <= 5 else 2)
    cols = -(-n_zoom // rows)                                   # ceil
    fs = args.font_size
    fig = plt.figure(figsize=(3.0 * (cols + 1.4), (3.0 + 0.027 * fs * 3) * rows), constrained_layout=True)
    gs = fig.add_gridspec(rows, 1 + cols, width_ratios=[1.4] + [1] * cols)
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(_to_pil(hr))
    ax.add_patch(Rectangle((px, py), ps, ps, fill=False, edgecolor='red', linewidth=2))
    ax.set_title(f"GT: {args.dataset}/{stem} (×{args.scale})", fontsize=fs, fontweight='bold')
    ax.axis('off')
    zooms = [("GT (zoom)", hr, None)] + [(lab, sr, (pf, sf, pp)) for lab, sr, pf, sf, pp in results]
    for i, (lab, img, m) in enumerate(zooms):
        axz = fig.add_subplot(gs[i // cols, 1 + i % cols])
        patch = _to_pil(_crop(img, px, py, ps)).resize((args.zoom_px, args.zoom_px), Image.NEAREST)
        axz.imshow(patch)
        if m is None or args.no_metrics:
            # Empty lines keep the title height equal to the panels that show metrics.
            title = f"{lab}\n\n"
        else:
            title = f"{lab}\n{m[0]:.2f} dB / {m[1]:.4f}\npatch: {m[2]:.2f} dB"
        axz.set_title(title, fontsize=fs, linespacing=1.15)
        axz.axis('off')
        for s in axz.spines.values():
            s.set_visible(True)
    out = args.out or os.path.join(RESULTS_DIR, 'figures', f"qual_{stem}_x{args.scale}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(os.path.splitext(out)[0] + '.pdf')
    plt.close(fig)
    info = env_info()
    print(f"figura: {out} (+ .pdf) | {info['gpu']} | {info['date']}")


if __name__ == '__main__':
    main()
