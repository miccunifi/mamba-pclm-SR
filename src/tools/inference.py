"""
inference.py — Super-resolution of images with a trained MambaIR.

Unlike test.py (which evaluates PSNR/SSIM against the HR images of a benchmark), here
we start from LR images without ground truth and save the super-resolved output.
It works on a single image or on a whole folder.

It loads a checkpoint from train.py ({'model': ...}), BasicSR weights
({'params'/'params_ema': ...}) or a 'bare' state_dict. For large images use
--tile to limit memory.

Usage
-----
  # single image
  python inference.py --variant classic_sr_x4 --ckpt best.pth --input foto.png --output sr.png
  # whole folder -> folder
  python inference.py --variant light_sr_x2 --ckpt best.pth --input ./lr_imgs --output ./sr_imgs
  # with tiling (large images)
  python inference.py --variant classic_sr_x4 --ckpt best.pth --input big.png --output sr.png --tile 256
  # with metrics: per-image PSNR/SSIM against the HR images + CSV (test.py protocol)
  python inference.py --ckpt best.pth --input ./Set5/LR_bicubic/X2 --output ./sr_set5 \
      --hr_dir ./Set5/HR --csv set5_x2.csv
"""
import argparse
import csv
import os
import statistics

import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from PIL import Image

from models import build_model, model_from_cfg, parse_overrides
from metrics import calc_psnr, calc_ssim


_IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


def _match_size(a, b):
    """Align a and b to the smallest common size with a top-left (0,0) crop: the LR of
    the benchmarks is generated from the HR mod-cropped at the top left, so the SR
    corresponds to that region (same protocol as test.py)."""
    H = min(a.shape[-2], b.shape[-2]); W = min(a.shape[-1], b.shape[-1])
    return a[..., :H, :W], b[..., :H, :W]


def _strip_scale_suffix(stem, scale):
    """babyx2 -> baby | frame_000001_x4 -> frame_000001 | name -> name.
    Suffixes with an underscore must be tried BEFORE those without; otherwise
    'frame_x4' would lose only 'x4' and keep a trailing underscore."""
    for suf in (f"_x{scale}", f"_X{scale}", f"x{scale}", f"X{scale}"):
        if stem.endswith(suf):
            return stem[:-len(suf)]
    return stem


def _find_hr(hr_dir, lr_filename, scale):
    """Find the HR file that matches an LR file: try the stem without the scale
    suffix, then the identical stem, with every known extension."""
    stem = os.path.splitext(lr_filename)[0]
    for s in (_strip_scale_suffix(stem, scale), stem):
        for ext in _IMG_EXTS:
            cand = os.path.join(hr_dir, s + ext)
            if os.path.isfile(cand):
                return cand
    return None


def _extract_state_dict(ck):
    # Key order matters: prefer EMA weights when a BasicSR checkpoint has both.
    if isinstance(ck, dict):
        for key in ('model', 'params_ema', 'params', 'state_dict'):
            if key in ck:
                return ck[key]
    return ck


def load_model(ckpt_path, device, arch=None, variant=None, overrides=None):
    """Rebuild the architecture from the checkpoint (arch/arch_cfg saved by
    train.py) when present, otherwise from arch/variant/overrides."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ck, dict) and ck.get('arch') and ck.get('arch_cfg'):
        model = model_from_cfg(ck['arch'], ck['arch_cfg'])
    else:
        if arch is None or variant is None:
            raise ValueError("checkpoint senza arch_cfg: specifica --arch e --variant")
        model, _ = build_model(arch, variant, **(overrides or {}))
    model.load_state_dict(_extract_state_dict(ck))
    return model.to(device).eval()


@torch.no_grad()
def sr_forward(model, lr, scale, tile=0, tile_overlap=32):
    """Super-resolve `lr` (1,3,H,W). With tile>0 process overlapping tiles."""
    if tile <= 0:
        return model(lr).clamp(0, 1)

    b, c, h, w = lr.shape
    stride = max(tile - tile_overlap, 1)
    # Tile start positions; append a last tile flush with the border so nothing is skipped.
    h_idx = list(range(0, max(h - tile, 0) + 1, stride)) or [0]
    w_idx = list(range(0, max(w - tile, 0) + 1, stride)) or [0]
    if h > tile and h_idx[-1] != h - tile:
        h_idx.append(h - tile)
    if w > tile and w_idx[-1] != w - tile:
        w_idx.append(w - tile)

    # Sum the SR patches and divide by the per-pixel overlap count to blend the seams.
    out = lr.new_zeros(b, c, h * scale, w * scale)
    weight = lr.new_zeros(b, 1, h * scale, w * scale)
    for hi in h_idx:
        for wi in w_idx:
            patch = lr[..., hi:hi + tile, wi:wi + tile]
            sr_patch = model(patch).clamp(0, 1)
            oh, ow = hi * scale, wi * scale
            ph, pw = sr_patch.shape[-2], sr_patch.shape[-1]
            out[..., oh:oh + ph, ow:ow + pw] += sr_patch
            weight[..., oh:oh + ph, ow:ow + pw] += 1.0
    return out / weight.clamp(min=1.0)


@torch.no_grad()
def super_resolve_file(model, in_path, out_path, scale, device, tile=0):
    """Load an LR image, super-resolve it and save the result."""
    img = Image.open(in_path).convert('RGB')
    lr = TF.to_tensor(img).unsqueeze(0).to(device)
    sr = sr_forward(model, lr, scale, tile=tile)
    save_image(sr, out_path)
    print(f"  {os.path.basename(in_path)}: "
          f"{lr.shape[-1]}x{lr.shape[-2]} -> {sr.shape[-1]}x{sr.shape[-2]}  ->  {out_path}")
    return sr


# ======================================================================================
# Entrypoint
# ======================================================================================
def main():
    p = argparse.ArgumentParser(description="Inference MambaIR / PCLM-SR (SR) su immagini")
    p.add_argument('--ckpt', required=True)
    p.add_argument('--arch', default=None, choices=['mambair', 'pclmsr'],
                   help="solo se il checkpoint non contiene arch_cfg")
    p.add_argument('--variant', default=None,
                   help="solo se il checkpoint non contiene arch_cfg")
    p.add_argument('--arch_override', nargs='*', default=None,
                   help="override architetturali (solo senza arch_cfg nel ckpt)")
    p.add_argument('--input', required=True, help="file immagine o cartella")
    p.add_argument('--output', required=True, help="file (input singolo) o cartella")
    p.add_argument('--scale', type=int, default=None, help="default: dedotto dal modello")
    p.add_argument('--tile', type=int, default=0, help=">0 abilita il tiling (px LR per riquadro)")
    p.add_argument('--hr_dir', default=None,
                   help="cartella delle immagini HR di riferimento: se data, calcola PSNR/SSIM "
                        "per immagine (protocollo di test.py) e scrive un CSV")
    p.add_argument('--csv', default=None,
                   help="percorso del CSV con le metriche (default: <output>/metrics.csv)")
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.ckpt, device, arch=args.arch, variant=args.variant,
                       overrides=parse_overrides(args.arch_override))
    scale = args.scale if args.scale is not None else model.upscale

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params/1e6:.3f}M parametri | scale={scale} | device={device}")

    rows = []

    def _measure(sr, lr_name, sr_path):
        """If --hr_dir is given, compute PSNR/SSIM of `sr` against the matching HR."""
        if not args.hr_dir:
            return
        hp = _find_hr(args.hr_dir, lr_name, scale)
        if hp is None:
            print(f"    (HR non trovato per {lr_name}: metriche saltate)")
            rows.append((lr_name, float('nan'), float('nan'), sr_path))
            return
        hr = TF.to_tensor(Image.open(hp).convert('RGB')).unsqueeze(0).to(device)
        a, b = _match_size(sr, hr)
        p_ = calc_psnr(a, b, crop_border=scale)
        s_ = calc_ssim(a, b, crop_border=scale)
        print(f"    PSNR {p_:6.2f} dB | SSIM {s_:.4f}")
        rows.append((lr_name, p_, s_, sr_path))

    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        files = sorted(f for f in os.listdir(args.input)
                       if f.lower().endswith(_IMG_EXTS))
        if not files:
            print(f"nessuna immagine in {args.input}")
            return
        print(f"super-risoluzione di {len(files)} immagini -> {args.output}")
        for f in files:
            out_path = os.path.join(args.output, f"{os.path.splitext(f)[0]}_x{scale}.png")
            sr = super_resolve_file(model, os.path.join(args.input, f), out_path,
                                    scale, device, tile=args.tile)
            _measure(sr, f, out_path)
    else:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        sr = super_resolve_file(model, args.input, args.output, scale, device, tile=args.tile)
        _measure(sr, os.path.basename(args.input), args.output)

    if args.hr_dir and rows:
        csv_path = args.csv or os.path.join(
            args.output if os.path.isdir(args.output) else (os.path.dirname(args.output) or '.'),
            'metrics.csv')
        # p_ == p_ is False only for NaN: drop the images without a matching HR.
        valid = [(p_, s_) for _, p_, s_, _ in rows if p_ == p_]
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or '.', exist_ok=True)
        with open(csv_path, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['image', 'psnr', 'ssim', 'sr_path'])
            for name, p_, s_, sp in rows:
                w.writerow([name, f"{p_:.4f}", f"{s_:.6f}", sp])
            if valid:
                w.writerow(['MEAN', f"{statistics.mean(v[0] for v in valid):.4f}",
                            f"{statistics.mean(v[1] for v in valid):.6f}", ''])
        if valid:
            print("-" * 56)
            print(f"MEDIA su {len(valid)} immagini: "
                  f"PSNR {statistics.mean(v[0] for v in valid):6.2f} dB | "
                  f"SSIM {statistics.mean(v[1] for v in valid):.4f}")
        print(f"CSV: {csv_path}")


if __name__ == '__main__':
    main()
