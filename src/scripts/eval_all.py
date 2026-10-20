"""
eval_all.py — PSNR/SSIM (Y channel, crop = scale) and results/quality.csv.

One metric function for everything: `evaluate_folder(sr_dir, hr_dir, scale)` compares
the PNGs of an SR folder with the HR ones (YCbCr conversion and PSNR/SSIM as in BasicSR:
metrics.py -> basicsr.metrics.calculate_psnr_pt / calculate_ssim_pt, test_y_channel=True,
crop_border=scale; SR and HR quantized to uint8; top-left alignment / mod-crop).
This way the baselines are generated with THEIR own code (test script + official weights) and
evaluated here with exactly the same metric as our models.

Subcommands
-----------
  ours    : run ONE of our checkpoints -> results/sr/{config}/seed{seed}/x{scale}/{Dataset}/
            and metrics on that folder (rho, seed, iters read from the checkpoint).
  sweep   : like `ours` for every checkpoint  experiments/rho_*_x*/seed*/iter_<N>.pth
  folders : metrics on SR folders already generated (baselines):  <sr_root>/<Dataset>/*.png
  metrics : metrics on a single pair of folders (debug)

Usage
-----
  python scripts/eval_all.py ours --ckpt experiments/rho_050_x2/seed0/iter_100000.pth
  python scripts/eval_all.py sweep --iters 100000 500000
  python scripts/eval_all.py folders --method SwinIR-light --scale 2 --sr_root results/sr_baselines/SwinIR-light/x2
  python scripts/eval_all.py metrics --sr_dir <dir> --hr_dir <dir> --scale 2
Benchmark HR/LR: --benchmark_dir (default ../datasets/SRBenchmarks/benchmark), layout
<Dataset>/HR/*.png and <Dataset>/LR_bicubic/X<s>/<stem>x<s>.png.
"""
import argparse
import glob
import os
import re
import statistics
import sys

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ROOT, RESULTS_DIR, DATASETS, QUALITY_FIELDS, QUALITY_KEY, env_info,  # noqa: E402
                    env_banner, upsert_csv, load_ours_checkpoint, sr_forward, ours_method_name,
                    fmt_rho, config_name, benchmark_pairs, benchmark_dirs, BENCHMARK_DIR)
from metrics import calc_psnr, calc_ssim, USE_BASICSR  # noqa: E402

_IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
DEFAULT_BENCH = BENCHMARK_DIR


def _open_rgb(path):
    return Image.open(path).convert('RGB')


def _index_images(root):
    idx = {}
    # Map stem -> path; the first file found for a stem wins.
    for cur, _, files in os.walk(root):
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() in _IMG_EXTS:
                idx.setdefault(stem, os.path.join(cur, f))
    return idx


def _strip_suffix(stem, scale):
    # Drop the suffixes baseline scripts add to SR file names so the stem matches the HR one.
    for suf in (f"_x{scale}", f"_X{scale}", f"x{scale}", f"X{scale}", f"_SRF_{scale}",
                f"_LR{scale}", "_sr", "_SR", f"_x{scale}_SR", "_rlt", "_out"):
        if stem.endswith(suf):
            return stem[:-len(suf)]
    return stem


def _match_size(a, b):
    """Crop top-left to the common size (benchmark mod-crop)."""
    H = min(a.shape[-2], b.shape[-2]); W = min(a.shape[-1], b.shape[-1])
    return a[..., :H, :W], b[..., :H, :W]


@torch.no_grad()
def evaluate_folder(sr_dir, hr_dir, scale, device='cuda', verbose=False):
    """Return (mean psnr_y, mean ssim_y, n, per_image[(name, psnr, ssim)])."""
    hr_idx = _index_images(hr_dir)
    sr_items = sorted(_index_images(sr_dir).items())
    if not sr_items:
        raise RuntimeError(f"nessuna immagine SR in {sr_dir}")
    rows, missing = [], []
    for stem, sp in sr_items:
        hp = hr_idx.get(stem) or hr_idx.get(_strip_suffix(stem, scale))
        if hp is None:
            missing.append(stem)
            continue
        sr = TF.to_tensor(_open_rgb(sp)).unsqueeze(0).to(device)
        hr = TF.to_tensor(_open_rgb(hp)).unsqueeze(0).to(device)
        sr, hr = _match_size(sr, hr)
        rows.append((stem, calc_psnr(sr, hr, crop_border=scale), calc_ssim(sr, hr, crop_border=scale)))
        if verbose:
            print(f"    {stem:<30} {rows[-1][1]:6.2f} {rows[-1][2]:.4f}")
    if missing:
        print(f"  ATTENZIONE: {len(missing)} SR senza HR corrispondente (es. {missing[:3]})")
    if not rows:
        raise RuntimeError(f"nessuna coppia SR/HR in {sr_dir} vs {hr_dir}")
    return (statistics.mean(r[1] for r in rows), statistics.mean(r[2] for r in rows), len(rows), rows)


bench_dirs = benchmark_dirs


@torch.no_grad()
def infer_pairs(model, pairs, out_dir, scale, device='cuda', tile=0):
    """SR of the LR images in `pairs` [(hr_path, lr_path)] -> out_dir/<HR stem>.png. Uses ONLY
    the LR '<stem>x<scale>' paired with an HR (the folders also hold copies with other names)."""
    os.makedirs(out_dir, exist_ok=True)
    if not pairs:
        raise RuntimeError("nessuna coppia HR/LR")
    for hr_path, lr_path in pairs:
        stem = os.path.splitext(os.path.basename(hr_path))[0]
        lr = TF.to_tensor(_open_rgb(lr_path)).unsqueeze(0).to(device)
        sr = sr_forward(model, lr, scale, tile=tile)
        save_image(sr, os.path.join(out_dir, f"{stem}.png"))
    return len(pairs)


def _ckpt_meta(ck, ckpt_path):
    """(config_name, rho, seed, iters, scale) from the checkpoint (run_meta/arch_cfg)."""
    arch_cfg = ck['arch_cfg']
    rho = float(arch_cfg['mamba_ratio'])
    scale = int(arch_cfg['upscale'])
    seed = ck.get('seed')
    iters = int(ck.get('iter', 0))
    meta = ck.get('run_meta') or {}
    name = (meta.get('config') or {}).get('name') or config_name(rho, scale)
    if seed is None:
        # Old checkpoints store no seed: recover it from the path (.../seed<N>/...).
        m = re.search(r'seed(\d+)', ckpt_path)
        seed = int(m.group(1)) if m else -1
    return name, rho, int(seed), iters, scale


def eval_ours(ckpt_path, benchmark_dir, datasets, device, tile, sr_root, env, verbose=False):
    model, ck = load_ours_checkpoint(ckpt_path, device)
    name, rho, seed, iters, scale = _ckpt_meta(ck, ckpt_path)
    print(f"\n[{name}] seed={seed} iter={iters} rho={rho:.2f} x{scale} | {ckpt_path}")
    rows = []
    for ds in datasets:
        hr_dir, lr_dir = bench_dirs(benchmark_dir, ds, scale)
        if not os.path.isdir(lr_dir):
            print(f"  {ds}: LR mancante ({lr_dir}), salto")
            continue
        out_dir = os.path.join(sr_root, name, f"seed{seed}", f"x{scale}", ds)
        pairs = benchmark_pairs(benchmark_dir, ds, scale)
        infer_pairs(model, pairs, out_dir, scale, device, tile)
        psnr, ssim, n, _ = evaluate_folder(out_dir, hr_dir, scale, device, verbose)
        print(f"  {ds:<9} {n:>3} img | PSNR {psnr:6.2f} | SSIM {ssim:.4f}   -> {out_dir}")
        rows.append(dict(method=ours_method_name(rho), rho=fmt_rho(rho), seed=seed, iters=iters,
                         scale=scale, dataset=ds, psnr_y=f"{psnr:.4f}", ssim_y=f"{ssim:.5f}",
                         checkpoint=os.path.relpath(os.path.abspath(ckpt_path), ROOT),
                         gpu=env['gpu'], commit=env['commit'], date=env['date']))
    return rows


# ======================================================================================
# main
# ======================================================================================
def main():
    p = argparse.ArgumentParser(description="PSNR/SSIM canale Y -> results/quality.csv")
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(sp):
        sp.add_argument('--benchmark_dir', default=DEFAULT_BENCH)
        sp.add_argument('--datasets', nargs='+', default=DATASETS)
        sp.add_argument('--out', default=os.path.join(RESULTS_DIR, 'quality.csv'))
        sp.add_argument('--verbose', action='store_true')

    s = sub.add_parser('ours'); common(s)
    s.add_argument('--ckpt', required=True, nargs='+')
    s.add_argument('--tile', type=int, default=0)
    s.add_argument('--sr_root', default=os.path.join(RESULTS_DIR, 'sr'))

    s = sub.add_parser('sweep'); common(s)
    s.add_argument('--experiments_root', default=os.path.join(ROOT, 'experiments'))
    s.add_argument('--iters', type=int, nargs='+', default=[100000],
                   help="quali iter_<N>.pth valutare (default 100000)")
    s.add_argument('--tile', type=int, default=0)
    s.add_argument('--sr_root', default=os.path.join(RESULTS_DIR, 'sr'))
    s.add_argument('--skip_done', action='store_true', help="salta i checkpoint gia' nel CSV")

    s = sub.add_parser('folders'); common(s)
    s.add_argument('--method', required=True, help="nome nel CSV (es. SwinIR-light)")
    s.add_argument('--scale', type=int, required=True)
    s.add_argument('--sr_root', required=True, help="cartella con <Dataset>/*.png")
    s.add_argument('--note', default='', help="va nella colonna checkpoint (es. URL pesi)")

    s = sub.add_parser('metrics')
    s.add_argument('--sr_dir', required=True); s.add_argument('--hr_dir', required=True)
    s.add_argument('--scale', type=int, required=True); s.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = env_info()
    print(env_banner(env))
    print(f"metriche: {'BasicSR calculate_psnr_pt/calculate_ssim_pt' if USE_BASICSR else 'interne (equivalenti)'} "
          f"| canale Y | crop_border = scale | uint8 | device={device}")

    if args.cmd == 'metrics':
        psnr, ssim, n, _ = evaluate_folder(args.sr_dir, args.hr_dir, args.scale, device, args.verbose)
        print(f"{n} img | PSNR {psnr:.4f} | SSIM {ssim:.5f}")
        return

    rows = []
    if args.cmd == 'ours':
        for ck in args.ckpt:
            rows += eval_ours(ck, args.benchmark_dir, args.datasets, device, args.tile,
                              args.sr_root, env, args.verbose)
    elif args.cmd == 'sweep':
        from common import read_csv
        done = set()
        if args.skip_done:
            done = {r['checkpoint'] for r in read_csv(args.out)}
        pat = os.path.join(args.experiments_root, 'rho_*_x*', 'seed*', 'iter_*.pth')
        ckpts = sorted(glob.glob(pat))
        ckpts = [c for c in ckpts if int(re.search(r'iter_(\d+)\.pth', c).group(1)) in args.iters]
        print(f"checkpoint trovati: {len(ckpts)} ({pat}, iters {args.iters})")
        for ck in ckpts:
            if os.path.relpath(os.path.abspath(ck), ROOT) in done:
                print(f"  gia' valutato: {ck}")
                continue
            rows += eval_ours(ck, args.benchmark_dir, args.datasets, device, args.tile,
                              args.sr_root, env, args.verbose)
            # Write the CSV after every checkpoint so an interrupted sweep keeps its results.
            upsert_csv(args.out, rows, QUALITY_FIELDS, QUALITY_KEY)
    elif args.cmd == 'folders':
        for ds in args.datasets:
            sr_dir = os.path.join(args.sr_root, ds)
            if not os.path.isdir(sr_dir):
                print(f"  {ds}: cartella SR mancante ({sr_dir}), salto")
                continue
            hr_dir, _ = bench_dirs(args.benchmark_dir, ds, args.scale)
            psnr, ssim, n, _ = evaluate_folder(sr_dir, hr_dir, args.scale, device, args.verbose)
            print(f"  [{args.method}] {ds:<9} {n:>3} img | PSNR {psnr:6.2f} | SSIM {ssim:.4f}")
            rows.append(dict(method=args.method, rho='', seed='', iters='', scale=args.scale,
                             dataset=ds, psnr_y=f"{psnr:.4f}", ssim_y=f"{ssim:.5f}",
                             checkpoint=args.note or os.path.relpath(os.path.abspath(sr_dir), ROOT),
                             gpu=env['gpu'], commit=env['commit'], date=env['date']))
    if rows:
        upsert_csv(args.out, rows, QUALITY_FIELDS, QUALITY_KEY)
        print(f"\nCSV aggiornato: {args.out} (+{len(rows)} righe)")


if __name__ == '__main__':
    main()
