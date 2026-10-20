"""
bench_memory_dataset.py — GPU memory used by full-resolution inference on the images of a
benchmark (default Urban100), for our models (rho in {0,.25,.5,.75,1}) and for the baselines
(same adapters as count_params_macs.py / bench_latency.py).

Protocol
--------
* Strict FP32 (TF32 off), batch 1, torch.no_grad(), forward only, random weights (activation
  memory depends on architecture and resolution, not on the weights).
* For EACH LR image of the benchmark: torch.cuda.reset_peak_memory_stats() -> forward ->
  torch.cuda.synchronize() -> torch.cuda.max_memory_allocated(). We record the maximum over
  the images (peak_mem_mb_max, with the name and LR size of the image that causes it), the
  mean (peak_mem_mb_mean), the static memory of the weights (weights_mem_mb) and the
  activations (act_mem_mb_max = peak_max - weights). max_memory_reserved is also reported
  (memory actually held by the caching allocator, which is what nvidia-smi shows).
* An image that goes OOM is counted (column n_oom) and the run moves on to the next image.
* --tile N: tiled inference (N px LR, overlap 32) as in the official tests with tiling;
  default 0 (whole image). Use the same value for all compared models.
* Warm-up: the first image runs twice (the first run is excluded from the statistics).

Output: results/memory_<dataset>.csv with schema
  method, rho, scale, dataset, n_images, n_oom, tile, peak_mem_mb_max, peak_mem_mb_mean,
  reserved_mb_max, weights_mem_mb, act_mem_mb_max, max_image, max_lr_hw, gpu, commit, date

Usage
-----
  python scripts/bench_memory_dataset.py                    # our models, Urban100, x2/x3/x4
  python scripts/bench_memory_dataset.py --baselines all    # + baselines
  python scripts/bench_memory_dataset.py --scales 4 --rhos 0 1 --baselines SwinIR-light MambaIR-light
  python scripts/bench_memory_dataset.py --dataset Manga109 --tile 256
"""
import argparse
import os
import statistics
import sys

import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RHOS, SCALES, RESULTS_DIR, BENCHMARK_DIR, build_ours, env_info,  # noqa: E402
                    env_banner, upsert_csv, gpu_other_processes, ours_method_name, fmt_rho, sr_forward,
                    benchmark_pairs)
import baselines as B  # noqa: E402

DEFAULT_BENCH = BENCHMARK_DIR
FIELDS = ['method', 'rho', 'scale', 'dataset', 'n_images', 'n_oom', 'tile', 'peak_mem_mb_max',
          'peak_mem_mb_mean', 'reserved_mb_max', 'weights_mem_mb', 'act_mem_mb_max', 'max_image',
          'max_lr_hw', 'gpu', 'commit', 'date']
KEY = ['method', 'rho', 'scale', 'dataset', 'tile']
MB = 2 ** 20


def list_lr(benchmark_dir, dataset, scale):
    """LR paths '<stem>x<scale>.png' that have a matching HR (one per benchmark image)."""
    return [lr for _, lr in benchmark_pairs(benchmark_dir, dataset, scale)]


def weights_bytes(model):
    return (sum(p.numel() * p.element_size() for p in model.parameters()) +
            sum(b.numel() * b.element_size() for b in model.buffers()))


@torch.no_grad()
def measure_model(label, model, scale, paths, tile, env, dataset, rho=None):
    wb = weights_bytes(model)
    peaks, reserved, oom = [], [], 0
    max_peak, max_img, max_hw = -1, '', ''
    first = True
    for pth in paths:
        lr = TF.to_tensor(Image.open(pth).convert('RGB')).unsqueeze(0).to('cuda')
        h, w = lr.shape[-2:]
        # Warm-up: the first image runs twice; only the last run of each image is recorded.
        reps = 2 if first else 1
        first = False
        for r in range(reps):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                y = sr_forward(model, lr, scale, tile=tile)
                torch.cuda.synchronize()
                pk = torch.cuda.max_memory_allocated()
                rs = torch.cuda.max_memory_reserved()
                del y
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if r == reps - 1:
                    oom += 1
                    print(f"    OOM: {os.path.basename(pth)} ({h}x{w})", flush=True)
                continue
            if r == reps - 1:
                peaks.append(pk); reserved.append(rs)
                if pk > max_peak:
                    max_peak, max_img, max_hw = pk, os.path.basename(pth), f"{h}x{w}"
        del lr
    n = len(peaks)
    row = dict(method=label, rho=fmt_rho(rho) if rho is not None else '', scale=scale, dataset=dataset,
               n_images=n, n_oom=oom, tile=tile,
               peak_mem_mb_max=f"{max_peak/MB:.1f}" if n else '',
               peak_mem_mb_mean=f"{statistics.mean(peaks)/MB:.1f}" if n else '',
               reserved_mb_max=f"{max(reserved)/MB:.1f}" if n else '',
               weights_mem_mb=f"{wb/MB:.2f}",
               act_mem_mb_max=f"{(max_peak-wb)/MB:.1f}" if n else '',
               max_image=max_img, max_lr_hw=max_hw,
               gpu=env['gpu'], commit=env['commit'], date=env['date'])
    print(f"{label:<18} x{scale} {dataset:<9} | {n:>3} img, {oom} OOM | peak max {row['peak_mem_mb_max']:>8} MB "
          f"({max_img} LR {max_hw}) | mean {row['peak_mem_mb_mean']:>8} MB | reserved max "
          f"{row['reserved_mb_max']:>8} MB | weights {row['weights_mem_mb']} MB", flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description="Memoria GPU dell'inferenza a piena risoluzione su un benchmark")
    p.add_argument('--dataset', default='Urban100')
    p.add_argument('--benchmark_dir', default=DEFAULT_BENCH)
    p.add_argument('--scales', type=int, nargs='+', default=SCALES)
    p.add_argument('--rhos', type=float, nargs='+', default=RHOS)
    p.add_argument('--baselines', nargs='*', default=[], help="'all' oppure nomi; default nessuna")
    p.add_argument('--tile', type=int, default=0, help="0 = immagine intera")
    p.add_argument('--out', default=None, help="default results/memory_<dataset>.csv")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("serve una GPU")
    out = args.out or os.path.join(RESULTS_DIR, f'memory_{args.dataset}.csv')

    # Strict FP32: TF32 off; cudnn autotuning off so its workspace does not inflate the peak.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    env = env_info()
    print(env_banner(env))
    print(f"protocollo: FP32 (TF32 off) | batch 1 | no_grad | forward only | tile={args.tile} | "
          f"reset_peak_memory_stats per immagine | dataset {args.dataset} @ {args.benchmark_dir}")
    others = gpu_other_processes()
    if others:
        print(f"WARNING: altri processi sulla GPU (max_memory_reserved include solo questo processo): {others}")

    rows = []
    for scale in args.scales:
        paths = list_lr(args.benchmark_dir, args.dataset, scale)
        print(f"\n=== x{scale} | {args.dataset}: {len(paths)} immagini ===")
        for rho in args.rhos:
            model, _ = build_ours(rho, scale, device='cuda')
            r = measure_model(ours_method_name(rho), model, scale, paths, args.tile, env, args.dataset, rho=rho)
            rows.append(r); upsert_csv(out, [r], FIELDS, KEY)
            del model; torch.cuda.empty_cache()
        for name in B.resolve_names(args.baselines):
            try:
                model, _ = B.build_baseline(name, scale, device='cuda')
            except Exception as e:
                print(f"{name:<18} x{scale} | ERRORE: {type(e).__name__}: {e}", flush=True)
                continue
            r = measure_model(name, model, scale, paths, args.tile, env, args.dataset)
            rows.append(r); upsert_csv(out, [r], FIELDS, KEY)
            del model; torch.cuda.empty_cache()
    print(f"\nCSV aggiornato: {out} ({len(rows)} righe)")


if __name__ == '__main__':
    main()
