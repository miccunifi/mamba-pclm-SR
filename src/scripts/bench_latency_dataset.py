"""
bench_latency_dataset.py — Per-image latency (ms/img) and throughput (FPS) of full-resolution
inference on the images of a benchmark (Urban100 and/or Manga109), for our models
(rho in {0,.25,.5,.75,1}) and for the baselines (same adapters as the other scripts).

Definitions
-----------
  latency_ms_mean = mean, over the dataset images, of the time of ONE forward (batch 1)
  fps             = 1000 / latency_ms_mean

Protocol
--------
* Strict FP32 (TF32 off), batch 1, torch.no_grad(), forward only, random weights (latency
  depends on architecture and resolution, not on the weights), cudnn.benchmark=False for all
  (--cudnn_benchmark turns it on for all: with images of different sizes it would cost one
  autotune for every new size).
* Images: ONLY the LR images '<stem>x<scale>' that have a paired HR (100 Urban100,
  109 Manga109), at their native resolution, read once and kept on GPU: disk loading and the
  host->device transfer are NOT timed.
* Warm-up: --warmup forwards (default 10) on the first image, before timing.
* Each image is timed --repeats times (default 3) with a pair of torch.cuda.Event and
  torch.cuda.synchronize() before/after; the image time is the mean of the repetitions.
  Mean, median, std, min and max over the images are reported, plus FPS.
* GPU idle: nvidia-smi is checked before measuring (WARNING if other processes are running).
* An image that goes OOM is counted (n_oom) and excluded from the statistics.

Output: results/latency_<dataset>.csv with schema
  method, rho, scale, dataset, n_images, n_oom, warmup, repeats, latency_ms_mean,
  latency_ms_median, latency_ms_std, latency_ms_min, latency_ms_max, fps, gpu, commit, date

Usage
-----
  python scripts/bench_latency_dataset.py                 # ours, Urban100+Manga109, x2/x3/x4
  python scripts/bench_latency_dataset.py --baselines all
  python scripts/bench_latency_dataset.py --datasets Urban100 --scales 4 --rhos 0 1 --baselines SwinIR-light
  python scripts/bench_latency_dataset.py --repeats 5 --warmup 20 --md results/latency_datasets.md
"""
import argparse
import os
import statistics
import sys

import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RHOS, SCALES, RESULTS_DIR, BENCHMARK_DIR, build_ours, env_info, env_banner,  # noqa: E402
                    upsert_csv, gpu_other_processes, ours_method_name, fmt_rho, sr_forward,
                    benchmark_pairs)
import baselines as B  # noqa: E402

FIELDS = ['method', 'rho', 'scale', 'dataset', 'n_images', 'n_oom', 'warmup', 'repeats',
          'latency_ms_mean', 'latency_ms_median', 'latency_ms_std', 'latency_ms_min', 'latency_ms_max',
          'fps', 'gpu', 'commit', 'date']
KEY = ['method', 'rho', 'scale', 'dataset']


def load_images(benchmark_dir, dataset, scale):
    """Return (filename, LR tensor (1,3,H,W) on GPU) pairs, loaded once per dataset/scale."""
    imgs = []
    for _, lr_path in benchmark_pairs(benchmark_dir, dataset, scale):
        imgs.append((os.path.basename(lr_path),
                     TF.to_tensor(Image.open(lr_path).convert('RGB')).unsqueeze(0).to('cuda')))
    return imgs


@torch.no_grad()
def time_once(model, x, scale, tile):
    # Sync before and after so the two events bracket only this forward.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    sr_forward(model, x, scale, tile=tile)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


@torch.no_grad()
def measure_model(label, model, scale, images, args, env, dataset, rho=None):
    for _ in range(args.warmup):
        sr_forward(model, images[0][1], scale, tile=args.tile)
    torch.cuda.synchronize()
    per_img, oom = [], 0
    for name, x in images:
        try:
            t = statistics.mean(time_once(model, x, scale, args.tile) for _ in range(args.repeats))
            per_img.append(t)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            oom += 1
            print(f"    OOM: {name} ({x.shape[-2]}x{x.shape[-1]})", flush=True)
    n = len(per_img)
    mean = statistics.mean(per_img) if n else float('nan')
    row = dict(method=label, rho=fmt_rho(rho) if rho is not None else '', scale=scale, dataset=dataset,
               n_images=n, n_oom=oom, warmup=args.warmup, repeats=args.repeats,
               latency_ms_mean=f"{mean:.3f}" if n else '',
               latency_ms_median=f"{statistics.median(per_img):.3f}" if n else '',
               latency_ms_std=f"{(statistics.stdev(per_img) if n > 1 else 0.0):.3f}" if n else '',
               latency_ms_min=f"{min(per_img):.3f}" if n else '',
               latency_ms_max=f"{max(per_img):.3f}" if n else '',
               fps=f"{1000.0/mean:.3f}" if n else '',
               gpu=env['gpu'], commit=env['commit'], date=env['date'])
    print(f"{label:<18} x{scale} {dataset:<9} | {n:>3} img, {oom} OOM | {mean:9.2f} ms/img "
          f"(median {row['latency_ms_median']:>9}, std {row['latency_ms_std']:>8}, "
          f"min {row['latency_ms_min']:>9}, max {row['latency_ms_max']:>9}) | {row['fps']:>8} FPS", flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description="Latenza (ms/img) e FPS = 1000/ms su Urban100 / Manga109")
    p.add_argument('--datasets', nargs='+', default=['Urban100', 'Manga109'])
    p.add_argument('--benchmark_dir', default=BENCHMARK_DIR)
    p.add_argument('--scales', type=int, nargs='+', default=SCALES)
    p.add_argument('--rhos', type=float, nargs='+', default=RHOS)
    p.add_argument('--baselines', nargs='*', default=[], help="'all' oppure nomi; default nessuna")
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeats', type=int, default=3, help="ripetizioni cronometrate per immagine")
    p.add_argument('--tile', type=int, default=0, help="0 = immagine intera")
    p.add_argument('--cudnn_benchmark', action='store_true')
    p.add_argument('--out_dir', default=RESULTS_DIR, help="scrive <out_dir>/latency_<dataset>.csv")
    p.add_argument('--md', default=None, help="tabella markdown riassuntiva opzionale")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("serve una GPU")

    # Strict FP32 for every model; cudnn autotune is off unless requested.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

    env = env_info()
    print(env_banner(env))
    print(f"protocollo: FP32 (TF32 off) | batch 1 | no_grad | forward only | warmup {args.warmup} | "
          f"repeats {args.repeats}/img | CUDA events + synchronize | cudnn.benchmark={torch.backends.cudnn.benchmark} "
          f"| tile={args.tile} | FPS = 1000 / mean ms")
    others = gpu_other_processes()
    print(f"WARNING: la GPU NON e' idle: {others}" if others else "GPU idle (nvidia-smi: nessun altro processo compute)")

    md = ["| Model | rho | scale | dataset | n | ms/img (mean) | median | std | FPS |", "|---|---|---|---|---|---|---|---|---|"]
    n_rows = 0
    for dataset in args.datasets:
        out = os.path.join(args.out_dir, f'latency_{dataset}.csv')
        for scale in args.scales:
            images = load_images(args.benchmark_dir, dataset, scale)
            print(f"\n=== {dataset} x{scale}: {len(images)} immagini ===")
            # Default args bind the loop variable so each lambda builds its own model.
            todo = [(ours_method_name(r), lambda r=r: build_ours(r, scale, device='cuda')[0], r) for r in args.rhos]
            todo += [(nm, lambda nm=nm: B.build_baseline(nm, scale, device='cuda')[0], None)
                     for nm in B.resolve_names(args.baselines)]
            for label, build, rho in todo:
                try:
                    model = build()
                except Exception as e:
                    print(f"{label:<18} x{scale} | ERRORE: {type(e).__name__}: {e}", flush=True)
                    continue
                row = measure_model(label, model, scale, images, args, env, dataset, rho=rho)
                upsert_csv(out, [row], FIELDS, KEY)
                n_rows += 1
                md.append(f"| {label} | {row['rho'] or '-'} | x{scale} | {dataset} | {row['n_images']} | "
                          f"{row['latency_ms_mean']} | {row['latency_ms_median']} | {row['latency_ms_std']} | {row['fps']} |")
                del model
                torch.cuda.empty_cache()
            del images
            torch.cuda.empty_cache()
        print(f"CSV aggiornato: {out}")
    if args.md:
        os.makedirs(os.path.dirname(os.path.abspath(args.md)) or '.', exist_ok=True)
        with open(args.md, 'w') as f:
            f.write(env_banner(env) + "\n\n" + "\n".join(md) + "\n")
        print(f"Markdown: {args.md}")
    print(f"righe scritte: {n_rows}")


if __name__ == '__main__':
    main()
