"""
train_baseline_memory.py — Short or long training of the OFFICIAL MambaIR-light and
MambaIRv2-light architectures with the SAME recipe as our models, to measure their GPU memory
and training speed (also watch them with nvtop / nvidia-smi).

Recipe (identical to train.py / configs/*.yml): DIV2K train (800 img), LR patch 64x64 (HR 128 at
x2, 192 at x3/x4), batch 32, h/v flip, Adam(2e-4, betas 0.9/0.99, wd 0), L1, FP32,
cudnn.benchmark=False.
The model comes from scripts/baselines.py (arch file of the official repo, lightweight config of
the official .yml): MambaIR-light = embed 60, depths 6x4, d_state 10, mlp 1.2;
MambaIRv2-light = embed 48, depths 5x4, heads 4, window 16, d_state 8. Weights are random.

What it prints
--------------
* every --print_freq iterations: loss, it/s, current and peak allocated/reserved memory
  (torch.cuda.max_memory_allocated / max_memory_reserved);
* at the end of the run: peak allocated (live tensors), peak reserved (= what nvtop/nvidia-smi
  shows for this process, minus the CUDA context of ~300-600 MB), weight memory, and an
  optional CSV row (--csv) with schema
    model, scale, batch_size, lr_patch, iters, params, peak_alloc_mb, peak_reserved_mb,
    weights_mb, it_per_s, gpu, commit, date
* with --ckpt_dir it saves final.pth (state_dict + config) at the end of the run.

Usage
-----
  python scripts/train_baseline_memory.py --model mambair_light --scale 2 --iters 200
  python scripts/train_baseline_memory.py --model mambairv2_light --scale 2 --iters 200
  python scripts/train_baseline_memory.py --model mambair_light --scale 4 --batch_size 16 --iters 100
  python scripts/train_baseline_memory.py --model rho:0.5 --scale 2 --iters 200  # compare with ours
  # long run (e.g. 100K iter) with checkpoint:
  python scripts/train_baseline_memory.py --model mambairv2_light --scale 2 --iters 100000 \
      --print_freq 200 --ckpt_dir experiments/mambairv2_light_x2 --csv results/train_memory.csv
Models: mambair_light | mambairv2_light | any name in baselines.BASELINES | rho:<value>
"""
import argparse
import os
import random
import statistics
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, RESULTS_DIR, build_ours, count_params, env_info, env_banner, upsert_csv  # noqa: E402
import baselines as B  # noqa: E402
from dataset import build_paired_dataset  # noqa: E402
from models import TRAIN_CONFIGS  # noqa: E402

MB = 2 ** 20
ALIASES = {'mambair_light': 'MambaIR-light', 'mambairv2_light': 'MambaIRv2-light'}
FIELDS = ['model', 'scale', 'batch_size', 'lr_patch', 'iters', 'params', 'peak_alloc_mb',
          'peak_reserved_mb', 'weights_mb', 'it_per_s', 'gpu', 'commit', 'date']


def build(name, scale):
    # 'rho:<v>' builds our PCLM-SR model; any other name resolves to an official baseline.
    if name.startswith('rho:'):
        model, _ = build_ours(float(name[4:]), scale, device='cuda')
        return model, f"PCLM-rho{float(name[4:]):.2f}"
    key = ALIASES.get(name.lower())
    if key is None:
        key = B.resolve_names([name])[0]
    model, _ = B.build_baseline(key, scale, device='cuda')
    return model, key


def _seed_worker(worker_id):
    # Seed Python's random in each DataLoader worker from that worker's torch seed.
    ws = torch.initial_seed() % (2 ** 32)
    random.seed(ws)


def mem_line():
    return (f"alloc {torch.cuda.memory_allocated()/MB:7.0f} MB (peak {torch.cuda.max_memory_allocated()/MB:7.0f}) | "
            f"reserved {torch.cuda.memory_reserved()/MB:7.0f} MB (peak {torch.cuda.max_memory_reserved()/MB:7.0f})")


def main():
    p = argparse.ArgumentParser(description="Training MambaIR-light / MambaIRv2-light (ricetta nostra) con report memoria")
    p.add_argument('--model', required=True, help="mambair_light | mambairv2_light | <baseline> | rho:<v>")
    p.add_argument('--scale', type=int, default=2, choices=[2, 3, 4])
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--milestones', type=int, nargs='+', default=None,
                   help="default: quelle della ricetta (250K 400K 450K 475K), gamma 0.5")
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--print_freq', type=int, default=20)
    p.add_argument('--data_root', default=os.path.normpath(os.path.join(ROOT, '..', 'datasets')))
    p.add_argument('--ckpt_dir', default=None, help="se impostato salva final.pth")
    p.add_argument('--csv', default=None, help="es. results/train_memory.csv (upsert su model/scale/batch)")
    p.add_argument('--use_checkpointing', action='store_true',
                   help="MambaIRv2Light: use_checkpoint=True (gradient checkpointing, meno memoria)")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("serve una GPU")

    random.seed(args.seed); torch.manual_seed(args.seed)
    # Strict FP32 (no TF32, no cudnn autotune), as in the training recipe.
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    env = env_info()
    print(env_banner(env))

    sched = TRAIN_CONFIGS[f'pclmsr_x{args.scale}']
    hr_patch = sched['hr_patch_size']
    milestones = args.milestones or list(sched['milestones'])

    train_hr = os.path.join(args.data_root, 'DIV2K', 'DIV2K_train_HR')
    train_lr = os.path.join(args.data_root, 'DIV2K', 'DIV2K_train_LR_bicubic', f'X{args.scale}')
    ds = build_paired_dataset([train_hr], [train_lr], scale=args.scale, hr_patch_size=hr_patch,
                              filename_tmpl=f'{{}}x{args.scale}', train_mode=True)
    g = torch.Generator(); g.manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=True, drop_last=True, generator=g, worker_init_fn=_seed_worker)

    model, label = build(args.model, args.scale)
    # Turn on gradient checkpointing in every block that supports it.
    if args.use_checkpointing:
        for m in model.modules():
            if hasattr(m, 'use_checkpoint'):
                m.use_checkpoint = True
    model.train()
    n_params = count_params(model)
    weights_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / MB
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.5)
    crit = torch.nn.L1Loss()
    print(f"[{label}] x{args.scale} | {n_params/1e6:.3f}M params ({weights_mb:.1f} MB) | batch {args.batch_size} | "
          f"patch LR {hr_patch//args.scale}x{hr_patch//args.scale} | iters {args.iters} | lr {args.lr} | "
          f"milestones {milestones} | workers {args.num_workers}")
    print(f"pre-training: {mem_line()}")
    print("-" * 100)

    torch.cuda.reset_peak_memory_stats()
    it, running, nb, t0 = 0, 0.0, 0, time.time()
    speeds = []
    try:
        while it < args.iters:
            for lr_img, hr_img in loader:
                it += 1
                lr_img, hr_img = lr_img.to('cuda', non_blocking=True), hr_img.to('cuda', non_blocking=True)
                opt.zero_grad(set_to_none=True)
                sr = model(lr_img)
                loss = crit(sr, hr_img)
                loss.backward()
                opt.step()
                lr_sched.step()
                running += loss.item(); nb += 1
                if it % args.print_freq == 0 or it == args.iters:
                    torch.cuda.synchronize()
                    sp = nb / (time.time() - t0)
                    # Skip the first window: it includes warm-up and loader start-up.
                    if it > args.print_freq:
                        speeds.append(sp)
                    print(f"iter {it:>7d}/{args.iters} | loss {running/nb:.4f} | {sp:5.2f} it/s | {mem_line()}",
                          flush=True)
                    running, nb, t0 = 0.0, 0, time.time()
                if it >= args.iters:
                    break
    except torch.OutOfMemoryError as e:
        print(f"\nOOM a iter {it} con batch {args.batch_size}: {str(e).splitlines()[0]}")
        print(f"picco prima dell'OOM: {mem_line()}")
        sys.exit(2)

    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated() / MB
    peak_res = torch.cuda.max_memory_reserved() / MB
    it_s = statistics.mean(speeds) if speeds else float('nan')
    # `it_s == it_s` is False only for NaN (no speed samples), which keeps the ETA as NaN.
    print("-" * 100)
    print(f"[{label}] x{args.scale} batch {args.batch_size}: picco allocato {peak_alloc:.0f} MB | picco riservato "
          f"{peak_res:.0f} MB (~ valore nvtop/nvidia-smi meno il contesto CUDA) | pesi {weights_mb:.1f} MB | "
          f"{it_s:.2f} it/s -> {args.iters/it_s/3600 if it_s == it_s else float('nan'):.2f} h per {args.iters} iter, "
          f"{500000/it_s/3600 if it_s == it_s else float('nan'):.1f} h per 500K")

    if args.ckpt_dir:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        torch.save({'model': model.state_dict(), 'iter': it, 'label': label, 'scale': args.scale,
                    'args': vars(args)}, os.path.join(args.ckpt_dir, 'final.pth'))
        print(f"salvato {os.path.join(args.ckpt_dir, 'final.pth')}")
    if args.csv:
        upsert_csv(args.csv, [dict(model=label, scale=args.scale, batch_size=args.batch_size,
                                   lr_patch=hr_patch // args.scale, iters=it, params=n_params,
                                   peak_alloc_mb=f"{peak_alloc:.0f}", peak_reserved_mb=f"{peak_res:.0f}",
                                   weights_mb=f"{weights_mb:.1f}", it_per_s=f"{it_s:.2f}",
                                   gpu=env['gpu'], commit=env['commit'], date=env['date'])],
                   FIELDS, ['model', 'scale', 'batch_size'])
        print(f"CSV aggiornato: {args.csv}")


if __name__ == '__main__':
    main()
