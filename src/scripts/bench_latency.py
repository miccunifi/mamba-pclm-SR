"""
bench_latency.py — Latency, peak memory and per-module profile on fixed inputs, for our
models (rho in {0,.25,.5,.75,1}) and for the baselines (same adapters as
count_params_macs.py). Updates results/efficiency.csv (latency_*/mem_* columns) and writes
results/profile.csv (time per module category).

Protocol (reported in full in README_experiments.md)
----------------------------------------------------
* Strict FP32 (TF32 disabled for matmul and cuDNN), batch 1, torch.no_grad(), forward only.
* Fixed LR inputs: 640x360 (x2), 427x240 (x3), 320x180 (x4) -> output 1280x720; random
  tensor; random weights (latency does not depend on the weights).
* cudnn.benchmark set to the SAME value for all models (default False; --cudnn_benchmark).
* 10 warm-up passes, then >= 30 timed runs (default 50) with torch.cuda.Event
  and torch.cuda.synchronize() at every run; mean and median are reported in ms.
* GPU idle: nvidia-smi is checked before the measurement; if other processes use the GPU
  a WARNING is printed (the measurement continues).
* Peak memory: torch.cuda.reset_peak_memory_stats() before the forward and
  torch.cuda.max_memory_allocated() after -> peak_mem_mb; weights_mem_mb = bytes of
  parameters+buffers; act_mem_mb = peak - weights (includes input/output and temporaries).
* Per-module profile (our models only): torch.profiler (CPU+CUDA) over
  --profile_iters forwards, with record_function inserted via hooks on the modules; CUDA time
  is aggregated per category: scan (selective scan kernel), vssm_proj (VSSM minus the scan:
  in_proj/x_proj/dt_proj/out_proj/DWConv/LN), local (convolutional branch + LN), fusion
  (1x1 conv), eca, le (Local Enhancement + LN), rlmg_conv (group 3x3 conv),
  shallow_head (conv_first + conv_after_body + upsampler), other (residuals, split/cat, mean).
  The profiler adds overhead: use the fractions; the latency is the one from CUDA events.
* Every run logs GPU, driver, torch/CUDA/cuDNN/mamba_ssm versions, commit and date
  (also in the CSVs).

Usage
-----
  python scripts/bench_latency.py                                # our models, x2/x3/x4
  python scripts/bench_latency.py --baselines all                # + baselines
  python scripts/bench_latency.py --scales 4 --rhos 0 1 --iters 100 --profile_iters 20
"""
import argparse
import os
import statistics
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RHOS, SCALES, FIXED_LR, RESULTS_DIR, EFFICIENCY_FIELDS, EFFICIENCY_KEY,  # noqa: E402
                    PROFILE_FIELDS, PROFILE_KEY, build_ours, count_params, env_info, env_banner,
                    upsert_csv, gpu_other_processes, ours_method_name, fmt_rho)
import baselines as B  # noqa: E402

CATEGORIES = ['scan', 'vssm_proj', 'local', 'fusion', 'eca', 'le', 'rlmg_conv', 'shallow_head', 'other']


@torch.no_grad()
def time_forward(model, x, warmup, iters):
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        model(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def weights_bytes(model):
    return (sum(p.numel() * p.element_size() for p in model.parameters()) +
            sum(b.numel() * b.element_size() for b in model.buffers()))


@torch.no_grad()
def peak_memory(model, x):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    y = model(x)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del y
    return peak


class _Annotator:
    """Opens/closes a record_function 'cat:<label>' around the forward of the modules."""

    def __init__(self):
        self.handles = []
        self.stacks = defaultdict(list)

    def add(self, module, label):
        if module is None:
            return

        # One stack per module id, so nested or repeated calls close the right scope.
        def pre(m, inp):
            rf = torch.autograd.profiler.record_function(f'cat:{label}')
            rf.__enter__()
            self.stacks[id(m)].append(rf)

        def post(m, inp, out):
            self.stacks[id(m)].pop().__exit__(None, None, None)

        self.handles.append(module.register_forward_pre_hook(pre))
        self.handles.append(module.register_forward_hook(post))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def annotate_ours(model):
    from pclmsr import PCLMB
    ann = _Annotator()
    ann.add(model, 'model')
    ann.add(model.conv_first, 'shallow_head')
    ann.add(model.conv_after_body, 'shallow_head')
    ann.add(model.upsample, 'shallow_head')
    for g in model.groups:
        ann.add(g.conv, 'rlmg_conv')
        if getattr(g, 'tfg', None) is not None:
            ann.add(g.tfg, 'rfr')
    # ln_m and litevssm share the 'vssm' label; the scan part is split out later by _ScanPatch.
    for blk in model.modules():
        if not isinstance(blk, PCLMB):
            continue
        ann.add(getattr(blk, 'ln_m', None), 'vssm')
        ann.add(getattr(blk, 'litevssm', None), 'vssm')
        ann.add(getattr(blk, 'ln_l', None), 'local')
        ann.add(getattr(blk, 'local', None), 'local')
        ann.add(blk.fuse, 'fusion')
        ann.add(blk.eca, 'eca')
        ann.add(blk.ln_le, 'le')
        ann.add(blk.le, 'le')
    return ann


class _ScanPatch:
    """Wraps mambair.selective_scan (used by SS2D.scan) in record_function('cat:scan')."""

    def __enter__(self):
        import mambair
        self.mod, self.orig = mambair, mambair.selective_scan

        def wrapped(*a, **k):
            with torch.autograd.profiler.record_function('cat:scan'):
                return self.orig(*a, **k)
        mambair.selective_scan = wrapped
        return self

    def __exit__(self, *exc):
        self.mod.selective_scan = self.orig


def _device_time_us(evt):
    # Newer torch versions renamed 'cuda_time_total' to 'device_time_total'.
    for attr in ('device_time_total', 'cuda_time_total'):
        v = getattr(evt, attr, None)
        if v is not None:
            return float(v)
    return 0.0


@torch.no_grad()
def profile_ours(model, x, iters):
    """Returns dict category -> ms per forward (aggregated CUDA time)."""
    from torch.profiler import profile, ProfilerActivity
    ann = annotate_ours(model)
    try:
        with _ScanPatch():
            # One warm-up forward, then profile `iters` forwards.
            model(x)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(iters):
                    model(x)
                torch.cuda.synchronize()
    finally:
        ann.remove()
    raw = defaultdict(float)
    for evt in prof.key_averages():
        if evt.key.startswith('cat:'):
            raw[evt.key[4:]] += _device_time_us(evt) / 1000.0 / iters
    # vssm_proj = vssm minus scan; other = model total minus every known category (incl. rfr).
    out = {}
    out['scan'] = raw.get('scan', 0.0)
    out['vssm_proj'] = max(raw.get('vssm', 0.0) - out['scan'], 0.0)
    for c in ('local', 'fusion', 'eca', 'le', 'rlmg_conv', 'shallow_head'):
        out[c] = raw.get(c, 0.0)
    known = sum(out.values()) + raw.get('rfr', 0.0)
    out['other'] = max(raw.get('model', 0.0) - known, 0.0)
    if raw.get('rfr', 0.0):
        out['rfr'] = raw['rfr']
    out['_model_total'] = raw.get('model', 0.0)
    return out


@torch.no_grad()
def profile_total(model, x, iters):
    from torch.profiler import profile, ProfilerActivity
    ann = _Annotator()
    ann.add(model, 'model')
    try:
        model(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                model(x)
            torch.cuda.synchronize()
    finally:
        ann.remove()
    tot = 0.0
    for evt in prof.key_averages():
        if evt.key == 'cat:model':
            tot += _device_time_us(evt) / 1000.0 / iters
    return {'_model_total': tot}


def measure(label, model, scale, args, env, rho=None, do_profile=True):
    w, h = FIXED_LR[scale]
    x = torch.randn(1, 3, h, w, device='cuda')
    times = time_forward(model, x, args.warmup, args.iters)
    mean, median = statistics.mean(times), statistics.median(times)
    sd = statistics.stdev(times) if len(times) > 1 else 0.0
    wb = weights_bytes(model)
    peak = peak_memory(model, x)
    row = dict(method=label, rho=fmt_rho(rho) if rho is not None else '', scale=scale,
               latency_ms_mean=f"{mean:.3f}", latency_ms_median=f"{median:.3f}",
               peak_mem_mb=f"{peak/2**20:.1f}", weights_mem_mb=f"{wb/2**20:.2f}",
               act_mem_mb=f"{(peak-wb)/2**20:.1f}", gpu=env['gpu'], commit=env['commit'],
               date=env['date'])
    print(f"{label:<18} x{scale} | {mean:8.2f} ms (median {median:8.2f}, sd {sd:5.2f}, n={len(times)}) "
          f"| peak {peak/2**20:8.1f} MB (weights {wb/2**20:6.2f}, act {(peak-wb)/2**20:8.1f})", flush=True)
    prof_rows = []
    # Baselines get only the 'total' profile row; our models also get the per-category rows.
    if do_profile and args.profile_iters > 0:
        prof = profile_ours(model, x, args.profile_iters) if rho is not None \
            else profile_total(model, x, args.profile_iters)
        tot = prof.pop('_model_total')
        if rho is not None:
            parts = "  ".join(f"{c} {prof[c]:.2f}" for c in prof)
            print(f"{'':<18}    profilo (ms/forward, CUDA, sotto profiler; tot {tot:.2f}): {parts}")
            for c, ms in prof.items():
                prof_rows.append(dict(method=label, rho=fmt_rho(rho), scale=scale, category=c,
                                      ms=f"{ms:.4f}", frac=f"{(ms/tot if tot else 0):.4f}",
                                      gpu=env['gpu'], commit=env['commit'], date=env['date']))
        prof_rows.append(dict(method=label, rho=fmt_rho(rho) if rho is not None else '',
                              scale=scale, category='total', ms=f"{tot:.4f}", frac='1.0000',
                              gpu=env['gpu'], commit=env['commit'], date=env['date']))
    return row, prof_rows


def main():
    p = argparse.ArgumentParser(description="Latenza / memoria / profilo per-modulo")
    p.add_argument('--scales', type=int, nargs='+', default=SCALES)
    p.add_argument('--rhos', type=float, nargs='+', default=RHOS)
    p.add_argument('--baselines', nargs='*', default=[], help="'all' oppure nomi; default nessuna")
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--iters', type=int, default=50, help="run cronometrate (>= 30)")
    p.add_argument('--profile_iters', type=int, default=10, help="0 = nessun profilo per-modulo")
    p.add_argument('--cudnn_benchmark', action='store_true',
                   help="cudnn.benchmark=True per TUTTI i modelli (default False per tutti)")
    p.add_argument('--out', default=os.path.join(RESULTS_DIR, 'efficiency.csv'))
    p.add_argument('--profile_out', default=os.path.join(RESULTS_DIR, 'profile.csv'))
    p.add_argument('--log', default=None)
    args = p.parse_args()
    if args.iters < 30:
        raise SystemExit("--iters deve essere >= 30 (protocollo)")
    if not torch.cuda.is_available():
        raise SystemExit("serve una GPU")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    torch.backends.cudnn.deterministic = False

    env = env_info()
    lines = [env_banner(env),
             f"protocollo: FP32 (TF32 off) | batch 1 | no_grad | forward only | warmup {args.warmup} | "
             f"timed {args.iters} (CUDA events + synchronize) | cudnn.benchmark={torch.backends.cudnn.benchmark} "
             f"| profile_iters {args.profile_iters} | input fissi {FIXED_LR}"]
    others = gpu_other_processes()
    if others:
        lines.append(f"WARNING: la GPU NON e' idle, altri processi attivi: {others}")
    else:
        lines.append("GPU idle (nvidia-smi: nessun altro processo compute)")
    print("\n".join(lines))

    rows, prof_rows = [], []

    # Write each row to the CSVs right away, so partial results survive a crash.
    def _flush(r, pr):
        rows.append(r); prof_rows.extend(pr)
        upsert_csv(args.out, [r], EFFICIENCY_FIELDS, EFFICIENCY_KEY)
        if pr:
            upsert_csv(args.profile_out, pr, PROFILE_FIELDS, PROFILE_KEY)

    for scale in args.scales:
        print(f"\n=== x{scale} | LR {FIXED_LR[scale][0]}x{FIXED_LR[scale][1]} ===", flush=True)
        for rho in args.rhos:
            model, info = build_ours(rho, scale, device='cuda')
            r, pr = measure(ours_method_name(rho), model, scale, args, env, rho=rho)
            _flush(r, pr)
            del model
            torch.cuda.empty_cache()
        for name in B.resolve_names(args.baselines):
            try:
                model, info = B.build_baseline(name, scale, device='cuda')
            except Exception as e:
                print(f"{name:<18} x{scale} | ERRORE: {type(e).__name__}: {e}", flush=True)
                continue
            r, pr = measure(name, model, scale, args, env)
            _flush(r, pr)
            del model
            torch.cuda.empty_cache()

    print(f"\nCSV aggiornato: {args.out} ({len(rows)} righe)")
    if prof_rows:
        print(f"profilo per-modulo: {args.profile_out} ({len(prof_rows)} righe)")
    if args.log:
        with open(args.log, 'a') as f:
            f.write("\n".join(lines) + "\n")


if __name__ == '__main__':
    main()
