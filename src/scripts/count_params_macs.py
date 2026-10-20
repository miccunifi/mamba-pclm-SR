"""
count_params_macs.py — Parameters (per scale) and MACs on fixed inputs, for our models
(rho in {0,.25,.5,.75,1}) and for the baselines, WITH THE SAME COUNTER.

Protocol
--------
* Fixed LR inputs: 640x360 (x2), 427x240 (x3), 320x180 (x4)  ->  output 1280x720.
* Counter: fvcore.nn.FlopCountAnalysis (1 fvcore "flop" = 1 multiply-accumulate =
  1 MAC: conv, linear, matmul/bmm/einsum, ...; element-wise ops, activations,
  normalizations and softmax are NOT counted, the standard convention in the SR literature).
* Selective scan: fvcore/ptflops/thop do NOT see the mamba_ssm CUDA kernel (it is a
  torch.autograd.Function, which appears in the traced graph as prim::PythonOp.<Name>).
  We register a handle for `prim::PythonOp.SelectiveScanFn` with the VMamba formula
  (`flops_selective_scan_fn`, analysis/flops.py in the VMamba repo):
        MACs_scan = 9 * B * L * D * N   (+ B * D * L for the D term)
  applied to EVERY kernel call (in our SS2D: 4 directions = 4 calls with
  D = d_inner; in the official MambaIR: 1 call with D = 4*d_inner -> same total).
  Similar handles: `CausalConv1dFn` (DVMSR, slow Mamba path: D*L*K) and `MambaInnerFn`
  (fused path; for DVMSR it is DISABLED -> use_fast_path=False -> separate ops).
* The count runs on GPU: only then is the CUDA kernel executed (with the PyTorch
  fallback of the scan the einsum would be counted, not the formula).

Acceptance criterion (checked on every run): at rho=0 macs_scan == 0; for
rho>0 macs_scan > 0 and strictly increasing in rho.

Usage
-----
  python scripts/count_params_macs.py                            # our models, x2/x3/x4
  python scripts/count_params_macs.py --baselines all            # + all baselines
  python scripts/count_params_macs.py --scales 4 --baselines SwinIR-light MambaIR-light
  python scripts/count_params_macs.py --out results/efficiency.csv --md results/params_macs.md
"""
import argparse
import os
import sys
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RHOS, SCALES, FIXED_LR, RESULTS_DIR, EFFICIENCY_FIELDS, EFFICIENCY_KEY,  # noqa: E402
                    build_ours, params_by_component, count_params, env_info, env_banner,
                    upsert_csv, ours_method_name, fmt_rho)
import baselines as B  # noqa: E402

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    raise SystemExit("fvcore non installato: pip install fvcore")


def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False,
                            with_Group=True, with_complex=False):
    """Copy of the VMamba function (MzeroMiko/VMamba, analysis/flops.py):
    u: (B, D, L), delta: (B, D, L), A: (D, N), B/C: (B, N, L), D: (D,), z: (B, D, L).
    Returns the number of MACs of the selective scan (without the projections)."""
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def _sizes(v):
    return list(v.type().sizes())


def selective_scan_flop_jit(inputs, outputs):
    """prim::PythonOp.SelectiveScanFn(u, delta, A, B, C, D, [z], [delta_bias], ...)."""
    Bsz, D, L = _sizes(inputs[0])
    N = _sizes(inputs[2])[1]
    with_D = len(inputs) > 5
    with_Z = len(inputs) > 6
    return flops_selective_scan_fn(B=Bsz, L=L, D=D, N=N, with_D=with_D, with_Z=with_Z)


def causal_conv1d_flop_jit(inputs, outputs):
    """prim::PythonOp.CausalConv1dFn(x (B,D,L), weight (D,K), ...) -> B*D*L*K MACs."""
    Bsz, D, L = _sizes(inputs[0])
    K = _sizes(inputs[1])[-1]
    return Bsz * D * L * K


def mamba_inner_flop_jit(inputs, outputs):
    """prim::PythonOp.MambaInnerFn (fused path of mamba_ssm.Mamba): causal conv1d +
    x_proj + dt_proj + scan + out_proj. Used only if a model stays on the fast path."""
    xz = _sizes(inputs[0])                      # (B, 2*d_inner, L)
    Bsz, two_di, L = xz
    di = two_di // 2
    K = _sizes(inputs[1])[-1]                   # conv1d_weight (d_inner, 1, K)
    xproj = _sizes(inputs[3])                   # (dt_rank + 2N, d_inner)
    dtproj = _sizes(inputs[4])                  # (d_inner, dt_rank)
    outproj = _sizes(inputs[5])                 # (d_model, d_inner)
    N = _sizes(inputs[7])[1]                    # A (d_inner, N)
    macs = Bsz * di * L * K
    macs += Bsz * L * di * xproj[0]
    macs += Bsz * L * dtproj[1] * di
    macs += flops_selective_scan_fn(B=Bsz, L=L, D=di, N=N, with_D=True, with_Z=True)
    macs += Bsz * L * di * outproj[0]
    return macs


SCAN_OP = 'prim::PythonOp.SelectiveScanFn'
HANDLES = {
    SCAN_OP: selective_scan_flop_jit,
    'prim::PythonOp.CausalConv1dFn': causal_conv1d_flop_jit,
    'prim::PythonOp.MambaInnerFn': mamba_inner_flop_jit,
}


def _disable_mamba_fast_path(model):
    """DVMSR uses mamba_ssm.Mamba with the fused kernel (MambaInnerFn): we disable it to
    count in_proj/conv1d/x_proj/dt_proj/scan/out_proj as separate ops (same total)."""
    n = 0
    for m in model.modules():
        if type(m).__name__ == 'Mamba' and hasattr(m, 'use_fast_path'):
            m.use_fast_path = False
            n += 1
    return n


@torch.no_grad()
def count_macs(model, h_lr, w_lr, device='cuda'):
    """Returns (macs_total, macs_scan, by_operator dict, unsupported dict)."""
    x = torch.randn(1, 3, h_lr, w_lr, device=device)
    fca = FlopCountAnalysis(model, (x,))
    for op, fn in HANDLES.items():
        fca.set_op_handle(op, fn)
    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    fca.tracer_warnings('none')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        total = int(fca.total())
        by_op = dict(fca.by_operator())
        unsupported = dict(fca.unsupported_ops())
    # fvcore may key the op with or without the 'prim::' prefix.
    scan = int(by_op.get(SCAN_OP, 0) or by_op.get(SCAN_OP.split('::', 1)[1], 0))
    return total, scan, by_op, unsupported


def _row(method, rho, scale, params, macs, scan, env):
    return dict(method=method, rho=fmt_rho(rho) if rho is not None else '', scale=scale,
                params=params, macs_g=f"{macs/1e9:.4f}", macs_scan_g=f"{scan/1e9:.4f}",
                gpu=env['gpu'], commit=env['commit'], date=env['date'])


def main():
    p = argparse.ArgumentParser(description="Params e MACs (fvcore + handle scan VMamba)")
    p.add_argument('--scales', type=int, nargs='+', default=SCALES)
    p.add_argument('--rhos', type=float, nargs='+', default=RHOS)
    p.add_argument('--baselines', nargs='*', default=[],
                   help="'all' oppure nomi (SwinIR-light MambaIR-light MambaIRv2-light SAFMN "
                        "DVMSR RLFN CARN IMDN LAPAR-A LatticeNet); default: nessuna")
    p.add_argument('--out', default=os.path.join(RESULTS_DIR, 'efficiency.csv'))
    p.add_argument('--md', default=None, help="tabella markdown opzionale")
    p.add_argument('--device', default='cuda')
    p.add_argument('--show_ops', action='store_true', help="stampa i MACs per operatore")
    args = p.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit("serve una GPU: il kernel CUDA della scan e' quello che viene contato")
    env = env_info()
    print(env_banner(env))
    print(f"input fissi (W x H): {FIXED_LR}  |  fvcore {__import__('fvcore').__version__}")

    rows, md = [], []
    hdr = (f"{'model':<18} {'scale':>5} {'rho':>5} {'params':>10} {'MACs(G)':>9} "
           f"{'scan(G)':>9} {'scan%':>6}")
    md.append("| Model | scale | rho | Params | MACs (G) | scan MACs (G) | scan % |")
    md.append("|---|---|---|---|---|---|---|")

    for scale in args.scales:
        w, h = FIXED_LR[scale]
        print(f"\n=== x{scale} | LR {w}x{h} -> HR {w*scale}x{h*scale} ===")
        print(hdr)
        scans = []
        for rho in args.rhos:
            model, info = build_ours(rho, scale, device=args.device)
            pc = params_by_component(model)
            total, scan, by_op, unsup = count_macs(model, h, w, args.device)
            scans.append(scan)
            frac = 100.0 * scan / total if total else 0.0
            print(f"{info['name']:<18} {scale:>5} {rho:>5.2f} {pc['total']:>10,} "
                  f"{total/1e9:>9.3f} {scan/1e9:>9.3f} {frac:>5.1f}%   "
                  f"[mamba {pc['mamba']:,} | local {pc['local']:,} | fusion {pc['fusion']:,} | "
                  f"eca {pc['eca']:,} | le {pc['le']:,} | head {pc['head']:,}]")
            if args.show_ops:
                for op, v in sorted(by_op.items(), key=lambda kv: -kv[1]):
                    print(f"      {op:<45} {v/1e9:9.4f} G")
                if unsup:
                    print(f"      (non contati: {sorted(unsup)})")
            rows.append(_row(ours_method_name(rho), rho, scale, pc['total'], total, scan, env))
            md.append(f"| PCLM rho={rho:.2f} | x{scale} | {rho:.2f} | {pc['total']/1e3:.1f}K | "
                      f"{total/1e9:.3f} | {scan/1e9:.3f} | {frac:.1f}% |")
            del model
            torch.cuda.empty_cache()
        # Acceptance check: scan MACs are 0 at rho=0 and strictly increasing in rho.
        rr = list(args.rhos)
        for r, s in zip(rr, scans):
            if r == 0.0:
                assert s == 0, f"x{scale}: a rho=0 lo scan deve essere 0, trovato {s}"
            else:
                assert s > 0, f"x{scale}: a rho={r} lo scan deve essere > 0"
        order = sorted(range(len(rr)), key=lambda i: rr[i])
        seq = [scans[i] for i in order]
        assert all(a < b for a, b in zip(seq, seq[1:])), f"x{scale}: scan non crescente in rho: {seq}"
        print(f"  accettazione OK: scan=0 a rho=0, crescente in rho: "
              f"{[f'{s/1e9:.3f}' for s in seq]} G")

    # ---- baseline ----
    names = B.resolve_names(args.baselines)
    for scale in args.scales:
        if not names:
            break
        w, h = FIXED_LR[scale]
        print(f"\n=== baseline x{scale} | LR {w}x{h} ===")
        print(hdr)
        for name in names:
            try:
                model, info = B.build_baseline(name, scale, device=args.device)
            except Exception as e:
                print(f"{name:<18} {scale:>5}   ERRORE: {type(e).__name__}: {e}")
                continue
            nfast = _disable_mamba_fast_path(model)
            params = count_params(model)
            total, scan, by_op, unsup = count_macs(model, h, w, args.device)
            frac = 100.0 * scan / total if total else 0.0
            tag = ('' if info['official_scale'] else ' (scala non ufficiale)') + \
                  (' [UNOFFICIAL impl]' if info['unofficial'] else '')
            print(f"{name:<18} {scale:>5} {'-':>5} {params:>10,} {total/1e9:>9.3f} "
                  f"{scan/1e9:>9.3f} {frac:>5.1f}%{tag}")
            if args.show_ops:
                for op, v in sorted(by_op.items(), key=lambda kv: -kv[1]):
                    print(f"      {op:<45} {v/1e9:9.4f} G")
                if unsup:
                    print(f"      (non contati: {sorted(unsup)})")
            rows.append(_row(name, None, scale, params, total, scan, env))
            md.append(f"| {name}{tag} | x{scale} | - | {params/1e3:.1f}K | {total/1e9:.3f} | "
                      f"{scan/1e9:.3f} | {frac:.1f}% |")
            del model
            torch.cuda.empty_cache()

    upsert_csv(args.out, rows, EFFICIENCY_FIELDS, EFFICIENCY_KEY)
    print(f"\nCSV aggiornato: {args.out} ({len(rows)} righe)")
    if args.md:
        os.makedirs(os.path.dirname(os.path.abspath(args.md)) or '.', exist_ok=True)
        with open(args.md, 'w') as f:
            f.write(f"{env_banner(env)}\n\nMACs @ output 1280x720 (fvcore + handle scan VMamba)\n\n")
            f.write("\n".join(md) + "\n")
        print(f"Markdown: {args.md}")


if __name__ == '__main__':
    main()
