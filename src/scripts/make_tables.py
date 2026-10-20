"""
make_tables.py — LaTeX tables (booktabs) and figures from the CSV files in results/. No number is
typed by hand: everything is read from quality.csv, efficiency.csv and profile.csv.

Output (results/tables/ and results/figures/)
---------------------------------------------
  tab_rho_x{2,3,4}.tex        PSNR/SSIM (Y) per rho and dataset, mean +- std over seeds, fixed iters
  tab_rho_paired_x{s}.tex     paired differences between adjacent rhos (same seed): mean +- std
  tab_joint_x{s}.tex          our models + baselines: params, MACs, latency, memory, PSNR/SSIM
  tab_efficiency.tex          params / MACs / scan / latency / memory per rho and scale
  fig_psnr_vs_rho_x{s}.pdf    PSNR vs rho with error bars (std over seeds), one panel per dataset
  fig_psnr_vs_params_x{s}.pdf PSNR (mean of datasets or --dataset) vs parameters: ours + baselines
  fig_latency_memory_vs_rho.pdf  latency and peak memory vs rho, one curve per scale
  fig_profile_x{s}.pdf        per-module profile (stacked bars) per rho
  summary.md                  text summary with provenance (gpu/commit/date of the CSV files)

Usage
-----
  python scripts/make_tables.py                     # iters=100000 (ablation, 3 seeds)
  python scripts/make_tables.py --iters 500000      # full run (seed 0)
  python scripts/make_tables.py --dataset Urban100  # PSNR-vs-params figure on one dataset
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS_DIR, DATASETS, RHOS, SCALES  # noqa: E402

CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
SEQ_RHO = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
OTHER = '#c3c2b7'
PROFILE_CATS = ['scan', 'vssm_proj', 'local', 'fusion', 'eca', 'le', 'rlmg_conv', 'shallow_head', 'other']
PROFILE_LABELS = {'scan': 'selective scan', 'vssm_proj': 'VSSM proj/DWConv', 'local': 'local branch',
                  'fusion': '1x1 fusion', 'eca': 'ECA', 'le': 'local enh. + LN',
                  'rlmg_conv': 'RLMG conv3x3', 'shallow_head': 'shallow conv + head', 'other': 'other'}


# ======================================================================================
# IO
# ======================================================================================
def _read(name, required=True):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.isfile(path):
        if required:
            raise SystemExit(f"manca {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str).fillna('')
    # Coerce numeric columns; seed/iters/scale become nullable ints.
    for c in ('rho', 'psnr_y', 'ssim_y', 'params', 'macs_g', 'macs_scan_g', 'latency_ms_mean',
              'latency_ms_median', 'peak_mem_mb', 'weights_mem_mb', 'act_mem_mb', 'ms', 'frac'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in ('seed', 'iters', 'scale'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
    return df


def _tex_escape(s):
    return str(s).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def _ms(mean, std, n, fmt='{:.2f}'):
    if n == 0 or math.isnan(mean):
        return '--'
    if n == 1:
        return fmt.format(mean)
    return (fmt + r' $\pm$ ' + fmt).format(mean, std)


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)
    print(f"scritto {path}")


def _provenance(dfs):
    out = []
    for name, df in dfs.items():
        if df.empty:
            continue
        gpus = sorted(set(df['gpu'])) if 'gpu' in df else []
        commits = sorted(set(df['commit'])) if 'commit' in df else []
        dates = sorted(set(df['date'])) if 'date' in df else []
        out.append(f"- {name}: {len(df)} righe | gpu {gpus} | commit {commits} | "
                   f"date {dates[0] if dates else '?'} .. {dates[-1] if dates else '?'}")
    return "\n".join(out)


def table_rho(q, scale, iters, datasets):
    """Mean +- std over seeds of PSNR/SSIM per rho x dataset (rows: rho, columns: dataset)."""
    d = q[(q['scale'] == scale) & (q['iters'] == iters) & q['rho'].notna()]
    cols = " ".join("cc" for _ in datasets)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{PSNR/SSIM (Y channel) vs.\ $\rho$ at $\times$%d, mean $\pm$ std over seeds "
             r"(%d iterations). Base-light backbone ($C{=}60$, $4\times6$ PCLMB, no RFR).}" % (scale, iters),
             r"\label{tab:rho_x%d}" % scale,
             r"\begin{tabular}{l r %s}" % cols, r"\toprule",
             r"$\rho$ & $C_m$ & " + " & ".join(r"\multicolumn{2}{c}{%s}" % ds for ds in datasets) + r" \\",
             " & & " + " & ".join("PSNR & SSIM" for _ in datasets) + r" \\", r"\midrule"]
    n_seeds_all = set()
    for rho in RHOS:
        dr = d[np.isclose(d['rho'], rho)]
        if dr.empty:
            continue
        cells = []
        for ds in datasets:
            g = dr[dr['dataset'] == ds]
            n = len(g)
            n_seeds_all.add(n)
            cells.append(_ms(g['psnr_y'].mean(), g['psnr_y'].std(ddof=1) if n > 1 else 0.0, n))
            cells.append(_ms(g['ssim_y'].mean(), g['ssim_y'].std(ddof=1) if n > 1 else 0.0, n, '{:.4f}'))
        lines.append(f"{rho:.2f} & {int(math.floor(rho * 60))} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\\[2pt]\footnotesize seeds per cell: %s" % sorted(n_seeds_all), r"\end{table}", ""]
    return "\n".join(lines)


def table_paired(q, scale, iters, datasets):
    """Paired differences PSNR(rho_b, s) - PSNR(rho_a, s) between adjacent rhos, same seed."""
    d = q[(q['scale'] == scale) & (q['iters'] == iters) & q['rho'].notna()]
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Paired PSNR differences (dB) between adjacent $\rho$ values at $\times$%d "
             r"(same seed, %d iterations): mean $\pm$ std over seeds; $n$ = number of seed pairs.}"
             % (scale, iters), r"\label{tab:rho_paired_x%d}" % scale,
             r"\begin{tabular}{l %s}" % " ".join("c" for _ in datasets), r"\toprule",
             r"$\rho_a \rightarrow \rho_b$ & " + " & ".join(datasets) + r" \\", r"\midrule"]
    for ra, rb in zip(RHOS, RHOS[1:]):
        cells = []
        for ds in datasets:
            a = d[np.isclose(d['rho'], ra) & (d['dataset'] == ds)].set_index('seed')['psnr_y']
            b = d[np.isclose(d['rho'], rb) & (d['dataset'] == ds)].set_index('seed')['psnr_y']
            common = a.index.intersection(b.index)
            diff = (b.loc[common] - a.loc[common]).astype(float)
            n = len(diff)
            if n == 0:
                cells.append('--')
            else:
                sd = diff.std(ddof=1) if n > 1 else 0.0
                cells.append(_ms(diff.mean(), sd, n, '{:+.3f}') + f" ($n{{=}}{n}$)")
        lines.append(f"{ra:.2f} $\\rightarrow$ {rb:.2f} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _quality_summary(q, scale, iters, datasets):
    """dict method -> {dataset: (psnr_mean, ssim_mean, n)}; ours (iters) + baselines (no iters)."""
    out = {}
    d = q[q['scale'] == scale]
    for method, g in d.groupby('method'):
        # Our models carry a rho; baselines do not and are not filtered by iters.
        is_ours = g['rho'].notna().all()
        if is_ours:
            g = g[g['iters'] == iters]
        row = {}
        for ds in datasets:
            gg = g[g['dataset'] == ds]
            if len(gg):
                row[ds] = (gg['psnr_y'].mean(), gg['ssim_y'].mean(), len(gg))
        if row:
            out[method] = row
    return out


def table_joint(q, e, scale, iters, datasets):
    d_e = e[e['scale'] == scale].set_index('method')
    qs = _quality_summary(q, scale, iters, datasets)
    methods = list(d_e.index)
    ours = [m for m in methods if not pd.isna(d_e.loc[m, 'rho'])]
    base = [m for m in methods if pd.isna(d_e.loc[m, 'rho'])]
    ours.sort(key=lambda m: d_e.loc[m, 'rho'])
    base.sort()
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
             r"\caption{Joint comparison at $\times$%d. Params and MACs on a %s LR input "
             r"(1280$\times$720 output, fvcore + VMamba scan formula); latency (mean of $\ge$30 runs, "
             r"FP32, batch 1) and peak memory on the GPU listed in the CSV; PSNR/SSIM on the Y channel. "
             r"Our rows: mean over seeds at %d iterations.}" % (scale, {2: '640$\\times$360', 3: '427$\\times$240',
                                                                     4: '320$\\times$180'}[scale], iters),
             r"\label{tab:joint_x%d}" % scale,
             r"\begin{tabular}{l r r r r r r %s}" % " ".join("c" for _ in datasets), r"\toprule",
             r"Method & $\rho$ & Params (K) & MACs (G) & scan (G) & Lat.\ (ms) & Peak mem (MB) & "
             + " & ".join(datasets) + r" \\", r"\midrule"]

    def row(m, label):
        r = d_e.loc[m]
        rho = '--' if pd.isna(r['rho']) else f"{r['rho']:.2f}"
        f = lambda v, fmt='{:.1f}': ('--' if pd.isna(v) else fmt.format(v))
        cells = [_tex_escape(label), rho, f(r['params'] / 1e3), f(r['macs_g']), f(r['macs_scan_g']),
                 f(r['latency_ms_mean'], '{:.2f}'), f(r['peak_mem_mb'], '{:.0f}')]
        for ds in datasets:
            v = qs.get(m, {}).get(ds)
            cells.append('--' if v is None else f"{v[0]:.2f}/{v[1]:.4f}")
        return " & ".join(cells) + r" \\"
    for m in ours:
        lines.append(row(m, f"PCLM-SR ($\\rho$={d_e.loc[m, 'rho']:.2f})"))
    if ours and base:
        lines.append(r"\midrule")
    for m in base:
        lines.append(row(m, m))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def table_efficiency(e):
    d = e[e['rho'].notna()].sort_values(['scale', 'rho'])
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Efficiency vs.\ $\rho$ (1280$\times$720 output). MACs from fvcore with the "
             r"VMamba selective-scan handle; latency mean/median over $\ge$30 timed runs (FP32, batch 1); "
             r"memory = peak allocated, weights, activations (peak $-$ weights).}",
             r"\label{tab:efficiency}", r"\begin{tabular}{c c r r r r r r r r}", r"\toprule",
             r"Scale & $\rho$ & Params (K) & MACs (G) & scan (G) & Lat.\ mean & Lat.\ median & "
             r"Peak (MB) & Weights (MB) & Act.\ (MB) \\", r"\midrule"]
    f = lambda v, fmt='{:.1f}': ('--' if pd.isna(v) else fmt.format(v))
    for _, r in d.iterrows():
        lines.append(f"$\\times${int(r['scale'])} & {r['rho']:.2f} & {f(r['params']/1e3)} & {f(r['macs_g'])} & "
                     f"{f(r['macs_scan_g'])} & {f(r['latency_ms_mean'], '{:.2f}')} & "
                     f"{f(r['latency_ms_median'], '{:.2f}')} & {f(r['peak_mem_mb'], '{:.0f}')} & "
                     f"{f(r['weights_mem_mb'], '{:.2f}')} & {f(r['act_mem_mb'], '{:.0f}')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _style(ax, title=None, xlabel=None, ylabel=None):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc='left')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=INK2)


def fig_psnr_vs_rho(q, scale, iters, datasets, out):
    import matplotlib.pyplot as plt
    d = q[(q['scale'] == scale) & (q['iters'] == iters) & q['rho'].notna()]
    if d.empty:
        print(f"  (fig_psnr_vs_rho x{scale}: nessun dato)")
        return
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, ds, col in zip(axes, datasets, CAT):
        g = d[d['dataset'] == ds].groupby('rho')['psnr_y'].agg(['mean', 'std', 'count']).reset_index()
        g['std'] = g['std'].fillna(0.0)
        ax.errorbar(g['rho'], g['mean'], yerr=g['std'], color=col, linewidth=2, marker='o',
                    markersize=5, capsize=3, markeredgecolor='white', markeredgewidth=1)
        _style(ax, title=ds, xlabel=r'$\rho$ (Mamba channel fraction)', ylabel='PSNR (dB)')
        ax.set_xticks(RHOS)
    fig.suptitle(f'PSNR vs ρ at ×{scale} ({iters} it., error bars = std over seeds)', fontsize=9,
                 color=INK, x=0.01, ha='left')
    fig.savefig(out + '.pdf'); fig.savefig(out + '.png', dpi=200)
    plt.close(fig)
    print(f"scritto {out}.pdf/.png")


def fig_psnr_vs_params(q, e, scale, iters, datasets, dataset, out):
    import matplotlib.pyplot as plt
    qs = _quality_summary(q, scale, iters, datasets)
    d_e = e[e['scale'] == scale].set_index('method')
    pts = []
    for m in d_e.index:
        vals = qs.get(m, {})
        # Without --dataset, PSNR is the mean over all datasets that have data.
        if dataset:
            v = vals.get(dataset)
            psnr = None if v is None else v[0]
        else:
            psnr = np.mean([v[0] for v in vals.values()]) if vals else None
        if psnr is None or pd.isna(d_e.loc[m, 'params']):
            continue
        pts.append((m, d_e.loc[m, 'params'] / 1e3, psnr, d_e.loc[m, 'rho']))
    if not pts:
        print(f"  (fig_psnr_vs_params x{scale}: nessun dato)")
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
    ours = [p for p in pts if not pd.isna(p[3])]
    base = [p for p in pts if pd.isna(p[3])]
    if ours:
        ours.sort(key=lambda p: p[3])
        ax.plot([p[1] for p in ours], [p[2] for p in ours], color=CAT[0], linewidth=2, zorder=2)
        for (m, x, y, rho), col in zip(ours, SEQ_RHO):
            ax.scatter([x], [y], s=46, color=col, edgecolor='white', linewidth=1, zorder=3)
            ax.annotate(f'ρ={rho:.2f}', (x, y), textcoords='offset points', xytext=(5, 4),
                        fontsize=7, color=INK2)
    for (m, x, y, _) in base:
        ax.scatter([x], [y], s=40, marker='s', color=CAT[1], edgecolor='white', linewidth=1, zorder=3)
        ax.annotate(m, (x, y), textcoords='offset points', xytext=(5, -8), fontsize=7, color=INK2)
    _style(ax, xlabel='Parameters (K)', ylabel=f"PSNR (dB){', ' + dataset if dataset else ', mean of datasets'}")
    ax.set_title(f'PSNR vs parameters at ×{scale} (ours: line, ρ sweep; baselines: squares)',
                 fontsize=9, color=INK, loc='left')
    fig.savefig(out + '.pdf'); fig.savefig(out + '.png', dpi=200)
    plt.close(fig)
    print(f"scritto {out}.pdf/.png")


def fig_latency_memory(e, out):
    import matplotlib.pyplot as plt
    d = e[e['rho'].notna()]
    if d['latency_ms_mean'].isna().all():
        print("  (fig_latency_memory_vs_rho: nessuna latenza in efficiency.csv)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), constrained_layout=True)
    for scale, col in zip(SCALES, CAT):
        g = d[d['scale'] == scale].sort_values('rho')
        if g.empty:
            continue
        axes[0].plot(g['rho'], g['latency_ms_mean'], color=col, linewidth=2, marker='o', markersize=5,
                     markeredgecolor='white', label=f'×{scale}')
        axes[1].plot(g['rho'], g['peak_mem_mb'], color=col, linewidth=2, marker='o', markersize=5,
                     markeredgecolor='white', label=f'×{scale}')
    _style(axes[0], title='Latency (ms, mean, FP32, batch 1)', xlabel=r'$\rho$')
    _style(axes[1], title='Peak GPU memory (MB)', xlabel=r'$\rho$')
    for ax in axes:
        ax.set_xticks(RHOS)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.savefig(out + '.pdf'); fig.savefig(out + '.png', dpi=200)
    plt.close(fig)
    print(f"scritto {out}.pdf/.png")


def fig_profile(pr, scale, out):
    import matplotlib.pyplot as plt
    d = pr[(pr['scale'] == scale) & pr['rho'].notna() & pr['category'].isin(PROFILE_CATS)]
    if d.empty:
        print(f"  (fig_profile x{scale}: nessun profilo)")
        return
    # One row per rho, one column per module category (mean ms over profile rows).
    piv = d.pivot_table(index='rho', columns='category', values='ms', aggfunc='mean').reindex(columns=PROFILE_CATS).fillna(0.0)
    fig, ax = plt.subplots(figsize=(6.2, 3.0), constrained_layout=True)
    x = np.arange(len(piv.index))
    bottom = np.zeros(len(piv.index))
    colors = CAT[:8] + [OTHER]
    for cat, col in zip(PROFILE_CATS, colors):
        v = piv[cat].values
        ax.bar(x, v, bottom=bottom, color=col, width=0.6, edgecolor='white', linewidth=1,
               label=PROFILE_LABELS[cat])
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r:.2f}' for r in piv.index])
    _style(ax, title=f'Per-module CUDA time per forward at ×{scale} (under profiler)',
           xlabel=r'$\rho$', ylabel='ms')
    ax.legend(frameon=False, fontsize=7, labelcolor=INK2, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    fig.savefig(out + '.pdf'); fig.savefig(out + '.png', dpi=200)
    plt.close(fig)
    print(f"scritto {out}.pdf/.png")


# ======================================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--iters', type=int, default=100000, help="iterazioni dei checkpoint (nostri modelli)")
    p.add_argument('--datasets', nargs='+', default=DATASETS)
    p.add_argument('--dataset', default=None, help="dataset per la figura PSNR-vs-params (default: media)")
    p.add_argument('--tables_dir', default=os.path.join(RESULTS_DIR, 'tables'))
    p.add_argument('--figures_dir', default=os.path.join(RESULTS_DIR, 'figures'))
    args = p.parse_args()

    import matplotlib
    # Headless backend; pdf.fonttype 42 embeds TrueType fonts so PDF text stays editable.
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'font.family': 'sans-serif', 'text.color': INK, 'axes.labelcolor': INK2,
                                'pdf.fonttype': 42})

    q = _read('quality.csv', required=False)
    e = _read('efficiency.csv', required=True)
    pr = _read('profile.csv', required=False)
    os.makedirs(args.tables_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    _write(os.path.join(args.tables_dir, 'tab_efficiency.tex'), table_efficiency(e))
    fig_latency_memory(e, os.path.join(args.figures_dir, 'fig_latency_memory_vs_rho'))
    for scale in SCALES:
        if not pr.empty:
            fig_profile(pr, scale, os.path.join(args.figures_dir, f'fig_profile_x{scale}'))
        if q.empty or q[q['scale'] == scale].empty:
            print(f"  x{scale}: quality.csv vuoto, salto tabelle di qualita'")
            continue
        _write(os.path.join(args.tables_dir, f'tab_rho_x{scale}.tex'),
               table_rho(q, scale, args.iters, args.datasets))
        _write(os.path.join(args.tables_dir, f'tab_rho_paired_x{scale}.tex'),
               table_paired(q, scale, args.iters, args.datasets))
        _write(os.path.join(args.tables_dir, f'tab_joint_x{scale}.tex'),
               table_joint(q, e, scale, args.iters, args.datasets))
        fig_psnr_vs_rho(q, scale, args.iters, args.datasets,
                        os.path.join(args.figures_dir, f'fig_psnr_vs_rho_x{scale}'))
        fig_psnr_vs_params(q, e, scale, args.iters, args.datasets, args.dataset,
                           os.path.join(args.figures_dir, f'fig_psnr_vs_params_x{scale}'))

    summary = ["# Provenance dei CSV usati", _provenance({'quality.csv': q, 'efficiency.csv': e,
                                                          'profile.csv': pr}),
               f"\niters (nostri modelli): {args.iters} | dataset: {args.datasets}"]
    _write(os.path.join(args.tables_dir, 'summary.md'), "\n".join(summary) + "\n")


if __name__ == '__main__':
    main()
