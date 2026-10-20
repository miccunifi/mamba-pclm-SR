"""
common.py — Shared utilities for the rho study scripts (scripts/*.py).

  * protocol constants (fixed inputs 1280x720 / scale, rho list, CSV schemas)
  * building our models from the .yml configs (random weights or from a checkpoint)
  * parameter count per component (Mamba / local / fusion / ECA / LE / head)
  * environment info (GPU, driver, versions, commit, date) for CSV provenance
  * upsert into fixed-schema CSVs (update the given columns, keep the others)
"""
import csv
import datetime as _dt
import os
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # src/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESULTS_DIR = os.path.join(ROOT, 'results')
CONFIG_DIR = os.path.join(ROOT, 'configs')
REPOS_DIR = os.path.join(ROOT, 'repos')

RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
SCALES = [2, 3, 4]
FIXED_LR = {2: (640, 360), 3: (427, 240), 4: (320, 180)}
DATASETS = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']

QUALITY_FIELDS = ['method', 'rho', 'seed', 'iters', 'scale', 'dataset', 'psnr_y', 'ssim_y',
                  'checkpoint', 'gpu', 'commit', 'date']
QUALITY_KEY = ['method', 'rho', 'seed', 'iters', 'scale', 'dataset']
EFFICIENCY_FIELDS = ['method', 'rho', 'scale', 'params', 'macs_g', 'macs_scan_g',
                     'latency_ms_mean', 'latency_ms_median', 'peak_mem_mb', 'weights_mem_mb',
                     'act_mem_mb', 'gpu', 'commit', 'date']
EFFICIENCY_KEY = ['method', 'rho', 'scale']
PROFILE_FIELDS = ['method', 'rho', 'scale', 'category', 'ms', 'frac', 'gpu', 'commit', 'date']
PROFILE_KEY = ['method', 'rho', 'scale', 'category']


def rho_tag(rho):
    return f"{int(round(float(rho) * 100)):03d}"


def config_name(rho, scale):
    return f"rho_{rho_tag(rho)}_x{scale}"


def ours_method_name(rho):
    """'method' name used in the CSVs for our models (one row per rho)."""
    return f"PCLM-rho{rho_tag(rho)}"


def fmt_rho(rho):
    return f"{float(rho):.2f}"


def load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def build_ours(rho=None, scale=None, config_path=None, device='cuda'):
    """Build the study's PCLM-SR model (random weights) from the .yml config.
    Return (model, info) with info = dict(name, rho, scale, config_path, arch_cfg)."""
    from models import build_model
    if config_path is None:
        config_path = os.path.join(CONFIG_DIR, config_name(rho, scale) + '.yml')
    yml = load_yaml(config_path)
    model, arch_cfg = build_model(yml['arch'], yml['variant'], **dict(yml['network_g']))
    info = dict(name=yml['name'], rho=float(yml['network_g']['mamba_ratio']),
                scale=int(yml['scale']), config_path=config_path, arch_cfg=arch_cfg)
    return model.to(device).eval(), info


def load_ours_checkpoint(ckpt_path, device='cuda'):
    """Rebuild the model from the arch_cfg saved in the checkpoint and load the weights.
    Return (model, ck) where ck is the checkpoint dict (iter, seed, run_meta, ...)."""
    from models import model_from_cfg
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not (isinstance(ck, dict) and ck.get('arch') and ck.get('arch_cfg')):
        raise ValueError(f"{ckpt_path}: checkpoint senza arch/arch_cfg (non salvato da train.py)")
    model = model_from_cfg(ck['arch'], ck['arch_cfg'])
    model.load_state_dict(ck['model'])
    return model.to(device).eval(), ck


COMPONENTS = ('mamba', 'local', 'fusion', 'eca', 'le', 'rlmg_conv', 'shallow_body', 'head', 'rfr')


def component_of(param_name):
    """Assign a parameter (by name) to a PCLM-SR component."""
    n = param_name
    parts = n.split('.')
    # Order matters: block-specific patterns come before the generic groups.*.conv rule.
    if '.litevssm.' in n or '.ln_m.' in n or n.endswith('.alpha_m'):
        return 'mamba'
    if '.local.' in n or '.ln_l.' in n or n.endswith('.alpha_l'):
        return 'local'
    if '.fuse.' in n or n.endswith('.alpha_f'):
        return 'fusion'
    if '.eca.' in n:
        return 'eca'
    if '.le.' in n or '.ln_le.' in n or n.endswith('.beta'):
        return 'le'
    if '.tfg.' in n:
        return 'rfr'
    if n.startswith('groups.') and len(parts) >= 3 and parts[2] == 'conv':
        return 'rlmg_conv'
    if n.startswith('upsample.'):
        return 'head'
    if n.startswith('conv_first.') or n.startswith('conv_after_body.'):
        return 'shallow_body'
    raise KeyError(f"parametro non classificato: {n}")


def params_by_component(model):
    out = {c: 0 for c in COMPONENTS}
    for name, p in model.named_parameters():
        out[component_of(name)] += p.numel()
    out['total'] = sum(p.numel() for p in model.parameters())
    assert sum(out[c] for c in COMPONENTS) == out['total']
    return out


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def _run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ''


def git_commit(path=ROOT):
    c = _run(['git', '-C', path, 'rev-parse', '--short=12', 'HEAD'])
    if not c:
        return 'nogit'
    dirty = _run(['git', '-C', path, 'status', '--porcelain'])
    return c + ('-dirty' if dirty else '')


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'


def driver_version():
    return _run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']).split('\n')[0]


def gpu_other_processes():
    """Processes using the GPU according to nvidia-smi (for the 'GPU idle' check)."""
    out = _run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                '--format=csv,noheader'])
    procs = []
    me = os.getpid()
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid = line.split(',')[0].strip()
        if pid.isdigit() and int(pid) != me:
            procs.append(line)
    return procs


def env_info():
    try:
        import mamba_ssm
        mamba_v = getattr(mamba_ssm, '__version__', '?')
    except Exception:
        mamba_v = 'absent'
    try:
        import causal_conv1d
        cc_v = getattr(causal_conv1d, '__version__', '?')
    except Exception:
        cc_v = 'absent'
    return dict(
        gpu=gpu_name(), driver=driver_version(),
        torch=torch.__version__, cuda=torch.version.cuda,
        cudnn=str(torch.backends.cudnn.version()) if torch.cuda.is_available() else '',
        mamba_ssm=mamba_v, causal_conv1d=cc_v,
        python=sys.version.split()[0],
        commit=git_commit(), date=_dt.datetime.now().isoformat(timespec='seconds'),
        hostname=_run(['hostname']),
    )


def env_banner(info=None):
    info = info or env_info()
    return (f"GPU: {info['gpu']} | driver {info['driver']} | torch {info['torch']} | "
            f"CUDA {info['cuda']} | cuDNN {info['cudnn']} | mamba_ssm {info['mamba_ssm']} | "
            f"causal_conv1d {info['causal_conv1d']} | python {info['python']} | "
            f"commit {info['commit']} | {info['date']}")


def read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def upsert_csv(path, rows, fieldnames, key):
    """Insert/update `rows` in `path`. The key is the tuple of the `key` fields;
    for rows already present, ONLY the non-empty fields of `rows` are overwritten
    (so count_params_macs and bench_latency can fill in the same row)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    existing = read_csv(path)
    index = {tuple(str(r.get(k, '')) for k in key): r for r in existing}
    for row in rows:
        row = {k: ('' if row.get(k) is None else str(row.get(k))) for k in fieldnames}
        kk = tuple(row[k] for k in key)
        if kk in index:
            for k, v in row.items():
                if v != '':
                    index[kk][k] = v
        else:
            existing.append(row)
            index[kk] = row
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in existing:
            w.writerow({k: r.get(k, '') for k in fieldnames})
    return path


BENCHMARK_DIR = os.path.normpath(os.path.join(ROOT, '..', 'datasets', 'SRBenchmarks', 'benchmark'))


def benchmark_dirs(benchmark_dir, dataset, scale):
    base = os.path.join(benchmark_dir, dataset)
    # Prefer a scale-specific HR folder (HR/X{scale}) when present, else the plain HR folder.
    hr = next((d for d in (os.path.join(base, 'HR', f'X{scale}'), os.path.join(base, 'HR'))
               if os.path.isdir(d)), os.path.join(base, 'HR'))
    return hr, os.path.join(base, 'LR_bicubic', f'X{scale}')


def benchmark_pairs(benchmark_dir, dataset, scale):
    """Sorted list of (hr_path, lr_path): for each HR '<stem>.png' the LR is '<stem>x<scale>.png'
    (filename_tmpl '{}x{scale}', as in train.py/test.py). The LR folders also contain copies
    with other names (e.g. 'baby.png', 'img_001.png'): they are IGNORED, as in the rest of
    the code, so each image is counted once and with the same LR used so far."""
    from dataset import PairedFolderDataset
    hr_dir, lr_dir = benchmark_dirs(benchmark_dir, dataset, scale)
    ds = PairedFolderDataset(hr_dir, lr_dir, scale=scale, hr_patch_size=scale,
                             filename_tmpl=f'{{}}x{scale}', train_mode=False, recursive=True)
    return list(ds.pairs)


@torch.no_grad()
def sr_forward(model, lr, scale, tile=0, tile_overlap=32):
    if tile <= 0:
        return model(lr).clamp(0, 1)
    b, c, h, w = lr.shape
    # Tiled inference: overlapping tiles are summed and then averaged via the weight map.
    stride = max(tile - tile_overlap, 1)
    h_idx = list(range(0, max(h - tile, 0) + 1, stride)) or [0]
    w_idx = list(range(0, max(w - tile, 0) + 1, stride)) or [0]
    if h_idx[-1] != h - tile and h > tile:
        h_idx.append(h - tile)
    if w_idx[-1] != w - tile and w > tile:
        w_idx.append(w - tile)
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
