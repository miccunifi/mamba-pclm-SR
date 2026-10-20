"""
train.py — Training of MambaIR for super-resolution.

Faithful to the authors' .yml files (organized by ITERATIONS, not by epochs):
  - optimizer   : Adam, betas=(0.9, 0.99), weight_decay=0
  - scheduler   : MultiStepLR (milestones in iterations), gamma=0.5
  - loss        : L1 (eq. 8 of the paper) via losses.build_loss
  - val metrics : PSNR/SSIM on the Y channel via metrics.calc_psnr / calc_ssim
  - patch / lr / milestones / total_iter / pretrain : per variant (see TRAIN_CONFIGS)

For scales x3/x4 the paper initializes from the x2 model weights (strict=False).
The classic variant halves lr and iterations (1e-4, 250k); the light variant uses
the full schedule.

Validation
----------
During training, validation runs on patches (center-crop) of the validation set:
it is a fast and stable signal. The "official" metrics on whole benchmark images
(Set5/Set14/B100/Urban100) come from test.py.

Usage
-----
  python train.py --variant light_sr_x2 --datasets div2k --root ./data
  python train.py --variant classic_sr_x4 --datasets both --root ./data \
                  --pretrain ./checkpoints/classic_sr_x2/final.pth
  # resume:
  python train.py --variant light_sr_x2 --resume ./checkpoints/light_sr_x2/latest.pth
"""
import argparse
import json
import math
import os
import random
import time

import torch
from torch.utils.data import DataLoader

from models import build_model, parse_overrides, list_variants, TRAIN_CONFIGS
from dataset import build_sr_dataset, build_paired_dataset
from losses import build_loss, FFTLoss
from metrics import calc_psnr, calc_ssim
from visualize import save_visuals


OPTIM = dict(betas=(0.9, 0.99), weight_decay=0.0)
SCHED_GAMMA = 0.5
PIXEL_OPT = dict(type='L1Loss', loss_weight=1.0, reduction='mean')  # eq. (8)


# ======================================================================================
# Utility
# ======================================================================================
def set_seed(seed, deterministic=False):
    """Seeds python/torch(+CUDA) for repeatable/independent runs (multi-seed ablation
    study). Note: with custom CUDA kernels (mamba-ssm) bit-exact reproducibility is
    NOT guaranteed even with deterministic=True; the goal is to define independent and
    traceable runs (weight init + data order + augmentation)."""
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32))
    except ImportError:
        pass
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id):
    """worker_init_fn for the DataLoaders: propagates the seed to the `random` module
    too (used by dataset.py for crop/flip), which PyTorch does not seed by itself."""
    ws = torch.initial_seed() % (2 ** 32)
    random.seed(ws)
    try:
        import numpy as np
        np.random.seed(ws)
    except ImportError:
        pass


def cycle(loader):
    """Iterates forever over a DataLoader (iteration-based training)."""
    while True:
        for batch in loader:
            yield batch


def _extract_state_dict(ck):
    """Extracts the state_dict from a checkpoint, accepting the most common formats:
    {'model': ...} (this train.py), {'params'/'params_ema': ...} (BasicSR), or
    a 'bare' state_dict."""
    if isinstance(ck, dict):
        for key in ('model', 'params_ema', 'params', 'state_dict'):
            if key in ck:
                return ck[key]
    return ck


def load_pretrain(model, path, device):
    """Loads pre-trained weights in NON-strict mode (for x3/x4 initialized from x2:
    the upsampler has different shapes and is simply re-initialized).
    Tensors present in both but with a DIFFERENT shape are dropped (otherwise
    load_state_dict would raise a size mismatch even with strict=False)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    state = _extract_state_dict(ck)
    model_sd = model.state_dict()

    filtered, shape_mismatch = {}, []
    for k, v in state.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
        elif k in model_sd:
            shape_mismatch.append(k)

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    loaded = len(filtered)
    print(f"pretrain caricato da {path} (strict=False): "
          f"{loaded}/{len(model_sd)} tensori trasferiti")
    if shape_mismatch:
        print(f"  forma incompatibile, reinizializzati ({len(shape_mismatch)}): "
              f"{', '.join(shape_mismatch[:6])}{' ...' if len(shape_mismatch) > 6 else ''}")
    only_missing = [m for m in missing if m not in shape_mismatch]
    if only_missing:
        print(f"  assenti nel checkpoint, reinizializzati: {len(only_missing)}")
    if unexpected:
        print(f"  extra nel checkpoint, ignorati: {len(unexpected)}")
    if loaded == 0:
        print("  ATTENZIONE: nessun peso trasferito (architettura diversa dal checkpoint?)")


@torch.no_grad()
def validate(model, val_loader, device, scale, criterion):
    """Returns (mean_loss, mean_psnr, mean_ssim) on the validation set.
    Loss = same loss as training (L1); PSNR/SSIM = Y channel, `scale` border
    pixels dropped (as in the paper)."""
    model.eval()
    losses, psnrs, ssims = [], [], []
    for lr_img, hr_img in val_loader:
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)
        sr = model(lr_img)
        losses.append(criterion(sr, hr_img).item())
        sr = sr.clamp(0, 1)
        psnrs.append(calc_psnr(sr, hr_img, crop_border=scale))
        ssims.append(calc_ssim(sr, hr_img, crop_border=scale))
    model.train()
    n = max(len(psnrs), 1)
    return sum(losses) / n, sum(psnrs) / n, sum(ssims) / n


def train(model, train_loader, val_loader, cfg, device, ckpt_dir,
          val_freq=5000, save_freq=5000, print_freq=200, resume=None,
          best_metric='psnr', vis_dir=None, vis_freq=5000, vis_samples=4,
          arch=None, arch_cfg=None, fft_weight=0.0, seed=None,
          ckpt_iter_freq=None, run_meta=None):
    """Trains MambaIR for `cfg['total_iter']` iterations.

    best_metric : 'psnr' (save at the MAXIMUM validation PSNR) or
                  'loss' (save at the MINIMUM validation loss).
    vis_dir     : if set, saves grids (images + Fourier) of one training batch
                  and one validation batch every `vis_freq` iterations.

    ckpt_iter_freq : if set, also saves an IMMUTABLE checkpoint
                  iter_<N>.pth every ckpt_iter_freq iterations (rho study:
                  iter_100000.pth of the full run = seed 0 of the ablations).
    run_meta    : optional dict stored in the checkpoint (e.g. the .yml used).

    Saves: best.pth, latest.pth (every save_freq, to resume after an
    interruption), final.pth and history.json (loss/psnr/ssim curves)."""
    assert best_metric in ('psnr', 'loss'), "best_metric: 'psnr' o 'loss'"
    model.to(device).train()
    scale = cfg['scale']

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], **OPTIM)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg['milestones'], gamma=cfg.get('gamma', SCHED_GAMMA))
    criterion = build_loss(PIXEL_OPT)
    if fft_weight and fft_weight > 0:
        pixel_crit, fft_crit = criterion, FFTLoss(loss_weight=fft_weight)

        def criterion(sr, hr):
            return pixel_crit(sr, hr) + fft_crit(sr, hr)
        print(f"loss: L1 + {fft_weight} * FFT (frequency-loss probe)")

    history = {'iter': [], 'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': []}
    best_value = -math.inf if best_metric == 'psnr' else math.inf
    start_iter = 0

    # Fixed validation batch, reused for every visual dump.
    vis_val_batch = None
    if vis_dir is not None and val_loader is not None:
        vis_val_batch = next(iter(val_loader))

    if resume is not None:
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        start_iter = ck.get('iter', 0)
        best_value = ck.get('best_value', best_value)
        history = ck.get('history', history)
        print(f"ripreso da {resume} (iter {start_iter})")

    os.makedirs(ckpt_dir, exist_ok=True)

    def save_ckpt(tag, it):
        torch.save({
            'iter': it, 'best_value': best_value, 'best_metric': best_metric,
            'history': history, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'config': cfg, 'arch': arch, 'arch_cfg': arch_cfg,
            'seed': seed, 'fft_weight': fft_weight, 'run_meta': run_meta,
        }, os.path.join(ckpt_dir, f"{tag}.pth"))

    @torch.no_grad()
    def dump_visuals(it, train_batch):
        model.eval()
        # training batch
        lr_t, hr_t = train_batch
        sr_t = model(lr_t.to(device)).clamp(0, 1)
        save_visuals(lr_t.to(device), hr_t.to(device), sr_t, vis_dir,
                     f"iter_{it:07d}_train", n=vis_samples)
        if vis_val_batch is not None:
            lr_v, hr_v = vis_val_batch
            sr_v = model(lr_v.to(device)).clamp(0, 1)
            save_visuals(lr_v.to(device), hr_v.to(device), sr_v, vis_dir,
                         f"iter_{it:07d}_val", n=vis_samples)
        model.train()

    total_iter = cfg['total_iter']
    data_iter = cycle(train_loader)
    running, nb = 0.0, 0
    last_train_batch = None
    t0 = time.time()

    for it in range(start_iter + 1, total_iter + 1):
        lr_img, hr_img = next(data_iter)
        last_train_batch = (lr_img, hr_img)
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)

        optimizer.zero_grad()
        sr = model(lr_img)
        loss = criterion(sr, hr_img)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running += loss.item(); nb += 1

        # ---- log ----
        if it % print_freq == 0:
            speed = nb / (time.time() - t0)
            print(f"iter {it:>7d}/{total_iter} | loss {running / nb:.4f} | "
                  f"lr {scheduler.get_last_lr()[0]:.2e} | {speed:.1f} it/s")
            running, nb, t0 = 0.0, 0, time.time()

        if vis_dir is not None and it % vis_freq == 0 and last_train_batch is not None:
            dump_visuals(it, last_train_batch)

        if val_loader is not None and it % val_freq == 0:
            v_loss, v_psnr, v_ssim = validate(model, val_loader, device, scale, criterion)
            history['iter'].append(it)
            history['train_loss'].append(loss.item())
            history['val_loss'].append(v_loss)
            history['val_psnr'].append(v_psnr)
            history['val_ssim'].append(v_ssim)

            monitored = v_psnr if best_metric == 'psnr' else v_loss
            improved = (monitored > best_value) if best_metric == 'psnr' \
                else (monitored < best_value)
            tag = ""
            if improved:
                best_value = monitored
                save_ckpt('best', it)
                tag = f"  *best ({best_metric}={best_value:.4f})"
            print(f"  [val] iter {it} | val_loss {v_loss:.4f} | "
                  f"PSNR {v_psnr:6.2f} dB | SSIM {v_ssim:.4f}{tag}")
            with open(os.path.join(ckpt_dir, 'history.json'), 'w') as f:
                json.dump(history, f, indent=2)

        if it % save_freq == 0:
            save_ckpt('latest', it)
        if ckpt_iter_freq and it % ckpt_iter_freq == 0:
            save_ckpt(f'iter_{it}', it)

    save_ckpt('final', total_iter)
    print(f"training terminato a {total_iter} iter | best {best_metric} {best_value:.4f}")
    return model, history


def load_yaml_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_yaml_config(args, yml):
    """Fills the CLI fields from the .yml (configs/rho_*_x*.yml). Flags passed
    EXPLICITLY on the command line (--seed, --total_iter, --ckpt_dir, --pretrain,
    --resume, --num_workers) take precedence over the file. Returns the training
    `cfg` dict (lr/milestones/total_iter/patch), which in config mode is NEVER
    rescaled: with --total_iter 100000 the milestones stay those of the full run,
    so the first 100K iterations are identical to those of the 500K run."""
    ds, tr, pth, lg = yml['datasets'], yml['train'], yml.get('path', {}), yml.get('logger', {})
    args.arch = yml['arch']
    args.variant = yml['variant']
    args.arch_override = None
    args.yaml_overrides = dict(yml.get('network_g', {}))
    if args.seed is None:
        args.seed = yml.get('manual_seed', 0)
    args.train_hr, args.train_lr = list(ds['train_hr']), list(ds['train_lr'])
    args.val_hr = list(ds['val_hr']) if ds.get('val_hr') else None
    args.val_lr = list(ds['val_lr']) if ds.get('val_lr') else None
    args.filename_tmpl = ds.get('filename_tmpl', '{}x{scale}')
    args.batch_size = int(ds['batch_size'])
    if args.num_workers_cli is None:
        args.num_workers = int(ds.get('num_workers', 8))
    if args.val_freq is None:
        args.val_freq = int(lg.get('val_freq', 5000))
    if args.print_freq is None:
        args.print_freq = int(lg.get('print_freq', 200))
    if args.save_freq is None:
        args.save_freq = int(lg.get('save_latest_freq', 5000))
    args.ckpt_iter_freq = int(lg.get('save_checkpoint_freq', 100000))
    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.join(pth.get('experiments_root', './experiments'),
                                     yml['name'], f"seed{args.seed}")
    if args.pretrain is None and pth.get('pretrain_network_g'):
        args.pretrain = str(pth['pretrain_network_g']).format(seed=args.seed)
    args.deterministic = bool(tr.get('deterministic', False)) or args.deterministic
    args.cudnn_benchmark = bool(tr.get('cudnn_benchmark', False))
    if tr.get('ema_decay', 0):
        raise SystemExit("EMA non e' implementata in questo train.py: usa ema_decay: 0 "
                         "in tutti i config (gestione identica per tutte le run)")
    loss = tr.get('loss')
    if loss and (loss.get('type') != 'L1Loss' or float(loss.get('loss_weight', 1.0)) != 1.0):
        raise SystemExit(f"ricetta fissa: la loss deve restare L1Loss (peso 1.0), non {loss}")

    total_iter = int(args.total_iter) if args.total_iter is not None else int(tr['total_iter'])
    cfg = dict(scale=int(yml['scale']), hr_patch_size=int(ds['hr_patch_size']),
               needs_pretrain=bool(pth.get('pretrain_network_g')),
               lr=float(tr['lr']), total_iter=total_iter,
               milestones=[int(m) for m in tr['milestones']],
               gamma=float(tr.get('gamma', SCHED_GAMMA)))
    return cfg


# ======================================================================================
# Entrypoint
# ======================================================================================
def _emit(lines, logfile=None):
    """Prints to console and, if given, appends the same lines to a log file."""
    text = "\n".join(lines)
    print(text, flush=True)
    if logfile:
        try:
            with open(logfile, 'a') as f:
                f.write(text + "\n")
        except Exception as e:
            print(f"(impossibile scrivere {logfile}: {e})", flush=True)


def _data_config_lines(scale, tmpl, use_folders, args):
    """Summary of the data paths: folder existence + scale consistency check."""
    def mark(d):
        return "OK      " if (d and os.path.isdir(d)) else "MANCANTE"

    def scale_warn(d):
        if d is None or f"X{scale}" in d or f"x{scale}" in os.path.basename(d):
            return ""
        return f"   <-- ATTENZIONE: il path non contiene X{scale}, scala sbagliata?"

    lines = ["=" * 78, f"CONFIG DATI  |  scale x{scale}  |  filename_tmpl='{tmpl}'"]
    if use_folders:
        lines.append("modalita': cartelle locali (no download)")
        lines.append("  TRAIN HR:")
        for d in args.train_hr:
            lines.append(f"    [{mark(d)}] {d}")
        lines.append("  TRAIN LR:")
        for d in args.train_lr:
            lines.append(f"    [{mark(d)}] {d}{scale_warn(d)}")
        lines.append("  VAL HR:")
        for d in (args.val_hr or ["(nessuna)"]):
            lines.append(f"    [{mark(d) if d != '(nessuna)' else '   --   '}] {d}")
        lines.append("  VAL LR:")
        for d in (args.val_lr or ["(nessuna)"]):
            suff = scale_warn(d) if d != "(nessuna)" else ""
            lines.append(f"    [{mark(d) if d != '(nessuna)' else '   --   '}] {d}{suff}")
    else:
        lines.append(f"modalita': torchSR (download={args.download})")
        lines.append(f"  datasets={args.datasets}")
        lines.append(f"  root={args.root}")
    lines.append("=" * 78)
    return lines


def _example_pair(ds):
    """First (hr_path, lr_path) pair, also handling ConcatDataset (multiple folders)."""
    if getattr(ds, 'pairs', None):
        return ds.pairs[0]
    for sub in getattr(ds, 'datasets', []):
        if getattr(sub, 'pairs', None):
            return sub.pairs[0]
    return None


def main():
    p = argparse.ArgumentParser(description="Training MambaIR / PCLM-SR per SR")
    p.add_argument('--config', default=None,
                   help="config .yml (configs/rho_*_x*.yml): arch, network_g, dati, "
                        "schedule, seed. I flag CLI espliciti hanno la precedenza.")
    p.add_argument('--arch', default='mambair', choices=['mambair', 'pclmsr'])
    p.add_argument('--variant', default='light_sr_x2')
    p.add_argument('--arch_override', nargs='*', default=None,
                   help="override architetturali per l'ablazione, es: "
                        "--arch_override mamba_ratio=0.5 use_eca=false d_state=12")
    p.add_argument('--datasets', default='div2k',
                   help="torchSR: 'div2k' | 'flickr2k' | 'both' (=DF2K). "
                        "Ignorato se usi le cartelle esplicite (--train_hr/--train_lr).")
    p.add_argument('--root', default='./data', help="cartella dei dataset (modalita' torchSR)")
    p.add_argument('--download', default='true', choices=['true', 'false'],
                   help="modalita' torchSR: scaricare i dati se mancano (default true)")
    p.add_argument('--train_hr', nargs='+', default=None,
                   help="cartelle HR di training (una o piu', es. DIV2K e Flickr2K)")
    p.add_argument('--train_lr', nargs='+', default=None,
                   help="cartelle LR di training (stessa lunghezza di --train_hr)")
    p.add_argument('--val_hr', nargs='+', default=None, help="cartella/e HR di validation")
    p.add_argument('--val_lr', nargs='+', default=None, help="cartella/e LR di validation")
    p.add_argument('--filename_tmpl', default='{}x{scale}',
                   help="template nome LR; '{scale}' viene sostituito con lo scale "
                        "(es. '{}x{scale}' -> '0001x2'). Usa '{}' se HR e LR hanno lo stesso nome.")
    p.add_argument('--batch_size', type=int, default=32, help="batch del paper per la SR")
    p.add_argument('--num_workers', type=int, default=None, dest='num_workers_cli',
                   help="worker del DataLoader (default 8; con --config: dal .yml)")
    p.add_argument('--ckpt_dir', default=None, help="default: ./checkpoints/<variant>")
    p.add_argument('--pretrain', default=None,
                   help="pesi x2 per inizializzare x3/x4 (strict=False)")
    p.add_argument('--resume', default=None, help="checkpoint da cui riprendere")
    p.add_argument('--val_freq', type=int, default=None, help="default 5000 (con --config: dal .yml)")
    p.add_argument('--save_freq', type=int, default=None, help="default 5000 (con --config: dal .yml)")
    p.add_argument('--print_freq', type=int, default=None, help="default 200 (con --config: dal .yml)")
    p.add_argument('--best_metric', default='psnr', choices=['psnr', 'loss'],
                   help="criterio del best model: PSNR massimo o loss di val minima")
    p.add_argument('--total_iter', type=int, default=None,
                   help="budget di iterazioni; se impostato, le milestone dello "
                        "scheduler vengono riscalate proporzionalmente (utile per "
                        "ablazioni a budget ridotto e uguale per tutte)")
    p.add_argument('--vis_dir', default=None,
                   help="se impostato, salva griglie immagini+Fourier qui")
    p.add_argument('--vis_freq', type=int, default=5000)
    p.add_argument('--vis_samples', type=int, default=4)
    p.add_argument('--lr', type=float, default=None,
                   help="learning rate iniziale (override del valore della variante)")
    p.add_argument('--milestones', type=int, nargs='+', default=None,
                   help="milestone (in iterazioni) del MultiStepLR; se passate, NON vengono "
                        "riscalate da --total_iter")
    p.add_argument('--gamma', type=float, default=None,
                   help=f"fattore di decadimento del MultiStepLR (default {SCHED_GAMMA})")
    p.add_argument('--seed', type=int, default=None,
                   help="seed di run (init pesi + ordine dati + augmentation): "
                        "indispensabile per lo studio multi-seed dell'ablazione")
    p.add_argument('--deterministic', action='store_true',
                   help="cudnn deterministico (piu' lento; i kernel mamba-ssm possono "
                        "restare non deterministici)")
    p.add_argument('--fft_weight', type=float, default=0.0,
                   help="se >0 aggiunge w*FFTLoss alla L1 (probe frequency-loss, "
                        "stile SAFMN: 0.05)")
    p.add_argument('--allow_slow_scan', action='store_true',
                   help="consente il training con il fallback PyTorch della selective scan "
                        "(lentissimo e con memoria enorme): di default si esce se il kernel "
                        "CUDA di mamba_ssm non e' importabile")
    args = p.parse_args()

    # Refuse to train on the slow PyTorch selective-scan fallback unless explicitly allowed.
    import mambair as _mb
    if torch.cuda.is_available() and not _mb._HAS_FAST_SCAN and not args.allow_slow_scan:
        raise SystemExit(
            "kernel CUDA della selective scan (mamba_ssm) NON disponibile: "
            f"{getattr(_mb, '_FAST_SCAN_ERROR', '?')}\n"
            "Reinstalla mamba-ssm/causal-conv1d per QUESTO torch (bash setup_env.sh, oppure: "
            "pip install --no-build-isolation --force-reinstall --no-deps causal-conv1d==<ver> "
            "mamba-ssm==<ver>) e verifica con:\n"
            "  python -c \"from mamba_ssm.ops.selective_scan_interface import selective_scan_fn\"\n"
            "(--allow_slow_scan forza comunque il training con il fallback PyTorch)")
    args.num_workers = args.num_workers_cli if args.num_workers_cli is not None else 8
    args.yaml_overrides = {}
    args.ckpt_iter_freq = None
    args.cudnn_benchmark = None

    yml = None
    if args.config is not None:
        yml = load_yaml_config(args.config)
        cfg = _apply_yaml_config(args, yml)
    else:
        cfg = TRAIN_CONFIGS[args.variant]
        # Variant mode: --total_iter rescales the milestones proportionally.
        if args.total_iter is not None:
            factor = args.total_iter / cfg['total_iter']
            cfg = dict(cfg, total_iter=args.total_iter,
                       milestones=[max(int(m * factor), 1) for m in cfg['milestones']])

    for k, dflt in (('val_freq', 5000), ('save_freq', 5000), ('print_freq', 200)):
        if getattr(args, k) is None:
            setattr(args, k, dflt)

    if args.seed is not None:
        set_seed(args.seed, deterministic=args.deterministic)
    if args.cudnn_benchmark is not None:
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

    if args.milestones is not None:
        cfg = dict(cfg, milestones=list(args.milestones))
    if args.lr is not None:
        cfg = dict(cfg, lr=args.lr)
    if args.gamma is not None:
        cfg = dict(cfg, gamma=args.gamma)
    scale = cfg['scale']
    ckpt_dir = args.ckpt_dir or os.path.join('./checkpoints', args.variant)
    os.makedirs(ckpt_dir, exist_ok=True)
    logfile = os.path.join(ckpt_dir, 'data_config.txt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    use_folders = args.train_hr is not None and args.train_lr is not None
    tmpl = args.filename_tmpl.replace('{scale}', str(scale))

    _emit(_data_config_lines(scale, tmpl, use_folders, args), logfile)

    if use_folders:
        train_ds = build_paired_dataset(
            args.train_hr, args.train_lr, scale=scale,
            hr_patch_size=cfg['hr_patch_size'], filename_tmpl=tmpl, train_mode=True)
        if args.val_hr and args.val_lr:
            val_ds = build_paired_dataset(
                args.val_hr, args.val_lr, scale=scale,
                hr_patch_size=cfg['hr_patch_size'], filename_tmpl=tmpl, train_mode=False)
        else:
            print("ATTENZIONE: nessuna cartella di validation (--val_hr/--val_lr); "
                  "validation disattivata.")
            val_ds = None
    else:
        download = args.download == 'true'
        train_ds = build_sr_dataset(args.datasets, args.root, scale=scale, split='train',
                                    hr_patch_size=cfg['hr_patch_size'], train_mode=True,
                                    download=download)
        val_ds = build_sr_dataset(args.datasets, args.root, scale=scale, split='val',
                                  hr_patch_size=cfg['hr_patch_size'], train_mode=False,
                                  download=download)

    summary = [f"TRAIN: {len(train_ds)} coppie HR/LR | patch HR={cfg['hr_patch_size']} "
               f"(LR={train_ds.lr_size}) px"]
    ex = _example_pair(train_ds)
    if ex:
        summary += [f"  esempio HR: {ex[0]}", f"  esempio LR: {ex[1]}"]
    if val_ds is not None:
        summary.append(f"VAL:   {len(val_ds)} coppie HR/LR")
        exv = _example_pair(val_ds)
        if exv:
            summary += [f"  esempio HR: {exv[0]}", f"  esempio LR: {exv[1]}"]
    summary.append("-" * 78)
    _emit(summary, logfile)

    # Seeded shuffling plus per-worker seeding make the data order reproducible.
    loader_kw = {}
    if args.seed is not None:
        g = torch.Generator()
        g.manual_seed(args.seed)
        loader_kw = dict(generator=g, worker_init_fn=_seed_worker)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              **loader_kw)
    val_loader = (DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
                  if val_ds is not None else None)

    overrides = dict(args.yaml_overrides)
    overrides.update(parse_overrides(args.arch_override))
    model, arch_cfg = build_model(args.arch, args.variant, **overrides)
    n_params = sum(p.numel() for p in model.parameters())

    if cfg['needs_pretrain'] and args.pretrain is None and args.resume is None:
        print(f"ATTENZIONE: {args.variant} dovrebbe partire dai pesi x2 "
              f"(--pretrain). Procedo comunque da zero.")
    if args.pretrain is not None and args.resume is None:
        load_pretrain(model, args.pretrain, device)

    print(f"[{args.arch}] {args.variant}: {n_params/1e6:.3f}M parametri | "
          f"dataset={train_ds.names} | scale={scale} | "
          f"patch HR={cfg['hr_patch_size']} (LR={train_ds.lr_size}) | device={device}")
    if overrides:
        print(f"override (ablazione): {overrides}")
    print(f"lr={cfg['lr']:.0e} | total_iter={cfg['total_iter']} | "
          f"milestones={cfg['milestones']} | batch={args.batch_size}")
    if args.seed is not None or (args.fft_weight and args.fft_weight > 0):
        det = " (cudnn deterministic)" if args.deterministic else ""
        print(f"seed={args.seed}{det} | fft_weight={args.fft_weight}")
    run_meta = None
    if yml is not None:
        print(f"config={args.config} | cudnn.benchmark={torch.backends.cudnn.benchmark} | "
              f"num_workers={args.num_workers} | ckpt ogni {args.ckpt_iter_freq} iter | "
              f"pretrain={args.pretrain}")
        run_meta = dict(config_path=os.path.abspath(args.config), config=yml,
                        cli=dict(seed=args.seed, total_iter=cfg['total_iter'],
                                 pretrain=args.pretrain))
        import shutil
        shutil.copy(args.config, os.path.join(ckpt_dir, 'config.yml'))   # provenance
    print("-" * 78)

    train(model, train_loader, val_loader, cfg, device, ckpt_dir,
          val_freq=args.val_freq, save_freq=args.save_freq,
          print_freq=args.print_freq, resume=args.resume,
          best_metric=args.best_metric, vis_dir=args.vis_dir,
          vis_freq=args.vis_freq, vis_samples=args.vis_samples,
          arch=args.arch, arch_cfg=arch_cfg,
          fft_weight=args.fft_weight, seed=args.seed,
          ckpt_iter_freq=args.ckpt_iter_freq, run_meta=run_meta)


if __name__ == '__main__':
    main()