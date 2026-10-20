"""
make_configs.py — Generate the 15 rho-study configs: configs/rho_{000,025,050,075,100}_x{2,3,4}.yml

All files are IDENTICAL except for `network_g.mamba_ratio` and `scale` (plus the paths that
depend on the scale: LR folders X<s>, variant pclmsr_x<s>, pretrain from x2).
The recipe (lr, milestones, iterations, patch, batch, loss, augmentation) comes from
models.TRAIN_CONFIGS, i.e. the same one used so far: it cannot diverge by mistake.

Usage
-----
  python scripts/make_configs.py                       # writes configs/*.yml
  python scripts/make_configs.py --data_root ../datasets --experiments_root ./experiments
  python scripts/make_configs.py --check                # checks that the files on disk match
"""
import argparse
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # src/
sys.path.insert(0, ROOT)
from models import TRAIN_CONFIGS  # noqa: E402

RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
SCALES = [2, 3, 4]


def rho_tag(rho):
    return f"{int(round(rho * 100)):03d}"


def config_name(rho, scale):
    return f"rho_{rho_tag(rho)}_x{scale}"


BACKBONE = dict(
    embed_dim=60, num_groups=4, blocks_per_group=6, d_state=10,
    expand=1.5, local_ratio=2, le_ratio=4, local_type='bottleneck',
    scan_directions=4, local_skip=True, residual_scaling=True, fuse_residual_scale=False,
    use_eca=True, use_le=True,
    use_rfr=False,
    upsampler='pixelshuffledirect', img_range=1.0, in_chans=3,
)


def make_config(rho, scale, data_root, experiments_root):
    sched = TRAIN_CONFIGS[f'pclmsr_x{scale}']
    # Seed ablation runs stop at 100K iter; no lr drop before that, so they match the full run.
    assert sched['milestones'][0] >= 100000, \
        "la prima milestone deve essere >= 100K (le prime 100K iter devono coincidere)"
    net = dict(BACKBONE)
    net['mamba_ratio'] = float(rho)
    cfg = {
        'name': config_name(rho, scale),
        'manual_seed': 0,                     # override: python train.py --config ... --seed 1
        'scale': scale,
        'arch': 'pclmsr',
        'variant': f'pclmsr_x{scale}',
        'network_g': net,
        'datasets': {
            'train_hr': [f'{data_root}/DIV2K/DIV2K_train_HR'],
            'train_lr': [f'{data_root}/DIV2K/DIV2K_train_LR_bicubic/X{scale}'],
            'val_hr': [f'{data_root}/DIV2K/DIV2K_valid_HR'],
            'val_lr': [f'{data_root}/DIV2K/DIV2K_valid_LR_bicubic/X{scale}'],
            'filename_tmpl': '{}x{scale}',
            'hr_patch_size': sched['hr_patch_size'],   # 128 (x2) / 192 (x3, x4) -> LR 64x64
            'batch_size': 32,
            'num_workers': 8,
        },
        'train': {
            'optimizer': 'Adam(betas=(0.9, 0.99), weight_decay=0)  # costante in train.py',
            'lr': sched['lr'],
            'milestones': list(sched['milestones']),
            'gamma': 0.5,
            'total_iter': sched['total_iter'],       # 500K; ablation seed 1/2: --total_iter 100000
            'loss': {'type': 'L1Loss', 'loss_weight': 1.0, 'reduction': 'mean'},
            'ema_decay': 0,
            'cudnn_benchmark': False,
            'deterministic': False,
        },
        'path': {
            'experiments_root': experiments_root,
            # x3/x4 start from the x2 run with the same rho; train.py fills {seed} from --seed.
            'pretrain_network_g': (None if scale == 2 else
                                   f'{experiments_root}/{config_name(rho, 2)}/seed{{seed}}/final.pth'),
        },
        'logger': {
            'save_checkpoint_freq': 100000,          # iter_100000.pth, iter_200000.pth, ...
            'save_latest_freq': 5000,
            'val_freq': 5000,
            'print_freq': 200,
        },
    }
    return cfg


def dump(cfg):
    return yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default=os.path.join(ROOT, 'configs'))
    p.add_argument('--data_root', default='../datasets',
                   help="radice dataset, relativa a src/ (default ../datasets)")
    p.add_argument('--experiments_root', default='./experiments')
    p.add_argument('--check', action='store_true', help="non scrive: verifica i file esistenti")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    n_bad = 0
    for scale in SCALES:
        for rho in RHOS:
            cfg = make_config(rho, scale, args.data_root, args.experiments_root)
            path = os.path.join(args.out_dir, cfg['name'] + '.yml')
            text = dump(cfg)
            if args.check:
                ok = os.path.isfile(path) and open(path).read() == text
                n_bad += (not ok)
                print(f"{'OK  ' if ok else 'DIFF'} {path}")
            else:
                with open(path, 'w') as f:
                    f.write(text)
                print(f"scritto {path}")
    if args.check:
        # Invariant: train/logger sections are shared and network_g differs only by mamba_ratio.
        base = yaml.safe_load(dump(make_config(0.5, 2, args.data_root, args.experiments_root)))
        for scale in SCALES:
            for rho in RHOS:
                c = make_config(rho, scale, args.data_root, args.experiments_root)
                for k in ('train', 'logger'):
                    assert c[k] == base[k], f"{c['name']}: sezione {k} diversa"
                net = dict(c['network_g']); net.pop('mamba_ratio')
                bnet = dict(base['network_g']); bnet.pop('mamba_ratio')
                assert net == bnet, f"{c['name']}: network_g differisce oltre mamba_ratio"
        print("invariante OK: i config differiscono solo per mamba_ratio e scala")
        sys.exit(1 if n_bad else 0)


if __name__ == '__main__':
    main()
