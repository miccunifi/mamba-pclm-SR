"""
baselines.py — BASELINE model adapters for count_params_macs.py and bench_latency.py.

Each adapter builds the OFFICIAL architecture (file from the repo cloned in src/repos/)
in the lightweight configuration of the requested scale and returns an nn.Module with
the uniform interface  sr = model(lr)  (lr: (1,3,H,W) in [0,1], sr: (1,3,sH,sW)).
Weights are NOT needed for parameters/MACs/latency (they depend only on the architecture);
the URLs of the official checkpoints are in BASELINES[...]['ckpt_url'] and in the README.

Adapters needed (and why):
  * SwinIR-light, MambaIRv2-light: the official tests pad the input to a multiple of the
    window (8 / 16) by reflection and crop the output: we replicate this (PadToMultiple)
    because 640x360 and 427x240 are not multiples of 16 / 8.
  * CARN: forward(x, scale) -> wrapper with a fixed scale; built with multi_scale=False
    (a single upsampler, which is how the paper counts parameters per scale).
  * LAPAR-A: Network(config) from the repo with the pkl kernels (absolute path).
  * DVMSR: the NTIRE repo defines it and ships weights ONLY at x4; for x2/x3 we build the
    architecture with a different upscale (no official results) -> flag `only_scales`.
  * LatticeNet: there is NO official PyTorch code (only MindSpore, VipSuperLab);
    we use the unofficial port ymff0592/super-resolution, marked `unofficial=True`.
  * RLFN: official weights for x2/x4 (rlfn_x{2,4}.pth); at x3 architecture only.

Import isolation: the repos use packages with conflicting names (`model`, `models`,
`utils`, `ops`): every import runs with the repo root at the head of sys.path, and the
modules with those names are removed from sys.modules right after.
"""
import importlib
import importlib.util
import os
import subprocess
import sys
import types

import torch
import torch.nn as nn

from common import REPOS_DIR

_PURGE = ('model', 'models', 'utils', 'ops', 'block', 'common', 'data', 'option', 'utility')


def _repo(name):
    d = os.path.join(REPOS_DIR, name)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"repo mancante: {d} (vedi README_experiments.md, sezione baseline)")
    return d


def repo_commit(name):
    try:
        return subprocess.check_output(['git', '-C', _repo(name), 'rev-parse', '--short=12', 'HEAD'],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return '?'


def _stub_basicsr():
    """Install minimal basicsr stubs (no-op registry + arch_util) when basicsr is not
    imported, so that BasicSR-based arch files import without the whole framework."""
    import collections.abc
    from itertools import repeat as _repeat
    if 'basicsr.utils.registry' not in sys.modules:
        for n in ('basicsr', 'basicsr.utils', 'basicsr.archs'):
            sys.modules.setdefault(n, types.ModuleType(n))
        reg = types.ModuleType('basicsr.utils.registry')

        class _Reg:
            def register(self, *a, **k):
                # supports both the bare `@register` and the `@register(...)` decorator forms
                if a and callable(a[0]):
                    return a[0]
                return lambda c: c
        reg.ARCH_REGISTRY = _Reg()
        sys.modules['basicsr.utils.registry'] = reg
    if 'basicsr.archs.arch_util' not in sys.modules:
        au = types.ModuleType('basicsr.archs.arch_util')

        def to_2tuple(x):
            if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
                return tuple(x)
            return tuple(_repeat(x, 2))
        au.to_2tuple = to_2tuple
        au.trunc_normal_ = nn.init.trunc_normal_
        sys.modules['basicsr.archs.arch_util'] = au


def _import_isolated(root, pyfile=None, package=None, extra_paths=()):
    """Import `pyfile` (by path) or `package` (dotted, e.g. 'model.architecture') with
    `root` (+extra_paths) at the head of sys.path. Afterwards, remove from sys.modules the
    generic-name packages this import added and restore the ones that were there before."""
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.split('.')[0] in _PURGE}
    before = set(sys.modules)
    paths = [root] + list(extra_paths)
    # insert in reverse order so that `root` ends up first in sys.path
    for p in reversed(paths):
        sys.path.insert(0, p)
    try:
        if package is not None:
            mod = importlib.import_module(package)
        else:
            modname = f"_baseline_{os.path.basename(root)}_{os.path.splitext(os.path.basename(pyfile))[0]}"
            spec = importlib.util.spec_from_file_location(modname, pyfile)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[modname] = mod
            spec.loader.exec_module(mod)
    finally:
        for p in paths:
            if p in sys.path:
                sys.path.remove(p)
        for k in list(sys.modules):
            if k not in before and k.split('.')[0] in _PURGE:
                del sys.modules[k]
        sys.modules.update(saved)
    return mod


# ======================================================================================
# wrapper
# ======================================================================================
class PadToMultiple(nn.Module):
    """Pad to a multiple of `mult` by reflection (flip-concat, as in main_test_swinir.py
    and the MambaIR/MambaIRv2 tests) + crop the output to scale*(H,W)."""

    def __init__(self, net, mult, scale):
        super().__init__()
        self.net, self.mult, self.upscale = net, mult, scale

    def forward(self, x):
        _, _, h, w = x.shape
        m = self.mult
        hp = (h // m + 1) * m - h if h % m else 0
        wp = (w // m + 1) * m - w if w % m else 0
        if hp:
            x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h + hp, :]
        if wp:
            x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w + wp]
        y = self.net(x)
        return y[..., :h * self.upscale, :w * self.upscale]


class FixedScale(nn.Module):
    """forward(x) -> net(x, scale) (CARN)."""

    def __init__(self, net, scale):
        super().__init__()
        self.net, self.upscale = net, scale

    def forward(self, x):
        return self.net(x, self.upscale)


# ======================================================================================
# adapter
# ======================================================================================
def swinir_light(scale):
    root = _repo('SwinIR')
    mod = _import_isolated(root, pyfile=os.path.join(root, 'models', 'network_swinir.py'))
    net = mod.SwinIR(upscale=scale, in_chans=3, img_size=64, window_size=8, img_range=1.,
                     depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2,
                     upsampler='pixelshuffledirect', resi_connection='1conv')
    return PadToMultiple(net, 8, scale)


def mambair_light(scale):
    root = _repo('MambaIR')
    _stub_basicsr()
    mod = _import_isolated(root, pyfile=os.path.join(root, 'basicsr', 'archs', 'mambair_arch.py'))
    # options/test/mambair/test_MambaIR_lightSR_x*.yml
    return mod.MambaIR(upscale=scale, in_chans=3, img_size=64, img_range=1., d_state=10,
                       depths=[6, 6, 6, 6], embed_dim=60, mlp_ratio=1.2,
                       upsampler='pixelshuffledirect', resi_connection='1conv')


def mambairv2_light(scale):
    root = _repo('MambaIR')
    _stub_basicsr()
    mod = _import_isolated(root, pyfile=os.path.join(root, 'basicsr', 'archs', 'mambairv2light_arch.py'))
    # options/train/mambairv2/train_MambaIRv2_lightSR_x*.yml (window 16)
    net = mod.MambaIRv2Light(upscale=scale, in_chans=3, img_size=64, img_range=1.,
                             embed_dim=48, d_state=8, depths=[5, 5, 5, 5], num_heads=[4, 4, 4, 4],
                             window_size=16, inner_rank=32, num_tokens=64, convffn_kernel_size=5,
                             mlp_ratio=1.0, upsampler='pixelshuffledirect', resi_connection='1conv')
    return PadToMultiple(net, 16, scale)


def safmn(scale):
    root = _repo('SAFMN')
    _stub_basicsr()
    mod = _import_isolated(root, pyfile=os.path.join(root, 'basicsr', 'archs', 'safmn_arch.py'))
    # options/train/SAFMN/train_DF2K_X*.yml: dim 36, n_blocks 8, ffn_scale 2.0
    return mod.SAFMN(dim=36, n_blocks=8, ffn_scale=2.0, upscaling_factor=scale)


def dvmsr(scale):
    root = _repo('DVMSR')
    mod = _import_isolated(root, pyfile=os.path.join(root, 'models', 'team07_DVMSR.py'))
    return mod.DVMSR(upscale=scale)


def rlfn(scale):
    root = _repo('RLFN')
    mod = _import_isolated(root, package='model.rlfn')
    return mod.RLFN(in_channels=3, out_channels=3, feature_channels=52, upscale=scale)


def carn(scale):
    root = os.path.join(_repo('CARN-pytorch'), 'carn')       # package `model` (model.ops)
    mod = _import_isolated(root, package='model.carn')
    return FixedScale(mod.Net(scale=scale, multi_scale=False, group=1), scale)


def imdn(scale):
    root = _repo('IMDN')
    mod = _import_isolated(root, package='model.architecture')
    return mod.IMDN(in_nc=3, nf=64, num_modules=6, out_nc=3, upscale=scale)


def lapar_a(scale):
    root = _repo('Simple-SR')
    exp = os.path.join(root, 'exps', f'LAPAR_A_x{scale}')
    cfgmod = _import_isolated(root, pyfile=os.path.join(exp, 'config.py'), extra_paths=[exp])
    netmod = _import_isolated(root, pyfile=os.path.join(exp, 'network.py'), extra_paths=[exp])
    config = cfgmod.config
    config.MODEL.KERNEL_PATH = os.path.join(root, 'kernel', 'kernel_72_k5.pkl')
    config.MODEL.SCALE = scale
    config.MODEL.DEVICE = 'cpu'
    return netmod.Network(config)


def latticenet(scale):
    root = _repo('LatticeNet-pytorch')
    mod = _import_isolated(root, package='model.latticenet')
    args = types.SimpleNamespace(n_feats=64, scale=[scale], n_colors=3, rgb_range=255)
    return mod.LatticeNet(args)


BASELINES = {
    'SwinIR-light': dict(build=swinir_light, repo='SwinIR', only_scales=(2, 3, 4),
                         url='https://github.com/JingyunLiang/SwinIR',
                         arch_file='models/network_swinir.py',
                         ckpt_url='https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/'
                                  '002_lightweightSR_DIV2K_s64w8_SwinIR-S_x{scale}.pth',
                         train_data='DIV2K', notes='padding a multipli di 8 come main_test_swinir.py'),
    'MambaIR-light': dict(build=mambair_light, repo='MambaIR', only_scales=(2, 3, 4),
                          url='https://github.com/csguoh/MambaIR',
                          arch_file='basicsr/archs/mambair_arch.py',
                          ckpt_url='https://huggingface.co/cguoh/MambaIR (MambaIR_light{scale}.pth; '
                                   'Google Drive nel README del repo)',
                          train_data='DIV2K+Flickr2K (DF2K)', notes=''),
    'MambaIRv2-light': dict(build=mambairv2_light, repo='MambaIR', only_scales=(2, 3, 4),
                            url='https://github.com/csguoh/MambaIR',
                            arch_file='basicsr/archs/mambairv2light_arch.py',
                            ckpt_url='https://github.com/csguoh/MambaIR/releases/tag/v1.0 '
                                     '(MambaIRv2_light{scale}.pth)',
                            train_data='DF2K', notes='padding a multipli di 16 (window 16)'),
    'SAFMN': dict(build=safmn, repo='SAFMN', only_scales=(2, 3, 4),
                  url='https://github.com/sunny2109/SAFMN', arch_file='basicsr/archs/safmn_arch.py',
                  ckpt_url='https://huggingface.co/Meloo/SAFMN/tree/main (SAFMN_DF2K_x{scale}.pth)',
                  train_data='DF2K', notes=''),
    'DVMSR': dict(build=dvmsr, repo='DVMSR', only_scales=(4,),
                  url='https://github.com/nathan66666/DVMSR', arch_file='models/team07_DVMSR.py',
                  ckpt_url='in-repo: model_zoo/team07_DVMSR.pth (NTIRE 2024 ESR, solo x4)',
                  train_data='DF2K+LSDIR (challenge)', notes='architettura definita solo a x4'),
    'RLFN': dict(build=rlfn, repo='RLFN', only_scales=(2, 3, 4),
                 url='https://github.com/bytedance/RLFN', arch_file='model/rlfn.py',
                 ckpt_url='in-repo: model_zoo/rlfn_x2.pth, model_zoo/rlfn_x4.pth (nessun x3)',
                 train_data='DIV2K', notes=''),
    'CARN': dict(build=carn, repo='CARN-pytorch', only_scales=(2, 3, 4),
                 url='https://github.com/nmhkahn/CARN-pytorch', arch_file='carn/model/carn.py',
                 ckpt_url='in-repo: checkpoint/carn.pth (multi-scala)', train_data='DIV2K',
                 notes='conteggio con multi_scale=False (un upsampler per scala)'),
    'IMDN': dict(build=imdn, repo='IMDN', only_scales=(2, 3, 4),
                 url='https://github.com/Zheng222/IMDN', arch_file='model/architecture.py',
                 ckpt_url='in-repo: checkpoints/IMDN_x{scale}.pth', train_data='DIV2K', notes=''),
    'LAPAR-A': dict(build=lapar_a, repo='Simple-SR', only_scales=(2, 3, 4),
                    url='https://github.com/Jia-Research-Lab/Simple-SR',
                    arch_file='exps/LAPAR_A_x{scale}/network.py',
                    ckpt_url='https://drive.google.com/drive/folders/1c-KUEPJl7pHs9btqHYoUJkcMPKViObgJ',
                    train_data='DF2K', notes=''),
    'LatticeNet': dict(build=latticenet, repo='LatticeNet-pytorch', only_scales=(2, 3, 4),
                       url='https://github.com/ymff0592/super-resolution (port NON ufficiale; '
                           'ufficiale solo MindSpore: https://github.com/VipSuperLab/LatticeNet)',
                       arch_file='model/latticenet.py', ckpt_url='nessun peso PyTorch ufficiale',
                       train_data='DIV2K (paper)', notes='UNOFFICIAL', unofficial=True),
}


def build_baseline(name, scale, device='cuda'):
    """Return (model.eval() on device, info)."""
    spec = BASELINES[name]
    model = spec['build'](scale).to(device).eval()
    info = dict(name=name, scale=scale, repo=spec['repo'], commit=repo_commit(spec['repo']),
                url=spec['url'], arch_file=spec['arch_file'].format(scale=scale),
                ckpt_url=spec['ckpt_url'].format(scale=scale), unofficial=spec.get('unofficial', False),
                official_scale=scale in spec['only_scales'], notes=spec['notes'])
    return model, info


def resolve_names(items):
    """'all' | list of names (case-insensitive) -> list of BASELINES keys."""
    if not items or items == ['none']:
        return []
    if items == ['all']:
        return list(BASELINES)
    low = {k.lower(): k for k in BASELINES}
    out = []
    for it in items:
        if it.lower() not in low:
            raise SystemExit(f"baseline sconosciuta: {it} (disponibili: {list(BASELINES)})")
        out.append(low[it.lower()])
    return out


if __name__ == '__main__':
    # smoke test: build every baseline at x4 on CPU and print its parameter count
    for name in BASELINES:
        try:
            m, info = build_baseline(name, 4, device='cpu')
            n = sum(p.numel() for p in m.parameters())
            print(f"{name:<16} x4  {n/1e3:8.1f}K params  ({info['repo']}@{info['commit']})")
        except Exception as e:
            print(f"{name:<16} ERRORE: {type(e).__name__}: {e}")
