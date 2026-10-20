"""
models.py — Unified model registry (MambaIR and PCLM-SR) for the
training/test/inference pipeline.

Exposes:
  - build_model(arch, variant, **overrides) -> (model, arch_cfg)
        builds the model and also returns the full dict of hyperparameters
        actually used (arch_cfg). Save it in the checkpoint to rebuild the exact
        architecture at test/inference time (essential for the ablation study).
  - model_from_cfg(arch, arch_cfg) -> model
        rebuilds a model from a saved arch_cfg.
  - parse_overrides(list[str]) -> dict
        converts ['mamba_ratio=0.5', 'use_eca=false', 'd_state=12'] into a typed
        dict, so ablations can be passed from the command line.
  - TRAIN_CONFIGS : training schedule for each variant (lr, milestones, iters,
        patch, pretrain), shared by the two models.
"""
from mambair import MambaIR, _VARIANTS as _MAMBAIR_VARIANTS
from pclmsr import PCLMSR, _VARIANTS as _PCLMSR_VARIANTS


_REGISTRY = {
    'mambair': (MambaIR, _MAMBAIR_VARIANTS),
    'pclmsr':  (PCLMSR,  _PCLMSR_VARIANTS),
}


def list_archs():
    return list(_REGISTRY)


def list_variants(arch):
    return list(_REGISTRY[arch][1])


def arch_of_variant(variant):
    """Infer the arch from the variant name (useful to avoid passing --arch)."""
    for arch, (_, variants) in _REGISTRY.items():
        if variant in variants:
            return arch
    raise ValueError(f"variante sconosciuta: {variant!r}")


def build_model(arch, variant, **overrides):
    """Build the model and return (model, arch_cfg). `arch_cfg` is the full dict
    of kwargs actually used (variant base + overrides)."""
    if arch not in _REGISTRY:
        raise ValueError(f"arch sconosciuta: {arch!r} (usa una di {list_archs()})")
    cls, variants = _REGISTRY[arch]
    if variant not in variants:
        raise ValueError(f"variante {variant!r} non valida per {arch} "
                         f"(usa una di {list(variants)})")
    arch_cfg = dict(variants[variant])
    arch_cfg.update(overrides)
    return cls(**arch_cfg), arch_cfg


def model_from_cfg(arch, arch_cfg):
    """Rebuild a model from an arch_cfg saved in the checkpoint."""
    cls, _ = _REGISTRY[arch]
    return cls(**arch_cfg)


def _coerce(v):
    # Try bool, None, int, float in this order; fall back to the raw string.
    low = v.lower()
    if low in ('true', 'false'):
        return low == 'true'
    if low in ('none', 'null', '~'):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def parse_overrides(items):
    """['mamba_ratio=0.5', 'use_eca=false'] -> {'mamba_ratio':0.5,'use_eca':False}."""
    out = {}
    for it in items or []:
        if '=' not in it:
            raise ValueError(f"override non valido: {it!r} (usa chiave=valore)")
        k, v = it.split('=', 1)
        out[k.strip()] = _coerce(v.strip())
    return out


_X2_SCHED = dict(lr=2e-4, total_iter=500000,
                 milestones=[250000, 400000, 450000, 475000])
# Classic x3/x4 fine-tune from the x2 checkpoint, so they use half the x2 schedule.
_CLASSIC_HALF = dict(lr=1e-4, total_iter=250000,
                     milestones=[125000, 200000, 225000, 237500])

TRAIN_CONFIGS = {
    # MambaIR lightweight
    'light_sr_x2':   dict(scale=2, hr_patch_size=128, needs_pretrain=False, **_X2_SCHED),
    'light_sr_x3':   dict(scale=3, hr_patch_size=192, needs_pretrain=True,  **_X2_SCHED),
    'light_sr_x4':   dict(scale=4, hr_patch_size=192, needs_pretrain=True,  **_X2_SCHED),
    # MambaIR classic
    'classic_sr_x2': dict(scale=2, hr_patch_size=128, needs_pretrain=False, **_X2_SCHED),
    'classic_sr_x3': dict(scale=3, hr_patch_size=192, needs_pretrain=True,  **_CLASSIC_HALF),
    'classic_sr_x4': dict(scale=4, hr_patch_size=192, needs_pretrain=True,  **_CLASSIC_HALF),
    # PCLM-SR: trained from scratch at every scale
    'pclmsr_x2':     dict(scale=2, hr_patch_size=128, needs_pretrain=False, **_X2_SCHED),
    'pclmsr_x3':     dict(scale=3, hr_patch_size=192, needs_pretrain=False, **_X2_SCHED),
    'pclmsr_x4':     dict(scale=4, hr_patch_size=192, needs_pretrain=False, **_X2_SCHED),
}