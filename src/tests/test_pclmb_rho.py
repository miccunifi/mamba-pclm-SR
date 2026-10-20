"""
test_pclmb_rho.py — Sanity test for the rho study (Base-light, no RFR).

For rho in {0, 0.25, 0.5, 0.75, 1} and scale in {2, 3, 4}:
  * build the model from the config configs/rho_XXX_xS.yml (C=60, 4x6 blocks, d_state=10);
  * print total and per-component parameters (Mamba / local / fusion / ECA / LE / head);
  * forward + backward on a 2x3x64x64 batch, check the output shapes;
  * rho=0: no parameter/module of the Mamba branch (no LN, no alpha_m, no SS2D);
    rho=1: no parameter/module of the local branch (no LN, no alpha_l);
  * RFR absent in all configurations; ECA and 1x1 fusion present in all of them;
  * C_m = floor(rho*C) and the VSSM works with C_m = 15 and 45 (no divisibility constraint).

Usage:  python -m pytest tests/test_pclmb_rho.py -v -s      (or: python tests/test_pclmb_rho.py)
"""
import math
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    # src/
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from common import RHOS, SCALES, build_ours, params_by_component, CONFIG_DIR  # noqa: E402
from pclmsr import PCLMB, LiteVSSM, TinyFrequencyGate, ECA  # noqa: E402
from mambair import SS2D  # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
C = 60


def _blocks(model):
    return [m for m in model.modules() if isinstance(m, PCLMB)]


@pytest.mark.parametrize('scale', SCALES)
@pytest.mark.parametrize('rho', RHOS)
def test_pclmb_rho(rho, scale):
    cfg_path = os.path.join(CONFIG_DIR, f"rho_{int(round(rho * 100)):03d}_x{scale}.yml")
    assert os.path.isfile(cfg_path), f"config mancante: {cfg_path} (python scripts/make_configs.py)"
    model, info = build_ours(config_path=cfg_path, device=DEVICE)
    assert info['rho'] == rho and info['scale'] == scale

    blocks = _blocks(model)
    assert len(blocks) == 24, f"attesi 4 RLMG x 6 PCLMB = 24 blocchi, trovati {len(blocks)}"
    assert all(b.dim == C for b in blocks)
    cm_expected = math.floor(rho * C)
    for b in blocks:
        assert b.c_mamba == cm_expected and b.c_local == C - cm_expected

    pc = params_by_component(model)
    print(f"\n[rho={rho:.2f} x{scale}] C_m={cm_expected} C_l={C - cm_expected} | "
          f"total={pc['total']:,} | mamba={pc['mamba']:,} local={pc['local']:,} "
          f"fusion={pc['fusion']:,} eca={pc['eca']:,} le={pc['le']:,} "
          f"rlmg_conv={pc['rlmg_conv']:,} shallow_body={pc['shallow_body']:,} "
          f"head={pc['head']:,} rfr={pc['rfr']:,}")

    assert pc['rfr'] == 0 and not any(isinstance(m, TinyFrequencyGate) for m in model.modules())
    assert all(g.tfg is None for g in model.groups)
    assert pc['eca'] > 0 and sum(isinstance(m, ECA) for m in model.modules()) == 24
    assert pc['fusion'] == 24 * (C * C + C)
    assert all(b.eca is not None and b.fuse is not None for b in blocks)

    if rho == 0.0:
        assert pc['mamba'] == 0
        assert not any(isinstance(m, (LiteVSSM, SS2D)) for m in model.modules())
        for b in blocks:
            assert not hasattr(b, 'litevssm') and not hasattr(b, 'ln_m') and not hasattr(b, 'alpha_m')
        assert not any(n.endswith('.alpha_m') or '.ln_m.' in n or '.litevssm.' in n
                       for n, _ in model.named_parameters())
        assert not any('alpha_m' in n for n, _ in model.named_buffers())
    else:
        assert pc['mamba'] > 0
        ss = [m for m in model.modules() if isinstance(m, SS2D)]
        assert len(ss) == 24
        assert all(s.d_model == cm_expected for s in ss)
        print(f"    VSSM: d_model={ss[0].d_model} d_inner={ss[0].d_inner} "
              f"dt_rank={ss[0].dt_rank} d_state={ss[0].d_state} K={ss[0].K}")
    if rho == 1.0:
        assert pc['local'] == 0
        for b in blocks:
            assert not hasattr(b, 'local') and not hasattr(b, 'ln_l') and not hasattr(b, 'alpha_l')
        assert not any(n.endswith('.alpha_l') or '.ln_l.' in n or '.local.' in n
                       for n, _ in model.named_parameters())
    else:
        assert pc['local'] > 0

    # ---- forward + backward ----
    model.train()
    x = torch.randn(2, 3, 64, 64, device=DEVICE)
    y = model(x)
    assert y.shape == (2, 3, 64 * scale, 64 * scale), y.shape
    assert torch.isfinite(y).all()
    loss = y.abs().mean()
    loss.backward()
    n_nograd = [n for n, p in model.named_parameters() if p.grad is None]
    assert not n_nograd, f"parametri senza gradiente: {n_nograd[:5]}"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    # Odd input sizes must still map to exactly scale x the input size.
    model.eval()
    with torch.no_grad():
        y2 = model(torch.randn(1, 3, 37, 53, device=DEVICE))
    assert y2.shape == (1, 3, 37 * scale, 53 * scale)


def test_params_monotone_in_rho():
    """At fixed scale the parameter count grows with rho (the VSSM is wider than the local
    branch) and the head depends ONLY on the scale."""
    for scale in SCALES:
        tot, head = [], []
        for rho in RHOS:
            m, _ = build_ours(rho, scale, device='cpu')
            pc = params_by_component(m)
            tot.append(pc['total']); head.append(pc['head'])
        assert all(a < b for a, b in zip(tot, tot[1:])), (scale, tot)
        assert len(set(head)) == 1, (scale, head)
    # The head is the only part that depends on scale, so its size must grow with scale.
    heads = [params_by_component(build_ours(0.5, s, device='cpu')[0])['head'] for s in SCALES]
    assert heads[0] < heads[1] < heads[2]


def test_configs_identical_except_rho_and_scale():
    import yaml
    base = None
    for scale in SCALES:
        for rho in RHOS:
            with open(os.path.join(CONFIG_DIR, f"rho_{int(round(rho * 100)):03d}_x{scale}.yml")) as f:
                c = yaml.safe_load(f)
            assert c['network_g']['mamba_ratio'] == rho and c['scale'] == scale
            assert c['network_g']['use_rfr'] is False and c['network_g']['use_eca'] is True
            assert c['train']['milestones'][0] >= 100000
            assert c['logger']['save_checkpoint_freq'] == 100000
            assert c['train']['ema_decay'] == 0 and c['train']['cudnn_benchmark'] is False
            # Strip mamba_ratio; everything else must be identical to the first config seen.
            net = dict(c['network_g']); net.pop('mamba_ratio')
            key = (net, c['train'], c['logger'], c['datasets']['batch_size'],
                   c['datasets']['num_workers'])
            if base is None:
                base = key
            assert key == base, c['name']


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v', '-s']))
