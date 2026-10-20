"""
pclmsr.py — PCLM-SR: Partial-Channel Lightweight Mamba for Image Super-Resolution.

PyTorch implementation that follows the paper formulation (sections 2-12).
It reuses the tested components of mambair.py: `selective_scan` (the mamba-ssm
CUDA kernel when available, otherwise a PyTorch fallback) and the `SS2D` core,
which matches the LiteVSSM of eq. (39) exactly:
    LiteVSSM(X) = W3 [ LN(2DSSM(SiLU(DWConv(W1 X)))) (.) SiLU(W2 X) ]

Architecture (eq. 3-6):
  F0   = Conv3x3(I_LR)
  F_g  = RLMG_g(F_{g-1}),  g = 1..G
  F_DF = Conv3x3(F_G) + F0
  I_SR = Conv3x3( PixelShuffle_s( Conv3x3(F_DF) ) )

Core block PCLMB (eq. 31-32):
  channel split: Cm = floor(rho*C) (Mamba), Cl = C - Cm (local)
  Xbm = LiteVSSM(LN(Xm)) + alpha_m (.) Xm                       (eq. 16)
  Xbl = Local(LN(Xl))    + alpha_l (.) Xl                       (eq. 18)
  Y   = Conv1x1(Concat[Xbm, Xbl]) + X                           (eq. 25)
  PCLMB(X) = ECA(LE(LN(Y))) + beta (.) Y                        (eq. 29)

alpha_m, alpha_l, beta (and gamma in the Tiny Frequency Gate) are learnable
per-channel residual scaling factors applied to the skip connections (MambaIR style).
"""
import math

import torch
import torch.nn as nn

from mambair import SS2D, selective_scan, UpsampleOneStep  # noqa: F401


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel axis for channel-first tensors (B,C,H,W)."""

    def __init__(self, num_channels):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)            # (B,H,W,C)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2).contiguous()


class ConvBottleneck(nn.Module):
    """Bottleneck with a depth-wise conv. Used both for the local branch (eq. 21)
    and for the Local Enhancement (eq. 54):
        out = W_up[ sigma( DWConv3x3( sigma( W_down x ) ) ) ]
    W_down: dim -> dim/ratio,  W_up: dim/ratio -> dim."""

    def __init__(self, dim, ratio=2, act=nn.GELU):
        super().__init__()
        hidden = max(dim // ratio, 1)
        self.down = nn.Conv2d(dim, hidden, 1)
        self.act1 = act()
        self.dw = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.act2 = act()
        self.up = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        return self.up(self.act2(self.dw(self.act1(self.down(x)))))


class LocalSimple(nn.Module):
    """Simple variant of the local branch (eq. 20):
        Local(x) = Conv1x1( sigma( DWConv3x3(x) ) )."""

    def __init__(self, dim, act=nn.GELU):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.act = act()
        self.pw = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        return self.pw(self.act(self.dw(x)))


class LiteVSSM(nn.Module):
    """Channel-first wrapper around SS2D (= eq. 39). Works on `dim`=Cm channels
    with expansion ratio `expand`=lambda."""

    def __init__(self, dim, d_state=8, expand=1.5, scan_directions=4):
        super().__init__()
        self.core = SS2D(d_model=dim, d_state=d_state, expand=expand,
                         scan_directions=scan_directions)

    def forward(self, x):                    # (B, Cm, H, W)
        x = x.permute(0, 2, 3, 1).contiguous()   # channel-last for SS2D
        x = self.core(x)
        return x.permute(0, 3, 1, 2).contiguous()


class ECA(nn.Module):
    """ECA(T) = Sigmoid(Conv1D_k(GAP(T))) (.) T. Adaptive kernel k (ECA-Net) when None."""

    def __init__(self, channels, k=None, gamma=2, b=1):
        super().__init__()
        # ECA-Net rule: k = |(log2(C) + b) / gamma|, rounded to an odd value >= 3.
        if k is None:
            t = int(abs((math.log2(channels) + b) / gamma))
            k = t if t % 2 else t + 1
            k = max(k, 3)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x):
        z = x.mean(dim=(-2, -1))                  # GAP -> (B, C)
        a = self.conv(z.unsqueeze(1))             # (B, 1, C)
        a = torch.sigmoid(a).squeeze(1)           # (B, C)
        return x * a.view(x.size(0), -1, 1, 1)


class TinyFrequencyGate(nn.Module):
    """TFG(X) = W_proj[ irFFT2D( GConv1x1( rFFT2D( W_red X ) ) ) ] + gamma (.) X.
    The GConv1x1 on the complex spectrum is applied (same weights) to the real and
    imaginary parts, i.e. a real-weight channel projection in the complex domain."""

    def __init__(self, channels, freq_ratio=0.25, groups=1):
        super().__init__()
        cf = max(int(freq_ratio * channels), 1)
        self.red = nn.Conv2d(channels, cf, 1)
        self.gconv = nn.Conv2d(cf, cf, 1, groups=groups)
        self.proj = nn.Conv2d(cf, channels, 1)
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        B, C, H, W = x.shape
        xr = self.red(x)
        spec = torch.fft.rfft2(xr, norm='ortho')
        spec = torch.complex(self.gconv(spec.real), self.gconv(spec.imag))
        xf = torch.fft.irfft2(spec, s=(H, W), norm='ortho')
        return self.proj(xf) + self.gamma.view(1, -1, 1, 1) * x


class PCLMB(nn.Module):
    """Partial-Channel Lite Mamba Block with ablation switches.

    Supported ablations:
      mamba_ratio        : fraction of channels sent to the Mamba branch (0..1).
                           1.0 -> Mamba only (no local branch); 0.0 -> local only.
      local_type         : 'bottleneck' (eq. 21) | 'simple' (eq. 20).
      use_eca / use_le   : enable/disable ECA and Local Enhancement (eq. 29).
      residual_scaling   : if False, alpha_m/alpha_l/beta are fixed to 1 (plain
                           residuals) instead of learnable.
      fuse_residual_scale: if True use Y = Fuse(T) + alpha_f (.) X (eq. 26),
                           otherwise Y = Fuse(T) + X (eq. 25, default).
      scan_directions    : number of 2D-scan directions in LiteVSSM (1 | 2 | 4).
      local_skip         : if False, remove the +alpha_l*Xl skip of the local branch
                           (eq. 18 -> Local(LN(Xl)) only).
    """

    def __init__(self, dim, mamba_ratio=0.25, d_state=8, expand=1.5,
                 local_ratio=2, le_ratio=4, local_type='bottleneck',
                 use_eca=True, use_le=True, residual_scaling=True,
                 fuse_residual_scale=False, scan_directions=4, local_skip=True):
        super().__init__()
        self.dim = dim
        self.c_mamba = int(dim * mamba_ratio)
        self.c_local = dim - self.c_mamba
        assert self.c_mamba > 0 or self.c_local > 0, "blocco vuoto"
        self.use_eca, self.use_le = use_eca, use_le
        self.fuse_residual_scale = fuse_residual_scale
        self.local_skip = local_skip

        # Per-channel residual scale: learnable, or a fixed tensor of ones.
        def scale_param(n):
            t = torch.ones(n)
            return nn.Parameter(t) if residual_scaling else t

        if self.c_mamba > 0:
            self.ln_m = LayerNorm2d(self.c_mamba)
            self.litevssm = LiteVSSM(self.c_mamba, d_state=d_state, expand=expand,
                                     scan_directions=scan_directions)
            self.alpha_m = scale_param(self.c_mamba)                   # eq. 16
            # Fixed scales are registered as buffers so they move with the module.
            if not isinstance(self.alpha_m, nn.Parameter):
                self.register_buffer('alpha_m_buf', self.alpha_m); self.alpha_m = self.alpha_m_buf

        if self.c_local > 0:
            self.ln_l = LayerNorm2d(self.c_local)
            self.local = (LocalSimple(self.c_local) if local_type == 'simple'
                          else ConvBottleneck(self.c_local, ratio=local_ratio))   # eq. 20/21
            self.alpha_l = scale_param(self.c_local)                   # eq. 18
            if not isinstance(self.alpha_l, nn.Parameter):
                self.register_buffer('alpha_l_buf', self.alpha_l); self.alpha_l = self.alpha_l_buf

        self.fuse = nn.Conv2d(dim, dim, 1)
        if fuse_residual_scale:
            self.alpha_f = nn.Parameter(torch.ones(dim))              # eq. 26

        # ---- local enhancement + channel attention (eq. 29) ----
        self.ln_le = LayerNorm2d(dim)
        self.le = ConvBottleneck(dim, ratio=le_ratio) if use_le else None  # eq. 54
        self.eca = ECA(dim) if use_eca else None
        self.beta = scale_param(dim)                                  # eq. 29
        if not isinstance(self.beta, nn.Parameter):
            self.register_buffer('beta_buf', self.beta); self.beta = self.beta_buf

    @staticmethod
    def _v(t):
        return t.view(1, -1, 1, 1)

    def forward(self, x):
        if self.c_mamba > 0 and self.c_local > 0:
            xm, xl = torch.split(x, [self.c_mamba, self.c_local], dim=1)
        elif self.c_mamba > 0:
            xm, xl = x, None
        else:
            xm, xl = None, x

        parts = []
        if xm is not None:
            parts.append(self.litevssm(self.ln_m(xm)) + self._v(self.alpha_m) * xm)
        if xl is not None:
            out_l = self.local(self.ln_l(xl))
            if self.local_skip:
                out_l = out_l + self._v(self.alpha_l) * xl
            parts.append(out_l)
        t = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]   # eq. 23

        skip = self._v(self.alpha_f) * x if self.fuse_residual_scale else x
        y = self.fuse(t) + skip                                       # eq. 25/26

        if self.use_le or self.use_eca:
            r = self.ln_le(y)
            if self.le is not None:
                r = self.le(r)
            if self.eca is not None:
                r = self.eca(r)
            return r + self._v(self.beta) * y                         # eq. 29
        return self._v(self.beta) * y


class RLMG(nn.Module):
    """Residual Lite Mamba Group (eq. 7-10):
    num_blocks x PCLMB -> Conv3x3 [-> TinyFrequencyGate] + input skip."""

    def __init__(self, dim, num_blocks=3, mamba_ratio=0.25, d_state=8, expand=1.5,
                 local_ratio=2, le_ratio=4, local_type='bottleneck',
                 use_eca=True, use_le=True, residual_scaling=True,
                 fuse_residual_scale=False, scan_directions=4, local_skip=True,
                 use_freq_gate=False, freq_ratio=0.25):
        super().__init__()
        self.blocks = nn.ModuleList([
            PCLMB(dim, mamba_ratio=mamba_ratio, d_state=d_state, expand=expand,
                  local_ratio=local_ratio, le_ratio=le_ratio, local_type=local_type,
                  use_eca=use_eca, use_le=use_le, residual_scaling=residual_scaling,
                  fuse_residual_scale=fuse_residual_scale,
                  scan_directions=scan_directions, local_skip=local_skip)
            for _ in range(num_blocks)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)                       # eq. 9/10
        self.tfg = TinyFrequencyGate(dim, freq_ratio=freq_ratio) if use_freq_gate else None

    def forward(self, x):
        u = x
        for blk in self.blocks:
            u = blk(u)                                                # eq. 8
        out = self.conv(u)
        if self.tfg is not None:
            out = self.tfg(out)                                       # eq. 10
        return out + x


# ======================================================================================
# Upsampler (eq. 6): Conv3x3 -> PixelShuffle_s -> Conv3x3
# ======================================================================================
class PixelShuffleUpsampler(nn.Module):
    def __init__(self, scale, dim, num_out_ch=3, up_feat=None):
        super().__init__()
        up_feat = up_feat or dim
        self.conv1 = nn.Conv2d(dim, up_feat * scale * scale, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(scale)
        self.conv2 = nn.Conv2d(up_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        return self.conv2(self.shuffle(self.conv1(x)))


class PCLMSR(nn.Module):
    """PCLM-SR network (eq. 3-6): Conv3x3 -> G x RLMG -> Conv3x3 + F0 -> upsampler."""

    def __init__(self,
                 upscale=2, in_chans=3, embed_dim=48,
                 num_groups=4, blocks_per_group=3,
                 mamba_ratio=0.25, d_state=8, expand=1.5,
                 local_ratio=2, le_ratio=4, local_type='bottleneck',
                 use_eca=True, use_le=True, residual_scaling=True,
                 fuse_residual_scale=False, scan_directions=4, local_skip=True,
                 use_freq_gate=False, freq_ratio=0.25,
                 img_range=1.0, upsampler='classic', use_rfr=None, **kwargs):
        super().__init__()
        # Backward-compatible alias: `use_rfr`, when given, overrides `use_freq_gate`.
        if use_rfr is not None:
            use_freq_gate = bool(use_rfr)
        self.img_range = img_range
        self.upscale = upscale

        # DIV2K RGB mean (same convention as EDSR / SwinIR / MambaIR).
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.register_buffer('mean', torch.tensor(rgb_mean).view(1, 3, 1, 1))
        else:
            self.register_buffer('mean', torch.zeros(1, in_chans, 1, 1))

        # shallow feature extraction (eq. 3)
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        self.groups = nn.ModuleList([
            RLMG(embed_dim, num_blocks=blocks_per_group, mamba_ratio=mamba_ratio,
                 d_state=d_state, expand=expand, local_ratio=local_ratio,
                 le_ratio=le_ratio, local_type=local_type, use_eca=use_eca,
                 use_le=use_le, residual_scaling=residual_scaling,
                 fuse_residual_scale=fuse_residual_scale,
                 scan_directions=scan_directions, local_skip=local_skip,
                 use_freq_gate=use_freq_gate, freq_ratio=freq_ratio)
            for _ in range(num_groups)
        ])
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)   # eq. 5

        self.upsampler = upsampler
        if upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, in_chans)
        else:
            self.upsample = PixelShuffleUpsampler(upscale, embed_dim, num_out_ch=in_chans)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # Subtract the dataset mean and scale by img_range; undone at the end.
        mean = self.mean.type_as(x)
        x = (x - mean) * self.img_range

        f0 = self.conv_first(x)                       # eq. 3
        f = f0
        for group in self.groups:                     # eq. 4
            f = group(f)
        f_df = self.conv_after_body(f) + f0           # eq. 5
        out = self.upsample(f_df)                     # eq. 6

        return out / self.img_range + mean


_BASE = dict(
    in_chans=3, embed_dim=48, num_groups=4, blocks_per_group=3,
    mamba_ratio=0.25, d_state=8, expand=1.5, local_ratio=2, le_ratio=4,
    scan_directions=4, local_skip=True,
    use_freq_gate=False, freq_ratio=0.25, img_range=1.0,
)

PCLMSR_X2 = dict(_BASE, upscale=2)
PCLMSR_X3 = dict(_BASE, upscale=3)
PCLMSR_X4 = dict(_BASE, upscale=4)

_VARIANTS = {'pclmsr_x2': PCLMSR_X2, 'pclmsr_x3': PCLMSR_X3, 'pclmsr_x4': PCLMSR_X4}


def build_pclmsr(variant='pclmsr_x2', **overrides):
    """Build PCLM-SR from a known config ('pclmsr_x2'|'pclmsr_x3'|'pclmsr_x4').
    Use `overrides` to change hyperparameters (e.g. mamba_ratio=0.5, use_freq_gate=True)."""
    if variant not in _VARIANTS:
        raise ValueError(f"variant sconosciuta: {variant!r} (usa una di {list(_VARIANTS)})")
    return PCLMSR(**dict(_VARIANTS[variant], **overrides))


# ======================================================================================
# Sanity check
# ======================================================================================
if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    size = 64 if cuda else 16
    device = 'cuda' if cuda else 'cpu'

    for name in ['pclmsr_x2', 'pclmsr_x3', 'pclmsr_x4']:
        for tfg in (False, True):
            model = build_pclmsr(name, use_freq_gate=tfg).to(device).eval()
            n = sum(p.numel() for p in model.parameters())
            x = torch.randn(1, 3, size, size, device=device)
            with torch.no_grad():
                y = model(x)
            print(f"{name:>10s} | TFG={int(tfg)} | {n/1e3:7.1f}K param | "
                  f"in {tuple(x.shape)} -> out {tuple(y.shape)}")