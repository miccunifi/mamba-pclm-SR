"""
MambaIR - A Simple Baseline for Image Restoration with State-Space Model
(Guo et al., ECCV 2024)  https://github.com/csguoh/MambaIR

Clean, self-contained PyTorch reimplementation for super-resolution.

Architecture (paper Sec. 3.2):
  1. Shallow feature extraction : a single 3x3 conv
  2. Deep feature extraction    : N x Residual State-Space Group (RSSG),
                                  each = several Residual State-Space Blocks (RSSB) + conv
  3. HQ reconstruction          : upsampling with pixel-shuffle

Core block RSSB (Fig. 2a, eq. 5-6):
    Z   = VSSM(LN(x))      + s  * x
    out = CA(Conv(LN(Z)))  + s' * Z
where VSSM is the Vision State-Space Module (Fig. 2b). Its inner 4-direction
scan is the 2D Selective Scan (2D-SSM, Fig. 2c).

NOTE on speed: the selective scan below has a pure PyTorch reference path
that runs everywhere but is SLOW (Python loop over the sequence).
For real training install `mamba-ssm`: the fused CUDA kernel is used
automatically when the tensors are on GPU.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Optional fused CUDA kernel from mamba-ssm; fall back to the pure PyTorch path if missing.
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _selective_scan_cuda
    _HAS_FAST_SCAN = True
    _FAST_SCAN_ERROR = None
except Exception as _e:
    _HAS_FAST_SCAN = False
    _FAST_SCAN_ERROR = f"{type(_e).__name__}: {_e}"


def selective_scan(u, delta, A, B, C, D, delta_softplus=True):
    """Selective state-space scan.

        h_t = exp(delta_t * A) * h_{t-1} + (delta_t * B_t) * u_t
        y_t = C_t . h_t + D * u_t

    Shapes:
        u, delta : (batch, dim, L)
        A        : (dim, state)        # negative
        B, C     : (batch, state, L)   # input-dependent, shared across `dim`
        D        : (dim,)
    Returns y : (batch, dim, L)
    """
    if _HAS_FAST_SCAN and u.is_cuda:
        return _selective_scan_cuda(
            u.contiguous(), delta.contiguous(), A.contiguous(),
            B.contiguous(), C.contiguous(), D.contiguous(),
            z=None, delta_bias=None, delta_softplus=delta_softplus, return_last_state=False,
        )

    # Reference path: discretize A and B, then run the recurrence step by step over L.
    if delta_softplus:
        delta = F.softplus(delta)
    dA = torch.exp(torch.einsum('b d l, d n -> b d l n', delta, A))       # (B, dim, L, N)
    dB_u = torch.einsum('b d l, b n l, b d l -> b d l n', delta, B, u)    # (B, dim, L, N)

    bsz, dim, L, N = dA.shape
    h = u.new_zeros(bsz, dim, N)
    ys = []
    for t in range(L):
        h = dA[:, :, t] * h + dB_u[:, :, t]
        ys.append(torch.einsum('b d n, b n -> b d', h, C[:, :, t]))
    y = torch.stack(ys, dim=2)                                           # (B, dim, L)
    return y + u * D.view(1, -1, 1)


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention."""

    def __init__(self, num_feat, squeeze_factor=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


class CAB(nn.Module):
    """Conv + channel attention: this is the `CA(Conv(.))` of eq. (6).
    The conv uses a bottleneck structure to stay efficient."""

    def __init__(self, num_feat, is_light_sr=False, compress_ratio=3, squeeze_factor=30):
        super().__init__()
        if is_light_sr:
            compress_ratio = 2
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, 1, 1, 0),                       # compress 1x1
                nn.Conv2d(num_feat // compress_ratio, num_feat // compress_ratio, 3, 1, 1,
                          groups=num_feat // compress_ratio),                                    # depth-wise 3x3
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, 1, 1, 0),                        # expand 1x1
                nn.Conv2d(num_feat, num_feat, 3, 1, padding=2, groups=num_feat, dilation=2),
                ChannelAttention(num_feat, squeeze_factor),
            )
        else:
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
                ChannelAttention(num_feat, squeeze_factor),
            )

    def forward(self, x):
        return self.cab(x)


# ======================================================================================
# Vision State-Space Module / SS2D  (paper Fig. 2b + 2c)
# ======================================================================================

class SS2D(nn.Module):
    """Vision State-Space Module (Fig. 2b). The `scan` method is the 2D-SSM (Fig. 2c).

    Implements eq. (7): two branches (SSM branch + gating branch) merged with a
    Hadamard product.
        X1 = LN(2D-SSM(SiLU(DWConv(Linear(X)))))
        X2 = SiLU(Linear(X))
        out = Linear(X1 (.) X2)
    """

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2.0,
                 dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                 dropout=0.0, bias=False, conv_bias=True, scan_directions=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        # Indices into the scan sequences: 0 row-major, 1 column-major, 2/3 their reverses.
        _DIR_SETS = {1: [0], 2: [0, 2], 4: [0, 1, 2, 3]}
        if scan_directions not in _DIR_SETS:
            raise ValueError(f"scan_directions deve essere 1, 2 o 4 (ricevuto {scan_directions})")
        self.dir_idx = _DIR_SETS[scan_directions]
        self.K = scan_directions

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                padding=(d_conv - 1) // 2, groups=self.d_inner, bias=conv_bias)
        self.act = nn.SiLU()

        self.x_proj = nn.ModuleList([
            nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
            for _ in range(self.K)
        ])
        self.dt_proj = nn.ModuleList([
            self._dt_init(self.dt_rank, self.d_inner, dt_min, dt_max, dt_init_floor)
            for _ in range(self.K)
        ])

        self.A_logs = nn.Parameter(self._A_log_init(self.d_state, self.d_inner, self.K))  # (K, d_inner, N)
        self.Ds = nn.Parameter(torch.ones(self.K, self.d_inner))                          # (K, d_inner)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _dt_init(dt_rank, d_inner, dt_min, dt_max, dt_init_floor, dt_scale=1.0):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        # Sample dt log-uniformly in [dt_min, dt_max]; the bias stores softplus^-1(dt).
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def _A_log_init(d_state, d_inner, copies):
        # S4D-real init: A = -(1..N) for every channel, stored as log(-A), one copy per direction.
        A = torch.arange(1, d_state + 1, dtype=torch.float32)        # (N,)
        A = A.view(1, d_state).repeat(d_inner, 1)                    # (d_inner, N)
        A_log = torch.log(A).unsqueeze(0).repeat(copies, 1, 1)       # (K, d_inner, N)
        return A_log.contiguous()

    def scan(self, x):
        """2D-SSM (Fig. 2c): scan in 4 directions, apply the selective SSM, merge by sum.

        The 4 directions: top-left->bottom-right, bottom-right->top-left, top-right->bottom-left,
        bottom-left->top-right (= row-wise and column-wise scans, plus their reverses).

        x: (B, d_inner, H, W) -> (B, d_inner, H, W)
        """
        B, Dim, H, W = x.shape
        L = H * W

        x_hw = x.reshape(B, Dim, L)
        x_wh = x.transpose(2, 3).reshape(B, Dim, L)
        seqs = [x_hw, x_wh, torch.flip(x_hw, dims=[-1]), torch.flip(x_wh, dims=[-1])]

        y_sum = None
        for k in range(self.K):
            d = self.dir_idx[k]
            seq = seqs[d]                                      # (B, Dim, L)
            proj = self.x_proj[k](seq.transpose(1, 2))         # (B, L, dt_rank + 2N)
            dt, Bk, Ck = torch.split(proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj[k](dt).transpose(1, 2)           # (B, Dim, L)
            Bk = Bk.transpose(1, 2)                            # (B, N, L)
            Ck = Ck.transpose(1, 2)                            # (B, N, L)
            A = -torch.exp(self.A_logs[k].float())             # (Dim, N)
            D = self.Ds[k].float()                             # (Dim,)
            out = selective_scan(seq, dt, A, Bk, Ck, D)        # (B, Dim, L)

            # Undo the direction's reordering so every output is back in (H, W) layout.
            if d == 0:
                y = out.reshape(B, Dim, H, W)
            elif d == 1:
                y = out.reshape(B, Dim, W, H).transpose(2, 3)
            elif d == 2:
                y = torch.flip(out, dims=[-1]).reshape(B, Dim, H, W)
            else:  # d == 3
                y = torch.flip(out, dims=[-1]).reshape(B, Dim, W, H).transpose(2, 3)
            y_sum = y if y_sum is None else y_sum + y

        return y_sum                                           # (B, Dim, H, W)

    def forward(self, x):
        """x: (B, H, W, C)"""
        B, H, W, C = x.shape
        xz = self.in_proj(x)                       # (B, H, W, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()     # (B, d_inner, H, W)
        x = self.act(self.conv2d(x))               # depth-wise conv + SiLU
        y = self.scan(x)                           # (B, d_inner, H, W)

        y = y.permute(0, 2, 3, 1).contiguous()     # (B, H, W, d_inner)
        y = self.out_norm(y)
        y = y * F.silu(z)
        return self.dropout(self.out_proj(y))

# ======================================================================================
# Stochastic depth
# ======================================================================================
class DropPath(nn.Module):
    """Stochastic depth: during training, zero the residual branch per sample."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device).add_(keep).floor_()
        return x.div(keep) * mask


# ======================================================================================
# Residual State-Space Block (RSSB) - Fig. 2a, eq. (5)-(6)
# ======================================================================================
class RSSB(nn.Module):
    """  Z   = VSSM(LN(x))      + s  * x
         out = CA(Conv(LN(Z)))  + s' * Z
    s and s' are learnable scale factors that control the skip connection."""

    def __init__(self, dim, d_state, expand, is_light_sr=False, drop_path=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.vssm = SS2D(d_model=dim, d_state=d_state, expand=expand)
        self.skip_scale = nn.Parameter(torch.ones(dim))     # s   (eq. 5)
        self.ln2 = nn.LayerNorm(dim)
        self.conv_blk = CAB(dim, is_light_sr=is_light_sr)
        self.skip_scale2 = nn.Parameter(torch.ones(dim))    # s'  (eq. 6)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, x_size):
        B, L, C = x.shape
        H, W = x_size
        x = x.reshape(B, H, W, C)
        z = x * self.skip_scale + self.drop_path(self.vssm(self.ln1(x)))
        conv_out = self.conv_blk(self.ln2(z).permute(0, 3, 1, 2).contiguous())
        z = z * self.skip_scale2 + conv_out.permute(0, 2, 3, 1)
        return z.reshape(B, L, C)


# ======================================================================================
# Residual State-Space Group (RSSG)
# ======================================================================================
class ResidualGroup(nn.Module):
    """Stack of RSSBs + a final conv, with a group-level residual."""

    def __init__(self, dim, depth, d_state, mlp_ratio, is_light_sr=False, drop_paths=None):
        super().__init__()
        if drop_paths is None:
            drop_paths = [0.0] * depth
        self.blocks = nn.ModuleList([
            RSSB(dim, d_state=d_state, expand=mlp_ratio,
                 is_light_sr=is_light_sr, drop_path=drop_paths[i])
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, x_size):
        B, L, C = x.shape
        H, W = x_size
        shortcut = x
        for blk in self.blocks:
            x = blk(x, x_size)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x + shortcut


# ======================================================================================
# Upsampler
# ======================================================================================
class Upsample(nn.Sequential):
    """Pixel-shuffle upsampler (classic SR)."""

    def __init__(self, scale, num_feat):
        layers = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log2(scale))):
                layers += [nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1), nn.PixelShuffle(2)]
        elif scale == 3:
            layers += [nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1), nn.PixelShuffle(3)]
        else:
            raise ValueError(f"scale {scale} non supportato (solo 2^n e 3).")
        super().__init__(*layers)


class UpsampleOneStep(nn.Sequential):
    """Lightweight one-step upsampler (lightweight SR, 'pixelshuffledirect')."""

    def __init__(self, scale, num_feat, num_out_ch):
        super().__init__(
            nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1),
            nn.PixelShuffle(scale),
        )


class MambaIR(nn.Module):
    def __init__(self,
                 img_size=64, in_chans=3, embed_dim=96,
                 depths=(6, 6, 6, 6), d_state=16, mlp_ratio=2.0,
                 drop_rate=0.0, drop_path_rate=0.1,
                 upscale=2, img_range=1.0,
                 upsampler='pixelshuffle', resi_connection='1conv',
                 **kwargs):
        super().__init__()
        self.img_range = img_range
        self.upscale = upscale
        self.upsampler = upsampler
        num_out_ch = in_chans

        # Center RGB inputs with the dataset mean (DIV2K values, as in EDSR/SwinIR).
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.register_buffer('mean', torch.tensor(rgb_mean).view(1, 3, 1, 1))
        else:
            self.register_buffer('mean', torch.zeros(1, in_chans, 1, 1))

        is_light_sr = (upsampler == 'pixelshuffledirect')

        # 1) shallow feature extraction
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # 2) deep feature extraction
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.pos_drop = nn.Dropout(drop_rate)
        total_blocks = sum(depths)
        # Stochastic depth rate grows linearly from 0 to drop_path_rate across all blocks.
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        self.layers = nn.ModuleList()
        cur = 0
        for depth in depths:
            self.layers.append(ResidualGroup(
                dim=embed_dim, depth=depth, d_state=d_state, mlp_ratio=mlp_ratio,
                is_light_sr=is_light_sr, drop_paths=dpr[cur:cur + depth],
            ))
            cur += depth
        self.norm = nn.LayerNorm(embed_dim)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(0.2, True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(0.2, True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )
        else:
            raise ValueError(f"resi_connection sconosciuto: {resi_connection}")

        # 3) high-quality reconstruction
        if upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, 64, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, 64)
            self.conv_last = nn.Conv2d(64, num_out_ch, 3, 1, 1)
        elif upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)
        else:
            raise ValueError(f"upsampler sconosciuto: {upsampler}")

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

    def forward_features(self, x):
        B, C, H, W = x.shape
        x_size = (H, W)
        x = x.flatten(2).transpose(1, 2)             # (B, C, H, W) -> (B, L, C)
        x = self.patch_norm(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

    def forward(self, x):
        mean = self.mean.type_as(x)
        x = (x - mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        else:  # 'pixelshuffledirect'
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)

        x = x / self.img_range + mean
        return x


# Paper configs: lightweight SR and classic SR.
_LIGHT = dict(
    in_chans=3, img_size=64, img_range=1.0,
    d_state=10, depths=(6, 6, 6, 6), embed_dim=60, mlp_ratio=1.2,
    upsampler='pixelshuffledirect', resi_connection='1conv',
)
LIGHT_SR_X2 = dict(_LIGHT, upscale=2)
LIGHT_SR_X3 = dict(_LIGHT, upscale=3)
LIGHT_SR_X4 = dict(_LIGHT, upscale=4)

_CLASSIC = dict(
    in_chans=3, img_size=64, img_range=1.0,
    d_state=16, depths=(6, 6, 6, 6, 6, 6), embed_dim=180, mlp_ratio=2.0,
    upsampler='pixelshuffle', resi_connection='1conv',
)
CLASSIC_SR_X2 = dict(_CLASSIC, upscale=2)
CLASSIC_SR_X3 = dict(_CLASSIC, upscale=3)
CLASSIC_SR_X4 = dict(_CLASSIC, upscale=4)

_VARIANTS = {
    'light_sr_x2': LIGHT_SR_X2, 'light_sr_x3': LIGHT_SR_X3, 'light_sr_x4': LIGHT_SR_X4,
    'classic_sr_x2': CLASSIC_SR_X2, 'classic_sr_x3': CLASSIC_SR_X3, 'classic_sr_x4': CLASSIC_SR_X4,
}


def build_mambair(variant='classic_sr_x2', **overrides):
    """Build the model from a known config.

    variant: one of
        'light_sr_x2'  | 'light_sr_x3'  | 'light_sr_x4'
        'classic_sr_x2'| 'classic_sr_x3'| 'classic_sr_x4'
    overrides: optional parameters to override (e.g. upscale=3).
    """
    if variant not in _VARIANTS:
        raise ValueError(f"variant sconosciuta: {variant!r} (usa una di {list(_VARIANTS)})")
    cfg = dict(_VARIANTS[variant], **overrides)
    return MambaIR(**cfg)


# ======================================================================================
# Sanity check
# ======================================================================================
if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    size = 64 if cuda else 8
    device = 'cuda' if cuda else 'cpu'

    for name in ['light_sr_x2', 'light_sr_x3', 'light_sr_x4',
                 'classic_sr_x2', 'classic_sr_x3', 'classic_sr_x4']:
        model = build_mambair(name).to(device).eval()
        n_params = sum(p.numel() for p in model.parameters())
        x = torch.randn(1, 3, size, size, device=device)
        with torch.no_grad():
            y = model(x)
        print(f"{name:>14s} | parametri {n_params/1e6:6.3f}M | "
              f"in {tuple(x.shape)} -> out {tuple(y.shape)}")
