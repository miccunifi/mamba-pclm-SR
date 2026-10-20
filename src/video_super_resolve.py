"""
video_super_resolve.py — Super-resolve a LR video with a PCLM-SR model, in streaming mode.

Reads the low-res video one frame at a time, applies the model (its scale factor must
match) and writes the super-resolved video directly through an ffmpeg pipe. It does NOT
save any frame to disk: the only file produced is the final SR video.

The model is rebuilt from the checkpoint (arch_cfg) as in test.py; the model input is
the LR frame in [0,1] RGB (same convention as inference.py). The scale factor of the
model must match --scale.

Usage
-----
  python video_super_resolve.py --input clip_x4.mp4 --scale 4 \
      --ckpt ./checkpoints/pclmsr_small_x4_div2k/best.pth --output clip_x4_SR.mp4
  # large frames / small GPU: enable tiling
  python video_super_resolve.py --input clip_x4.mp4 --scale 4 --ckpt best.pth --tile 256

Requires torch (+ mamba_ssm for the fast scan) and ffmpeg/ffprobe. On CPU or without
mamba_ssm the inference is very slow.
"""
import argparse
import os
import subprocess

import numpy as np
import torch
import torchvision.transforms.functional as TF

from models import build_model, model_from_cfg, parse_overrides


def _extract_state_dict(ck):
    if isinstance(ck, dict):
        for key in ('model', 'params_ema', 'params', 'state_dict'):
            if key in ck:
                return ck[key]
    return ck


def load_model(ckpt_path, device, arch=None, variant=None, overrides=None):
    """Build the model and load the weights. Checkpoints written by train.py carry
    'arch' and 'arch_cfg', so the exact trained architecture is rebuilt from them;
    otherwise arch/variant/overrides from the CLI are used."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ck, dict) and ck.get('arch') and ck.get('arch_cfg'):
        model = model_from_cfg(ck['arch'], ck['arch_cfg'])
        info = [f"arch={ck['arch']}"]
        if ck['arch_cfg'].get('mamba_ratio') is not None:
            info.append(f"rho={ck['arch_cfg']['mamba_ratio']}")
        if ck.get('iter') is not None:
            info.append(f"iter={ck['iter']}")
        print(f"architettura ricostruita dal checkpoint: {' | '.join(info)}")
    else:
        if arch is None or variant is None:
            raise ValueError("checkpoint senza arch_cfg: specifica --arch e --variant")
        model, _ = build_model(arch, variant, **(overrides or {}))
    model.load_state_dict(_extract_state_dict(ck))
    return model.to(device).eval()


@torch.no_grad()
def sr_forward(model, lr, scale, tile=0, tile_overlap=32):
    """Super-resolve `lr` (1,3,H,W). With tile > 0 the frame is processed as
    overlapping tiles and the outputs are stitched (averaged on the overlaps)."""
    if tile <= 0:
        return model(lr).clamp(0, 1)
    b, c, h, w = lr.shape
    stride = max(tile - tile_overlap, 1)
    h_idx = list(range(0, max(h - tile, 0) + 1, stride)) or [0]
    w_idx = list(range(0, max(w - tile, 0) + 1, stride)) or [0]
    # append a border-aligned last tile so the whole frame is covered
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


def _probe_video(video):
    """width, height, fps, nb_frames, color_space (tag) via ffprobe."""
    import json
    try:
        out = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,color_space,color_transfer,color_primaries',
             '-show_entries', 'format=duration', '-of', 'json', video],
            capture_output=True, text=True, check=True).stdout
        data = json.loads(out)
        st = data.get('streams', [{}])[0]
        def frac(s):
            try:
                n, d = str(s).split('/'); return float(n) / float(d) if float(d) else 0.0
            except Exception:
                try:
                    return float(s)
                except Exception:
                    return 0.0
        fps = frac(st.get('avg_frame_rate', '0/1')) or frac(st.get('r_frame_rate', '0/1')) or 25.0
        try:
            total = int(st.get('nb_frames') or 0)
        except Exception:
            total = 0
        # nb_frames is missing for some containers: estimate it from duration * fps
        if not total:
            try:
                total = int(float(data.get('format', {}).get('duration', 0)) * fps)
            except Exception:
                total = 0
        return (int(st.get('width', 0)), int(st.get('height', 0)), fps, total,
                st.get('color_space'), st.get('color_transfer'), st.get('color_primaries'))
    except Exception as e:
        raise RuntimeError(f"ffprobe fallito su {video}: {e}")


def _resolve_matrix(choice, height, tag):
    """Effective colour matrix: explicit choice > stream tag > HD/SD heuristic."""
    if choice in ('bt601', 'bt709'):
        return choice
    if tag == 'bt709':
        return 'bt709'
    if tag in ('smpte170m', 'bt470bg'):
        return 'bt601'
    return 'bt709' if (height or 0) >= 720 else 'bt601'


def _pq_to_sdr(rgb01, white_nits=203.0, knee=0.75):
    """HDR10 -> SDR tone-mapping in numpy (independent of the ffmpeg filters, VALIDATED
    on synthetic PQ values: MAE ~2.6/255, exact hues): PQ (ST 2084) -> linear
    luminance -> BT.2020 primaries -> BT.709 -> "shoulder" curve (identity up to the
    knee, soft compression of the highlights only) -> sRGB.
    white_nits: luminance of the SDR reference white (BT.2408: 203 nit)."""
    # ST 2084 (PQ) EOTF constants
    m1, m2 = 0.1593017578125, 78.84375
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    Ep = np.clip(rgb01, 0.0, 1.0) ** (1.0 / m2)
    lin = (np.clip(Ep - c1, 0.0, None) / (c2 - c3 * Ep)) ** (1.0 / m1) * 10000.0  # nit
    M = np.array([[ 1.6605, -0.5876, -0.0728],
                  [-0.1246,  1.1329, -0.0083],
                  [-0.0182, -0.1006,  1.1187]])                                   # 2020 -> 709
    x = np.clip(lin.reshape(-1, 3) @ M.T, 0.0, None).reshape(rgb01.shape) / white_nits
    k = knee
    y = np.where(x <= k, x, k + (1.0 - k) * (1.0 - np.exp(-(x - k) / (1.0 - k))))
    y = np.clip(y, 0.0, 1.0)
    srgb = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)
    return (np.clip(srgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _open_reader(video, matrix='auto', hdr_mode='preserve', hdr_white=203.0):
    """Read RGB frames through an ffmpeg pipe, decoding with the CORRECT colour matrix
    (bt709 for HD content, bt601 for SD, or the one tagged in the stream).

    HDR sources (HDR10/HLG: BT.2020 + PQ/HLG) are detected from the tags:
      - 'preserve' (default, safe): values untouched, bt2020 matrix, HDR tags
        propagated to the output (LR and SR look the same in the player).
      - 'tonemap': HDR -> SDR BT.709 conversion done IN NUMPY (_pq_to_sdr):
        ffmpeg only provides the raw 16-bit PQ values (matrix-only conversion),
        the tone-mapping math is ours and does not depend on the version of
        the ffmpeg filters.
    Returns (fps, total, frame_generator, matrix_used)."""
    w, h, fps, total, tag, trc, prim = _probe_video(video)
    if not (w and h):
        raise RuntimeError(f"dimensioni non rilevate per {video}")
    # HDR if the transfer is PQ/HLG or the stream is tagged BT.2020
    is_hdr = (trc in ('smpte2084', 'arib-std-b67')) or tag == 'bt2020nc' or prim == 'bt2020'
    post = None
    # swscale's default YUV->RGB path truncates instead of rounding: the decoded frames come
    # out ~1.7/255 too dark (10-bit input) or ~1.0/255 (8-bit input). full_chroma_int +
    # accurate_rnd make the conversion exact to rounding at negligible CPU cost.
    sws = ':flags=full_chroma_int+accurate_rnd'
    if is_hdr and hdr_mode == 'tonemap':
        vf, pixfmt, bpp = 'scale=in_color_matrix=bt2020' + sws, 'rgb48le', 6
        post = lambda a: _pq_to_sdr(a, white_nits=hdr_white)
        mtx = 'bt709'
        print(f"sorgente HDR rilevata ({tag}/{trc}) -> tone-mapping numpy a SDR BT.709 "
              f"(bianco riferimento {hdr_white:.0f} nit)")
    elif is_hdr:
        vf, pixfmt, bpp = 'scale=in_color_matrix=bt2020' + sws, 'rgb24', 3
        mtx = 'bt2020'
        print(f"sorgente HDR rilevata ({tag}/{trc}) -> modalita' preserve (tag HDR propagati)")
    else:
        mtx = _resolve_matrix(matrix, h, tag)
        vf, pixfmt, bpp = f'scale=in_color_matrix={mtx}' + sws, 'rgb24', 3
    proc = subprocess.Popen(
        ['ffmpeg', '-v', 'error', '-i', video, '-map', '0:v:0',
         '-vf', vf,
         '-f', 'rawvideo', '-pix_fmt', pixfmt, '-'],
        stdout=subprocess.PIPE)

    def gen():
        fsize = w * h * bpp
        try:
            while True:
                buf = proc.stdout.read(fsize)
                if len(buf) < fsize:
                    break
                if bpp == 6:
                    fr = np.frombuffer(buf, '<u2').reshape(h, w, 3).astype(np.float64) / 65535.0
                    yield post(fr)
                else:
                    yield np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
        finally:
            # clean shutdown also on early stop (e.g. --max_frames)
            try:
                proc.stdout.close()
            except Exception:
                pass
            if proc.poll() is None:
                proc.kill()
            proc.wait()
    return fps, total, gen(), mtx


_MATRIX_TAGS = {
    # matrix -> (colorspace, primaries, trc) for tagging the stream
    'bt709': ('bt709', 'bt709', 'bt709'),
    'bt601': ('smpte170m', 'smpte170m', 'smpte170m'),
    'bt2020': ('bt2020nc', 'bt2020', 'smpte2084'),   # HDR10 (preserve mode)
}


def _open_writer(path, w, h, fps, crf, lossless, codec, audio_src, matrix='bt709'):
    """Write the SR video from RGB rawvideo with an EXPLICIT-matrix RGB->YUV conversion
    and a TAGGED stream: without tags the players pick the matrix by resolution
    (SD->601, HD->709) and the colours of the SR video come out warm/saturated."""
    cs, pr, trc = _MATRIX_TAGS[matrix]
    is_nvenc = 'nvenc' in codec
    cmd = ['ffmpeg', '-y', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-']
    if audio_src:
        cmd += ['-i', audio_src]
    cmd += ['-map', '0:v:0']
    if audio_src:
        cmd += ['-map', '1:a:0?', '-c:a', 'aac', '-shortest']
    cmd += ['-c:v', codec]
    # ffmpeg >= 7 takes the colour properties of the encoded stream from the frames, not from
    # the -color_* output options, and rawvideo frames carry none: without this explicit tagging
    # the output ends up with color_transfer/color_primaries = unknown (verified with 7.1.1),
    # which for HDR sources means the SR video would be displayed as SDR.
    vf_conv = (f'scale=out_color_matrix={matrix}:out_range=tv,'
               f'setparams=range=tv:color_primaries={pr}:color_trc={trc}:colorspace={cs}')
    vf_crop = f'crop=trunc(iw/2)*2:trunc(ih/2)*2,{vf_conv}'
    if is_nvenc:
        # GPU hardware encoder (very fast at 4K)
        if lossless:
            cmd += ['-vf', vf_conv, '-preset', 'lossless', '-pix_fmt', 'yuv444p']
        else:
            cmd += ['-vf', vf_crop,
                    '-preset', 'p5', '-tune', 'hq', '-rc', 'vbr', '-cq', str(crf), '-b:v', '0',
                    '-pix_fmt', 'yuv420p']
    else:
        if lossless:
            cmd += ['-vf', vf_conv, '-qp', '0', '-pix_fmt', 'yuv444p']
        else:
            # yuv420p needs even dimensions: crop to even (no-op if already even)
            cmd += ['-vf', vf_crop, '-crf', str(crf), '-pix_fmt', 'yuv420p']
    cmd += ['-colorspace', cs, '-color_primaries', pr, '-color_trc', trc, '-color_range', 'tv',
            '-r', str(fps), path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Super-risoluzione video con PCLM-SR, in streaming")
    p.add_argument('--input', required=True, help="video low-res di input")
    p.add_argument('--ckpt', required=True, help="checkpoint del modello (arch dedotta da arch_cfg)")
    p.add_argument('--scale', type=int, default=None, help="fattore di scala (default: dedotto dal modello)")
    p.add_argument('--arch', default=None, choices=['mambair', 'pclmsr'])
    p.add_argument('--variant', default=None)
    p.add_argument('--arch_override', nargs='*', default=None)
    p.add_argument('--output', default=None, help="video SR di output (default: <input>_SR.<container>)")
    p.add_argument('--tile', type=int, default=0, help=">0 abilita il tiling (px LR per riquadro)")
    p.add_argument('--batch', type=int, default=4,
                   help="frame elaborati insieme in una forward (riempie la GPU). Se OOM, abbassa")
    p.add_argument('--prefetch', type=int, default=4,
                   help="buffer di frame per sovrapporre decodifica/codifica al calcolo GPU "
                        "(>=2 attiva il pipelining a thread; 0/1 = sequenziale)")
    p.add_argument('--precision', choices=['fp32', 'bf16', 'fp16'], default='fp32',
                   help="precisione di calcolo (autocast). CONSIGLIATO: bf16 sulla RTX 30/40 "
                        "(veloce e numericamente stabile). fp16 e' INSTABILE con i modelli "
                        "Mamba (overflow nella selective scan -> output collassato alla media, "
                        "grigio con bordi fantasma): evitalo. Default fp32.")
    p.add_argument('--half', action='store_true',
                   help="DEPRECATO: alias di --precision bf16 (il vecchio FP16 puro corrompe "
                        "l'output dei modelli Mamba)")
    p.add_argument('--compile', action='store_true',
                   help="torch.compile del modello (fusione kernel; primo batch lento per la compilazione)")
    p.add_argument('--channels_last', action='store_true',
                   help="layout channels_last (conv piu' veloci sui tensor core)")
    p.add_argument('--crf', type=int, default=15, help="qualita' H.264 del video SR (default 15; piu' basso=migliore)")
    p.add_argument('--lossless', action='store_true', help="video SR senza perdita (yuv444p, qp 0): grande")
    p.add_argument('--codec', default='libx264')
    p.add_argument('--container', default='mp4')
    p.add_argument('--keep_audio', action='store_true', help="copia l'audio del video LR nel video SR")
    p.add_argument('--hdr_mode', choices=['preserve', 'tonemap'], default='preserve',
                   help="per sorgenti HDR (BT.2020/PQ o HLG): 'preserve' (default, SICURO) non tocca "
                        "i valori e propaga i tag HDR cosi' LR e SR appaiono uguali nel player; "
                        "'tonemap' (validato, in numpy) converte a SDR BT.709")
    p.add_argument('--hdr_white', type=float, default=203.0,
                   help="(solo --hdr_mode tonemap) luminanza in nit del bianco SDR di riferimento "
                        "(default 203, BT.2408). Piu' alto = immagine piu' scura, piu' basso = piu' chiara")
    p.add_argument('--color_matrix', choices=['auto', 'bt709', 'bt601'], default='auto',
                   help="matrice colore YUV<->RGB in lettura e scrittura: auto = tag dello stream "
                        "o euristica (HD->bt709, SD->bt601). L'output viene taggato di conseguenza")
    p.add_argument('--fps', type=float, default=0.0, help="forza l'fps di output (default: quello del sorgente)")
    p.add_argument('--max_frames', type=int, default=0, help=">0 limita i frame (per test)")
    p.add_argument('--no_progress', action='store_true')
    args = p.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"video non trovato: {args.input}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True   # fixed sizes -> conv autotuning
    model = load_model(args.ckpt, device, arch=args.arch, variant=args.variant,
                       overrides=parse_overrides(args.arch_override))

    # precision: weights ALWAYS in fp32; bf16/fp16 only via autocast (stable).
    prec = args.precision
    if args.half and prec == 'fp32':
        prec = 'bf16'
        print("nota: --half e' deprecato -> uso --precision bf16 (il FP16 puro collassa i modelli Mamba).")
    if prec == 'bf16' and device == 'cuda' and not torch.cuda.is_bf16_supported():
        print("ATTENZIONE: bf16 non supportato da questa GPU -> ripiego su fp32.")
        prec = 'fp32'
    if prec == 'fp16':
        print("ATTENZIONE: fp16 con modelli Mamba e' numericamente instabile (output grigio/collassato);"
              " se vedi artefatti usa --precision bf16 o fp32.")
    amp_dtype = {'fp32': None, 'bf16': torch.bfloat16, 'fp16': torch.float16}[prec]
    use_amp = amp_dtype is not None and device == 'cuda'

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        model = torch.compile(model)
    up = int(getattr(model, 'upscale', args.scale or 0))
    scale = args.scale or up
    if scale != up:
        raise SystemExit(f"il modello ha upscale x{up} ma hai chiesto x{scale}: usa un modello concorde con la scala")

    fps_src, total, frames, matrix = _open_reader(args.input, args.color_matrix, args.hdr_mode, args.hdr_white)
    fps = args.fps or fps_src
    audio_src = os.path.abspath(args.input) if args.keep_audio else None

    gen_iter = iter(frames)
    try:
        first = next(gen_iter)
    except StopIteration:
        raise SystemExit("il video non contiene frame leggibili")
    lr_h, lr_w = first.shape[:2]
    out_w, out_h = lr_w * scale, lr_h * scale

    stem = os.path.splitext(os.path.basename(args.input))[0]
    out_path = args.output or os.path.join(os.path.dirname(os.path.abspath(args.input)),
                                           f"{stem}_SR.{args.container}")
    writer = _open_writer(out_path, out_w, out_h, fps, args.crf, args.lossless, args.codec, audio_src, matrix=matrix)

    n_params = sum(x.numel() for x in model.parameters()) / 1e6
    batch = max(1, args.batch)
    use_batch = batch > 1 and args.tile == 0
    opts = [prec if use_amp or prec == 'fp32' else f"{prec}(cpu:off)"]
    if args.channels_last:
        opts.append('channels_last')
    if args.compile:
        opts.append('compile')
    print(f"modello: {n_params:.3f}M param | scale x{scale} | device={device} | {'+'.join(opts)} "
          f"| batch={batch if use_batch else 1} | tile={args.tile or 'no'}")
    print(f"LR {lr_w}x{lr_h} -> SR {out_w}x{out_h} | fps {fps:.3f} | frame {total or '?'}"
          f" | pipeline={'thread (prefetch %d)' % args.prefetch if args.prefetch >= 2 else 'sequenziale'}")
    enc = f"NVENC ({args.codec})" if 'nvenc' in args.codec else args.codec
    print(f"encoder: {enc} | matrice colore: {matrix} (taggata) | "
          f"qualita': {'LOSSLESS' if args.lossless else f'cq/crf={args.crf}'}"
          f"{' + audio' if args.keep_audio else ''} -> {out_path}")
    if args.tile and batch > 1:
        print("  nota: tiling attivo -> batching disabilitato (elaboro un frame per volta).")
    if device != 'cuda':
        print("ATTENZIONE: device=CPU (senza mamba_ssm l'inferenza e' lentissima).")
    print("-" * 60)

    # frames expected by the progress bar: capped by --max_frames when set
    total_expected = (min(total, args.max_frames) if args.max_frames else total) if total else (args.max_frames or None)
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None
    pbar = tqdm(total=total_expected, unit='frame', desc='SR') if (tqdm and not args.no_progress) else None

    def all_frames():
        yield first
        yield from gen_iter

    _nan_warned = [False]

    def _forward(x):
        """Forward with autocast (if enabled) + guard: if the output is not finite
        (overflow in low precision), redo the computation in fp32."""
        with torch.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            y = model(x)
        y = y.float()
        if use_amp and not torch.isfinite(y).all():
            if not _nan_warned[0]:
                _nan_warned[0] = True
                print("\nATTENZIONE: output non finito in bassa precisione -> ricalcolo in fp32 "
                      "(se accade spesso, usa --precision fp32).", flush=True)
            with torch.autocast('cuda', enabled=False):
                y = model(x).float()
        return y

    def _sr_batch(buf):
        """buf: list of CPU tensors (1,3,H,W) [0,1] -> list of RGB bytes (one per frame)."""
        x = torch.cat(buf, 0).pin_memory().to(device, non_blocking=True)
        if args.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        y = _forward(x).clamp(0, 1)
        y = (y * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
        return [np.ascontiguousarray(y[i]).tobytes() for i in range(y.shape[0])]

    def _sr_one(lr_t):
        """tiling path: one frame -> bytes."""
        lr = lr_t.pin_memory().to(device, non_blocking=True)
        if args.channels_last:
            lr = lr.contiguous(memory_format=torch.channels_last)
        with torch.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            sr = sr_forward(model, lr, scale, tile=args.tile)
        sr = sr.float().clamp(0, 1)
        if use_amp and not torch.isfinite(sr).all():
            with torch.autocast('cuda', enabled=False):
                sr = sr_forward(model, lr, scale, tile=args.tile).float().clamp(0, 1)
        a = (sr.squeeze(0) * 255.0).round().to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
        return np.ascontiguousarray(a).tobytes()

    def _process(frame_source, sink):
        """Consume the frames, run SR (batched or per-frame) and pass the bytes to sink().
        Returns n."""
        n = 0
        buf = []

        def flush():
            nonlocal n
            if not buf:
                return
            for b in _sr_batch(buf):
                sink(b)
            n += len(buf)
            if pbar is not None:
                pbar.update(len(buf))
            buf.clear()

        cnt = 0
        for rgb in frame_source:
            cnt += 1
            lr_t = TF.to_tensor(rgb).unsqueeze(0)          # CPU (1,3,H,W) [0,1]
            if use_batch:
                buf.append(lr_t)
                if len(buf) >= batch:
                    flush()
            else:
                sink(_sr_one(lr_t))
                n += 1
                if pbar is not None:
                    pbar.update(1)
            if pbar is None and not args.no_progress and cnt % 20 == 0:
                print(f"  {cnt} frame...", flush=True)
            if args.max_frames and cnt >= args.max_frames:
                break
        flush()
        return n

    model.eval()
    prefetch = args.prefetch
    n = 0
    try:
        if prefetch >= 2:
            # ---- threaded pipeline: decoding and encoding overlap the GPU compute ----
            import threading
            import queue
            _SENT = object()
            in_q = queue.Queue(maxsize=prefetch)                     # decoded LR frames
            out_q = queue.Queue(maxsize=2 * batch + prefetch)        # SR bytes ready to write
            err = threading.Event()

            def _reader():
                try:
                    c = 0
                    for rgb in all_frames():
                        if err.is_set():
                            break
                        in_q.put(rgb)
                        c += 1
                        if args.max_frames and c >= args.max_frames:
                            break
                finally:
                    in_q.put(_SENT)

            def _writer():
                while True:
                    item = out_q.get()
                    if item is _SENT:
                        break
                    if err.is_set():
                        continue   # after an error keep draining, so the producer never blocks
                    try:
                        writer.stdin.write(item)
                    except Exception:
                        # Any write failure (broken pipe, OS error) aborts the run. The thread must
                        # stay alive and keep draining, otherwise out_q.put() would block forever.
                        err.set()

            rt = threading.Thread(target=_reader, daemon=True)
            wt = threading.Thread(target=_writer, daemon=True)
            rt.start()
            wt.start()

            src_done = [False]

            def _src():
                # feeds _process from in_q; stops at the sentinel or on a writer error
                while True:
                    item = in_q.get()
                    if item is _SENT:
                        src_done[0] = True
                        break
                    if err.is_set():
                        break
                    yield item

            try:
                n = _process(_src(), out_q.put)
            finally:
                # Shut both threads down cleanly even after an error. The writer drains out_q up
                # to its sentinel. The reader may be blocked on a full in_q: drain it until its
                # sentinel (it stops decoding as soon as err is set), or rt.join() would hang.
                out_q.put(_SENT)
                wt.join()
                if not src_done[0]:
                    err.set()
                    while in_q.get() is not _SENT:
                        pass
                rt.join()
            if err.is_set():
                raise BrokenPipeError
        else:
            # ---- sequential (no threads) ----
            n = _process(all_frames(), writer.stdin.write)
    except BrokenPipeError:
        raise SystemExit("ffmpeg ha chiuso la pipe (errore di codifica).")
    finally:
        if pbar is not None:
            pbar.close()
        try:
            writer.stdin.close()
        except Exception:
            pass
    code = writer.wait()

    print("-" * 60)
    ok = code == 0 and os.path.isfile(out_path)
    size = os.path.getsize(out_path) / 1e6 if os.path.isfile(out_path) else 0
    print(f"{'OK' if ok else f'ERRORE (ffmpeg code {code})'} | {n} frame | {size:.1f} MB | {out_path}")


if __name__ == '__main__':
    main()