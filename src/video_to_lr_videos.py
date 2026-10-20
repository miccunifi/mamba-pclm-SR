"""
video_to_lr_videos.py — From an HR video, generate the bicubic (MATLAB) LR videos at x2/x3/x4.

Everything runs in STREAMING mode: it reads frames one at a time, applies the same MATLAB
bicubic used in training (make_lr_bicubic.downscale_to_uint8) and writes them directly
into one LR video per scale through an ffmpeg pipe. It does NOT save any frame to disk:
it uses very little space and memory (one frame at a time).

Output (in the output folder):
  <video_name>_x2.mp4 , <video_name>_x3.mp4 , <video_name>_x4.mp4

Usage
-----
  python video_to_lr_videos.py --input clip.mp4 --scales 2 3 4
  python video_to_lr_videos.py --input clip.mp4 --scales 2 3 4 --workers 8   # parallel on 8 cores
  python video_to_lr_videos.py --input clip.mp4 --output_dir ./lr --crf 15 --keep_audio

Note on fidelity: for super-resolution the LR frames should be as free as possible from
compression artifacts (the model is trained on clean LR PNGs).
For this reason the DEFAULT is lossless (yuv444p, qp 0): the LR frames are practically
identical to the MATLAB bicubic (difference < ~1/255, only due to RGB<->YUV rounding).
With --crf N the files are smaller but the LR frames are less faithful (not recommended for SR).
Requires ffmpeg/ffprobe (OpenCV is no longer needed).
"""
import os
# Limit BLAS threads per process: parallelism comes from the workers (avoids oversubscription).
# setdefault -> the user can still override it from the environment.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import math
import subprocess
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import reduce

import numpy as np

from make_lr_bicubic import mod_crop, downscale_to_uint8


def _mp_worker(task):
    """Compute the LR frames (as bytes) for all scales. Runs in the worker processes."""
    frame, scales, L, no_mod_crop = task
    hr = frame if no_mod_crop else mod_crop(frame, L)
    out = []
    for s in scales:
        # with --no_mod_crop each scale crops only to its own multiple, so the LR size is exact
        base = mod_crop(frame, s) if no_mod_crop else hr
        out.append(np.ascontiguousarray(downscale_to_uint8(base, s)).tobytes())
    return out


def _lcm_list(nums):
    return reduce(lambda a, b: a * b // math.gcd(a, b), nums, 1)


def _probe_video(video):
    """width, height, fps, nb_frames, color_space (tag) via ffprobe."""
    import json
    try:
        out = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,color_space',
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
        # nb_frames is missing in some containers: fall back to duration * fps
        if not total:
            try:
                total = int(float(data.get('format', {}).get('duration', 0)) * fps)
            except Exception:
                total = 0
        return int(st.get('width', 0)), int(st.get('height', 0)), fps, total, st.get('color_space')
    except Exception as e:
        raise RuntimeError(f"ffprobe fallito su {video}: {e}")


def _resolve_matrix(choice, height, tag):
    """Effective color matrix: explicit choice > stream tag > HD/SD heuristic."""
    if choice in ('bt601', 'bt709'):
        return choice
    if tag == 'bt709':
        return 'bt709'
    if tag in ('smpte170m', 'bt470bg'):
        return 'bt601'
    return 'bt709' if (height or 0) >= 720 else 'bt601'


def _open_reader(video, matrix='auto'):
    """Read RGB frames through an ffmpeg pipe, decoding with the CORRECT color matrix
    (bt709 for HD content, bt601 for SD, or the one tagged in the stream). OpenCV always
    uses bt601 and ignores the tags: on HD (bt709) sources it yields RGB with shifted hues.
    Returns (fps, total, frame_generator, matrix_used)."""
    w, h, fps, total, tag = _probe_video(video)
    if not (w and h):
        raise RuntimeError(f"dimensioni non rilevate per {video}")
    mtx = _resolve_matrix(matrix, h, tag)
    proc = subprocess.Popen(
        ['ffmpeg', '-v', 'error', '-i', video, '-map', '0:v:0',
         '-vf', f'scale=in_color_matrix={mtx}',
         '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
        stdout=subprocess.PIPE)

    def gen():
        fsize = w * h * 3
        try:
            while True:
                buf = proc.stdout.read(fsize)
                if len(buf) < fsize:
                    break
                yield np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
        finally:
            # clean shutdown also on early stop (e.g. --max_frames)
            try:
                proc.stdout.close()
            except Exception:
                pass
            if proc.poll() is None:
                proc.kill()          # kill, not terminate: terminate prints pipe errors
            proc.wait()
    return fps, total, gen(), mtx


_MATRIX_TAGS = {
    # matrix -> (colorspace, primaries, trc) for tagging the stream
    'bt709': ('bt709', 'bt709', 'bt709'),
    'bt601': ('smpte170m', 'smpte170m', 'smpte170m'),
}


def _open_writer(path, w, h, fps, crf, lossless, codec, audio_src, matrix='bt709'):
    """Write a video from RGB rawvideo with an EXPLICIT-matrix RGB->YUV conversion and a
    TAGGED stream (colorspace/primaries/trc/range): without tags, players pick the matrix
    from the resolution (SD->601, HD->709) and the colors differ between LR and SR
    (warm/saturated hues). With the tags the interpretation is always unambiguous."""
    cs, pr, trc = _MATRIX_TAGS[matrix]
    cmd = ['ffmpeg', '-y', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-']
    if audio_src:
        cmd += ['-i', audio_src]
    cmd += ['-map', '0:v:0']
    if audio_src:
        cmd += ['-map', '1:a:0?', '-c:a', 'aac', '-shortest']
    cmd += ['-c:v', codec]
    if lossless:
        cmd += ['-vf', f'scale=out_color_matrix={matrix}:out_range=tv',
                '-qp', '0', '-pix_fmt', 'yuv444p']          # exact: no loss, no chroma subsampling
    else:
        cmd += ['-vf', f'crop=trunc(iw/2)*2:trunc(ih/2)*2,scale=out_color_matrix={matrix}:out_range=tv',
                '-crf', str(crf), '-pix_fmt', 'yuv420p']
    cmd += ['-colorspace', cs, '-color_primaries', pr, '-color_trc', trc, '-color_range', 'tv',
            '-r', str(fps), path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main():
    p = argparse.ArgumentParser(description="Video HR -> video LR bicubici (MATLAB) a piu' scale, in streaming")
    p.add_argument('--input', required=True, help="video HR di input")
    p.add_argument('--output_dir', default=None, help="cartella di output (default: quella del video)")
    p.add_argument('--scales', type=int, nargs='+', default=[2, 3, 4])
    p.add_argument('--suffix', default='_x{scale}', help="suffisso nome output (default: _x<scala>)")
    p.add_argument('--container', default='mp4', help="estensione/container di output (default mp4)")
    p.add_argument('--codec', default='libx264', help="codec video ffmpeg (default libx264)")
    p.add_argument('--crf', type=int, default=None,
                   help="se impostato usa H.264 lossy a questo CRF (file piu' piccoli ma LR meno fedeli "
                        "al bicubico); default: LOSSLESS (fedele, consigliato per la super-risoluzione)")
    p.add_argument('--keep_audio', action='store_true', help="copia l'audio dell'originale nei video LR")
    p.add_argument('--no_mod_crop', action='store_true',
                   help="non ritagliare a multiplo di lcm(scale) (LR non piu' esatte)")
    p.add_argument('--max_frames', type=int, default=0, help=">0 limita i frame (per test)")
    p.add_argument('--fps', type=float, default=0.0, help="forza l'fps di output (default: quello del sorgente)")
    p.add_argument('--no_progress', action='store_true', help="disabilita la barra tqdm")
    p.add_argument('--color_matrix', choices=['auto', 'bt709', 'bt601'], default='auto',
                   help="matrice colore YUV<->RGB: auto = tag dello stream o euristica "
                        "(HD->bt709, SD->bt601). I video di output vengono taggati di conseguenza")
    p.add_argument('--workers', type=int, default=os.cpu_count() or 1,
                   help="processi paralleli per il bicubico (default: tutti i core CPU). 1 = sequenziale")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"video non trovato: {args.input}")
    scales = args.scales
    L = _lcm_list(scales) if not args.no_mod_crop else 1
    stem = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.output_dir or (os.path.dirname(os.path.abspath(args.input)))
    os.makedirs(out_dir, exist_ok=True)

    fps_src, total, frames, matrix = _open_reader(args.input, args.color_matrix)
    fps = args.fps or fps_src
    audio_src = os.path.abspath(args.input) if args.keep_audio else None
    lossless = args.crf is None

    gen_iter = iter(frames)
    try:
        first = next(gen_iter)
    except StopIteration:
        raise SystemExit("il video non contiene frame leggibili")

    hr0 = mod_crop(first, L) if not args.no_mod_crop else first
    Hc, Wc = hr0.shape[:2]

    # open one ffmpeg writer per scale, with the correct LR size
    writers, out_paths = {}, {}
    for s in scales:
        w, h = Wc // s, Hc // s
        out_paths[s] = os.path.join(out_dir, f"{stem}{args.suffix.format(scale=s)}.{args.container}")
        writers[s] = _open_writer(out_paths[s], w, h, fps, (args.crf or 0), lossless, args.codec, audio_src, matrix=matrix)

    print(f"video: {args.input}")
    print(f"fps: {fps:.3f} | frame: {total or '?'} | HR ritagliato: {Wc}x{Hc} (multiplo di {L}) "
          f"| matrice colore: {matrix} (output taggato)")
    print(f"qualita': {'LOSSLESS (yuv444p, fedele al bicubico)' if lossless else f'H.264 lossy crf={args.crf}'}"
          f"{' + audio' if args.keep_audio else ''}")
    if not lossless:
        print("  ATTENZIONE: encoding lossy -> i frame LR si discostano dal bicubico esatto "
              "(puo' penalizzare la super-risoluzione). Ometti --crf per LR fedeli.")
    for s in scales:
        print(f"  x{s}: {Wc//s}x{Hc//s} -> {out_paths[s]}")
    print(f"workers: {max(1, args.workers)} (bicubico su CPU; la GPU non viene usata)")
    print("-" * 60)

    # expected frame count for the progress bar (None when unknown)
    total_expected = (min(total, args.max_frames) if args.max_frames else total) if total else (args.max_frames or None)
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None
    pbar = tqdm(total=total_expected, unit='frame', desc='frame') if (tqdm and not args.no_progress) else None

    def all_frames():
        yield first
        yield from gen_iter

    def _wrote(cnt):
        if pbar is not None:
            pbar.update(1)
        elif not args.no_progress and cnt % 50 == 0:
            print(f"  {cnt} frame...", flush=True)

    workers = max(1, args.workers)
    frames_iter = all_frames()
    n = 0
    try:
        if workers > 1:
            # parallel pipeline: at most ~workers*3 frames in flight (bounded RAM), order preserved
            src = ((rgb, scales, L, args.no_mod_crop) for rgb in frames_iter)
            with ProcessPoolExecutor(max_workers=workers) as ex:
                inflight = deque()
                for _ in range(workers * 3):
                    try:
                        inflight.append(ex.submit(_mp_worker, next(src)))
                    except StopIteration:
                        break
                while inflight:
                    outs = inflight.popleft().result()      # results in submission order
                    for i, s in enumerate(scales):
                        writers[s].stdin.write(outs[i])
                    n += 1
                    _wrote(n)
                    if args.max_frames and n >= args.max_frames:
                        break
                    try:
                        inflight.append(ex.submit(_mp_worker, next(src)))
                    except StopIteration:
                        pass
        else:
            for rgb in frames_iter:
                hr = mod_crop(rgb, L) if not args.no_mod_crop else rgb
                for s in scales:
                    base = hr if not args.no_mod_crop else mod_crop(rgb, s)
                    writers[s].stdin.write(np.ascontiguousarray(downscale_to_uint8(base, s)).tobytes())
                n += 1
                _wrote(n)
                if args.max_frames and n >= args.max_frames:
                    break
    except BrokenPipeError:
        raise SystemExit("ffmpeg ha chiuso la pipe (errore di codifica). Controlla codec/percorsi.")
    finally:
        if pbar is not None:
            pbar.close()
        for s in scales:
            try:
                writers[s].stdin.close()
            except Exception:
                pass
    codes = {s: writers[s].wait() for s in scales}

    print("-" * 60)
    ok = True
    for s in scales:
        status = "OK" if codes[s] == 0 and os.path.isfile(out_paths[s]) else f"ERRORE (ffmpeg code {codes[s]})"
        if codes[s] != 0:
            ok = False
        size = os.path.getsize(out_paths[s]) / 1e6 if os.path.isfile(out_paths[s]) else 0
        print(f"  x{s}: {status} | {size:.1f} MB | {out_paths[s]}")
    print(f"{n} frame elaborati a {fps:.3f} fps." + ("" if ok else "  (alcuni output non validi)"))


if __name__ == '__main__':
    main()