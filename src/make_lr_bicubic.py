#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_lr_bicubic.py
==================
Generate the bicubic LR (low-resolution) versions of an HR dataset (e.g. Manga109)
at scales X2, X3, X4, following the DIV2K/EDSR/BasicSR convention.

Features:
  * MATLAB-compatible bicubic (cubic kernel + antialiasing): it produces EXACTLY
    the same pixels as `imresize(img, 1/scale, 'bicubic')`, so the results are
    comparable with published SR benchmarks.
  * mod-crop per scale: the HR is cropped so that H and W are divisible by the
    scale; then LR = HR/scale exactly (needed for clean HR/LR pairs).
  * Preserves the sub-folder structure (one folder per manga).
  * Optional multiprocessing + progress bar.

Typical usage (Manga109):
    python make_lr_bicubic.py /path/to/images
    python make_lr_bicubic.py /path/to/images --out /path/to/output --scales 2 3 4 --workers 8

Output layout (default, --out next to the input folder):
    output/
    ├── LR_bicubic/
    │   ├── X2/AisazuNihaIrarenai/000.png ...
    │   ├── X3/AisazuNihaIrarenai/000.png ...
    │   └── X4/AisazuNihaIrarenai/000.png ...
    └── HR/AisazuNihaIrarenai/000.png ...        (only with --save-hr; HR/X<s>/ if several scales)

Notes:
  * Default output is PNG (lossless) to avoid double JPEG compression.
  * With --div2k-naming the LR files are renamed as <name>x<scale>.png
    (the DIV2K convention where everything is "flat").
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# tqdm is optional: if it is missing, use a "silent" fallback
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kwargs):
        return it


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm"}


# ---------------------------------------------------------------------------
# MATLAB-compatible bicubic (NumPy port of imresize, as in BasicSR/mmsr)
# ---------------------------------------------------------------------------
def _cubic(x):
    # cubic convolution kernel with a = -0.5 (MATLAB's 'bicubic')
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return ((1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1)
            + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((absx > 1) & (absx <= 2)))


def _calculate_weights_indices(in_length, out_length, scale, kernel_width, antialiasing):
    if (scale < 1) and antialiasing:
        # antialiasing: widen the kernel to filter before downsampling
        kernel_width = kernel_width / scale

    # map each output pixel center to input coordinates (MATLAB convention, 1-based)
    x = np.linspace(1, out_length, out_length)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(np.ceil(kernel_width)) + 2

    indices = left.reshape(out_length, 1) + np.arange(P).reshape(1, P)          # out_length x P
    distance_to_center = u.reshape(out_length, 1) - indices                      # out_length x P

    if (scale < 1) and antialiasing:
        weights = scale * _cubic(distance_to_center * scale)
    else:
        weights = _cubic(distance_to_center)

    weights_sum = np.sum(weights, axis=1).reshape(out_length, 1)
    weights = weights / weights_sum

    # drop the all-zero columns at the borders (P is oversized by ~2)
    weights_zero_tmp = np.sum((weights == 0), axis=0)
    if not np.isclose(weights_zero_tmp[0], 0, rtol=1e-6):
        indices = indices[:, 1:1 + P - 2]
        weights = weights[:, 1:1 + P - 2]
    if not np.isclose(weights_zero_tmp[-1], 0, rtol=1e-6):
        indices = indices[:, 0:0 + P - 2]
        weights = weights[:, 0:0 + P - 2]

    weights = np.ascontiguousarray(weights)
    indices = np.ascontiguousarray(indices)

    # mirror padding needed before the first and after the last input pixel
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length + 1
    indices = indices + sym_len_s - 1          # -> 0-based indices into the mirror-padded array
    return weights, indices.astype(np.int64), int(sym_len_s), int(sym_len_e)


def imresize_matlab(img, scale, antialiasing=True):
    """Replica of MATLAB imresize (bicubic). img: HxW or HxWxC ndarray, float64.
    Returns a float64 ndarray (not clamped)."""
    squeeze = False
    if img.ndim == 2:
        img = img[:, :, None]
        squeeze = True
    img = img.astype(np.float64)
    in_H, in_W, in_C = img.shape
    out_H = int(np.ceil(in_H * scale))
    out_W = int(np.ceil(in_W * scale))
    kernel_width = 4  # support of the bicubic kernel

    w_H, idx_H, s_Hs, s_He = _calculate_weights_indices(in_H, out_H, scale, kernel_width, antialiasing)
    w_W, idx_W, s_Ws, s_We = _calculate_weights_indices(in_W, out_W, scale, kernel_width, antialiasing)

    # --- step 1: resize along H, with symmetric (mirror) padding ---
    aug = np.zeros((in_H + s_Hs + s_He, in_W, in_C))
    aug[s_Hs:s_Hs + in_H] = img
    if s_Hs > 0:
        aug[:s_Hs] = img[:s_Hs][::-1]
    if s_He > 0:
        aug[s_Hs + in_H:] = img[-s_He:][::-1]

    out1 = np.zeros((out_H, in_W, in_C))
    kw_H = w_H.shape[1]
    for i in range(out_H):
        a = int(idx_H[i, 0])
        out1[i] = np.tensordot(w_H[i], aug[a:a + kw_H], axes=([0], [0]))

    # --- step 2: resize along W ---
    aug2 = np.zeros((out_H, in_W + s_Ws + s_We, in_C))
    aug2[:, s_Ws:s_Ws + in_W] = out1
    if s_Ws > 0:
        aug2[:, :s_Ws] = out1[:, :s_Ws][:, ::-1]
    if s_We > 0:
        aug2[:, s_Ws + in_W:] = out1[:, -s_We:][:, ::-1]

    out2 = np.zeros((out_H, out_W, in_C))
    kw_W = w_W.shape[1]
    for j in range(out_W):
        a = int(idx_W[j, 0])
        out2[:, j] = np.tensordot(aug2[:, a:a + kw_W], w_W[j], axes=([1], [0]))

    if squeeze:
        out2 = out2[:, :, 0]
    return out2


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------
def load_image(path):
    im = Image.open(path)
    # palette/other formats -> RGB; keep 'L' (grayscale) and 'RGB' as they are
    if im.mode not in ("L", "RGB"):
        im = im.convert("RGB")
    return np.array(im)


def mod_crop(arr, scale):
    """Crop the HR so that H and W are multiples of scale (top-left aligned)."""
    if arr.ndim == 2:
        H, W = arr.shape
        return arr[:H - H % scale, :W - W % scale]
    H, W = arr.shape[:2]
    return arr[:H - H % scale, :W - W % scale, :]


def downscale_to_uint8(hr_uint8, scale):
    """Apply imresize 1/scale and convert back to uint8 (round + clip), like MATLAB."""
    lr = imresize_matlab(hr_uint8.astype(np.float64), 1.0 / scale, antialiasing=True)
    lr = np.clip(np.round(lr), 0, 255).astype(np.uint8)
    return lr


# ---------------------------------------------------------------------------
# Worker (must live at module level for multiprocessing)
# ---------------------------------------------------------------------------
def _process_one(task):
    src, rel, out_root, scales, ext, div2k_naming, save_hr, overwrite = task
    try:
        hr_full = load_image(src)
        stem = Path(rel).stem
        parent = Path(rel).parent  # sub-folder structure preserved

        for s in scales:
            hr = mod_crop(hr_full, s)

            if div2k_naming:
                lr_name = f"{stem}x{s}{ext}"
            else:
                lr_name = f"{stem}{ext}"
            lr_path = out_root / "LR_bicubic" / f"X{s}" / parent / lr_name
            lr_path.parent.mkdir(parents=True, exist_ok=True)

            if save_hr:
                # the mod-cropped HR depends on s: one HR folder per scale when several are given
                hr_path = out_root / "HR" / (f"X{s}" if len(scales) > 1 else "") / parent / f"{stem}{ext}"
                hr_path.parent.mkdir(parents=True, exist_ok=True)
                if overwrite or not hr_path.exists():
                    Image.fromarray(hr).save(hr_path)

            if not overwrite and lr_path.exists():
                continue
            lr = downscale_to_uint8(hr, s)
            Image.fromarray(lr).save(lr_path)

        return (rel, True, "")
    except Exception as e:  # do not abort the whole job because of one broken file
        return (rel, False, repr(e))


def collect_images(root):
    root = Path(root)
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append((p, p.relative_to(root)))
    return files


def main():
    ap = argparse.ArgumentParser(
        description="Genera LR_bicubic X2/X3/X4 (bicubico MATLAB-compatibile) da un dataset HR.")
    ap.add_argument("input", type=str, help="Cartella con le immagini HR (cerca ricorsivamente).")
    ap.add_argument("--out", type=str, default=None,
                    help="Cartella di output (default: <input>/../<input_name>_SR).")
    ap.add_argument("--scales", type=int, nargs="+", default=[2, 3, 4],
                    help="Scale da generare (default: 2 3 4).")
    ap.add_argument("--ext", type=str, default=".png",
                    help="Estensione output (default .png; usa .jpg per risparmiare spazio).")
    ap.add_argument("--div2k-naming", action="store_true",
                    help="Rinomina i file LR come <nome>x<scala> (convenzione DIV2K).")
    ap.add_argument("--save-hr", action="store_true",
                    help="Salva anche l'HR mod-cropped (utile per valutazione/coppie pulite).")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)),
                    help="Processi paralleli (default: tutti i core). Usa 1 per esecuzione seriale.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Rigenera anche i file già esistenti.")
    args = ap.parse_args()

    in_root = Path(args.input).expanduser().resolve()
    if not in_root.is_dir():
        sys.exit(f"[ERRORE] '{in_root}' non è una cartella.")

    out_root = Path(args.out).expanduser().resolve() if args.out \
        else in_root.parent / f"{in_root.name}_SR"
    out_root.mkdir(parents=True, exist_ok=True)

    ext = args.ext if args.ext.startswith(".") else "." + args.ext

    files = collect_images(in_root)
    if not files:
        sys.exit(f"[ERRORE] Nessuna immagine trovata in '{in_root}'.")

    print(f"Input        : {in_root}")
    print(f"Output       : {out_root}")
    print(f"Immagini     : {len(files)}")
    print(f"Scale        : {args.scales}")
    print(f"Output ext   : {ext}  |  naming DIV2K: {args.div2k_naming}  |  salva HR: {args.save_hr}")
    print(f"Workers      : {args.workers}\n")

    tasks = [(src, rel, out_root, args.scales, ext, args.div2k_naming,
              args.save_hr, args.overwrite) for src, rel in files]

    failures = []
    if args.workers <= 1:
        for t in tqdm(tasks, desc="LR_bicubic", unit="img"):
            rel, ok, err = _process_one(t)
            if not ok:
                failures.append((rel, err))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_process_one, t) for t in tasks]
            for f in tqdm(as_completed(futs), total=len(futs), desc="LR_bicubic", unit="img"):
                rel, ok, err = f.result()
                if not ok:
                    failures.append((rel, err))

    print(f"\nFatto. Immagini elaborate: {len(files) - len(failures)}/{len(files)}")
    if failures:
        print(f"[ATTENZIONE] {len(failures)} file non elaborati:")
        for rel, err in failures[:20]:
            print(f"  - {rel}: {err}")


if __name__ == "__main__":
    main()