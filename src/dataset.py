"""
Super-Resolution dataset built on torchSR (DIV2K and/or Flickr2K), meant to
feed the MambaIR model (see mambair.py / train_mambair.py).

- Downloads DIV2K and/or Flickr2K automatically (on first use).
- Performs aligned LR/HR crops (random in train, center in val).
- Augmentation: ONLY horizontal/vertical flips (p=0.5 each).
- Returns (LR, HR) pairs as float tensors in [0, 1], already in the format that
  MambaIR expects (the model normalizes internally with the RGB mean).

Patches (paper convention, Sec. 4.1)
------------------------------------
MambaIR is isotropic: it keeps the resolution constant, has no internal
downsampling and accepts ANY patch size. So the patches are simple:
    lr_size = hr_patch_size // scale     # e.g. 128//2 = 64  -> LR patch 64x64
    hr_size = hr_patch_size              # e.g. 128         -> HR patch 128x128
The paper uses 64x64 LR patches for SR, i.e. hr_patch_size = 64 * scale.

DIV2K vs Flickr2K
-----------------
* DIV2K   : 800 train images + 100 val (it has the 'train'/'val' splits).
* Flickr2K: 2650 images, ALL for training (it has NO validation split).

For this reason:
* In TRAIN you can use 'div2k', 'flickr2k' or 'both' (= DF2K, the standard in
  the literature: DIV2K-train + Flickr2K, 800+2650 = 3450 images).
* VAL ALWAYS uses only DIV2K val (Flickr2K has no validation set).
  The factory `build_sr_dataset` drops flickr2k automatically when
  split='val'.

Use the factory `build_sr_dataset(...)` to get the right dataset (single or
concatenated); `SRDataset` stays available for the single case. In both cases
the attributes lr_size/hr_size/scale/names are accessible on the returned
object.
"""
import os
import glob
import random

import torch
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms.functional as TF


def _crop_and_augment(lr, hr, scale, lr_size, train_mode):
    """Aligned LR/HR crop (random in train, center in val) + flips in train.
    lr,hr: tensors (C,H,W) in [0,1]. Returns (lr_patch, hr_patch)."""
    _, h_lr, w_lr = lr.shape
    if h_lr < lr_size or w_lr < lr_size:
        raise RuntimeError(f"Immagine LR troppo piccola ({h_lr}x{w_lr}) per patch {lr_size}")
    if train_mode:
        x = random.randint(0, w_lr - lr_size)
        y = random.randint(0, h_lr - lr_size)
    else:
        x = (w_lr - lr_size) // 2
        y = (h_lr - lr_size) // 2
    lr = lr[:, y:y + lr_size, x:x + lr_size]
    hr = hr[:, y * scale:(y + lr_size) * scale, x * scale:(x + lr_size) * scale]

    if train_mode:
        if random.random() < 0.5:
            lr, hr = TF.hflip(lr), TF.hflip(hr)
        if random.random() < 0.5:
            lr, hr = TF.vflip(lr), TF.vflip(hr)
    return lr, hr


class SRDataset(Dataset):
    """
    Wrapper around a single torchSR dataset (DIV2K or Flickr2K).

    Args:
        root: download folder.
        scale: upscaling factor (2, 3 or 4).
        dataset: 'div2k' or 'flickr2k'.
        split: 'train' or 'val'. IGNORED for flickr2k (always 'train').
        hr_patch_size: side of the HR patch. Must be divisible by scale
                       (the LR patch will be hr_patch_size // scale).
        train_mode: True -> random crop + flips; False -> center crop, deterministic.
        download: if True, download the data when missing; if False, use only the
                  data already present in `root` (offline / after a manual download).
    """

    def __init__(
        self,
        root,
        scale=2,
        dataset='div2k',
        split='train',
        hr_patch_size=128,
        train_mode=True,
        download=True,
    ):
        assert hr_patch_size % scale == 0, (
            f"hr_patch_size ({hr_patch_size}) deve essere divisibile per scale ({scale})"
        )
        self.scale = scale
        self.train_mode = train_mode
        self.name = dataset.lower()

        self.hr_size = hr_patch_size
        self.lr_size = hr_patch_size // scale

        from torchsr.datasets import Div2K, Flickr2K
        if self.name == 'div2k':
            self.dataset = Div2K(
                root=root, scale=scale, track='bicubic',
                split=split, download=download,
            )
        elif self.name == 'flickr2k':
            self.dataset = Flickr2K(
                root=root, scale=scale, track='bicubic',
                download=download,
            )
        else:
            raise ValueError(
                f"dataset sconosciuto: {dataset!r} (usa 'div2k' o 'flickr2k')"
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        hr_pil, lr_pil = sample[0], sample[1]
        # Guard: keep the first image if the LR entry comes back as a list.
        if isinstance(lr_pil, (list, tuple)):
            lr_pil = lr_pil[0]
        hr = TF.to_tensor(hr_pil)          # [0,1]
        lr = TF.to_tensor(lr_pil)          # [0,1]
        return _crop_and_augment(lr, hr, self.scale, self.lr_size, self.train_mode)


def _resolve_dataset_names(datasets, split):
    """
    Normalize the dataset request into a list of concrete names.

    `datasets` can be: 'div2k' | 'flickr2k' | 'both' (=DF2K) | a list of these.
    In split='val', Flickr2K is removed (it has no validation set); if the list
    would then be empty, fall back to DIV2K.

    Pure function (no download): easy to test.
    """
    if isinstance(datasets, str):
        datasets = [datasets]
    names = []
    for d in datasets:
        d = d.lower()
        if d in ('both', 'df2k'):
            names += ['div2k', 'flickr2k']
        elif d in ('div2k', 'flickr2k'):
            names.append(d)
        else:
            raise ValueError(
                f"dataset sconosciuto: {d!r} (usa 'div2k', 'flickr2k' o 'both')"
            )
    # Drop duplicates while keeping the order.
    names = list(dict.fromkeys(names))

    if split == 'val' and 'flickr2k' in names:
        names = [n for n in names if n != 'flickr2k']
        if not names:
            names = ['div2k']
    return names


def build_sr_dataset(
    datasets,
    root,
    scale=2,
    split='train',
    hr_patch_size=128,
    train_mode=True,
    download=True,
):
    """
    Build the requested SR dataset (single or concatenated).

    Examples:
        # DIV2K only (train)
        ds = build_sr_dataset('div2k', root, split='train', train_mode=True)
        # DF2K = DIV2K-train + Flickr2K
        ds = build_sr_dataset('both',  root, split='train', train_mode=True)
        # validation (always DIV2K val, even when passing 'both'/'flickr2k')
        ds = build_sr_dataset('both',  root, split='val',  train_mode=False)

    `download=False` uses only the data already present in `root` (no network).

    Returns an `SRDataset` (for a single dataset) or a `ConcatDataset`. In both
    cases the attributes lr_size/hr_size/scale/names are accessible on the
    returned object.
    """
    names = _resolve_dataset_names(datasets, split)
    subsets = [
        SRDataset(
            root=root, scale=scale, dataset=n, split=split,
            hr_patch_size=hr_patch_size, train_mode=train_mode, download=download,
        )
        for n in names
    ]

    if len(subsets) == 1:
        out = subsets[0]
    else:
        out = ConcatDataset(subsets)
        # ConcatDataset has no patch geometry: copy it from the first subset.
        for attr in ('lr_size', 'hr_size', 'scale'):
            setattr(out, attr, getattr(subsets[0], attr))

    out.names = names
    return out


_IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


class PairedFolderDataset(Dataset):
    """(LR, HR) pairs read from TWO explicit folders, as in the original .yml files:
        dataroot_gt -> hr_dir,  dataroot_lq -> lr_dir,  filename_tmpl -> '{}x2'

    No download: uses the files already on disk. For each HR image 'name.ext',
    the matching LR is looked up as '<filename_tmpl.format(name)>' in the LR
    folder (trying the known image extensions).

    Example (BasicSR-style DIV2K bicubic x2):
        hr_dir = .../DIV2K/HR
        lr_dir = .../DIV2K/LR_bicubic/X2
        filename_tmpl = '{}x2'        # 0001.png  <->  0001x2.png

    Args as in SRDataset: scale, hr_patch_size, train_mode.
    """

    def __init__(self, hr_dir, lr_dir, scale=2, hr_patch_size=128,
                 filename_tmpl='{}', train_mode=True, recursive=True):
        assert hr_patch_size % scale == 0, (
            f"hr_patch_size ({hr_patch_size}) deve essere divisibile per scale ({scale})")
        self.scale = scale
        self.train_mode = train_mode
        self.hr_size = hr_patch_size
        self.lr_size = hr_patch_size // scale
        self.hr_dir, self.lr_dir = hr_dir, lr_dir

        if not os.path.isdir(hr_dir):
            raise FileNotFoundError(f"cartella HR inesistente: {hr_dir}")
        if not os.path.isdir(lr_dir):
            raise FileNotFoundError(f"cartella LR inesistente: {lr_dir}")

        # Map (relative dir, file stem) -> path, so HR/LR files match across subfolders.
        def _index(root_dir):
            idx = {}
            for cur, _, files in os.walk(root_dir):
                rel = os.path.relpath(cur, root_dir)
                for f in files:
                    stem, ext = os.path.splitext(f)
                    if ext.lower() in _IMG_EXTS:
                        idx[(rel, stem)] = os.path.join(cur, f)
                if not recursive:
                    break
            return idx

        lr_index = _index(lr_dir)

        self.pairs = []
        for (rel, stem), hr_path in _index(hr_dir).items():
            lr_path = lr_index.get((rel, filename_tmpl.format(stem)))
            if lr_path is not None:
                self.pairs.append((hr_path, lr_path))
        self.pairs.sort()

        if not self.pairs:
            raise RuntimeError(
                f"nessuna coppia HR/LR trovata.\n  HR: {hr_dir}\n  LR: {lr_dir}\n"
                f"  filename_tmpl={filename_tmpl!r} (es. '{{}}x{scale}' per le LR bicubiche)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        from PIL import Image
        hr_path, lr_path = self.pairs[idx]
        hr = TF.to_tensor(Image.open(hr_path).convert('RGB'))
        lr = TF.to_tensor(Image.open(lr_path).convert('RGB'))
        return _crop_and_augment(lr, hr, self.scale, self.lr_size, self.train_mode)


def build_paired_dataset(hr_dirs, lr_dirs, scale=2, hr_patch_size=128,
                         filename_tmpl='{}', train_mode=True):
    """Build a PairedFolderDataset (or a ConcatDataset for several folder pairs,
    e.g. DF2K = DIV2K + Flickr2K). `hr_dirs`/`lr_dirs` can be strings or lists of
    equal length. Exposes lr_size/hr_size/scale/names on the returned object."""
    if isinstance(hr_dirs, str):
        hr_dirs = [hr_dirs]
    if isinstance(lr_dirs, str):
        lr_dirs = [lr_dirs]
    if len(hr_dirs) != len(lr_dirs):
        raise ValueError(f"hr_dirs ({len(hr_dirs)}) e lr_dirs ({len(lr_dirs)}) "
                         f"devono avere la stessa lunghezza")

    subsets = [
        PairedFolderDataset(hr, lr, scale=scale, hr_patch_size=hr_patch_size,
                            filename_tmpl=filename_tmpl, train_mode=train_mode)
        for hr, lr in zip(hr_dirs, lr_dirs)
    ]
    if len(subsets) == 1:
        out = subsets[0]
    else:
        out = ConcatDataset(subsets)
        # ConcatDataset has no patch geometry: copy it from the first subset.
        for attr in ('lr_size', 'hr_size', 'scale'):
            setattr(out, attr, getattr(subsets[0], attr))
    # Dataset name: use the parent folder when the HR folder is just called 'HR'.
    def _name(h):
        h = os.path.normpath(h)
        base = os.path.basename(h)
        return os.path.basename(os.path.dirname(h)) if base.lower() == 'hr' else base
    out.names = [_name(h) for h in hr_dirs]
    return out