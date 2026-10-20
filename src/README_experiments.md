# How Much Mamba Does Lightweight Super-Resolution Need? — experiment runbook

Analysis study: Base-light backbone (C = 60, 4 RLMG × 6 PCLMB, d_state = 10, ECA kept,
**RFR removed**), varying only the Mamba channel fraction ρ ∈ {0, 0.25, 0.5, 0.75, 1}
(C_m = ⌊ρ·C⌋ = 0 / 15 / 30 / 45 / 60). Training recipe, metrics, and all other
architectural settings are fixed. Every number in the paper is produced by a command
listed here and lands in a CSV with a fixed schema (`results/*.csv`); tables and figures
are generated from those CSVs only (`scripts/make_tables.py`), nothing is transcribed by hand.

All commands run from `src/` with the `mambair-sr` conda environment created by
`bash setup_env.sh` (Python 3.10, basicsr 1.4.2, fvcore 0.1.5, pytest, PyYAML, pandas, matplotlib,
easydict; see `requirements.txt`). The torch/Mamba stack depends on the GPU generation and is picked
automatically from the compute capability: torch 2.6.0+cu124 with mamba_ssm 2.2.4 /
causal-conv1d 1.5.0.post5 on Ampere/Ada/Hopper (the laptop numbers in `results/` were produced with
this stack), torch 2.8.0+cu128 with mamba_ssm 2.2.5 / causal-conv1d 1.5.2 on Blackwell (sm_120,
e.g. RTX PRO 5000: the cu124 build has no kernels for it). The `gpu` column of every CSV records
which machine produced each row; keep latency rows from one machine only.

```
src/
  pclmsr.py, mambair.py, models.py, train.py, metrics.py   # model + training (recipe unchanged)
  configs/rho_{000,025,050,075,100}_x{2,3,4}.yml           # 15 configs (generated)
  tests/test_pclmb_rho.py
  scripts/{make_configs,common,baselines,count_params_macs,bench_latency,eval_all,make_tables}.py
  scripts/train_sweep.sh
  repos/<baseline repos>                                   # official code (see §7)
  experiments/<config>/seed<k>/iter_<N>.pth                # training outputs
  results/{quality,efficiency,profile}.csv, results/tables/*.tex, results/figures/*.pdf
```

---

## 0. One-time checks

```bash
python scripts/make_configs.py            # (re)generate the 15 configs; --check verifies they only differ in rho/scale
python -m pytest tests/test_pclmb_rho.py -v -s
```
The test instantiates every (ρ, scale), prints total and per-component parameters, runs
forward/backward on 2×3×64×64, checks output shapes, and asserts: at ρ = 0 no Mamba-branch
parameter/module exists (no LN, no α_m, no SS2D); at ρ = 1 no local-branch parameter/module
exists; RFR absent everywhere; ECA and the 1×1 fusion present everywhere; the VSSM runs with
C_m = 15 and 45 (d_inner = 22 and 67, no divisibility constraint), and also on a 37×53 input.

Per-component parameter counts (identical for all scales except the head):

| ρ | C_m | total (×2) | Mamba | local | fusion | ECA | LE+LN | RLMG conv | shallow/body | head ×2/×3/×4 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0 | 411,384 | 0 | 100,080 | 87,840 | 72 | 52,920 | 129,840 | 34,140 | 6,492 / 14,607 / 25,968 |
| 0.25 | 15 | 471,936 | 102,984 | 57,648 | 87,840 | 72 | 52,920 | 129,840 | 34,140 | " |
| 0.50 | 30 | 607,584 | 267,840 | 28,440 | 87,840 | 72 | 52,920 | 129,840 | 34,140 | " |
| 0.75 | 45 | 803,664 | 484,032 | 8,328 | 87,840 | 72 | 52,920 | 129,840 | 34,140 | " |
| 1.00 | 60 | 1,075,944 | 764,640 | 0 | 87,840 | 72 | 52,920 | 129,840 | 34,140 | " |

(these come from the test output; the CSV numbers come from `count_params_macs.py`.)

## 1. Training

Recipe (unchanged, from `models.TRAIN_CONFIGS['pclmsr_x*']`): Adam(β = 0.9/0.99, wd 0),
lr 2e-4, MultiStepLR milestones [250K, 400K, 450K, 475K] γ = 0.5, 500K iterations, batch 32,
HR patch 128 (×2) / 192 (×3, ×4) → LR patch 64×64, random crop + h/v flips, L1 loss,
no EMA (not implemented in this codebase → identical for all runs), `cudnn.benchmark = False`,
8 dataloader workers, DIV2K train (800) / DIV2K valid (100) for the in-training validation.
Checkpoints: `iter_100000.pth … iter_500000.pth` (immutable), `latest.pth`, `best.pth`, `final.pth`.

```bash
# full run (seed 0): 500K iterations, checkpoint every 100K
python train.py --config configs/rho_050_x2.yml --seed 0
# ablation seeds: 100K iterations, milestones NOT rescaled -> identical to the first 100K of the full run
python train.py --config configs/rho_050_x2.yml --seed 1 --total_iter 100000
python train.py --config configs/rho_050_x2.yml --seed 2 --total_iter 100000
# x3 / x4: pretrain_network_g = experiments/rho_050_x2/seed{seed}/final.pth (same rho, same seed)
python train.py --config configs/rho_050_x3.yml --seed 0
# everything, in order (x2 -> x3 -> x4, seed 0 full then seeds 1,2 at 100K):
./scripts/train_sweep.sh                    # env: SCALES, RHOS, SEEDS, ABL_ITERS, PYTHON, DRY=1
```
The three samples per (ρ, scale) at 100K iterations are: `seed0/iter_100000.pth` (from the
full run), `seed1/iter_100000.pth`, `seed2/iter_100000.pth`. The full-schedule number is
`seed0/iter_500000.pth`. The `.yml` used is copied into the run folder and stored inside every
checkpoint (`run_meta`), together with `seed`, `iter`, `arch_cfg`.

GPU memory at batch 32 / LR 64×64 (measured on the RTX 4090 Laptop 16 GB, forward+backward+Adam):
ρ = 0: 6.7 GB, ρ = 0.25: 13.8 GB, ρ ≥ 0.5: out of memory (ρ = 1 does not fit even at batch 16).
The sweep therefore needs a ≥ 24 GB GPU (or a change of the recipe, which is not done here).

### 1b. Training memory / speed of the official MambaIR-light and MambaIRv2-light

```bash
python scripts/train_baseline_memory.py --model mambair_light   --scale 2 --iters 200 --csv results/train_memory.csv
python scripts/train_baseline_memory.py --model mambairv2_light --scale 2 --iters 200 --csv results/train_memory.csv
python scripts/train_baseline_memory.py --model rho:0.5 --scale 2 --iters 200 --csv results/train_memory.csv   # ours, same loop
```
Trains the official architectures (from `src/repos/MambaIR`, lightweight configs) with the project
recipe (DIV2K, LR patch 64, batch 32, Adam 2e-4, L1, FP32) and prints per-iteration speed plus
PyTorch's peak allocated / reserved memory; reserved memory is what nvtop / nvidia-smi show for
the process, minus the CUDA context. Out-of-memory is reported with the peak reached (exit code 2).
`--batch_size`, `--iters`, `--ckpt_dir` (saves `final.pth`) and `--use_checkpointing` (MambaIRv2
gradient checkpointing) are available. On the 16 GB laptop both exceed memory at batch 32
(MambaIR-light 6.4 GB and MambaIRv2-light 7.1 GB peak at batch 8).

## 2. Quality: PSNR / SSIM → `results/quality.csv`

Metric: Y channel of YCbCr (BasicSR conversion, `basicsr.metrics.calculate_psnr_pt /
calculate_ssim_pt`, `test_y_channel=True`), border crop = scale, SR and HR quantised to uint8
(SR images are saved as PNG and re-read), top-left mod-crop alignment. The **same function**
(`eval_all.evaluate_folder`) scores our models and the baselines.

```bash
# our models: inference from checkpoint -> results/sr/<config>/seed<k>/x<s>/<Dataset>/<stem>.png, then metrics
python scripts/eval_all.py ours --ckpt experiments/rho_050_x2/seed0/iter_100000.pth
python scripts/eval_all.py sweep --iters 100000 500000          # all experiments/rho_*_x*/seed*/iter_*.pth
# baselines: generate SR images with the official test script + official weights, then
python scripts/eval_all.py folders --method SwinIR-light --scale 2 --sr_root <dir with Set5/ Set14/ B100/ Urban100/ Manga109/> \
       --note "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth"
# a single folder pair (debug)
python scripts/eval_all.py metrics --sr_dir <SR> --hr_dir ../datasets/SRBenchmarks/benchmark/Set5/HR --scale 2
```
Schema: `method, rho, seed, iters, scale, dataset, psnr_y, ssim_y, checkpoint, gpu, commit, date`
(baseline rows have empty `rho/seed/iters`; `checkpoint` holds the weight file / URL note).
Large images (Urban100 ×2) fit in 16 GB without tiling; `--tile 256` is available if needed
(tiling changes the result slightly, so use the same setting for all ρ).

## 3. Parameters and MACs → `results/efficiency.csv`

```bash
python scripts/count_params_macs.py --baselines all --md results/params_macs.md     # all rho, all scales, all baselines
python scripts/count_params_macs.py --scales 4 --rhos 0 0.5 1 --show_ops               # per-operator breakdown
```
Protocol:
* Fixed LR inputs 640×360 (×2), 427×240 (×3), 320×180 (×4) → 1280×720 output (×3: 1281×720).
* Counter: `fvcore.nn.FlopCountAnalysis`; one fvcore "flop" = one multiply-accumulate = 1 MAC
  (conv, linear, matmul/bmm/einsum, layer_norm). Element-wise ops, activations, softmax, pixel
  shuffle, flips are not counted (standard SR convention; the same for every model).
* Selective scan: fvcore / ptflops / thop do not see the mamba_ssm CUDA kernel (an
  `autograd.Function`, visible in the traced graph only as `prim::PythonOp.SelectiveScanFn`).
  We register the VMamba handle (`flops_selective_scan_fn` from `analysis/flops.py` of VMamba):
  `MACs = 9·B·L·D·N + B·D·L` per kernel call, with `(B, D, L)` from `u` and `N` from `A`.
  Our SS2D calls the kernel once per direction (4 calls, D = d_inner); the official MambaIR
  calls it once with D = 4·d_inner — same total. Extra handles: `CausalConv1dFn` (D·L·K) and
  `MambaInnerFn` (fused path; for DVMSR the fast path is disabled during counting so that
  in_proj / conv1d / x_proj / dt_proj / scan / out_proj are counted as separate ops).
* The count runs on the GPU so that the CUDA kernel path is the one traced.
* Acceptance check (asserted at every run): scan MACs = 0 at ρ = 0, > 0 and strictly
  increasing in ρ. Measured at ×2: 0 / 44.3 / 90.6 / 134.9 / 181.1 G (29–45 % of the total).
* Sanity: the same counter reproduces published figures for the baselines at ×4
  (SwinIR-light 63.6 G, IMDN 40.9 G, RLFN 29.8 G, CARN 90.9 G, LatticeNet 43.6 G, SAFMN 13.6 G,
  MambaIR-light 85.0 G with the fvcore count the MambaIR authors adopted after their thop bug).

## 4. Latency, memory, per-module profile → `results/efficiency.csv`, `results/profile.csv`

```bash
python scripts/bench_latency.py --baselines all --log results/bench_latency.log
python scripts/bench_latency.py --scales 4 --rhos 0 1 --iters 100 --profile_iters 20
```
Protocol (printed at the top of every run and stored in the CSV columns `gpu, commit, date`):
* FP32 strict (`torch.backends.cuda.matmul.allow_tf32 = False`, `cudnn.allow_tf32 = False`),
  batch size 1, `torch.no_grad()`, forward pass only, random weights, random input tensor at the
  fixed LR sizes above.
* `cudnn.benchmark` set to the same value for every model (False; `--cudnn_benchmark` flips it
  for all).
* 10 warm-up passes, then 50 timed runs (≥ 30 required) with `torch.cuda.Event` pairs and
  `torch.cuda.synchronize()` before/after every run; mean and median reported.
* GPU idle check: `nvidia-smi --query-compute-apps` before the run; other compute processes
  trigger a WARNING in the log.
* Peak memory: `torch.cuda.empty_cache(); reset_peak_memory_stats()` before one forward,
  `max_memory_allocated()` after → `peak_mem_mb`; `weights_mem_mb` = bytes of parameters +
  buffers; `act_mem_mb` = peak − weights (includes input/output tensors and temporaries).
* Per-module profile (our models): `torch.profiler` (CPU + CUDA activities) over 10 forwards
  with `record_function` ranges inserted through forward hooks; CUDA time aggregated by category:
  `scan` (selective-scan kernel), `vssm_proj` (VSSM minus scan: in_proj, DWConv, x_proj, dt_proj,
  out_proj, LN), `local` (LN + local conv branch), `fusion` (1×1 conv), `eca`, `le` (local
  enhancement + LN), `rlmg_conv`, `shallow_head` (conv_first + conv_after_body + upsampler),
  `other` (residual adds, split/cat, mean shift). Profiler overhead inflates absolute times:
  report the fractions (`frac`) and use the CUDA-event latency as the headline number.
  Baselines get only the `total` row.
* Logged automatically: GPU name, driver, torch / CUDA / cuDNN / mamba_ssm / causal_conv1d
  versions, python, commit (`nogit` if `src/` is not a git repository — run `git init` and commit
  before the final measurements), date.
* Environment used for the numbers currently in `results/`: NVIDIA GeForce RTX 4090 Laptop GPU,
  driver 580.159.03, torch 2.6.0+cu124, CUDA 12.4, cuDNN 90100, mamba_ssm 2.2.4.

Schema: `method, rho, scale, params, macs_g, macs_scan_g, latency_ms_mean, latency_ms_median,
peak_mem_mb, weights_mem_mb, act_mem_mb, gpu, commit, date` (`count_params_macs.py` fills the
first columns, `bench_latency.py` the latency/memory ones; both upsert on `(method, rho, scale)`).
`profile.csv`: `method, rho, scale, category, ms, frac, gpu, commit, date`.

## 4b. Inference memory on a benchmark → `results/memory_<dataset>.csv`

```bash
python scripts/bench_memory_dataset.py --baselines all                 # Urban100, x2/x3/x4, all rho + baselines
python scripts/bench_memory_dataset.py --dataset Manga109 --scales 4 --rhos 0 1 --tile 256
```
Full-resolution inference (batch 1, FP32 strict, `no_grad`, random weights) on every image of the
benchmark, using only the `<stem>x<scale>` LR files paired with an HR (the folders also contain
differently generated copies with other names; those are ignored everywhere, as in training).
Per image: `reset_peak_memory_stats()` → forward → `max_memory_allocated()`. Reported per model:
maximum and mean peak over images, the image and LR size that cause the maximum, maximum
`max_memory_reserved()` (what `nvidia-smi` shows), weights memory and activation memory
(peak − weights). Images that go out of memory are counted (`n_oom`) and skipped. Use the same
`--tile` for all models being compared (default 0 = whole image).

## 4c. Latency per image and FPS on a benchmark → `results/latency_<dataset>.csv`

```bash
python scripts/bench_latency_dataset.py --baselines all                       # Urban100 + Manga109, x2/x3/x4
python scripts/bench_latency_dataset.py --datasets Urban100 --scales 4 --repeats 5 --md results/latency_datasets.md
```
Every paired LR image of the benchmark is processed once at native resolution (batch 1, FP32
strict, `no_grad`, random weights, images pre-loaded on the GPU so I/O is excluded), after 10
warm-up forwards; each image is timed `--repeats` times (default 3) with CUDA events and
`torch.cuda.synchronize()`. Reported: mean, median, std, min, max of the per-image time (ms/img)
and **FPS = 1000 / mean ms**. Same `--tile` and `cudnn.benchmark` for every model.

## 4d. Qualitative comparison (zoomed patch) → `results/figures/qual_<image>_x<s>.png/.pdf`

```bash
python scripts/qualitative_compare.py --scale 4 --image img062 --patch_size 80 \
    --models "MambaIR-light=mambair_light:repos/MambaIR/weights/MambaIR_lightSR4.pth" \
             "MambaIRv2-light=mambairv2_light:repos/MambaIR/weights/mambairv2_lightSR_x4.pth" \
             "PCLM rho=0.5=ours:experiments/rho_050_x4/seed0/iter_500000.pth" bicubic
```
Ground truth with the patch marked in red, then the zoomed patch for GT and each model, with
full-image PSNR/SSIM (Y, crop = scale) and patch PSNR under each zoom. `--seed` picks a random
image and patch, `--image <stem>` / `--patch_xy X Y` fix them. Official weights:
`repos/MambaIR/weights/MambaIR_lightSR4.pth` (https://huggingface.co/cguoh/MambaIR, folder
MambaIRv1_ckpt) and `mambairv2_lightSR_x4.pth` (https://github.com/csguoh/MambaIR/releases/tag/v1.0);
×2/×3 weights are at the same locations (`MambaIR_lightSR{2,3}.pth`, `mambairv2_lightSR_x{2,3}.pth`).

## 5. Tables and figures → `results/tables/`, `results/figures/`

```bash
python scripts/make_tables.py --iters 100000          # 3-seed ablation tables (default)
python scripts/make_tables.py --iters 500000          # full-schedule rows (seed 0)
python scripts/make_tables.py --dataset Urban100      # PSNR-vs-params figure on one dataset
```
* `tab_rho_x{2,3,4}.tex`: PSNR/SSIM per ρ and dataset, mean ± std over seeds (booktabs).
* `tab_rho_paired_x{s}.tex`: paired differences PSNR(ρ_b, seed) − PSNR(ρ_a, seed) for adjacent ρ,
  mean ± std over seeds and the number of seed pairs.
* `tab_joint_x{s}.tex`: our five ρ rows + baselines: params, MACs, scan MACs, latency, peak
  memory, PSNR/SSIM per dataset.
* `tab_efficiency.tex`: params / MACs / latency / memory per ρ and scale.
* `fig_psnr_vs_rho_x{s}`: PSNR vs ρ with std error bars, one panel per dataset.
* `fig_psnr_vs_params_x{s}`: PSNR vs parameters, ρ sweep as a line + baselines as squares.
* `fig_latency_memory_vs_rho`: latency and peak memory vs ρ, one curve per scale.
* `fig_profile_x{s}`: per-module stacked bars for each ρ.
* `summary.md`: provenance (row counts, GPUs, commits, dates) of the CSVs used.

## 6. Order of execution

1. `python scripts/make_configs.py && python -m pytest tests/test_pclmb_rho.py`
2. `python scripts/count_params_macs.py --baselines all --md results/params_macs.md`
3. `python scripts/bench_latency.py --baselines all --log results/bench_latency.log` (idle GPU)
4. `./scripts/train_sweep.sh` (≥ 24 GB GPU; ×2 before ×3/×4)
5. `python scripts/eval_all.py sweep --iters 100000 500000`
6. Baseline SR images with each official test script (§7), then
   `python scripts/eval_all.py folders --method <name> --scale <s> --sr_root <dir>` per baseline and scale
7. `python scripts/make_tables.py` (and `--iters 500000`)

## 7. Baselines: repositories, commits, checkpoints

All architectures are built from the official code in `src/repos/` (adapters in
`scripts/baselines.py`; weights are irrelevant for params/MACs/latency). For PSNR/SSIM, run the
official test script with the official weights, save the PNGs, and score them with `eval_all.py folders`.

| Method | Repository (commit) | Arch file / config used | Official weights |
|---|---|---|---|
| SwinIR-light | https://github.com/JingyunLiang/SwinIR (`6545850fbf8d`) | `models/network_swinir.py`, task `lightweight_sr` of `main_test_swinir.py`: embed 60, depths 6×4, heads 6, window 8, mlp 2, pixelshuffledirect; input padded to a multiple of 8 as in the test script | https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x{2,3,4}.pth |
| MambaIR-light | https://github.com/csguoh/MambaIR (`33d7b3460c46`) | `basicsr/archs/mambair_arch.py`, `options/test/mambair/test_MambaIR_lightSR_x*.yml`: embed 60, depths 6×4, d_state 10, mlp 1.2 | https://huggingface.co/cguoh/MambaIR (`MambaIR_light{2,3,4}.pth`; Google-Drive links in the README) |
| MambaIRv2-light | same repo | `basicsr/archs/mambairv2light_arch.py`, `options/train/mambairv2/train_MambaIRv2_lightSR_x*.yml`: embed 48, depths 5×4, heads 4, window 16, d_state 8, inner_rank 32, num_tokens 64; input padded to a multiple of 16 | https://github.com/csguoh/MambaIR/releases/tag/v1.0 (`MambaIRv2_light{2,3,4}.pth`) |
| SAFMN | https://github.com/sunny2109/SAFMN (`1a412066e927`) | `basicsr/archs/safmn_arch.py`, `options/train/SAFMN/train_DF2K_X*.yml`: dim 36, 8 blocks, ffn 2.0 | https://huggingface.co/Meloo/SAFMN/tree/main (`SAFMN_DF2K_x{2,3,4}.pth`) |
| DVMSR | https://github.com/nathan66666/DVMSR (`86663ce906aa`) | `models/team07_DVMSR.py`, defaults of `test_demo.py` (depths 2×4, embed 60); **defined and weighted only at ×4** (×2/×3 rows are architecture-only) | in-repo `model_zoo/team07_DVMSR.pth` |
| RLFN | https://github.com/bytedance/RLFN (`bccfde91bf8b`) | `model/rlfn.py`, `RLFN(feature_channels=52)` | in-repo `model_zoo/rlfn_x2.pth`, `rlfn_x4.pth` (no ×3 weights) |
| CARN | https://github.com/nmhkahn/CARN-pytorch (`40d3514f81cc`) | `carn/model/carn.py`, `Net(scale, multi_scale=False, group=1)` — one upsampler per scale (the paper's 1,592K counts the 3-scale model) | in-repo `checkpoint/carn.pth` (multi-scale) |
| IMDN | https://github.com/Zheng222/IMDN (`8f158e6a5ac9`) | `model/architecture.py`, `IMDN(nf=64, num_modules=6)` | in-repo `checkpoints/IMDN_x{2,3,4}.pth` |
| LAPAR-A | https://github.com/Jia-Research-Lab/Simple-SR (`08c71e9e46ba`) | `exps/LAPAR_A_x*/network.py` + `config.py` (kernel `kernel/kernel_72_k5.pkl`) | https://drive.google.com/drive/folders/1c-KUEPJl7pHs9btqHYoUJkcMPKViObgJ (authors' re-trained models) |
| LatticeNet | **no official PyTorch code** (official = MindSpore: https://github.com/VipSuperLab/LatticeNet); unofficial port https://github.com/ymff0592/super-resolution (`7bb1ce05566b`) | `model/latticenet.py`, `n_feats=64` (777K params at ×4, matching the paper) | none in PyTorch; use the paper's PSNR numbers and mark the row |

Local clones: `src/repos/{SwinIR,MambaIR,SAFMN,DVMSR,RLFN,CARN-pytorch,IMDN,Simple-SR,LatticeNet-pytorch}`.
Training data of the baselines differs (DIV2K vs DF2K, see `BASELINES[...]['train_data']`); the
joint table should carry a footnote.

## 8. Reproducibility notes

* Bit-exact reproducibility is not guaranteed with the mamba_ssm CUDA kernels; the seed fixes
  weight init, data order and augmentation.
* `results/*.csv` carry `gpu, commit, date` per row; re-running a script upserts rows in place.
* `src/` is not a git repository at the time of writing: `commit` is `nogit`. Initialise git and
  commit before producing the final numbers so the column is meaningful.
