# Results & Experiments

This document collects the evaluation protocols, the final submitted results, and the
development-time ablations behind the final recipe. All numbers are Top-1 accuracy.

## 1. Evaluation protocols (three different rulers)

| Protocol | Labels | Distribution | Role |
|---|---|---|---|
| Official test set | human-verified | class-balanced | leaderboard score |
| Stratified class-balanced holdout | human-verified | class-balanced | honest offline evaluation |
| Internal noisy validation | noisy (train-split) | long-tailed | training-time monitoring only |

- **Stratified holdout**: 9:1 per-class split of the training set (`seed=42`), at least
  one held-out sample per class — every tail class is represented, staying close to the
  class-balanced test distribution. Implemented in `data/dataset.py::stratified_split`,
  reproducible across processes.
- **Internal noisy validation**: random 10% of the noisy training data with its noisy
  labels; convenient during development but optimistic — it shares the noise and the
  long-tail distribution of the training pool.

## 2. Final submission results

| Protocol | Inference | Top-1 |
|---|---|---|
| Official test set (leaderboard) | MultiScaleTTA, 20 views | **71** pts |
| Stratified class-balanced holdout | single view | 74.25% |
| Stratified class-balanced holdout | MultiScaleTTA, 20 views | **75.09%** |
| Internal noisy validation | single view | 79.49% |

Submitted artifacts: `swa_model.pth` (SWA weights of the 384 chain,
`exp_name = fixres384_hd512_dsp_deft`) and `results/pred_results.csv` — the official
prediction file: 37,444 headerless `filename,class_id` rows (class ids 0–749), produced by
`scripts/predict_test.py --use_tta` with the 20-view MultiScaleTTA. The checkpoint is
attached to the [`v1.0.0` release](https://github.com/fx517/robust-cls-noisy-label-multiscale/releases/tag/v1.0.0).

## 3. Multi-scale TTA

Design (`data/transforms.py::MultiScaleTTA`):

- Each image is resized to two scales — `image_size + 32` and `image_size + 64`
  (416 and 448 for the 384 model) — and each scale is expanded with `TenCrop`
  (4 corners + center, plus their horizontal flips) → **2 × 10 = 20 views**.
- Crops are taken from images **larger than the crop size**, so views are genuinely
  distinct (an earlier plain 224-px TenCrop overlapped almost completely and gave
  little benefit).
- Logits are summed/averaged over views; the forward pass runs in fp16 chunks
  (`tta_chunk=512`) to bound the memory peak at 384–448 px.

Effect: **+0.84 pt** on the stratified class-balanced holdout (74.25% → 75.09%),
without touching the training pipeline — the most reliable inference-side gain under
the single-model constraint.

## 4. Development-time scale sweeps

Held-out split of 14,869 samples (frozen manifest), center view only, model rebuilt at
each target scale via FixRes positional-embedding interpolation. `acc_given` uses the
noisy given labels, `acc_clean` uses denoised (corrected) labels; `resq` re-encodes
images to 375 px + JPEG q78 to mimic the official test-image pipeline (resolution and
compression).

(a) 384-trained model (`fusa_224H_384fix`):

| Test scale | acc_given (orig) | acc_clean (orig) | acc_given (resq) | acc_clean (resq) |
|---|---|---|---|---|
| 320 | 75.11 | 78.08 | 74.69 | 77.64 |
| 352 | 75.34 | 78.26 | 74.96 | 77.87 |
| 384 | 75.61 | 78.63 | 75.09 | 78.12 |
| 416 | 75.55 | 78.48 | — | — |

The optimum sits at (or slightly above) the native resolution, which is why the
MultiScaleTTA scales start at `image_size + 32`.

(b) An earlier 224-native model prefers an upscale of ≈ 1.29×:

| Test scale | 224 | 256 | 288 | 320 | 352 | 384 |
|---|---|---|---|---|---|---|
| acc_given | 73.99 | 74.01 | **74.30** | 74.03 | 73.20 | 72.06 |

These sweeps were run during development on the fine-tuning chain and are reported for
completeness; only the numbers in §2 come from the submitted evaluation.

## 5. Development history (internal validation protocol)

Single-variable experiments on the project's internal validation protocol
(single-crop val unless noted), documented in the experiment scripts:

| Experiment | Change | Val Top-1 |
|---|---|---|
| Baseline (`reproduce_default_baseline`) | default config, 224 px | ~74.8%* |
| `sv1_fusion_neg1` | `prompt_fusion_weight` 0.2 → -1.0 | 75.81% |
| `sv2_fusion_neg3` | fusion -1.0 → -3.0 (text branch nearly off) | < sv1 |
| `fixres384` | 224 → 384 px (FixRes) | 78.64% (+2.83) |
| MultiScaleTTA (fix of the earlier overlapping TenCrop) | inference-only | 75.81% → 84.50%* |

\* Numbers as reported in the project's internal experiment logs; they use the internal
noisy-validation protocol and are not directly comparable to §2.

Take-aways:

- `prompt_fusion_weight = -1.0` (sigmoid ≈ 0.27) was the best of the tested fusion
  initializations — the text branch helps, but only with a conservative weight.
- FixRes 384 was the single strongest training-side change (+2.83 pt internal).
- The multi-scale TTA rebuild (distinct crops on upscaled images + flips) was the
  single strongest inference-side change.
