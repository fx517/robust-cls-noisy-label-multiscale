# Design Notes

Engineering decisions behind the implementation, and pointers for extending it.

## Architecture overview

```
main.py                          # single entry point: train / evaluate / noise detection
config/default.py                # all hyperparameters (dataclasses + per-dataset overrides)
data/dataset.py                  # IndexedImageDataset — returns (image, label, index)
data/transforms.py               # RandAugment / MixUp-CutMix / MultiScaleTTA / CLIP norm
data/sampler.py                  # class-balanced sampler (long-tail mitigation)
models/clip_lora.py              # CLIPLoRAModel — frozen ViT + LoRA + classifier (+prompt learner)
models/lora.py                   # LoRALinear + apply_lora_to_clip()
losses/elr.py                    # ELRLoss — per-sample EMA targets (N, C)
losses/double_softmax.py         # DoubleSoftmaxLoss (DSPT)
losses/gce.py                    # GCELoss — between CE and MAE
losses/combined.py               # CombinedLoss — composes the above (+ distillation)
losses/noise_filter.py           # GMM clean/noisy selection + label co-refinement
losses/ema_teacher.py            # EMA teacher (mean-teacher self-distillation)
losses/supcon.py                 # supervised contrastive loss (clean samples)
trainers/trainer.py              # training loop: AMP / accumulation / GMM / refinement / SWA
trainers/evaluator.py            # evaluation + TTA multi-view voting
utils/noise_detector.py          # post-training noise-label analysis
```

## Key design decisions

### 1. Why does the dataset return sample indices?

ELR maintains a per-sample EMA target vector `(C,)`. Since the DataLoader shuffles
batches, the identity of each sample must be carried along to locate its EMA target in
memory. `IndexedImageDataset.__getitem__` therefore returns `(image, label, index)`,
and every loss that needs per-sample state consumes `indices`.

### 2. Why are MixUp/CutMix run in "hard-label" mode?

MixUp/CutMix normally produce soft labels, but ELR's cross-entropy component needs
integer labels. The collator is therefore configured with `keep_hard_labels=True`:
images are still interpolated (strong regularization), while labels stay hard and ELR
stays exact. See `data/transforms.py::MixUpCutMixCollator`.

### 3. Where is LoRA applied?

`model.lora_target_modules` names the usual suspects (`q_proj`, `v_proj`, `k_proj`,
`out_proj`, `c_fc`, `c_proj`), and the adapter hook replaces every matching `nn.Linear` in
the frozen ViT. In practice open_clip's CLIP attention stores q/k/v as a single fused
`in_proj` parameter (`nn.MultiheadAttention`), so only three layers per block match:
`attn.out_proj`, `mlp.c_fc` and `mlp.c_proj` — **36 adapted layers** across the 12 blocks
(the training log prints `[LoRA] Applied LoRA to 36 layers`). The attention output
projection and the MLP are where the adapter capacity matters most in practice.

### 4. Why a cosine head with `hidden_dim = 512`?

The head hidden dimension is set equal to CLIP's feature dimension so that **no
projection layer** is inserted: CLIP text-encoder prototypes (from the 750 class names,
`"a photo of a {class}."`) can be injected losslessly as the classifier initialization.
A frozen copy of these prototypes (`zero_shot_anchor`) also serves as the clean
reference used by the DeFT soft labels. See `models/classifier.py` and
`models/clip_lora.py::_encode_class_names`.

### 5. How does self-distillation work without a second model?

There is no separately trained teacher: an EMA copy of the model parameters
(`use_self_distillation`, `ema_teacher_alpha=0.999`, warmup 200 steps) acts as the
teacher, providing soft targets that smooth the noisy gradient signal.

### 6. GMM selection, refinement and the semi-supervised loop

After `gmm_warmup_epochs`, a 2-component GMM is fitted on the per-sample loss history
to estimate each sample's posterior probability of being clean. The clean subset drives
SupCon; the noisy subset drives pseudo-label cross-entropy (ELR target argmax as
pseudo-label) and progressive label co-refinement. All three components are switchable
via `LossConfig`.

### 7. Honest validation

`train.honest_validation=True` evaluates the model against the denoised EMA targets of
the validation split instead of the noisy given labels, giving a more faithful signal
for model selection when labels are unreliable.

## Common development tasks

### Add a new loss

1. Create a file under `losses/` implementing `forward(outputs, labels)` or
   `forward(indices, outputs, labels)`.
2. Export it from `losses/__init__.py`.
3. Wire it into `losses/combined.py::CombinedLoss` (initialization + forward term).
4. Add its hyperparameters to `LossConfig` in `config/default.py`.

### Add a new augmentation

1. Extend `get_train_transforms()` in `data/transforms.py`.
2. If it is label-level (MixUp-family), adjust `MixUpCutMixCollator`.
3. Add the switch to `DataConfig`.

### Hyperparameter search

Any config field can be overridden from the command line without touching the code:

```bash
python main.py --dataset webfg400 --data_root ./data \
    --overrides train.lr=5e-5 loss.elr_lambda=4.0 model.lora_r=8 model.classifier_dropout=0.5
```

### Quick debugging run

```bash
python main.py --dataset webfg400 --data_root ./data \
    --overrides train.epochs=5 train.warmup_epochs=1 \
        data.use_randaug=False train.use_self_distillation=False train.batch_size=16
```

## Reproduction notes

1. **Seeds** — `--seed 42` for reproducibility.
2. **CUDA** — PyTorch 2.0+, CUDA 11.8+.
3. **Memory** — 224-px training with batch 64 needs ~12 GB; 384-px needs more; enable
   gradient accumulation / AMP as needed (the final chain used effective batch 128).
4. **CLIP weights** — use the OpenAI `ViT-B-32` weights, either downloaded by
   `open-clip-torch` or placed as `data/ViT-B-32_openai.bin`.
5. **Training time** — reference order of magnitude: tens of hours for the full 40-epoch
   384 chain on a single modern GPU (the competition run used a single 24 GB card).
