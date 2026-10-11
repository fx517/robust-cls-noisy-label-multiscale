*English translation of the team's Chinese technical report submitted to AIC-2026 Track 4 (网络监督细粒度图像识别).*

# Fine-Grained Image Recognition with Noisy Labels Based on Vision-Language Models and Robust Loss Optimization

Global Campus Artificial Intelligence Algorithm Elite Competition · Algorithm Challenge · Problem 4 (Web-Supervised Fine-Grained Image Recognition)

Team ID: AIC-2026-98713574

Team name: Team Xuanruihan (轩瑞涵)

Team captain: Feng Xuan (冯轩)

Submitted model: CLIP ViT-B/32 + LoRA + ELR + DSPT + DeFT (single model · single weight)

## Abstract

This solution addresses the web-supervised fine-grained image recognition task: the training data is crawled from the web, its labels are noisy and its class distribution is long-tailed, whereas the test set used for evaluation consists of uniformly distributed, manually annotated data, requiring the model to be discriminative over the entire long-tailed range. Targeting the three core challenges of label noise, long-tailed distribution and subtle inter-class differences, the solution takes the frozen vision-language model CLIP ViT-B/32[1] as its backbone, achieves parameter-efficient fine-tuning through LoRA[2] low-rank adaptation, and injects the zero-shot class priors provided by the CLIP text encoder into the cosine classification head.

In terms of loss design, the solution combines early-learning regularization ELR[3] and the dual softmax loss DSPT[4] to suppress the model's memorization of noisy labels at the gradient level; drawing on the idea of DeFT[5], it constructs clean soft labels from CLIP zero-shot text prototypes to strip away the interference of noisy hard labels, supplemented by supervised contrastive learning SupCon[7] and semi-supervised pseudo-label regularization. At the training level it adopts Gaussian mixture model based dynamic sample selection[6], collaborative label correction, exponential moving average teacher self-distillation[8] and stochastic weight averaging SWA[9] to achieve stable noise-robust training; in the inference stage it introduces multi-scale ten-crop test-time augmentation (20-view voting).

On the official test set it achieves a Top-1 accuracy of 71 points; under class-balanced evaluation on the stratified hold-out set, single-scale inference reaches 74.25% and multi-scale TTA reaches 75.09%, validating the effectiveness of each component of the solution for the noisy-label fine-grained recognition task.

**Keywords:** noisy label learning; fine-grained image recognition; vision-language models; LoRA; early-learning regularization; dual softmax; test-time augmentation

## 1 Introduction

### 1.1 Background

Fine-grained image recognition requires a model to distinguish subcategories that look extremely similar within the same superclass (e.g. different models of aircraft or cars), and is a classic hard problem in computer vision. This competition problem superimposes two additional real-world constraints: first, the training data is crawled from the web and its labels have not been manually verified, so there are a large number of mislabeled (noisy) samples; second, the number of samples per class in the training set is extremely imbalanced (a long-tailed distribution), with head classes having abundant samples while tail classes have very few, and some tail classes having only a handful of training images. The test set used for evaluation, by contrast, consists of uniformly distributed, manually annotated data — precisely the scoring dimension explicitly stated in the problem statement: whether the model can maintain generalization under the distribution shift of long-tailed training and balanced testing.

### 1.2 Analysis of the Difficulties

The above setting gives rise to three intertwined difficulties: first, under noisy labels the standard cross-entropy causes the model to gradually "memorize" the wrong labels, leading to degraded generalization; second, the long-tailed distribution leaves the representations of tail classes insufficiently learned in high-dimensional space and highly prone to collapse; third, the inter-class differences in fine-grained tasks are subtle, placing higher demands on feature discriminability and class priors. At the same time, the competition explicitly forbids multi-model ensembling and allows only a single model and a single weight, further restricting the available means of improvement.

### 1.3 Approach and Main Contributions

Based on the above analysis, the solution takes "strong priors + conservative adaptation" as its core idea: it introduces CLIP[1]'s zero-shot prior as a clean reference, and constrains the model's adaptation to noisy labels with noise-robust losses and training strategies. The main contributions are as follows:

- It proposes a joint training objective of ELR and the dual softmax loss DSPT, suppressing the contamination of the model by noisy labels at the gradient level;
- Drawing on the idea of DeFT, it constructs clean soft labels from CLIP zero-shot text prototypes, achieving label denoising under weak supervision;
- It builds a multi-strategy collaborative robust training framework of "GMM sample selection + label correction + EMA self-distillation + SWA";
- It designs multi-scale ten-crop test-time augmentation, stably improving inference accuracy under the single-model constraint.

## 2 Related Work

### 2.1 Efficient Fine-Tuning of Vision-Language Models

Contrastively pre-trained vision-language models such as CLIP[1] learn strongly generalizable aligned representations from massive image-text pairs, significantly improving downstream zero-shot capability. To further improve performance on downstream tasks, researchers have proposed prompt learning such as CoOp[10] and parameter-efficient fine-tuning such as LoRA[2], which complete adaptation with very few trainable parameters while keeping the backbone frozen. This solution adopts a combination of LoRA and zero-shot text priors, completing fine-grained adaptation at a 384 resolution with about 5.25% trainable parameters.

### 2.2 Noisy Label Learning

Noisy label learning is broadly divided into two categories: loss correction and sample selection. In terms of loss correction, early-learning regularization ELR[3] prevents memorization of wrong labels in later stages through an exponential moving average of each sample's predicted distribution; more recently, DSPT[4] further points out that in vision-language models with strong priors, wrong labels produce abnormally large gradients, and proposes a dual softmax to adaptively suppress these gradients. In terms of sample selection, DivideMix[6] uses a two-component Gaussian mixture model to dynamically separate clean and noisy samples and combines this with semi-supervised learning. In addition, DeFT[5] shows that the alignment prior of vision-language models can serve as an effective noisy-label detection signal. This solution comprehensively absorbs the above ideas.

### 2.3 Representation Enhancement and Stabilization

Supervised contrastive learning SupCon[7] strengthens discriminative features by pulling same-class samples closer and pushing different-class samples apart; the moving-average teacher[8] and stochastic weight averaging SWA[9] provide smooth and stable signals at the prediction level and at the weight level respectively. These methods are integrated in this solution to improve feature quality and training stability.

## 3 Data Processing

### 3.1 Data Overview

The competition provides a desensitized fine-grained dataset with 750 classes in total; the training set contains about 149,000 images (about 200 per class on average), organized by class folders, with noisy labels and a long-tailed distribution; the test set is class-balanced manually annotated data with 37,444 images in total. The training labels have not been manually verified and are the main source of noise that the solution must address; the distribution shift between the two kinds of data is shown in Figure 1.

*Figure 1 Comparison of the class distributions of the training set (long-tailed) and the test set (class-balanced) (schematic)*

### 3.2 Data Loading and Deduplication

The data is organized in a "train/{class name}/image" class directory structure. The loader traverses each class directory, collects images by whitelisted suffixes (.jpg/.jpeg/.png/.bmp/.webp, including uppercase), and sorts and deduplicates the sample list to guarantee cross-process reproducibility. Each image is uniformly converted to RGB on read and normalized with the standard CLIP normalization (mean (0.4815, 0.4578, 0.4082), standard deviation (0.2686, 0.2613, 0.2758)). To satisfy the requirement of the ELR loss to maintain a per-sample target index, the dataset returns a triple (image, label, global index) for each sample.

### 3.3 Stratified Hold-Out Split

To obtain an offline evaluation close to true generalization, the solution splits the training/validation sets in a class-stratified manner (ratio 9:1, seed=42) and ensures that at least 1 held-out sample not involved in training is kept for every class, giving full coverage of the long-tailed tail classes. Compared with a random split, this split avoids leakage of training samples into the validation set, thus enabling an honest evaluation on the class-balanced held-out set, which is identically distributed with the official balanced test set.

### 3.4 Augmentation and Mixing

The training stage uses RandAugment (n=3, m=15) together with random resized cropping, horizontal flipping, color jitter and random erasing to improve robustness to noisy labels. In addition, MixUp (α=0.5) and CutMix (α=1.0) are used for image-level mixing regularization; to avoid a conflict between the mixed soft labels and the per-sample EMA target of ELR, the original hard labels are retained during mixing (keep_hard_labels=True), using only the strong regularization effect of the mixed images themselves.

### 3.5 Long-Tail and Balanced Sampling

The training set has a long-tailed distribution while the test set is uniformly distributed, so a distribution shift exists between them. The solution provides class-balanced weighted sampling based on the effective number of samples (Class-Balanced WeightedRandomSampler) as an optional switch, used to oversample tail classes when needed. Because the readings of this direction on the noisy validation set are distorted, the solution keeps it as a configurable option, and the final submission is based on the class-balanced evaluation of the stratified hold-out set.

## 4 Model Architecture

### 4.1 Overall Framework

The overall structure of the model is shown in Figure 2: a frozen CLIP ViT-B/32 vision encoder with LoRA adapters extracts 512-dimensional normalized features, which pass through a cosine classification head to output class logits; the classification head prototypes are initialized zero-shot by the CLIP text encoder, and a frozen copy of the zero-shot prototypes is retained as a clean reference for the DeFT soft labels.

*Figure 2 Overall model architecture*

### 4.2 Backbone and Efficient LoRA Fine-Tuning

The backbone adopts the OpenAI pre-trained CLIP ViT-B/32[1]; the vision encoder is a 12-layer Transformer that outputs 512-dimensional features. The backbone parameters remain frozen during training, and low-rank trainable matrices are injected via LoRA[2] into 36 linear layers in total — the attention output projection (`out_proj`) and the MLP projections (`c_fc`, `c_proj`) of every one of the 12 transformer blocks (in open_clip's CLIP attention, q/k/v are a single fused `in_proj` parameter rather than separate layers, so no LoRA is attached to them). Rank r=64, scaling alpha=64, dropout=0.1:

$$ W' = W + \frac{\alpha}{r} B A $$

Thanks to LoRA, only about 8.38 million of the model's roughly 160 million parameters need to be trained (5.25%), obtaining sufficient adaptation capacity while keeping the pre-trained knowledge from being damaged.

### 4.3 High-Resolution FixRes

To further improve the perception of local details needed for fine-grained discrimination, the input resolution is raised from 224×224 to 384×384 (FixRes). The pre-trained ViT-B/32 positional embedding corresponds to 7×7=49 patches; the solution bicubically interpolates the positional embedding to 12×12=144 patches while keeping the CLS position unchanged, thereby gaining high-resolution capability on the frozen backbone at almost zero cost.

### 4.4 Cosine Classification Head and Zero-Shot Prior

The classification head uses a cosine prototype classifier rather than an ordinary fully connected layer: after L2-normalizing the features and each class prototype separately, the cosine similarity is computed and then multiplied by a learnable logit_scale to obtain the logits. This design is robust to feature scale and more stable under noisy labels. Crucially, the solution sets the hidden dimension of the classification head equal to the CLIP feature dimension (512 dimensions, no projection), so that the prototypes obtained by encoding the 750 class names (template "a photo of a {class}") with the CLIP text encoder can be injected "losslessly" as initialization; at the same time, this zero-shot prototype is stored as a frozen anchor buffer for use by the DeFT soft labels. This zero-shot initialization endows the model with an accurate class semantic prior, significantly alleviating the representation collapse of tail classes caused by insufficient samples. The overall model structure and parameter configuration are summarized in Table 1.

| Component | Configuration |
| --- | --- |
| Backbone | CLIP ViT-B/32 (OpenAI pre-trained, frozen) |
| Parameter-efficient fine-tuning | LoRA, r=64, alpha=64, dropout=0.1 — 36 adapted layers (`attn.out_proj` + MLP `c_fc`/`c_proj` per block) |
| Input resolution | 384×384 (FixRes, positional embedding bicubically interpolated 49→144) |
| Classification head | cosine prototype classifier, hidden_dim=512, CLIP text zero-shot initialization |
| Trainable parameters | about 8.38 million / 160 million for the whole model (5.25%) |

*Table 1 Model structure and parameter configuration*

## 5 Loss Functions

### 5.1 Early-Learning Regularization ELR

ELR[3] maintains an exponential moving average (EMA) target of the predicted distribution for each training sample and treats it as an "early stable belief":

$$ \mathcal{L}_{\text{ELR}} = \mathcal{L}_{\text{CE}}(f(x), y) + \lambda \log\bigl(1 - \langle t_{\text{ema}}, \mathrm{softmax}(f(x)) \rangle\bigr) $$

where $t_{\text{ema}}$ is the per-sample EMA target (β=0.9) and λ=5.0. As shown in Figure 3, the standard cross-entropy gradually memorizes noisy labels in the later stage of training, causing generalization degradation, whereas ELR preserves the clean patterns acquired from early learning while penalizing predictions that deviate from the EMA target through the regularization term, thereby preventing error accumulation and stabilizing generalization.

*Figure 3 The suppressing effect of ELR on memorization of noisy labels (schematic)*

### 5.2 Dual Softmax Loss DSPT

DSPT[4] normalizes the logits twice in sequence before computing the cross-entropy:

$$ p = \mathrm{softmax}(z), \qquad q = \mathrm{softmax}(p), \qquad \mathcal{L}_{\text{DSPT}} = -\log q[y] $$

The "saturation region" it introduces adaptively suppresses to near zero the gradients of high-error noisy samples that conflict with CLIP's strong prior and produce extreme gradients, while retaining effective gradients for samples that carry discriminative information. This loss has no extra hyperparameters and is bounded, fitting naturally with the prior characteristics of vision-language models (weight 1.0).

### 5.3 DeFT Soft Label Fusion

Inspired by DeFT[5]'s idea of "using vision-language model alignment priors to resist noise", the solution takes the frozen zero-shot text prototypes (anchor) as a clean reference, computes the clean probability distribution $p_{\text{zs}}$ of the visual features over these prototypes, and uses it as a soft label to supervise the model output:

$$ p_{\text{zs}} = \mathrm{softmax}\!\Bigl(\frac{\text{features} \cdot W_{\text{anchor}}^{\top} \cdot \text{scale}}{T}\Bigr), \qquad \mathcal{L}_{\text{DeFT}} = -\sum p_{\text{zs}} \cdot \log \mathrm{softmax}(z) $$

Since $p_{\text{zs}}$ originates from the CLIP zero-shot prior rather than from the noisy hard labels, this loss guides the model toward the clean class distribution and strips away the interference of hard-label noise (weight 0.3, temperature T=1.0).

### 5.4 Supervised Contrastive Loss SupCon

A supervised contrastive loss SupCon[7] (weight 0.1, temperature 0.07) is applied on the clean samples determined by the GMM, using the features extracted by the moving-average teacher as positive pairs, pulling same-class features closer to enhance fine-grained discriminability.

### 5.5 Semi-Supervised Pseudo-Label Regularization

For the noisy subset determined by the GMM, the noisy hard labels are no longer used; instead, the argmax of the ELR target is used as a pseudo-label to apply cross-entropy (weight 0.3), steering the noisy samples toward the model's stable early belief and achieving self-correction.

## 6 Training Strategy

### 6.1 GMM Dynamic Sample Selection

Following DivideMix[6], a two-component Gaussian mixture model is fitted to the history of the per-sample cross-entropy loss. As shown in Figure 4, clean samples are concentrated in the low-loss region while noisy samples are concentrated in the high-loss region; based on this, the posterior probability of each sample belonging to the "clean/noisy" component is estimated online, achieving dynamic sample separation (10-epoch warmup, momentum 0.9, posterior threshold 0.5). The separation results are also used to drive the clean-sample constraint of SupCon and the semi-supervised pseudo-label regularization.

*Figure 4 GMM-based separation of clean/noisy samples (schematic)*

### 6.2 Collaborative Label Correction

For samples judged to be noisy, their labels are progressively corrected using the model's predicted probabilities, with an initial trust level of alpha=0.3 that gradually rises to 0.7 during training, correcting errors progressively without over-trusting the model.

### 6.3 EMA Teacher Self-Distillation

A momentum moving-average teacher model[8] is maintained (alpha=0.999, 200-step warmup), and its stably produced soft labels are used to perform KL distillation on the student model (temperature 3.0, weight 0.3), smoothing the noisy gradient signal.

### 6.4 Stochastic Weight Averaging SWA

In the final stage of training (the last 25% of epochs) the model weights are averaged[9], yielding a flatter solution with better generalization; the final submission uses the SWA weights.

### 6.5 Optimizer and Schedule

The AdamW optimizer is used (lr=5e-4, weight_decay=0.05, betas=(0.9,0.999)), together with a OneCycle learning-rate schedule (5-epoch warmup, minimum lr 1e-7); training runs for 40 epochs, with a single-card batch of 64 and gradient accumulation over 2 steps giving an effective batch of 128; AMP mixed precision (fp16) is used globally, with a gradient clipping norm of 1.0. The overall configuration of training and inference is summarized in Table 2.

| Item | Configuration |
| --- | --- |
| Loss function | ELR(λ=5.0, β=0.9) + DSPT(1.0) + DeFT(0.3, T=1.0) + SupCon(0.1, τ=0.07) + semi-supervised (0.3) |
| Optimizer | AdamW, lr=5e-4, weight_decay=0.05, betas=(0.9, 0.999) |
| Learning rate schedule | OneCycle, 5-epoch warmup, min_lr=1e-7 |
| Training setup | 40 epochs, effective batch 128, AMP fp16, gradient clipping norm 1.0 |
| Sample selection | GMM, 10-epoch warmup, momentum 0.9, posterior threshold 0.5 |
| Distillation and averaging | EMA teacher α=0.999, 200-step warmup; SWA over the last 25% of epochs |
| Inference augmentation | MultiScaleTTA, scales 416/448, ten-crop + horizontal flip, 20 views |

*Table 2 Training and inference configuration*

## 7 Inference Details

The inference stage uses multi-scale ten-crop test-time augmentation (MultiScaleTTA): the input is resized to two scales, 416 and 448 (i.e. 384+32 and 384+64), and ten crops are taken at each scale (four corners + center, and their horizontal flips), giving 20 views in total; the logits of the views are summed and averaged, and then voted on to obtain the final prediction. At test time, fp16 chunked forward passes are likewise used to control the peak memory footprint, and the final predictions are written to pred_results.csv in the form of "filename, predicted class index". Without increasing model capacity or modifying the training pipeline, this strategy stably improves single-model accuracy through multi-view aggregation, and is the most effective inference-side gain under the single-model constraint.

## 8 Experiments

### 8.1 Datasets and Evaluation Protocol

The evaluation adopts a triple protocol in order to distinguish different yardsticks: in-training-set noisy validation (noisy labels, long-tailed distribution), stratified hold-out class-balanced evaluation (manually annotated, unseen samples present for every class, class-balanced), and the official test set (manually annotated, entirely new test images). The latter two better reflect true generalization ability, and all evaluation metrics are Top-1 accuracy.

### 8.2 Experimental Results

Figure 5 reports the Top-1 accuracy of this solution under different evaluation protocols. The labels of the in-training-set noisy validation come from the same source as training and are noisy, so its values are relatively high; the stratified hold-out set and the official test set are both manually annotated and class-balanced, and therefore better reflect true generalization ability.

*Figure 5 Comparison of Top-1 accuracy under different evaluation protocols*

As can be seen from Figure 5, multi-scale TTA brings a stable improvement of about +0.84 points on the class-balanced hold-out set (74.25% → 75.09%), validating the effectiveness of the inference-side augmentation; the gap between the official test score and the hold-out set is mainly caused by the distribution shift of the entirely new test images and is within the expected range.

## 9 Conclusion

Targeting the noisy-label long-tailed fine-grained recognition task, this solution takes "strong priors + conservative adaptation" as its main thread and integrates a series of techniques including LoRA efficient fine-tuning, zero-shot prior injection, the joint robust losses ELR and DSPT, DeFT soft-label denoising, GMM sample selection, EMA self-distillation and SWA, forming a training scheme that converges stably and generalizes well under the single-model constraint, and ultimately achieves 71 points on the official test set. Future work may further explore more accurate representation enhancement for tail classes, as well as robustification strategies against distribution shift on entirely new data.

## References

[1] Radford A, Kim J W, Hallacy C, et al. Learning transferable visual models from natural language supervision[C]//Proceedings of the 38th International Conference on Machine Learning (ICML). PMLR, 2021: 8748-8763.

[2] Hu E J, Shen Y, Wallis P, et al. LoRA: Low-rank adaptation of large language models[C]//International Conference on Learning Representations (ICLR). 2022.

[3] Liu S, Niles-Weed J, Razavian N, et al. Early-learning regularization prevents memorization of noisy labels[C]//Advances in Neural Information Processing Systems 33 (NeurIPS). 2020: 5869-5879.

[4] Li J, Qi J, Zhou S, et al. Intrinsic gradient suppression for label-noise prompt tuning in vision-language models[C]//Proceedings of the 43rd International Conference on Machine Learning (ICML). 2026. arXiv:2605.00591.

[5] Wei T, Li H T, Li C S, et al. Vision-language models are strong noisy label detectors[C]//Advances in Neural Information Processing Systems 37 (NeurIPS). 2024. arXiv:2409.19696.

[6] Li J, Socher R, Hoi S C H. DivideMix: Learning with noisy labels as semi-supervised learning[C]//International Conference on Learning Representations (ICLR). 2020.

[7] Khosla P, Teterwak P, Wang C, et al. Supervised contrastive learning[C]//Advances in Neural Information Processing Systems 33 (NeurIPS). 2020: 18661-18673.

[8] Tarvainen A, Valpola H. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results[C]//Advances in Neural Information Processing Systems 30 (NeurIPS). 2017.

[9] Izmailov P, Podoprikhin D, Garipov T, et al. Averaging weights leads to wider optima and better generalization[C]//Proceedings of the 34th Conference on Uncertainty in Artificial Intelligence (UAI). 2018: 876-885.

[10] Zhou K, Yang J, Loy C C, et al. Learning to prompt for vision-language models[J]. International Journal of Computer Vision, 2022, 130(9): 2337-2348.
