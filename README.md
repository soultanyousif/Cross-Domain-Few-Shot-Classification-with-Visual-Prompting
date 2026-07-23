# Cross-Domain Few-Shot Classification with Visual Prompting

A controlled comparison of four adaptation strategies for CLIP ViT-B/16 under few-shot constraints: Linear Probing, VPT-Shallow, VPT-Deep, and Full Fine-Tuning. The goal is to understand how each method behaves as the number of labeled examples per class shrinks, and how much of full fine-tuning's performance can be recovered by training a few thousand prompt parameters instead of eighty-five million backbone parameters.

## Motivation

Full fine-tuning of a vision transformer is expensive and prone to overfitting when only a handful of labeled examples are available per class, which is the regime that matters most for real-world domain adaptation. Visual Prompt Tuning (VPT) offers an alternative: freeze the backbone entirely and learn a small set of continuous prompt tokens that are injected into the transformer's input sequence. This project measures whether that trade actually pays off, and under what conditions, across three datasets with very different visual characteristics.

## Datasets

- **EuroSAT** — 10-class satellite land-use classification, 16,200 train images
- **DTD (Describable Textures Dataset)** — 47-class texture recognition, 3,760 train images
- **Flowers102** — 102-class fine-grained flower classification, 1,020 train images

These were chosen deliberately for contrast: EuroSAT is far from CLIP's pretraining distribution, DTD has a large label space with subtle inter-class differences, and Flowers102 is fine-grained but visually close to natural image data CLIP was trained on.

## Methods Compared

| Method | Trainable Parameters | % of Full Fine-Tune |
|---|---|---|
| Linear Probe | 7,690 | 0.009% |
| VPT-Shallow | 8,458 | 0.010% |
| VPT-Deep | 597,514 | 0.696% |
| Full Fine-Tune | 85,807,114 | 100% |

All methods share a frozen, pretrained CLIP ViT-B/16 vision encoder as the backbone. VPT-Shallow injects one set of learnable prompt tokens after the CLS token at the input layer only. VPT-Deep injects a fresh set of prompt tokens at every one of the 12 transformer layers, discarding the previous layer's prompts each time. Full Fine-Tune unfreezes the entire backbone and trains at a substantially lower learning rate to limit catastrophic forgetting on tiny support sets.

## Experimental Design

The project runs in two phases:

**Phase 1 — Prompt Length Ablation.** Before comparing methods head to head, the optimal number of prompt tokens is not obvious, so both VPT variants are swept across prompt lengths of {1, 4, 8, 16, 32, 64} at every shot count on EuroSAT (60 runs). The best-performing prompt length for each (method, shot count) pair is carried forward into Phase 2, rather than assuming a single default value works everywhere.

**Phase 2 — Full Cross-Method Evaluation.** All four methods are evaluated across all three datasets at shot counts {1, 2, 4, 8, 16} per class, using the prompt lengths selected in Phase 1 (60 runs). Accuracy is tracked, along with training time and parameter count, and detailed 16-shot diagnostics (per-class precision/recall and confusion matrices) are generated for the highest-data setting.

Few-shot subsets are drawn with stratified, seeded sampling to guarantee an exact and reproducible number of examples per class. Training uses mixed-precision (AMP), AdamW, and cosine learning rate annealing, with per-method learning rates and epoch counts tuned to their respective parameter budgets.

## Key Findings

**VPT-Deep wins on average, and by a meaningful margin.** Averaged across all three datasets and all shot counts, VPT-Deep reaches 73.6% mean accuracy versus 71.3% for VPT-Shallow, 70.8% for Linear Probing, and only 64.8% for Full Fine-Tuning — while training under 0.7% of the parameters that full fine-tuning requires.

**Full fine-tuning is the worst choice in the extreme low-data regime.** At 1-shot on Flowers102, full fine-tuning collapses to 31.3% accuracy while linear probing reaches 74.9% and VPT-Shallow reaches 74.2%. With 102 classes and one example each, unfreezing 85 million parameters overfits almost immediately despite the reduced learning rate.

**The advantage of full fine-tuning only appears with more data and fewer classes.** On EuroSAT, a comparatively low-class-count dataset, full fine-tuning catches up to and slightly exceeds VPT-Deep by 16-shot (93.7% vs 93.3%). On DTD and Flowers102, which have far more classes relative to the available shots, VPT-Deep stays ahead at every shot count tested.

**Prompt length interacts with shot count in a non-obvious way.** Longer prompts do not uniformly help. At 1-shot, both VPT variants show noisy, sometimes non-monotonic behavior as prompt length increases, and VPT-Deep's best 1-shot result (61.2% at length 64) is far more volatile than its 16-shot result (93.8% at length 16). At higher shot counts, moderate prompt lengths (8-16) consistently perform best, suggesting the optimal prompt length should scale with the number of available examples rather than be fixed.

**Parameter efficiency has a clear ordering.** VPT-Deep delivers the best accuracy-per-parameter trade-off among all four methods, followed by VPT-Shallow and Linear Probing at near-identical trainable parameter counts but a meaningful accuracy gap between them, with Full Fine-Tuning trailing on both axes.

<img width="1790" height="463" alt="accuracy_vs_shots" src="https://github.com/user-attachments/assets/1d7f9838-a9f7-4756-b3ef-09b8571e654d" />


## Results Summary

Mean accuracy across all shot counts, by method and dataset:

| Dataset | Full Fine-Tune | Linear Probe | VPT-Deep | VPT-Shallow |
|---|---|---|---|---|
| EuroSAT | 79.28% | 69.49% | 77.74% | 71.98% |
| DTD | 46.82% | 54.45% | 56.52% | 53.32% |
| Flowers102 | 68.34% | 88.34% | 86.64% | 88.62% |

<img width="989" height="440" alt="mean_accuracy_by_dataset" src="https://github.com/user-attachments/assets/2fb7b4a1-ae11-48b3-b499-68202e13af85" />


16-shot accuracy (highest-data setting):

| Dataset | Full Fine-Tune | Linear Probe | VPT-Deep | VPT-Shallow |
|---|---|---|---|---|
| EuroSAT | 93.74% | 81.81% | 93.30% | 87.96% |
| DTD | 71.33% | 70.64% | 73.83% | 71.01% |
| Flowers102 | 92.03% | 95.82% | 96.50% | 96.08% |

<img width="705" height="290" alt="heatmap_16shot" src="https://github.com/user-attachments/assets/7bb8784a-6344-4718-b133-fb7452549a25" />


<img width="689" height="440" alt="parameter_efficiency" src="https://github.com/user-attachments/assets/5d969837-0e69-491e-b247-f8b9ad5f0d0c" />


## Ablation Study

Before the main cross-method comparison, prompt length was swept across {1, 4, 8, 16, 32, 64} for both VPT variants at every shot count on EuroSAT, to identify an appropriate prompt length rather than assuming a single default works everywhere. The optimal prompt length shifts with shot count rather than staying fixed, which is why Phase 2 uses a per-(method, shot count) prompt length instead of one constant value.

<img width="1389" height="446" alt="phase1_prompt_length_curves" src="https://github.com/user-attachments/assets/d9ddbfbd-646e-466b-9e5f-e9fd4c13aea4" />


## Detailed Diagnostics

Row-normalized confusion matrices and full per-class precision/recall reports were generated for all four methods at the 16-shot setting. EuroSAT is shown below as a representative example; the equivalent DTD and Flowers102 confusion matrices are included in the repository under the results directory.

<img width="2380" height="509" alt="confusion_matrix_eurosat_16shot" src="https://github.com/user-attachments/assets/1eadc237-c37a-4db7-8eac-80154fb7e9bc" />


## Tech Stack

Python, PyTorch, Hugging Face Transformers (CLIP), Hugging Face Datasets, torchvision, scikit-learn, pandas, matplotlib, seaborn. Trained on a single Tesla T4 GPU.

## Reproducibility

All random seeds are fixed (`SEED = 42`) across Python, NumPy, and PyTorch, and few-shot subsets are drawn deterministically, so results are reproducible end to end from a single script.

## Possible Extensions

- Extend the prompt-length ablation to DTD and Flowers102 to check whether the optimal prompt length generalizes across domains or is EuroSAT-specific
- Add a hybrid method that combines VPT-Deep with a partially unfrozen final backbone block
- Evaluate cross-dataset transfer: train prompts on one dataset and test zero-shot on another
