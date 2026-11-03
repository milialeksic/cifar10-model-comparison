# CIFAR-10 Model Comparison

 Milica Aleksic May 2025

## Introduction

Comparison of three models on the CIFAR-10 image classification dataset: a custom CNN, ResNet (library), and a Vision Transformer (ViT). The goal is to understand how different architectures and training techniques affect classification accuracy. Experiments cover data augmentation, learning rate, and weight decay variations.

## Training Strategy

All models trained with Adam optimizer, cross-entropy loss, batch size 128, up to 30 epochs. Best model checkpoint saved based on validation accuracy.

**Techniques used:** data augmentation (random horizontal flips, random cropping), L2 regularization (weight decay), early stopping via best-checkpoint saving.

**Configurations tested:**

| Experiment Type | Data Augmentation | Weight Decay | Learning Rate |
|---|---|---|---|
| Default | No | 0.0 | 0.001 |
| Augmentation | Yes | 0.0 | 0.001 |
| Weight Decay | No | 0.0005 | 0.001 |
| Learning Rate | No | 0.0 | 0.0005 |
| Aug. + Weight Decay | Yes | 0.0005 | 0.001 |
| Aug. + Learning Rate | Yes | 0.0 | 0.0005 |
| Aug. + Weight Decay + LR | Yes | 0.0005 | 0.0005 |

## Results Summary

### ResNet
| Experiment Type | Test Accuracy | Test Loss |
|---|---|---|
| Augmentation | 0.8129 | 0.5638 |
| Aug. + Weight Decay + LR | 0.8027 | 0.5803 |
| Aug. + Weight Decay | 0.8004 | 0.6011 |
| Aug. + LR | 0.7972 | 0.6397 |
| Weight Decay | 0.7492 | 1.1956 |
| Default | 0.7394 | 1.1992 |
| Learning Rate | 0.7166 | 1.4732 |

### Custom CNN
| Experiment Type | Test Accuracy | Test Loss |
|---|---|---|
| Augmentation | 0.8162 | 0.5403 |
| Aug. + Weight Decay | 0.8137 | 0.5566 |
| Aug. + LR | 0.8130 | 0.5507 |
| Aug. + Weight Decay + LR | 0.8072 | 0.5647 |
| Learning Rate | 0.7541 | 1.1898 |
| Weight Decay | 0.7481 | 1.0040 |
| Default | 0.7468 | 1.0322 |

### Vision Transformer (ViT)
| Experiment Type | Test Accuracy | Test Loss |
|---|---|---|
| Aug. + Weight Decay | 0.7611 | 0.6976 |
| Augmentation | 0.7588 | 0.6980 |
| Aug. + LR | 0.7410 | 0.7546 |
| Learning Rate | 0.4248 | 2.8901 |
| Weight Decay | 0.4137 | 3.0906 |
| Aug. + Weight Decay + LR | 0.4342 | 2.3417 |
| Default | 0.2917 | 2.5535 |

## Conclusions

- **ResNet** was the most stable and best-performing overall.
- **CNN** performed well with augmentation and was sensitive to hyperparameter tuning.
- **ViT** required careful tuning; default and over-regularized settings led to poor results. Augmentation alone did not help ViT as much as it did the other models.
- Data augmentation was the single most effective technique across all models.
