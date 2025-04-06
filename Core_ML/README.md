

# Robust Loss Functions for Noisy Label Learning 

This repository contains a PyTorch implementation for training neural networks on datasets with noisy labels. The code evaluates various loss functions designed to be robust against label noise.

## Overview

Training deep neural networks requires large amounts of labeled data, but the process of collecting accurate labels can be challenging. This project explores different loss functions designed to handle noisy labels effectively, improving model robustness in real-world scenarios.

## Features

- Implementation of multiple loss functions for dealing with noisy labels
- Support for both symmetric and asymmetric noise
- Complete training pipeline for CIFAR-10 dataset
- Visualization tools for result analysis

## Loss Functions

The implemented loss functions are organized into three groups:

### Vanilla Losses
- **CE**: Standard Cross-Entropy
- **MAE**: Mean Absolute Error 
- **RCE**: Reverse Cross-Entropy

### Normalized Losses
- **NCE**: Normalized Cross-Entropy
- **NRCE**: Normalized Reverse Cross-Entropy
- **NFL**: Normalized Focal Loss

### APL (Active Passive Losses)
- **NCE+RCE**: Normalized Cross-Entropy + Reverse Cross-Entropy
- **NCE+MAE**: Normalized Cross-Entropy + Mean Absolute Error
- **NFL+RCE**: Normalized Focal Loss + Reverse Cross-Entropy
- **NFL+MAE**: Normalized Focal Loss + Mean Absolute Error
- **NFL+NCE**: Normalized Focal Loss + Normalized Cross-Entropy
- **MAE+RCE**: Mean Absolute Error + Reverse Cross-Entropy

## Noise Types

The code supports two types of label noise:

- **Symmetric Noise**: Labels are randomly flipped to any other class with equal probability
- **Asymmetric Noise**: Labels are flipped to specific classes based on a pre-defined transition matrix (mimicking real-world confusion patterns)

## Experiments

The experiment suite evaluates all loss functions against varying noise levels:

- **Symmetric Noise**: 0%, 20%, 40%, 60%, 80%
- **Asymmetric Noise**: 10%, 20%, 30%, 40%

## Requirements

- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [tqdm](https://github.com/tqdm/tqdm)

Install dependencies with:
```bash
pip install torch torchvision numpy matplotlib tqdm
```
