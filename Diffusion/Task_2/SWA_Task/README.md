
# Sliding Window Attention (SWA) for Diffusion Models

## Overview

This project implements and evaluates Sliding Window Attention mechanisms in diffusion models. Sliding Window Attention is an efficient attention variant that restricts the receptive field of self-attention to a local window around each token, making it computationally efficient for processing long sequences while maintaining good performance.

## Implementation Details

The implementation is contained in the Jupyter notebook `SWA.py`, which includes:

- Model architecture with Sliding Window Attention
- Training pipeline for diffusion models
- Evaluation metrics and visualization
- Hyperparameter configurations

## Key Features

- Efficient attention mechanism for diffusion models
- Reduced computational complexity compared to full attention
- Scalable approach for generating high-quality images
- Compatible with various diffusion model architectures

## Results

After training for 120 epochs, the model demonstrates:
- Stable convergence with consistent sample quality
- Learning of image distributions
- Slight Computational efficiency compared to baseline models
  
## Model Checkpoints

Trained model checkpoints (after 120 epochs) are available.

## Dependencies

- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm
- diffusers



## Acknowledgements

This work was completed as part of the SAiDL Spring Assignment 2025.
```
