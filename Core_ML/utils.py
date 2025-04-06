"""General utility functions."""

import torch
import numpy as np

# === Seeding Function ===
# Prioritize performance over strict determinism for large experiments.
def set_seed(seed=42):
    """Sets random seeds for PyTorch, NumPy for reproducibility.
       Enables cuDNN benchmark mode and disables deterministic mode for performance.
    """
    torch.manual_seed(seed)                     # Set seed for PyTorch CPU ops
    np.random.seed(seed)                        # Set seed for NumPy
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)        # Set seed for all GPUs
        # Performance Optimization: Allow non-deterministic cuDNN algorithms (faster).
        torch.backends.cudnn.deterministic = False
        # Performance Optimization: Allow cuDNN to find fastest algorithms for input sizes (faster).
        torch.backends.cudnn.benchmark = True