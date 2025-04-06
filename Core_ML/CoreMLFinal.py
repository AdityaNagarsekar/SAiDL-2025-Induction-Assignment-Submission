# === Import Statements ===
# --- Core PyTorch & NN ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# --- Distributed Training (Placeholders, not fully implemented in main path) ---
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
# --- Vision & Data Handling ---
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
# --- Numerical & Utilities ---
import numpy as np
from tqdm import tqdm # Progress Bars
import os               # File system operations
import gc               # Garbage collector interface (for memory management)
import time             # Time tracking
from pathlib import Path # Object-oriented paths
from collections import defaultdict # Default dictionary for results storage
# --- Experiment Configuration & Reporting ---
import argparse         # Command-line argument parsing
import json             # Reading/writing experiment configs and results
import pandas as pd     # Data analysis and creating summary tables
from datetime import datetime # Timestamping output directories
# --- Performance ---
# Import AMP utilities correctly based on PyTorch version (>=1.6)
from torch.amp import autocast, GradScaler
# --- External Dependencies (Plotting) ---
# Assumes a separate file 'plotting_utils.py' contains plotting functions.
# A dummy version is created if not found, allowing the core logic to run.
from plotting_utils import plot_comprehensive_results

# === Seeding Function ===
# Design Choice (Iter 6): Prioritize performance over strict determinism for large experiments.
def set_seed(seed=42):
    """Sets random seeds for PyTorch, NumPy for reproducibility.
       Enables cuDNN benchmark mode and disables deterministic mode for performance.
    """
    torch.manual_seed(seed)                     # Set seed for PyTorch CPU ops
    np.random.seed(seed)                        # Set seed for NumPy
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)        # Set seed for all GPUs
        # Performance Optimization (Iter 6): Allow non-deterministic cuDNN algorithms (faster).
        torch.backends.cudnn.deterministic = False
        # Performance Optimization (Iter 6): Allow cuDNN to find fastest algorithms for input sizes (faster).
        torch.backends.cudnn.benchmark = True

# === Noisy Dataset Class ===
# Design Approach (Iter 1-6): Create a Dataset wrapper to inject controllable noise.
# Refined over iterations to handle subsets, symmetric/asymmetric noise accurately,
# and provide noise statistics. Aligned with Ma et al./Nagarsekar report specifics in Iter 6.
class CIFAR10Noisy(Dataset):
    """
    Wraps the CIFAR-10 dataset to inject symmetric or asymmetric label noise.
    Provides methods to get noise statistics. Aligned with Iteration 6 methodology.
    """
    def __init__(self, root, train=True, transform=None, noise_rate=0.0,
                 symmetric=True, random_state=None, indices=None):
        """
        Args:
            root (str): Path to the dataset directory.
            train (bool): Load training set (True) or test set (False).
            transform (callable, optional): Transformations to apply to images.
            noise_rate (float): Target proportion of labels to corrupt (0.0 to 1.0).
            symmetric (bool): True for symmetric noise, False for asymmetric noise.
            random_state (int, optional): Seed for reproducible noise generation.
            indices (list, optional): List of indices from the original dataset to use.
                                      If None, uses the entire specified set (train/test).
        """
        # Load the underlying CIFAR-10 dataset
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=True, transform=transform
        )
        # Store noise configuration
        self.noise_rate = noise_rate
        self.symmetric = symmetric
        self.num_classes = 10 # CIFAR-10 has 10 classes

        # Setup random number generator for noise injection reproducibility
        if random_state is not None:
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = np.random # Use default NumPy generator if no seed provided

        # Determine the subset of indices to use (support for potential train/val splits in earlier iterations)
        # Design Choice (Iter 6): Typically used with all training indices in the final version.
        if indices is not None:
            self.indices = indices
        else:
            # Default to using all indices if none are specified
            self.indices = list(range(len(self.dataset)))

        # Store the targets corresponding *only* to the selected indices
        self.targets = np.array([self.dataset.targets[i] for i in self.indices])
        # Design Choice (Iter 4 onwards): Keep a clean copy of the original targets for this subset.
        # This is ESSENTIAL for accurately calculating noise statistics later.
        self.original_targets = self.targets.copy()

        # Inject noise if noise_rate > 0
        self.corrupted_indices = None # Track which indices within the subset were actually corrupted
        if self.noise_rate > 0:
            # Overwrite self.targets with the noisy labels generated by _add_noise
            self.targets = self._add_noise()

    def _add_noise(self):
        """Internal method to generate noisy labels based on configuration."""
        noisy_targets = np.copy(self.targets) # Operate on a copy
        n_samples = len(self.indices)         # Number of samples in the current subset
        actual_noise_count = 0                # Counter for labels actually changed

        if self.symmetric:
            # Symmetric Noise: Uniformly flip label to any *other* class.
            # Calculate the target number of samples to make noisy
            n_noisy = int(self.noise_rate * n_samples)
            # Randomly select unique indices *within the current subset* (0 to n_samples-1)
            noise_indices = self.random_state.choice(n_samples, n_noisy, replace=False)
            # Store the indices (relative to the subset) that were selected for corruption
            self.corrupted_indices = noise_indices

            # Iterate through the selected indices and apply noise
            for idx in noise_indices:
                current_label = noisy_targets[idx]
                # Define possible new labels (all classes except the current one)
                possible_labels = [i for i in range(self.num_classes) if i != current_label]
                if not possible_labels: continue # Safety check (only fails if K=1)
                # Choose a new label uniformly from the possible ones
                new_label = self.random_state.choice(possible_labels)
                noisy_targets[idx] = new_label
                actual_noise_count += 1 # Increment count of successfully changed labels

        else:
            # Asymmetric Noise: Class-conditional flips based on visual similarity.
            # Design Choice (Iter 6): Specific map strictly follows Ma et al./Nagarsekar report for CIFAR-10.
            # TRUCK(9) -> AUTOMOBILE(1), BIRD(2) -> AIRPLANE(0), DEER(4) -> HORSE(7), CAT(3) <-> DOG(5)
            noise_map = {9: 1, 2: 0, 4: 7, 3: 5, 5: 3} # Note CAT<->DOG is bidirectional
            corrupted_indices_list = [] # List to store subset indices that get corrupted

            # Iterate through all samples *in the subset*
            for i in range(n_samples):
                current_label = noisy_targets[i]
                # Check if this label is in the noise map (i.e., subject to flipping)
                # AND if a random draw falls below the noise rate threshold
                if current_label in noise_map and self.random_state.rand() < self.noise_rate:
                     # Apply the specific flip defined in the map
                     noisy_targets[i] = noise_map[current_label]
                     # Record the index (within the subset) that was corrupted
                     corrupted_indices_list.append(i)
                     actual_noise_count += 1 # Increment count

            # Store the array of corrupted subset indices
            self.corrupted_indices = np.array(corrupted_indices_list)

        # Optional: Calculate and print the effective noise rate (useful for asymmetric comparison)
        effective_noise_rate = actual_noise_count / n_samples if n_samples > 0 else 0
        # print(f"DEBUG Noise: Target={self.noise_rate}, Type={'Symm' if self.symmetric else 'Asymm'}, Effective={effective_noise_rate:.4f}")

        return noisy_targets # Return the array containing noisy labels

    def get_noise_statistics(self):
        """Calculates and returns statistics about the noise actually applied.
           Crucial for verifying noise generation, especially asymmetric.
        """
        # Handle cases where no noise was applied or recorded
        if self.noise_rate == 0.0 or self.corrupted_indices is None or len(self.corrupted_indices) == 0:
            return {"actual_noise_rate": 0.0, "noise_per_class": {}}

        # Overall actual noise rate = (number of corrupted samples) / (total samples in subset)
        actual_noise_rate = len(self.corrupted_indices) / len(self.indices)

        # Calculate per-class noise rate: P(label is corrupted | original label was c)
        # Design Choice (Iter 4 onwards): Use the stored clean original targets for accurate calculation.
        original_targets_in_subset = self.original_targets

        noise_per_class = {} # Dictionary to store {original_class_index: noise_rate_for_that_class}
        for c in range(self.num_classes):
            # Find the indices *within the subset* that originally belonged to class c
            original_class_subset_indices = np.where(original_targets_in_subset == c)[0]

            # Handle case where a class might not be present in the subset (though unlikely for full train set)
            if len(original_class_subset_indices) == 0:
                noise_per_class[c] = 0.0
                continue

            # Check how many of these specific indices (belonging originally to class c)
            # are present in the list of corrupted indices.
            corrupted_in_class = np.isin(original_class_subset_indices, self.corrupted_indices).sum()
            # Calculate noise rate specific to original class c
            noise_per_class[c] = corrupted_in_class / len(original_class_subset_indices)

        return {
            "actual_noise_rate": actual_noise_rate, # Overall fraction of corrupted labels
            "noise_per_class": noise_per_class      # Fraction of corrupted labels for each original class
        }

    def __getitem__(self, index):
        """Retrieves a single sample (image, label) from the dataset subset."""
        # Map the requested subset index (0 to len(self)-1) to the index in the original CIFAR-10 dataset
        original_dataset_index = self.indices[index]
        # Load the image from the base dataset using the original index
        img, _ = self.dataset[original_dataset_index] # Original label is ignored here
        # Get the potentially noisy target associated with the requested subset index
        target = self.targets[index]
        return img, target # Return image and its (potentially noisy) label

    def __len__(self):
        """Returns the number of samples in this specific dataset instance (subset size)."""
        return len(self.indices)


# === CNN Model ===
# Design Choice (Iter 6): Switched from 'ImprovedCNN' to this specific architecture
# to strictly align with the visualized/described model in the reference empirical study
# report (Nagarsekar Appendix A.1), assumed to match Ma et al.'s "8-layer CNN".
# Notable absence of Dropout layers, likely based on diagram interpretation.
class CNN_From_Diagram(nn.Module):
    """
    CNN architecture based on a standard VGG-like pattern (Conv-Conv-Pool blocks).
    Designed for CIFAR-10 (32x32 input). 6 Conv, 3 MaxPool, 1 Hidden FC, 1 Output FC.
    Includes BatchNorm, lacks Dropout. Matches Nagarsekar report Appendix A.1.
    """
    def __init__(self, num_classes=10):
        super(CNN_From_Diagram, self).__init__()

        # --- Block 1 --- (Input: B x 3 x 32 x 32)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: B x 64 x 16 x 16

        # --- Block 2 --- (Input: B x 64 x 16 x 16)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: B x 128 x 8 x 8

        # --- Block 3 --- (Input: B x 128 x 8 x 8)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: B x 256 x 4 x 4

        # --- Classifier Head ---
        # Calculate the flattened feature size after the last pooling layer
        self.flattened_size = 256 * 4 * 4 # = 4096

        # Fully Connected Layer 1 (Hidden Layer)
        self.fc1 = nn.Linear(self.flattened_size, 512)
        # Fully Connected Layer 2 (Output Layer)
        self.fc2 = nn.Linear(512, num_classes)
        # Design Choice: No Dropout layers included, following the reference diagram strictly.

    def forward(self, x):
        """Defines the forward pass of the CNN."""
        # Block 1 Pass: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> Pool
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))

        # Block 2 Pass
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))

        # Block 3 Pass
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(F.relu(self.bn6(self.conv6(x))))

        # Flatten the output for the classifier
        x = x.view(-1, self.flattened_size) # Reshape (B, C, H, W) -> (B, C*H*W)

        # Classifier Pass
        x = F.relu(self.fc1(x)) # Hidden layer with ReLU activation
        x = self.fc2(x)         # Output layer (returns logits)

        return x


# === Common Helper Function ===
# Design Choice (Iter 6): Centralize the normalization formula for reuse in NCE, NFL, etc.
def normalize_loss(numerator, denominator, epsilon=1e-7):
    """
    Applies the core normalization logic: Numerator / (Denominator + epsilon).
    Used by normalized loss functions (NCE, NFL, NMAE, NRCE).

    Args:
        numerator (Tensor): The loss term for the target class (or equivalent).
        denominator (Tensor): The sum of loss terms over all classes (or equivalent).
        epsilon (float): Small value for numerical stability to prevent division by zero.

    Returns:
        Tensor: The normalized loss value(s).
    """
    # Ensure stability by adding epsilon to the denominator before division
    return numerator / (denominator + epsilon)


# === Loss Function Implementations ===
# Design Approach (Iter 6): Implement standard, robust, normalized, and APL losses,
# prioritizing stability (esp. FL/NFL) and alignment with paper definitions/parameters.

# --- Standard Losses (Baseline, Active) ---
class CrossEntropyLoss(nn.Module):
    """Standard Cross Entropy Loss wrapper."""
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, logits, targets):
        """Calculates CE loss using PyTorch's functional implementation."""
        return F.cross_entropy(logits, targets)

class FocalLoss(nn.Module):
    """Focal Loss (FL) implementation with numerical stability enhancements. Active Loss.
       Uses gamma=0.5 default, aligning with Ma et al. CIFAR experiments.
    """
    def __init__(self, gamma=0.5, reduction='mean', epsilon=1e-6):
        super(FocalLoss, self).__init__()
        # gamma: Focusing parameter. Higher values focus more on hard examples.
        # reduction: Specifies aggregation method ('mean', 'sum', 'none').
        # epsilon: Small value for numerical stability in intermediate calculations.
        # prob_clip: Value to clip probabilities away from exact 0 or 1.
        self.gamma = gamma
        self.reduction = reduction
        self.epsilon = epsilon
        self.prob_clip = 1e-6 # Stability: Avoid log(0) or (1-p)**gamma when p=1

    def forward(self, logits, targets):
        # Stability: Use log_softmax for better numerical precision than softmax -> log.
        log_probs = F.log_softmax(logits, dim=1) # Shape: [N, C]
        # Get the log-probabilities corresponding to the target classes. Shape: [N]
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1) # Use squeeze(1)

        # Calculate Cross Entropy loss per sample in a stable way: CE = -log(p_target)
        ce_loss = -target_log_probs # Shape: [N]

        # Calculate probabilities (pt) for the target classes, ensuring stability.
        pt = torch.exp(target_log_probs) # Shape: [N]
        # Stability: Clamp probabilities to avoid issues at the boundaries (0 or 1).
        pt = torch.clamp(pt, min=self.prob_clip, max=1.0 - self.prob_clip)

        # Calculate the focal loss modulating factor: (1 - pt)^gamma
        focal_weight = (1.0 - pt) ** self.gamma # Shape: [N]
        # Stability: Clip focal weight to prevent potential explosion if pt is tiny and gamma<0 (though gamma>=0 here)
        # or just generally keep weights reasonable. Max value chosen empirically/heuristically.
        focal_weight = torch.clamp(focal_weight, min=0.0, max=1e3)

        # Compute the final Focal Loss per sample: weight * CE_loss
        focal_loss = focal_weight * ce_loss # Shape: [N]

        # Stability: Final safety check for NaNs. If NaN occurs (should be rare with fixes), fallback to CE.
        if torch.isnan(focal_loss).any():
            print(f"Warning: NaN detected in FocalLoss (gamma={self.gamma}). Falling back to CE loss for this batch.")
            focal_loss = ce_loss # Use the pre-calculated stable CE loss

        # Apply reduction based on the specified mode.
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss


# --- Baseline Robust Losses (Passive) ---
class MAELoss(nn.Module):
    """Mean Absolute Error (MAE) Loss. Known to be robust to noise. Passive Loss.
       Calculates Sum_k |p_k - q_k|, where q_k is one-hot target.
    """
    def __init__(self, num_classes=10, reduction='mean'):
        super(MAELoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, logits, targets):
        device = logits.device
        # Calculate predicted probabilities
        probs = F.softmax(logits, dim=1) # Shape: [N, C]
        # Create one-hot encoded targets
        targets_one_hot = F.one_hot(targets, self.num_classes).float().to(device) # Shape: [N, C]
        # Calculate MAE per sample: sum of absolute differences across classes
        mae_per_sample = torch.abs(probs - targets_one_hot).sum(dim=1) # Shape: [N]

        # Apply reduction
        if self.reduction == 'mean':
            return mae_per_sample.mean()
        elif self.reduction == 'sum':
            return mae_per_sample.sum()
        else: # 'none'
            return mae_per_sample

class RCELoss(nn.Module):
    """Reverse Cross Entropy (RCE) Loss. Known to be robust. Passive Loss.
       Uses the simplified form derived from definition: RCE = -A * (1 - p_y).
       Uses A=-4.0 default, aligning with Ma et al. CIFAR experiments.
    """
    def __init__(self, num_classes=10, A=-4.0, reduction='mean'):
        super(RCELoss, self).__init__()
        if A >= 0: raise ValueError("RCE parameter 'A' must be negative.")
        self.num_classes = num_classes
        self.A = A # Log-value for pseudo-targets of non-labeled classes
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1) # Shape: [N, C]
        # Get probabilities corresponding to the target classes
        p_y = probs.gather(1, targets.unsqueeze(1)).squeeze(1) # Shape: [N]

        # Calculate RCE per sample using the simplified formula
        # Derived from -Sum[p_k * log(q_k)] where log(q_y)=log(1)=0 and log(q_{k!=y})=A
        # -> - [ p_y*log(q_y) + Sum_{k!=y} p_k*log(q_k) ]
        # -> - [ 0 + Sum_{k!=y} p_k*A ] = -A * Sum_{k!=y} p_k = -A * (1 - p_y)
        rce_per_sample = -self.A * (1.0 - p_y) # Shape: [N]

        # Apply reduction
        if self.reduction == 'mean':
            return rce_per_sample.mean()
        elif self.reduction == 'sum':
            return rce_per_sample.sum()
        else: # 'none'
            return rce_per_sample


# --- Normalized Losses (Active or Passive depending on base loss) ---
class NCELoss(nn.Module):
    """Normalized Cross Entropy (NCE) Loss. Active Loss.
       Implements L_norm = L_CE(y) / Sum_j L_CE(j) = (-log p_y) / (-Sum_j log p_j).
    """
    def __init__(self, epsilon=1e-7, reduction='mean'):
        super(NCELoss, self).__init__()
        self.epsilon = epsilon # For numerical stability in division
        self.reduction = reduction

    def forward(self, logits, targets):
        # Use log_softmax for stability
        log_probs = F.log_softmax(logits, dim=1) # Shape: [N, C]
        # Get log-probability of the target class
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1) # Shape: [N]

        # Numerator for normalization: CE loss for the target class
        numerator = -target_log_probs # Shape: [N]
        # Denominator for normalization: Sum of CE losses over all classes
        denominator = -torch.sum(log_probs, dim=1) # Shape: [N]

        # Apply normalization using the helper function
        nce_loss_per_sample = normalize_loss(numerator, denominator, self.epsilon) # Shape: [N]

        # Apply reduction
        if self.reduction == 'mean':
            return nce_loss_per_sample.mean()
        elif self.reduction == 'sum':
            return nce_loss_per_sample.sum()
        else: # 'none'
            return nce_loss_per_sample

class NFLLoss(nn.Module):
    """Normalized Focal Loss (NFL). Active Loss.
       Implements L_norm = L_FL(y) / Sum_j L_FL(j).
       Uses gamma=0.5 default. Includes extensive stability enhancements.
    """
    def __init__(self, gamma=0.5, epsilon=1e-6, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon # Epsilon for the final normalization division
        self.reduction = reduction
        self.prob_clip = 1e-6 # Probability clipping value

    def forward(self, logits, targets):
        # Stability: Use log_softmax
        log_probs = F.log_softmax(logits, dim=1) # Shape: [N, C]
        # Stability: Calculate probabilities and clip them
        probs = torch.exp(log_probs)
        probs = torch.clamp(probs, min=self.prob_clip, max=1.0 - self.prob_clip)
        # Stability: Recompute log_probs from clipped probs for consistency
        log_probs = torch.log(probs)

        # Calculate the focal loss term -(1-p_k)^gamma * log(p_k) for ALL classes
        focal_term_all_classes = -( (1.0 - probs) ** self.gamma ) * log_probs # Shape: [N, C]

        # Numerator: Focal loss term for the target class y
        fl_y = focal_term_all_classes.gather(1, targets.unsqueeze(1)).squeeze(1) # Shape: [N]
        # Denominator: Sum of focal loss terms over all classes j
        fl_all_sum = torch.sum(focal_term_all_classes, dim=1) # Shape: [N]

        # Stability Check: Warn if denominator is extremely small, indicating potential instability.
        if (fl_all_sum.abs() < self.epsilon).any():
             print(f"Warning: NFLLoss denominator near zero (min abs={fl_all_sum.abs().min()}).")

        # Apply normalization
        nfl_loss_per_sample = normalize_loss(fl_y, fl_all_sum, self.epsilon) # Shape: [N]

        # Stability Check: Final check for NaNs AFTER normalization. Raise error if found.
        if torch.isnan(nfl_loss_per_sample).any():
            # Provide more context if possible (consider printing intermediate values here if debugging)
            raise ValueError("NaN detected in NFLLoss calculation AFTER normalization. Training cannot proceed.")

        # Apply reduction
        if self.reduction == 'mean':
            return nfl_loss_per_sample.mean()
        elif self.reduction == 'sum':
            return nfl_loss_per_sample.sum()
        else: # 'none'
            return nfl_loss_per_sample


class NMAELoss(nn.Module):
    """Normalized Mean Absolute Error (NMAE) Loss. Passive Loss.
       Uses the simplified analytical form: NMAE = MAE / (2*(K-1)).
    """
    def __init__(self, num_classes=10, reduction='mean'):
        super(NMAELoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        # Pre-compute the scaling factor denominator for efficiency
        self.scale_factor = 2.0 * (self.num_classes - 1.0) # Use float
        # Safety check for K=1 case (though unlikely)
        if abs(self.scale_factor) < 1e-9:
            print("Warning: NMAE scale factor near zero (K=1?). Setting scale factor to 1.")
            self.scale_factor = 1.0

    def forward(self, logits, targets):
        device = logits.device
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, self.num_classes).float().to(device)
        # Calculate MAE per sample first
        mae_per_sample = torch.abs(probs - targets_one_hot).sum(dim=1) # Shape: [N]
        # Apply the pre-computed scaling factor
        nmae_per_sample = mae_per_sample / self.scale_factor # Shape: [N]

        # Apply reduction
        if self.reduction == 'mean':
            return nmae_per_sample.mean()
        elif self.reduction == 'sum':
            return nmae_per_sample.sum()
        else: # 'none'
            return nmae_per_sample

class NRCELoss(nn.Module):
    """Normalized Reverse Cross Entropy (NRCE) Loss. Passive Loss.
       Uses the simplified analytical form: NRCE = RCE / |A*(K-1)| = (1-p_y)/(K-1) (for A<0).
    """
    def __init__(self, num_classes=10, A=-4.0, reduction='mean'):
        super(NRCELoss, self).__init__()
        if A >= 0: raise ValueError("NRCE parameter 'A' must be negative.")
        self.num_classes = num_classes
        self.A = A
        self.reduction = reduction
        # Pre-compute the scaling factor denominator: |A * (K-1)|
        self.scale_denominator = abs(self.A * (self.num_classes - 1.0)) # Use float
        # Safety check
        if abs(self.scale_denominator) < 1e-9:
             print("Warning: NRCE scale denominator near zero (K=1 or A=0?). Setting denominator to 1.")
             self.scale_denominator = 1.0
        # Design Choice: Instantiate RCE internally to get per-sample values easily.
        # Could also recalculate RCE here, but this reuses code.
        self.rce_internal = RCELoss(num_classes=self.num_classes, A=self.A, reduction='none')

    def forward(self, logits, targets):
        # Calculate RCE per sample using the internal instance
        rce_per_sample = self.rce_internal(logits, targets) # Shape: [N]
        # Apply the pre-computed scaling factor denominator
        nrce_per_sample = rce_per_sample / self.scale_denominator # Shape: [N]

        # Apply reduction
        if self.reduction == 'mean':
            return nrce_per_sample.mean()
        elif self.reduction == 'sum':
            return nrce_per_sample.sum()
        else: # 'none'
            return nrce_per_sample


# --- Active Passive Loss (APL) Framework ---
class APLLoss(nn.Module):
    """Implements the Active Passive Loss framework: alpha * Active + beta * Passive.
       Requires constituent active and passive losses to be robust (inherently or normalized).
    """
    def __init__(self, active_loss, passive_loss, alpha=1.0, beta=1.0, loss_name=None):
        super(APLLoss, self).__init__()
        self.active_loss = active_loss   # The robust active loss component (e.g., NCE, NFL instance)
        self.passive_loss = passive_loss # The robust passive loss component (e.g., MAE, RCE instance)
        self.alpha = alpha               # Weight for the active term
        self.beta = beta                 # Weight for the passive term
        self.loss_name = loss_name       # String name for identification (e.g., "NCE+MAE")

    def forward(self, logits, targets):
        """Calculates the combined APL loss."""
        # Calculate the active component loss
        active_term = self.active_loss(logits, targets)
        # Calculate the passive component loss
        passive_term = self.passive_loss(logits, targets)
        # Return the weighted sum
        return self.alpha * active_term + self.beta * passive_term


# === Training and Evaluation Functions ===
# Design Approach (Iter 6): Refined train/eval loops integrating AMP, gradient clipping,
# NaN handling, and optimized for the no-validation, last-epoch reporting methodology.

def train(model, train_loader, criterion, optimizer, device,
          scheduler=None, use_amp=True, scaler=None):
    """Performs one epoch of training with AMP, gradient clipping, and NaN checks."""
    model.train() # Set model to training mode (enables dropout if present, uses batch stats for BN)
    running_loss = 0.0
    correct = 0
    total = 0
    # Hyperparameter: Gradient clipping threshold (chosen heuristically)
    grad_clip_max_norm = 0.05

    # --- Determine if gradient clipping should be applied for this criterion ---
    # Design Choice (Iter 6): Apply clipping only to FL/NFL based losses, as they are
    # empirically more prone to gradient explosion.
    apply_clipping = False
    loss_id = getattr(criterion, 'loss_name', type(criterion).__name__) # Get loss name/type for checking
    # Check if it's a standalone FL/NFL or an APL loss involving FL/NFL
    if isinstance(criterion, (FocalLoss, NFLLoss)) or ('FL' in loss_id): # Simple check using 'FL' substring
        apply_clipping = True
    # --- End Clipping Check ---

    # Initialize progress bar
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for inputs, targets in pbar:
        # Performance: Move data asynchronously if possible (requires pinned memory)
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        # Performance: Reset gradients efficiently
        optimizer.zero_grad(set_to_none=True)

        # --- Automatic Mixed Precision (AMP) Path ---
        # Condition: AMP enabled, GradScaler provided, and running on CUDA
        if use_amp and scaler and device.type == 'cuda':
            # AMP context manager: Operations inside run in lower precision (e.g., FP16)
            with autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, targets) # Loss computed in mixed precision

            # Stability: Check for NaN loss *before* backpropagation
            if torch.isnan(loss):
                print(f"WARNING: NaN loss detected BEFORE backward (AMP) for {loss_id}. Skipping batch.")
                optimizer.zero_grad(set_to_none=True) # Clear any partial gradients if needed
                continue # Skip this batch entirely

            # AMP: Scale the loss, then perform backward pass to calculate scaled gradients
            scaler.scale(loss).backward()

            # Stability: Apply gradient clipping if needed for this loss type
            if apply_clipping:
                # AMP: Unscale gradients before clipping to operate on original scale
                scaler.unscale_(optimizer)
                # Clip gradients in-place
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)

            # AMP: Optimizer step (automatically handles unscaling if not already done)
            scaler.step(optimizer)
            # AMP: Update the scale factor for the next iteration
            scaler.update()

        # --- Standard Precision Path (CPU or AMP disabled) ---
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets) # Loss computed in FP32

            # Stability: Check for NaN loss *before* backpropagation
            if torch.isnan(loss):
                print(f"WARNING: NaN loss detected BEFORE backward (non-AMP) for {loss_id}. Skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            # Standard backward pass
            loss.backward()

            # Stability: Apply gradient clipping if needed
            if apply_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)

            # Standard optimizer step
            optimizer.step()

        # --- Learning Rate Scheduler Step (Batch-wise) ---
        # Design Choice: Support schedulers that update per batch (e.g., OneCycleLR)
        if scheduler is not None and isinstance(scheduler, (optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.CyclicLR)):
            scheduler.step()
        # --- End Scheduler Step ---

        # --- Track Statistics ---
        batch_loss = loss.item() # Get Python float value of the loss
        running_loss += batch_loss * inputs.size(0) # Accumulate loss, weighted by batch size
        _, predicted = outputs.max(1) # Get predicted class index
        total += targets.size(0) # Accumulate total samples processed
        correct += predicted.eq(targets).sum().item() # Accumulate correct predictions
        # Update progress bar display
        pbar.set_postfix({'Loss': f'{batch_loss:.4f}', 'Acc': f'{100. * correct / total:.2f}%'})
        # --- End Statistics ---

    # Calculate average loss and accuracy for the epoch
    train_loss = running_loss / total if total > 0 else float('nan') # Avoid division by zero
    train_acc = 100.0 * correct / total if total > 0 else 0.0
    return train_loss, train_acc


def evaluate(model, data_loader, criterion, device, use_amp=True, desc="Evaluating"):
    """Evaluates the model on the given dataloader, calculating loss and accuracy.
       Uses AMP if enabled. Handles optional criterion. Returns ACCURACY.
    """
    model.eval() # Set model to evaluation mode (disables dropout, BN uses running stats)
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(data_loader, desc=desc, leave=False) # Progress bar
    # Disable gradient computations for efficiency and correctness during evaluation
    with torch.no_grad():
        for inputs, targets in pbar:
            # Performance: Move data asynchronously
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # Determine appropriate device type for autocast (supports 'cuda', 'mps')
            amp_device_type = device.type if device.type in ['cuda', 'mps'] else 'cpu'

            # --- AMP Path for Evaluation ---
            if use_amp and amp_device_type != 'cpu':
                with autocast(device_type=amp_device_type):
                    outputs = model(inputs)
                    # Calculate loss only if a criterion is provided
                    if criterion:
                        loss = criterion(outputs, targets)
                    else:
                        loss = torch.tensor(0.0, device=device) # Placeholder if no loss needed
            # --- Standard Precision Path ---
            else:
                outputs = model(inputs)
                if criterion:
                    loss = criterion(outputs, targets)
                else:
                    loss = torch.tensor(0.0, device=device)

            # --- Track Statistics ---
            batch_loss = loss.item() if criterion else 0.0 # Get loss value if calculated
            running_loss += batch_loss * inputs.size(0) # Accumulate loss
            _, predicted = outputs.max(1) # Get predictions
            total += targets.size(0) # Count samples
            correct += predicted.eq(targets).sum().item() # Count correct predictions
            # Update progress bar - NOTE: Reports Accuracy!
            pbar.set_postfix({'Loss': f'{batch_loss:.4f}', 'Acc': f'{100. * correct / total:.2f}%'})
            # --- End Statistics ---

    # Calculate overall epoch results
    eval_loss = running_loss / total if total > 0 else 0.0 # Avoid division by zero
    # Design Choice / Correction (Iter 6): Calculate and return ACCURACY
    eval_acc = 100.0 * correct / total if total > 0 else 0.0
    return eval_loss, eval_acc


# === Data Loader Creation Function ===
# Design Choice (Iter 6): Removed validation split. Optimized for performance.
def get_data_loaders(noise_rate, symmetric=True, batch_size=128, random_state=42,
                     num_workers=4, pin_memory=True):
    """Creates optimized train and test DataLoaders for CIFAR-10 with noise.
       No validation loader is created in this version.
    """
    # Standard CIFAR-10 transforms with augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),      # Random crop augmentation
        transforms.RandomHorizontalFlip(),          # Random horizontal flip augmentation
        transforms.ToTensor(),                      # Convert PIL image to tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # Normalize
    ])
    # Test transform without augmentation, only normalization
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # --- No Validation Split Logic ---
    # Load the base dataset details to get the total number of training samples
    cifar_train_full = torchvision.datasets.CIFAR10(
        root='./data_coreML', train=True, download=True # Load metadata, download if needed
    )
    num_train = len(cifar_train_full)
    # Use all available training indices
    train_idx = list(range(num_train))

    # Instantiate the noisy training dataset using ALL training indices
    train_dataset = CIFAR10Noisy(
        root='./data_coreML', train=True, transform=transform_train,
        noise_rate=noise_rate, symmetric=symmetric, random_state=random_state,
        indices=train_idx # Critical: Pass all indices
    )
    # Get noise statistics for the generated training set
    noise_statistics = train_dataset.get_noise_statistics()

    # Instantiate the standard clean test dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data_coreML', train=False, download=True, transform=transform_test
    )

    # --- Create DataLoaders with Performance Optimizations ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, # Shuffle training data each epoch
        num_workers=num_workers, # Use multiple processes for data loading
        pin_memory=pin_memory, # Speeds up host-to-GPU transfer if True
        drop_last=True, # Drop last incomplete batch for consistent batch sizes (maybe better stability/performance)
        persistent_workers=(num_workers > 0), # Keep worker processes alive between epochs (reduces overhead)
        prefetch_factor=2 if num_workers > 0 else None # Number of batches to preload per worker
    )
    # No validation loader created
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size, # Often use same or larger batch size for evaluation
        shuffle=False, # No need to shuffle test data
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False, # Evaluate on ALL test samples
        persistent_workers=(num_workers > 0)
    )

    # Return only train/test loaders and noise statistics
    return train_loader, test_loader, noise_statistics


# === Model Training Orchestration ===
# Design Choice (Iter 6): Simplified to remove validation loop and best model tracking.
# Reports performance based on the final epoch. Adds timing. Saves final model state.
def train_model(model, train_loader, test_loader, criterion,
                optimizer, scheduler, num_epochs, device, save_dir=None,
                loss_name=None, use_amp=True):
    """
    Orchestrates the training and evaluation of a model for a fixed number of epochs.
    Reports loss/accuracy based on the final epoch's test set evaluation.
    No validation or early stopping is performed in this version.
    """
    # Create model saving directory if specified
    if save_dir: os.makedirs(save_dir, exist_ok=True)

    # Initialize GradScaler for AMP (enabled based on use_amp flag and device type)
    scaler = GradScaler(enabled=use_amp and device.type == 'cuda')

    # History dictionary to store metrics per epoch (simplified: no validation)
    history = {
        'train_losses': [], 'train_accs': [],
        'test_losses': [], 'test_accs': [],
        'learning_rates': [], 'epoch_times': []
    }

    print(f"Starting training for {loss_name}...") # Indicate start
    # --- Main Training Loop ---
    for epoch in range(num_epochs):
        epoch_start_time = time.time() # Track epoch duration

        # --- Train for one epoch ---
        # Pass scheduler only if it's batch-based (handled inside train)
        batch_scheduler = scheduler if isinstance(scheduler, (optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.CyclicLR)) else None
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device,
                                      batch_scheduler, use_amp, scaler)

        # --- Evaluate on the Test Set ---
        # No validation evaluation in this iteration
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, use_amp, desc="Testing")

        # --- Step Learning Rate Scheduler (Epoch-wise) ---
        # Step scheduler if it's not batch-based (e.g., CosineAnnealingLR, StepLR)
        if scheduler is not None and not isinstance(scheduler, (optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.CyclicLR)):
            scheduler.step()

        # --- Record History ---
        current_lr = optimizer.param_groups[0]['lr'] # Get current learning rate
        epoch_time = time.time() - epoch_start_time # Calculate epoch duration
        history['train_losses'].append(train_loss)
        history['train_accs'].append(train_acc)
        history['test_losses'].append(test_loss)
        history['test_accs'].append(test_acc) # Recording test accuracy
        history['learning_rates'].append(current_lr)
        history['epoch_times'].append(epoch_time)
        # --- End Record History ---

        # --- Console Output ---
        # Print summary for the current epoch (Train Loss/Acc, Test Loss/Acc, LR, Time)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}] '
            f'Time: {epoch_time:.1f}s | '
            f'LR: {current_lr:.6f} | '
            f'Train L/A: {train_loss:.4f}/{train_acc:.2f}% | '
            f'Test L/A: {test_loss:.4f}/{test_acc:.2f}%' # Reporting test accuracy
        )
        # --- End Console Output ---

    # --- End Training Loop ---

    # --- Final Result Calculation (Based on Last Epoch) ---
    # Design Choice (Iter 6): Report performance from the very last epoch.
    final_test_acc = history['test_accs'][-1] if history['test_accs'] else -1.0
    final_test_loss = history['test_losses'][-1] if history['test_losses'] else -1.0

    # --- Prepare Result Dictionary ---
    result = {
        'history': history, # Full history for potential later analysis/plotting
        'final_test_acc': final_test_acc, # Primary metric: Test accuracy after num_epochs
        'final_test_loss': final_test_loss, # Test loss after num_epochs
        'avg_epoch_time_s': np.mean(history['epoch_times']) if history['epoch_times'] else 0,
        'total_training_time_s': sum(history['epoch_times']) if history['epoch_times'] else 0
    }

    # --- Save Final Model State (Optional) ---
    # Design Choice (Iter 6): Save the model state at the *end* of training.
    if save_dir and loss_name:
         # Handle potential DDP model wrapping if DDP were fully implemented
         final_model_state = (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()).copy()
         final_model_path = os.path.join(save_dir, f"{loss_name}_final.pth")
         torch.save(final_model_state, final_model_path)
         print(f"Saved final model state to {final_model_path}")

    # --- Final Print Summary ---
    print(f"\n{loss_name}: Training Complete.")
    print(f"{loss_name}: Final Test Acc (Epoch {num_epochs}): {final_test_acc:.2f}%")

    return result # Return the dictionary containing history and final results

# === Experiment Set Orchestration Function ===
# Design Choice (Iter 6): Encapsulate the logic for running all experiments
# (all losses, all noise rates) for a single noise type (symmetric or asymmetric)
# to keep the main function cleaner and manage resources effectively.
def run_experiment_set(
    # --- Configuration passed from main ---
    symmetric,              # bool: True for symmetric, False for asymmetric
    noise_rates_to_run,     # list: Noise rates to iterate through
    device,                 # torch.device: CPU or CUDA device
    base_output_dir_root,   # str: Parent directory for saving results (e.g., './results_all')
    num_epochs,             # int: Number of training epochs
    learning_rate,          # float: Initial learning rate
    batch_size,             # int: Batch size
    random_state,           # int: Random seed
    num_workers,            # int: DataLoader workers
    pin_memory,             # bool: DataLoader pin_memory setting
    use_amp,                # bool: Enable Automatic Mixed Precision
    subset_losses,          # bool: Flag to run only a subset of losses (for quick tests)
    loss_defs = None        # dict, optional: Pre-defined loss function dictionary to override defaults
    ):
    """
    Runs a complete set of experiments (multiple noise rates, multiple losses)
    for a specific noise type (symmetric or asymmetric). Handles setup, execution,
    resource management, results saving, and plotting invocation.
    """
    # --- Setup Run Directory and Logging ---
    noise_type = "symmetric" if symmetric else "asymmetric"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create a unique name and directory for this specific run set
    exp_name = f"cifar10_{noise_type}_noise_epochs{num_epochs}_{timestamp}"
    run_output_dir = Path(base_output_dir_root) / exp_name # Use pathlib for robustness
    run_output_dir.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist

    print(f"\n{'='*80}")
    print(f"STARTING EXPERIMENT SET: {noise_type.upper()}")
    print(f"Output Directory: {run_output_dir}")
    print(f"Noise Rates: {noise_rates_to_run}")
    print(f"Device: {device}")
    print(f"AMP Enabled: {use_amp}")
    print(f"{'='*80}\n")
    # --- End Setup ---

    # --- Define Loss Functions for this Run ---
    # Design Choice: Define losses within the run function or allow passing them in.
    # Allows flexibility while having defaults aligned with the paper/report.
    if loss_defs is None:
        num_classes = 10
        reduction = 'mean' # Use mean reduction for training
        # Default parameters aligned with Ma et al./Nagarsekar report
        fl_gamma, rce_A, apl_alpha, apl_beta = 0.5, -4.0, 1.0, 1.0

        # Define base losses using corrected/stabilized classes from Iteration 6
        base_loss_functions_defs = {
            'CE': {'loss_fn': CrossEntropyLoss(), 'type': 'standard', 'category': 'active'},
            'FL': {'loss_fn': FocalLoss(gamma=fl_gamma, reduction=reduction), 'type': 'standard', 'category': 'active'},
            'MAE': {'loss_fn': MAELoss(num_classes=num_classes, reduction=reduction), 'type': 'robust', 'category': 'passive'},
            'RCE': {'loss_fn': RCELoss(num_classes=num_classes, A=rce_A, reduction=reduction), 'type': 'robust', 'category': 'passive'},
            'NCE': {'loss_fn': NCELoss(reduction=reduction), 'type': 'normalized', 'category': 'active'},
            'NFL': {'loss_fn': NFLLoss(gamma=fl_gamma, reduction=reduction), 'type': 'normalized', 'category': 'active'},
            'NMAE': {'loss_fn': NMAELoss(num_classes=num_classes, reduction=reduction), 'type': 'normalized', 'category': 'passive'},
            'NRCE': {'loss_fn': NRCELoss(num_classes=num_classes, A=rce_A, reduction=reduction), 'type': 'normalized', 'category': 'passive'}
        }

        # Optional: Select only a subset of base losses if flag is set (for quick tests)
        base_losses_subset = ['CE', 'MAE', 'NCE', 'RCE', 'NFL'] # Example subset
        if subset_losses:
            print(f"Running SUBSET of base losses: {base_losses_subset}")
            base_loss_functions = {name: base_loss_functions_defs[name] for name in base_losses_subset if name in base_loss_functions_defs}
        else:
            base_loss_functions = base_loss_functions_defs

        # Define Active/Passive groups based on the selected base losses
        active_losses = {name: config for name, config in base_loss_functions.items() if config['category'] == 'active'}
        passive_losses = {name: config for name, config in base_loss_functions.items() if config['category'] == 'passive'}

        # Define standard APL combinations using available active/passive components
        apl_combinations = {}
        # Standard combinations tested in the papers
        apl_pairs_to_test = [('NCE', 'MAE'), ('NCE', 'RCE'), ('NFL', 'MAE'), ('NFL', 'RCE')]
        for active_name, passive_name in apl_pairs_to_test:
            # Only create the combination if both components are in the selected base_loss_functions
            if active_name in active_losses and passive_name in passive_losses:
                comb_name = f"{active_name}+{passive_name}"
                apl_combinations[comb_name] = {
                    'loss_fn': APLLoss(
                        active_loss=active_losses[active_name]['loss_fn'],
                        passive_loss=passive_losses[passive_name]['loss_fn'],
                        alpha=apl_alpha, beta=apl_beta, loss_name=comb_name
                    ),
                    'type': 'apl', 'category': 'combined',
                    'active': active_name, 'passive': passive_name, # Store components for analysis
                    'alpha': apl_alpha, 'beta': apl_beta
                }

        # Combine base losses and APL combinations into the final dictionary for this run
        loss_functions_to_run = {**base_loss_functions, **apl_combinations}
    else:
         # Use externally provided loss definitions if passed
         loss_functions_to_run = loss_defs

    print(f"Loss functions included in this run set: {list(loss_functions_to_run.keys())}")
    # --- End Loss Definition ---

    # --- Main Experiment Loop (Noise Rates -> Losses) ---
    results = defaultdict(lambda: defaultdict(dict)) # Nested dict for results[noise_rate][loss_name]
    all_noise_statistics = {}                         # Store noise stats per rate

    # Performance: Clear CUDA cache before starting the potentially long loops
    if device.type == 'cuda': torch.cuda.empty_cache()

    # Outer loop: Iterate through specified noise rates
    for noise_rate in noise_rates_to_run:
        print(f"\n--- Starting Noise Rate Loop: {noise_rate:.2f} ({noise_type} noise) ---")
        # Get data loaders and actual noise statistics for this rate
        train_loader, test_loader, noise_stats = get_data_loaders(
            noise_rate, symmetric, batch_size, random_state, num_workers, pin_memory
        )
        # Store noise stats, converting float rate key to string for JSON compatibility
        all_noise_statistics[str(noise_rate)] = noise_stats
        print(f"DataLoaders created. Effective Noise Rate: {noise_stats.get('actual_noise_rate', 'N/A'):.4f}")

        # Define directory for saving models specific to this noise rate
        models_dir = run_output_dir / 'models' / f"noise_{noise_rate:.1f}"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Inner loop: Iterate through all loss functions defined for this run
        for loss_name, config in loss_functions_to_run.items():
            print(f"\n..... Training Model with Loss: {loss_name} | Noise Rate: {noise_rate:.2f} .....")
            # --- Prepare for Single Model Training ---
            # Instantiate the loss function and move it to the device
            criterion = config['loss_fn'].to(device)
            # Instantiate a *new* model for each loss function run and move to device
            # Design Choice: Fresh model per run ensures fair comparison, avoids transfer effects.
            model = CNN_From_Diagram(num_classes=10).to(device)

            # Instantiate optimizer (SGD) and LR scheduler (Cosine Annealing)
            # Design Choice: Using standard SGD with momentum and weight decay (1e-4) aligned with practice.
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
            # --- End Preparation ---

            # --- Run Training and Evaluation for this Model/Loss/Noise ---
            try:
                # Call the main training loop function
                result = train_model(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    num_epochs=num_epochs,
                    device=device,
                    save_dir=models_dir, # Pass directory to save final model state
                    loss_name=loss_name,  # Pass loss name for saving file
                    use_amp=use_amp       # Pass AMP flag
                )
                # Store the results dictionary (contains history, final metrics, times)
                # Use string key for noise rate for JSON compatibility
                results[str(noise_rate)][loss_name] = result
            # --- Error Handling ---
            except Exception as e:
                # Catch potential errors during training (e.g., from NaN loss in NFL)
                print(f"CRITICAL ERROR during training: {loss_name} @ {noise_rate} ({noise_type}). Run failed.")
                print(f"Error details: {e}")
                import traceback
                traceback.print_exc() # Print full stack trace
                # Store None to indicate failure for this run
                results[str(noise_rate)][loss_name] = None # Mark as failed
            # --- Resource Cleanup (CRITICAL for large sweeps) ---
            finally:
                # Ensure cleanup happens even if errors occur
                # Design Choice (Iter 6): Explicit memory management to prevent leaks/OOM.
                try: del model, optimizer, scheduler, criterion # Delete large objects
                except NameError: pass # Ignore if they weren't defined due to early error
                gc.collect() # Force Python garbage collection
                if device.type == 'cuda':
                    torch.cuda.empty_cache() # Release unused cached memory on GPU
            # --- End Resource Cleanup ---
        # --- End Loss Loop ---

        print(f"--- Finished Noise Rate Loop: {noise_rate:.2f} ({noise_type} noise) ---")

        # --- Resource Cleanup (DataLoaders) ---
        # Design Choice (Iter 6): Clean up loaders after finishing all losses for a rate.
        try: del train_loader, test_loader
        except NameError: pass
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        # --- End Loader Cleanup ---
    # --- End Noise Rate Loop ---

    # --- Save Comprehensive Results ---
    results_filepath = run_output_dir / 'experiment_results.json'
    try:
        # Helper function to make NumPy/Path types JSON serializable
        def json_default(o):
            if isinstance(o,(np.int_,np.intc,np.intp,np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32,np.uint64)): return int(o)
            elif isinstance(o,(np.float_,np.float16,np.float32,np.float64)): return float(o)
            elif isinstance(o,np.ndarray): return o.tolist()
            elif isinstance(o, Path): return str(o) # Handle pathlib.Path object
            elif isinstance(o, torch.device): return str(o) # Handle device object
            return f'<not serializable: {type(o).__name__}>' # Fallback for unknown types

        # Save run configuration arguments separately for traceability
        run_config = {
            'symmetric': symmetric, 'noise_rates_run': noise_rates_to_run,
            'num_epochs': num_epochs, 'learning_rate': learning_rate,
            'batch_size': batch_size, 'random_state': random_state,
            'num_workers': num_workers, 'pin_memory': pin_memory,
            'use_amp': use_amp, 'subset_losses': subset_losses,
            'device': str(device) # Convert device object to string
        }
        config_path = run_output_dir / 'config_run.json'
        with open(config_path, 'w') as f:
            json.dump(run_config, f, indent=4)

        # Prepare the final output dictionary, ensuring contents are serializable
        # Convert results/stats dicts using json.loads(json.dumps(...)) trick with the default handler
        final_output = {
            'config_run': run_config,
            'results': json.loads(json.dumps(results, default=json_default)),
            'noise_statistics': json.loads(json.dumps(all_noise_statistics, default=json_default)),
            'loss_functions_run': list(loss_functions_to_run.keys()) # List of losses actually used
        }
        # Save the main results JSON file
        with open(results_filepath, 'w') as f:
            json.dump(final_output, f, indent=4)
        print(f"\n{noise_type.upper()} comprehensive results saved to {results_filepath}")

    except Exception as e:
        print(f"\nError saving {noise_type} results JSON: {e}")
        import traceback; traceback.print_exc()
    # --- End Saving ---

    # --- Generate Final Summary Table ---
    print(f"\n\n=== Final Test Accuracy (%) - {noise_type.upper()} ===")
    summary_data = {}
    # Extract final test accuracy for each loss and noise rate
    for loss_name in loss_functions_to_run.keys():
        row_data = {}
        for noise_rate_str in results: # Iterate through string keys ('0.0', '0.2'...)
             # Safely get result, defaulting to NaN if run failed or metric missing
             acc = results[noise_rate_str].get(loss_name, {}).get('final_test_acc', np.nan) \
                   if results[noise_rate_str].get(loss_name) is not None else np.nan
             row_data[float(noise_rate_str)] = acc # Use float for column sorting
        summary_data[loss_name] = row_data

    try:
        # Create pandas DataFrame
        df = pd.DataFrame.from_dict(summary_data, orient='index')
        # Ensure columns are sorted numerically by noise rate
        df = df.reindex(columns=sorted(noise_rates_to_run), fill_value=np.nan)
        df.index.name = 'Loss Function' # Label the index
        # Print formatted table to console
        print(df.round(2).to_string())
        # Save table to CSV file
        summary_csv_path = run_output_dir / 'summary_final_test_accuracy.csv'
        df.round(2).to_csv(summary_csv_path)
        print(f"Summary table saved to {summary_csv_path}")
    except Exception as e:
        print(f"Error creating or saving {noise_type} summary table: {e}")
    # --- End Summary Table ---

    # --- Invoke External Plotting Function ---
    # Design Choice (Iter 6): Delegate plotting for modularity.
    try:
        plot_dir = run_output_dir / 'plots' # Define specific plots subdirectory
        # Check if the plotting function is available
        if 'plot_comprehensive_results' in globals() and callable(plot_comprehensive_results):
             # Call the function, passing necessary data
             plot_comprehensive_results(
                 results=results, # The main results dictionary
                 loss_functions=loss_functions_to_run, # Dictionary defining losses run
                 noise_rates=noise_rates_to_run, # List of noise rates
                 output_dir=plot_dir, # Directory to save plots
                 symmetric=symmetric, # Noise type flag
                 exp_prefix=f"CIFAR10_{noise_type}" # Prefix for plot filenames
             )
             print(f"\n{noise_type.upper()} plots saved to {plot_dir}")
        else:
             # Handle case where plotting utility is missing
             print("\nWarning: Plotting function 'plot_comprehensive_results' not found or not callable. Skipping plot generation.")
    except Exception as e:
        # Catch errors during plotting
        print(f"\nError during {noise_type} plot generation: {e}")
        import traceback; traceback.print_exc()
    # --- End Plotting ---

    print(f"\n--- COMPLETED EXPERIMENT SET: {noise_type.upper()} ---")


# === Main Execution Block ===
def main():
    """Parses arguments and orchestrates the running of experiment sets."""
    # --- Argument Parsing ---
    # Design Choice (Iter 5 onwards): Use argparse for flexible configuration.
    parser = argparse.ArgumentParser(description='Run Noise Robustness Experiments (CIFAR-10)')
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate for SGD.')
    parser.add_argument('--num_epochs', type=int, default=120, help='Number of training epochs.')
    # Reproducibility & Setup
    parser.add_argument('--random_state', type=int, default=42, help='Seed for random number generators.')
    parser.add_argument('--base_output_dir', type=str, default='results_final_run', help='Base directory to save all results.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader worker processes.')
    parser.add_argument('--pin_memory', action='store_true', default=True, help='Use pinned memory for DataLoader (GPU only).') # Default changed to True
    # Performance & Device
    parser.add_argument('--use_amp', action='store_true', default=True, help='Enable Automatic Mixed Precision (CUDA only).') # Default changed to True
    parser.add_argument('--force_cpu', action='store_true', default=False, help='Force execution on CPU even if CUDA is available.')
    # Experiment Scope Control
    parser.add_argument('--subset_losses', action='store_true', default=False, help='Run only a small subset of loss functions for quick testing.')
    parser.add_argument('--subset_noise_rates', action='store_true', default=False, help='Run only a small subset of noise rates for quick testing.')
    # --- End Argument Parsing ---

    args = parser.parse_args() # Parse command-line arguments

    # Apply random seed globally
    set_seed(args.random_state)

    # --- Device Setup ---
    # Design Choice: Auto-select CUDA if available, allow forcing CPU. Enable TF32 for performance on CUDA.
    if torch.cuda.is_available() and not args.force_cpu:
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        # Performance Optimization (Iter 6): Enable TF32 on compatible GPUs (Ampere+) for matmul/conv.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
        print("Using CPU device.")
        # Disable AMP if running on CPU
        if args.use_amp:
            print("Warning: AMP requested but CPU selected. Disabling AMP.")
            args.use_amp = False
    # --- End Device Setup ---

    # --- Define Noise Rates ---
    # Design Choice: Allow running full or subset of rates via CLI flag.
    if args.subset_noise_rates:
        # Define smaller sets for quick tests
        symmetric_rates = [0.2, 0.6] # Example subset
        asymmetric_rates = [0.1, 0.4] # Example subset
        print("Running with SUBSET of noise rates.")
    else:
        # Define the full sets used in the papers/report
        symmetric_rates = [0.0, 0.2, 0.4, 0.6, 0.8]
        asymmetric_rates = [0.1, 0.2, 0.3, 0.4] # Note: 0.0 noise often omitted for asymmetric as it's same as clean
        print("Running with FULL set of noise rates.")
    # --- End Noise Rates ---

    # --- Execute Experiment Sets ---
    # Design Choice: Run symmetric and asymmetric experiments sequentially.
    # Can comment out one or the other as needed.

    # Run Symmetric Noise Experiments
    print("\n>>> Starting Symmetric Noise Experiments <<<")
    run_experiment_set(
        symmetric=True, # Flag for noise type
        noise_rates_to_run=symmetric_rates, # Noise levels for this set
        device=device, # Determined device
        base_output_dir_root=args.base_output_dir, # Parent output dir
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        random_state=args.random_state,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        use_amp=args.use_amp,
        subset_losses=args.subset_losses # Pass subset flag
    )

    # Run Asymmetric Noise Experiments
    print("\n>>> Starting Asymmetric Noise Experiments <<<")
    run_experiment_set(
        symmetric=False, # Flag for noise type
        noise_rates_to_run=asymmetric_rates, # Noise levels for this set
        device=device,
        base_output_dir_root=args.base_output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        random_state=args.random_state, # Use same base seed, noise function uses it internally
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        use_amp=args.use_amp,
        subset_losses=args.subset_losses # Pass subset flag
    )
    # --- End Experiment Execution ---

    print("\n\n--- ALL EXPERIMENTS COMPLETED ---")

# === Entry Point Guard ===
if __name__ == '__main__':
    # Design Choice (Iter 6): Check for plotting utility and create a dummy if missing.
    # This makes the core training script runnable even without the plotting dependency,
    # useful for testing or running in environments where plotting libraries aren't installed.
    plotting_utils_path = Path('plotting_utils.py')
    if not plotting_utils_path.exists():
         print(f"Warning: '{plotting_utils_path}' not found. Creating dummy plotting function.")
         try:
             with open(plotting_utils_path, 'w') as f:
                 # Write minimal Python code for a dummy function
                 f.write("import os\n")
                 f.write("from pathlib import Path\n")
                 f.write("try:\n") # Make matplotlib import optional
                 f.write("    import matplotlib.pyplot as plt\n")
                 f.write("    MPL_AVAILABLE = True\n")
                 f.write("except ImportError:\n")
                 f.write("    MPL_AVAILABLE = False\n\n")
                 f.write("def plot_comprehensive_results(*args, **kwargs):\n")
                 f.write("    print('[Plotting Utils] Dummy plot function called. No plots generated.')\n")
                 f.write("    output_dir = kwargs.get('output_dir', Path('./plots_dummy'))\n")
                 f.write("    os.makedirs(output_dir, exist_ok=True)\n")
                 f.write("    # Create a dummy file to indicate it ran\n")
                 f.write("    if MPL_AVAILABLE:\n")
                 f.write("        try:\n")
                 f.write("            fig, ax = plt.subplots(figsize=(2,1))\n") # Smaller dummy plot
                 f.write("            ax.text(0.5, 0.5, 'Dummy Plot', ha='center', va='center', fontsize=8)\n")
                 f.write("            ax.axis('off')\n") # Hide axes
                 f.write("            plt.savefig(output_dir / 'dummy_plot.png')\n")
                 f.write("            plt.close(fig)\n")
                 f.write("        except Exception as e:\n")
                 f.write("            print(f'Dummy plot generation failed: {e}')\n")
                 f.write("    else:\n")
                 f.write("        print('Matplotlib not available, skipping dummy plot generation.')\n")
         except Exception as e_write:
              print(f"Error writing dummy plotting_utils.py: {e_write}")

    # Call the main function to start the process
    main()