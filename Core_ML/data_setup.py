# data_setup.py
"""Contains the function to create CIFAR-10 DataLoaders."""

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Import the custom dataset class
from datasets import CIFAR10Noisy

# === Data Loader Creation Function ===
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