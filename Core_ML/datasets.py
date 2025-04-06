"""Contains the Noisy CIFAR-10 Dataset class."""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np

# === Noisy Dataset Class ===
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

        # Determine the subset of indices to use
        if indices is not None:
            self.indices = indices
        else:
            # Default to using all indices if none are specified
            self.indices = list(range(len(self.dataset)))

        # Store the targets corresponding *only* to the selected indices
        self.targets = np.array([self.dataset.targets[i] for i in self.indices])
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