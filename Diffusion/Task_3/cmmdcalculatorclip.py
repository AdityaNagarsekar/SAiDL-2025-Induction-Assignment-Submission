"""
Calculate CLIP Mean Maximum Discrepancy (CMMD) between two sets of images.

Uses features extracted from a CLIP model and computes the MMD^2 score.
"""

import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import math

# --- Helper Functions ---
def print_header(text): print("\n" + "=" * 80 + f"\n=== {text.upper()} \n" + "=" * 80)
def print_error(text): print(f"!!! ERROR: {text}", file=sys.stderr)
def print_warning(text): print(f"!!! WARNING: {text}")
def print_info(text): print(f">>> {text}")

# --- Image Loading and Feature Extraction ---

class ImageDataset(torch.utils.data.Dataset):
    """Simple dataset to load images from a directory."""
    def __init__(self, image_dir, processor, target_size=(224, 224)):
        self.image_dir = image_dir
        self.processor = processor
        self.target_size = target_size # Should match CLIP model expected input
        self.image_files = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))
        ])
        if not self.image_files:
            raise ValueError(f"No valid image files found in {image_dir}")
        print_info(f"Found {len(self.image_files)} images in {os.path.basename(image_dir)}.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert("RGB")
             # Preprocess using the CLIP processor
            processed = self.processor(images=image, return_tensors="pt")
            # Return pixel values (usually already normalized by processor)
            # Squeeze removes the batch dimension added by the processor
            return processed['pixel_values'].squeeze(0)
        except (UnidentifiedImageError, OSError, Exception) as e:
            print_warning(f"Skipping corrupted/unreadable image: {img_path} ({e})")
            # Return a tensor of zeros or handle differently if needed
            # Returning None and filtering in collate_fn might be safer
            return None # Signal error

def collate_fn(batch):
    """Collate function to filter out None items (corrupted images)."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None # Return None if the whole batch failed
    return torch.stack(batch)

@torch.no_grad()
def get_clip_features(image_dir, model, processor, device, batch_size=64):
    """Extracts CLIP image features for all images in a directory."""
    model.eval()
    dataset = ImageDataset(image_dir, processor)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4, # Adjust based on your system
        pin_memory=True,
        collate_fn=collate_fn # Use the custom collate_fn
    )

    all_features = []
    print_info(f"Extracting CLIP features from {os.path.basename(image_dir)}...")
    for batch in tqdm(dataloader, desc=f"Feature Extraction ({os.path.basename(image_dir)})"):
        if batch is None: # Skip if collate_fn returned None
             print_warning("Skipping an empty/failed batch.")
             continue
        try:
            inputs = batch.to(device)
            features = model.get_image_features(pixel_values=inputs)
            all_features.append(features.cpu()) # Move features to CPU
        except Exception as e:
            print_error(f"Error processing a batch: {e}")
            # Decide whether to continue or raise

    if not all_features:
        print_error(f"No features extracted from {image_dir}. Check image files and logs.")
        return None

    return torch.cat(all_features, dim=0)

# --- MMD Calculation ---

def rbf_kernel(X, Y, sigma_sq):
    """Computes RBF kernel matrix between X and Y."""
    # Using squared Euclidean distance for efficiency with exp
    dist_sq = torch.cdist(X, Y, p=2)**2
    gamma = -1.0 / (2.0 * sigma_sq)
    K = torch.exp(gamma * dist_sq)
    return K

def calculate_mmd_sq(features1, features2, sigma_sq_list=None):
    """Calculates the squared Maximum Mean Discrepancy (MMD^2) using RBF kernel(s)."""
    if features1 is None or features2 is None:
        return float('nan')
    if features1.shape[0] == 0 or features2.shape[0] == 0:
        print_warning("One or both feature sets are empty, cannot compute MMD.")
        return float('nan')

    n = features1.shape[0]
    m = features2.shape[0]
    print_info(f"Calculating MMD^2 between {n} and {m} samples.")

    # Estimate sigma if not provided (median heuristic)
    if sigma_sq_list is None:
        print_info("Estimating sigma using median heuristic...")
        combined_features = torch.cat([features1, features2], dim=0)
        dists = torch.pdist(combined_features) # Pairwise distances within combined set
        if len(dists) == 0: # Only one point total
             sigma_sq = torch.tensor(1.0) # Default fallback
        else:
             median_dist = torch.median(dists)
             sigma_sq = (median_dist**2) / math.log(n + m) # Bandwidth heuristic adjustment
             if sigma_sq == 0: # Handle case where median is 0
                 sigma_sq = torch.tensor(1e-3) # Use a small value instead
        sigma_sq_list = [sigma_sq.item()]
        print_info(f"Using estimated sigma_sq list: {sigma_sq_list}")

    mmd2_total = 0.0
    for sigma_sq in sigma_sq_list:
        if sigma_sq <= 0:
            print_warning(f"Skipping invalid sigma_sq value: {sigma_sq}")
            continue

        K_XX = rbf_kernel(features1, features1, sigma_sq)
        K_YY = rbf_kernel(features2, features2, sigma_sq)
        K_XY = rbf_kernel(features1, features2, sigma_sq)

        # Unbiased estimate for MMD^2
        # Sum over off-diagonal elements for K_XX and K_YY
        term1 = (K_XX.sum() - K_XX.diag().sum()) / (n * (n - 1)) if n > 1 else 0.0
        term2 = (K_YY.sum() - K_YY.diag().sum()) / (m * (m - 1)) if m > 1 else 0.0
        term3 = K_XY.mean() * 2.0

        mmd2 = term1 + term2 - term3
        mmd2_total += mmd2

    # Average over kernels if multiple sigmas were used
    final_mmd2 = mmd2_total / len(sigma_sq_list)

    # MMD can sometimes be slightly negative due to estimator variance, clamp at 0.
    final_mmd2_clamped = max(0, final_mmd2.item())

    print_info(f"MMD^2 = {final_mmd2_clamped:.6f} (using sigma_sq={sigma_sq_list})")
    return final_mmd2_clamped

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Calculate CLIP-MMD.")
    parser.add_argument("--real-path", type=str, required=True, help="Path to directory with real images.")
    parser.add_argument("--fake-path", type=str, required=True, help="Path to directory with generated images.")
    parser.add_argument("--model-id", type=str, default="openai/clip-vit-base-patch32", help="HuggingFace model ID for CLIP.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for feature extraction.")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu).")
    # Optional: Provide fixed sigma values instead of estimating
    parser.add_argument("--sigma-sq", type=float, nargs='+', default=None, help="List of sigma squared values for RBF kernel(s). If None, uses median heuristic.")

    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == 'cuda':
        print_warning("CUDA not available, switching to CPU.")
        args.device = 'cpu'

    # --- Load Model ---
    print_header(f"Loading Model: {args.model_id}")
    try:
        processor = CLIPProcessor.from_pretrained(args.model_id)
        model = CLIPModel.from_pretrained(args.model_id).to(args.device)
        model.eval() # Set to evaluation mode
    except Exception as e:
        print_error(f"Failed to load CLIP model or processor: {e}")
        sys.exit(1)

    # --- Validate Paths ---
    print_header("Validating Input Paths")
    if not os.path.isdir(args.real_path):
        print_error(f"Real image path is not a valid directory: {args.real_path}")
        sys.exit(1)
    if not os.path.isdir(args.fake_path):
        print_error(f"Generated image path is not a valid directory: {args.fake_path}")
        sys.exit(1)
    print_info(f"Real path: {args.real_path}")
    print_info(f"Fake path: {args.fake_path}")

    # --- Extract Features ---
    print_header("Extracting Features")
    try:
        real_features = get_clip_features(args.real_path, model, processor, args.device, args.batch_size)
        fake_features = get_clip_features(args.fake_path, model, processor, args.device, args.batch_size)
    except Exception as e:
        print_error(f"Error during feature extraction: {e}")
        sys.exit(1)

    if real_features is None or fake_features is None:
        print_error("Feature extraction failed for one or both paths. Cannot calculate CMMD.")
        sys.exit(1)

    # --- Calculate CMMD ---
    print_header("Calculating CMMD")
    cmmd_score_sq = calculate_mmd_sq(real_features, fake_features, args.sigma_sq)

    print("\n" + "-" * 30)
    print(f"Final CMMD^2 Score: {cmmd_score_sq:.6f}")
    print("-" * 30)

if __name__ == "__main__":
    main()