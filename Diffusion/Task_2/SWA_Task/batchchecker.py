"""
Inspects a saved batch tensor ('batch_sample_rank0.pt')
created during DiT training to check preprocessing.
"""

import torch
import argparse
import os
import matplotlib.pyplot as plt
import torchvision.utils

# --- Helper Functions ---
def print_info(text): print(f">>> {text}")
def print_warning(text): print(f"!!! WARNING: {text}")
def print_error(text): print(f"!!! ERROR: {text}")

def inspect_batch(file_path="dit_workspace/DiT/batch_sample_rank0.pt", num_images_to_show=4):
    """Loads and inspects the saved batch tensor."""

    # Adjust path relative to where this script is run
    full_path = os.path.abspath(file_path)
    print_info(f"Attempting to load batch file: {full_path}")

    if not os.path.exists(full_path):
        print_error(f"Batch file not found.")
        print_error("Please ensure you have run the training script (which saves the batch)")
        print_error("and that this inspection script is run from the correct directory.")
        return

    try:
        # Load the tensor (saved as CPU tensor)
        batch_tensor = torch.load(full_path, map_location='cpu')
        print_info(f"Successfully loaded batch tensor.")

        # --- Basic Checks ---
        print("\n--- Tensor Statistics ---")
        print(f"Shape: {batch_tensor.shape}")
        print(f"Data Type: {batch_tensor.dtype}")

        if batch_tensor.ndim != 4 or batch_tensor.shape[1] != 3:
            print_warning(f"Expected shape like (B, 3, H, W), but got {batch_tensor.shape}.")

        if batch_tensor.dtype != torch.float32:
            print_warning(f"Expected dtype torch.float32, but got {batch_tensor.dtype}.")

        # Calculate value range
        min_val = torch.min(batch_tensor).item()
        max_val = torch.max(batch_tensor).item()
        mean_val = torch.mean(batch_tensor).item()

        print(f"Minimum value: {min_val:.4f}")
        print(f"Maximum value: {max_val:.4f}")
        print(f"Mean value: {mean_val:.4f}")

        # --- Range Check ---
        # Check if values are approximately within [-1, 1]
        # Allow for slight floating point inaccuracies
        if min_val < -1.01 or max_val > 1.01:
            print_warning("Values seem outside the expected [-1, 1] range after normalization!")
        else:
            print_info("Values are within the expected [-1, 1] range.")

        # --- Visualization (Optional) ---
        try:
            print("\n--- Visualizing Sample Images (Denormalized) ---")
            num_images = min(num_images_to_show, batch_tensor.shape[0])
            if num_images <= 0:
                 print_info("No images to show.")
                 return

            print_info(f"Displaying first {num_images} images from the batch...")

            # Select subset, denormalize: [-1, 1] -> [0, 1]
            images_to_show = batch_tensor[:num_images]
            images_denormalized = (images_to_show * 0.5) + 0.5
            images_denormalized = torch.clamp(images_denormalized, 0.0, 1.0) # Clamp just in case

            # Create a grid
            grid = torchvision.utils.make_grid(images_denormalized, nrow=int(num_images**0.5))

            # Plot
            plt.figure(figsize=(10, 10))
            plt.imshow(grid.permute(1, 2, 0).numpy()) # Permute to HWC for matplotlib
            plt.title(f"Sample Batch Images (Denormalized, First {num_images})")
            plt.axis('off')
            plt.tight_layout()

            # Save the figure
            output_filename = "dit_workspace/inspected_batch_visualization.png"
            plt.savefig(output_filename)
            print_info(f"Saved visualization to: {os.path.abspath(output_filename)}")
            # plt.show() # Uncomment if you want interactive display

        except ImportError:
            print_warning("Matplotlib or torchvision not found. Skipping visualization.")
            print_warning("Install them with: pip install matplotlib torchvision")
        except Exception as viz_e:
            print_error(f"Error during visualization: {viz_e}")

    except Exception as e:
        print_error(f"Failed to load or process batch file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect saved training batch.")
    parser.add_argument(
        "--file-path",
        type=str,
        # Default assumes you run inspect_batch.py from the parent dir of dit_workspace
        default="dit_workspace/DiT/batch_sample_rank0.pt",
        help="Path to the saved batch file (relative to script location or absolute)."
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=20,
        help="Number of images from the batch to visualize."
    )
    args = parser.parse_args()
    inspect_batch(args.file_path, args.num_images)