"""
Script to check VAE encoding and decoding functionality.

Loads a specified Stable Diffusion VAE, processes an input image,
encodes it to the latent space, decodes it back, and saves the
original (processed) and reconstructed images for visual comparison.
"""

import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
from diffusers.models import AutoencoderKL
import numpy as np
import traceback # Import traceback for detailed error printing

# VAE scaling factor (from original SD/DiT usage)
VAE_SCALE_FACTOR = 0.18215

def print_info(text):
    """Prints an info message."""
    print(f">>> {text}")

def print_error(text):
    """Prints an error message."""
    print(f"!!! ERROR: {text}")

def main(args):
    # --- Device Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # Uncomment below if you have MPS (Apple Silicon) and want to try it
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print_info(f"Using device: {device}")

    # --- Load Image ---
    if not os.path.isfile(args.image_path):
        print_error(f"Input image not found at: {args.image_path}")
        return
    try:
        input_image_pil = Image.open(args.image_path).convert("RGB")
        print_info(f"Loaded image: {args.image_path} (Original size: {input_image_pil.size})")
    except Exception as e:
        print_error(f"Failed to load image: {e}")
        return

    # --- Preprocessing ---
    # Mimic the preprocessing from training: resize, normalize
    # Note: center_crop_arr logic is simplified here by just resizing directly
    preprocess = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.LANCZOS),
        # Ensure square aspect ratio by resizing (center crop is implicit)
        transforms.CenterCrop(args.image_size), # Ensure exactly image_size x image_size
        transforms.ToTensor(), # Scales pixels to [0.0, 1.0]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Scales to [-1.0, 1.0]
    ])

    try:
        # Apply preprocessing
        processed_image_tensor = preprocess(input_image_pil)
        # Add batch dimension
        input_tensor = processed_image_tensor.unsqueeze(0).to(device)
        print_info(f"Image preprocessed to tensor shape: {input_tensor.shape}")

        # Also save the *processed* PIL image (after resize/crop but before normalize)
        # for accurate visual comparison of what went *into* the VAE tensor-wise.
        # Denormalize tensor from [-1, 1] back to [0, 1] range for saving
        processed_image_pil = transforms.ToPILImage()( (processed_image_tensor * 0.5) + 0.5 )

    except Exception as e:
        print_error(f"Failed during image preprocessing: {e}")
        return

    # --- Load VAE ---
    vae_model_name = f"stabilityai/sd-vae-ft-{args.vae}"
    try:
        print_info(f"Loading VAE: {vae_model_name}...")
        vae = AutoencoderKL.from_pretrained(vae_model_name).to(device)
        vae.eval() # Set to evaluation mode
        print_info("VAE loaded successfully.")
    except Exception as e:
        print_error(f"Could not load VAE '{vae_model_name}': {e}")
        return

    # --- Encoding & Decoding ---
    try:
        print_info("Encoding image to latent space...")
        with torch.no_grad(): # Disable gradients for inference
            # Encode
            latent_dist = vae.encode(input_tensor).latent_dist
            # Sample from the distribution (like in training)
            latents = latent_dist.sample()
            # Apply scaling factor
            scaled_latents = latents * VAE_SCALE_FACTOR
            print_info(f"Encoded to latents shape: {scaled_latents.shape}")

            print_info("Decoding latents back to pixel space...")
            # Decode requires removing the scaling factor first
            # --- THIS IS THE CORRECTED LINE ---
            decoded_output = vae.decode(scaled_latents / VAE_SCALE_FACTOR).sample
            # --- END CORRECTION ---

            # Move reconstructed tensor to CPU for postprocessing
            reconstructed_tensor_norm = decoded_output.detach().cpu().float() # Ensure float
            print_info(f"Decoded tensor shape: {reconstructed_tensor_norm.shape}")

    except Exception as e:
        print_error(f"Error during VAE encode/decode: {e}")
        traceback.print_exc() # Print detailed traceback
        return

    # --- Postprocessing ---
    try:
        # Remove batch dimension
        reconstructed_tensor_norm = reconstructed_tensor_norm.squeeze(0)

        # Denormalize: Bring [-1, 1] back to [0, 1]
        reconstructed_tensor_0_1 = (reconstructed_tensor_norm * 0.5) + 0.5
        # Clamp values to ensure they are within [0, 1] range
        reconstructed_tensor_0_1 = torch.clamp(reconstructed_tensor_0_1, 0.0, 1.0)

        # Convert tensor to PIL Image
        postprocess = transforms.ToPILImage()
        reconstructed_image_pil = postprocess(reconstructed_tensor_0_1)
        print_info("Reconstructed tensor converted back to PIL Image.")

    except Exception as e:
        print_error(f"Error during postprocessing: {e}")
        return

    # --- Save Results ---
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        in_filename = os.path.join(args.output_dir, f"vae_{args.vae}_input_resized.png")
        out_filename = os.path.join(args.output_dir, f"vae_{args.vae}_reconstructed.png")
        combined_filename = os.path.join(args.output_dir, f"vae_{args.vae}_comparison.png")

        # Save the processed input image
        processed_image_pil.save(in_filename)
        print_info(f"Saved processed input image to: {in_filename}")

        # Save the reconstructed image
        reconstructed_image_pil.save(out_filename)
        print_info(f"Saved reconstructed image to: {out_filename}")

        # Save side-by-side comparison
        img_width, img_height = processed_image_pil.size
        comparison_img = Image.new('RGB', (img_width * 2, img_height))
        comparison_img.paste(processed_image_pil, (0, 0))
        comparison_img.paste(reconstructed_image_pil, (img_width, 0))
        comparison_img.save(combined_filename)
        print_info(f"Saved side-by-side comparison to: {combined_filename}")

    except Exception as e:
        print_error(f"Error saving output images: {e}")

    print_info("VAE check script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check VAE Encoding/Decoding")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--output-dir", type=str, default="./vae_check_output", help="Directory to save output images.")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema", help="Which VAE weights to use (ema or mse).")
    parser.add_argument("--image-size", type=int, default=256, help="Image size to resize input to (matching DiT training).")

    args = parser.parse_args()
    main(args)