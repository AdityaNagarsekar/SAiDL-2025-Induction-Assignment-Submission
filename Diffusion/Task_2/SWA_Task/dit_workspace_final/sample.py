
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
# from download import find_model # Removed, loading custom checkpoints
from models import DiT_models # Ensure this imports the modified DiT
import argparse
import os
import math # Added math import
import torch # Ensure torch is imported for type hints/checks


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False) # Important for inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: Sampling on CPU is possible but very slow.")
    print(f"Using device: {device}")

    # Load checkpoint:
    ckpt_path = args.ckpt
    if ckpt_path is None:
        raise ValueError("Must provide a checkpoint path using --ckpt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint path '{ckpt_path}' does not exist or is not a file.")

    print(f"Loading checkpoint from: {ckpt_path}")
    # --- MODIFICATION: Added weights_only=False ---
    # This is crucial if the checkpoint contains more than just the state_dict,
    # like optimizer state, args, etc., which is the case for our saved checkpoints.
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print("Checkpoint loaded successfully (weights_only=False).")
    except Exception as e:
         print(f"Error loading checkpoint with torch.load(..., weights_only=False): {e}")
         print("Attempting to load with weights_only=True as fallback (might fail if it's not just a state_dict)...")
         try:
             # Fallback attempt, less likely to work for training checkpoints
             checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
             print("Fallback load with weights_only=True succeeded (checkpoint might be just a state_dict).")
         except Exception as e2:
             raise RuntimeError(f"Failed to load checkpoint with both weights_only=False and True. Error: {e2}") from e

    # Determine which state dict to use (EMA preferred)
    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        print("Using EMA weights from checkpoint.")
        state_dict = checkpoint["ema"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        print("Using 'model' weights from checkpoint (possibly from DDP training).")
        # Remove 'module.' prefix if present (from DDP)
        state_dict = checkpoint["model"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    elif isinstance(checkpoint, dict) and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in checkpoint.items()):
         # Check if the loaded object itself is a state_dict
        print("Using raw checkpoint weights (assuming the loaded object is the state_dict).")
        state_dict = checkpoint
    else:
        ckpt_keys = list(checkpoint.keys()) if isinstance(checkpoint, dict) else "N/A (not a dict)"
        raise ValueError(f"Checkpoint format not recognized or missing required keys ('ema' or 'model'). Keys found: {ckpt_keys}")

    # Get model configuration from checkpoint args if available
    ckpt_args = checkpoint.get("args", None) if isinstance(checkpoint, dict) else None
    if ckpt_args:
        # Prioritize args from checkpoint
        try:
             model_name = ckpt_args.model
             image_size = ckpt_args.image_size
             # Handle potential missing attributes gracefully using getattr
             attention_type = getattr(ckpt_args, 'attention_type', args.attention_type) # Fallback to cli arg
             window_size_ckpt = getattr(ckpt_args, 'window_size', args.window_size)     # Fallback to cli arg
             num_classes_ckpt = getattr(ckpt_args, 'num_classes', 0) # Assume 0 if missing
             vae_ckpt = getattr(ckpt_args, 'vae', args.vae) # Fallback to cli arg VAE

             print(f"Model config loaded from checkpoint args:")
             print(f"  model: {model_name}, image_size: {image_size}, num_classes: {num_classes_ckpt}")
             print(f"  attention_type: {attention_type}, window_size: {window_size_ckpt}, vae: {vae_ckpt}")
        except AttributeError as e:
             print(f"Warning: Attribute missing in checkpoint args ({e}). Falling back to command-line arguments for some settings.")
             # Fallback to command-line args if ckpt_args is malformed or missing fields
             model_name = args.model or parser.get_default('model') # Use default if not provided
             image_size = args.image_size
             attention_type = args.attention_type
             window_size_ckpt = args.window_size
             num_classes_ckpt = args.num_classes # Use CLI arg for num_classes
             vae_ckpt = args.vae
             if model_name is None:
                  raise ValueError("Need to specify --model if 'model' not found in checkpoint args.")
             print(f"Using command-line/default args for config.")

    else:
        print("Warning: Model args not found in checkpoint. Using command-line arguments.")
        model_name = args.model
        image_size = args.image_size
        attention_type = args.attention_type
        window_size_ckpt = args.window_size if attention_type == 'swa' else None
        num_classes_ckpt = args.num_classes # Use CLI arg for num_classes
        vae_ckpt = args.vae
        if model_name is None:
            raise ValueError("Need to specify --model if args not found in checkpoint.")
        print(f"Using command-line args for config: {model_name}, size {image_size}, attn {attention_type}, win {window_size_ckpt}, classes {num_classes_ckpt}, vae {vae_ckpt}")


    # Final window size based on attention type
    window_size_final = window_size_ckpt if attention_type == 'swa' else None

    # Create DiT model instance
    latent_size = image_size // 8
    model = DiT_models[model_name](
        input_size=latent_size,
        num_classes=num_classes_ckpt, # Use num_classes from checkpoint/args
        window_size=window_size_final # Use derived window size
    ).to(device)

    # Load the state dict
    # Use strict=False initially; can help if EMA keys slightly differ or if loading non-EMA weights
    load_result = model.load_state_dict(state_dict, strict=False)
    print(f"Weight loading results (strict=False): {load_result}")
    # If strict=False had missing keys, they remain initialized.
    # If it had unexpected keys, they are ignored. Check the output carefully.
    if load_result.missing_keys:
        print(f"Warning: Missing keys during state_dict load: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"Warning: Unexpected keys during state_dict load: {load_result.unexpected_keys}")

    model.eval() # Set model to evaluation mode

    # Create diffusion sampler
    # timestep_respacing determines the sampling schedule (e.g., "ddim100", "250")
    diffusion = create_diffusion(str(args.num_sampling_steps))
    print(f"Diffusion sampler created with {args.num_sampling_steps} steps.")

    # Load VAE for decoding
    vae_model_name = f"stabilityai/sd-vae-ft-{vae_ckpt}"
    try:
        print(f"Loading VAE: {vae_model_name}")
        vae = AutoencoderKL.from_pretrained(vae_model_name).to(device)
        vae.eval()
        print("VAE loaded successfully.")
    except Exception as e:
        print(f"Error: Could not load VAE '{vae_model_name}': {e}")
        return # Cannot proceed without VAE

    # Prepare for sampling (unconditional)
    n = args.num_samples
    z = torch.randn(n, 4, latent_size, latent_size, device=device) # Start with random noise in latent space
    model_kwargs = dict() # No class labels for unconditional sampling

    # Classifier-Free Guidance (CFG) setup if scale > 1.0
    if args.cfg_scale > 1.0:
         if num_classes_ckpt == 0:
              print("Warning: CFG scale > 1.0 requested, but model seems unconditional (num_classes=0). CFG will have no effect.")
              # Proceed without CFG logic or error out? Let's proceed but warn.
              sample_fn = model.forward # Use standard forward pass
         else:
              print(f"Using Classifier-Free Guidance with scale {args.cfg_scale}")
              # Prepare class labels for CFG (requires a placeholder for unconditional)
              # Assume class index `num_classes_ckpt` is the unconditional embedding token index
              z = torch.cat([z, z], 0) # Double the batch for conditional/unconditional passes
              model_kwargs["y"] = torch.randint(0, num_classes_ckpt, (n,), device=device) # Example: random classes for conditional part
              model_kwargs["y"] = torch.cat([model_kwargs["y"], torch.full_like(model_kwargs["y"], num_classes_ckpt)], 0) # Add unconditional tokens
              model_kwargs["cfg_scale"] = args.cfg_scale
              sample_fn = model.forward_with_cfg # Use the CFG forward method
    else:
         print("CFG scale <= 1.0, using standard forward pass.")
         sample_fn = model.forward # Use standard forward pass


    # Start sampling
    print(f"Starting sampling loop for {n} images...")
    samples = diffusion.p_sample_loop(
        sample_fn, # Use the selected forward function (with or without CFG)
        z.shape,   # Shape of the noise tensor (B, C, H, W)
        z,         # Starting noise
        clip_denoised=False, # Standard DiT setting
        model_kwargs=model_kwargs, # Pass labels and CFG scale if used
        progress=True, # Show progress bar
        device=device
    )
    # If CFG was used, take the first half of the results (the guided samples)
    if args.cfg_scale > 1.0 and num_classes_ckpt > 0:
         samples, _ = samples.chunk(2, dim=0) # Need to handle this correctly based on p_sample_loop output structure if CFG used inside it vs outside

    print("Sampling loop complete.")


    # Decode samples from latent space to pixel space using VAE
    try:
        print("Decoding samples with VAE...")
        # Magic number 0.18215 is VAE scaling factor
        samples = vae.decode(samples / 0.18215).sample
        print("Decoding complete.")
    except Exception as e:
         print(f"Error during VAE decoding: {e}")
         # Samples are likely still in latent space, saving them might not be useful visually
         return

    # Save generated images as a grid
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir}")

    # Determine grid size (e.g., 4x4 for 16 samples)
    try:
         grid_size = int(math.sqrt(n))
         if grid_size * grid_size != n: # Handle non-perfect squares if necessary
              grid_size = math.ceil(math.sqrt(n))
              print(f"Adjusting grid size for {n} samples to {grid_size}x{grid_size}")
    except ValueError:
         grid_size = 4 # Default grid size if calculation fails
         print(f"Using default grid size: {grid_size}")


    try:
        save_image(samples, args.output_path, nrow=grid_size, normalize=True, value_range=(-1, 1))
        print(f"Saved {n} samples as a grid to: {args.output_path}")
    except Exception as e:
        print(f"Error saving image grid: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model/Checkpoint Args
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the DiT checkpoint (.pt file).")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default=None, help="DiT model architecture. If None, loads from checkpoint args.")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256, help="Image size (loaded from checkpoint if available).")
    parser.add_argument("--num-classes", type=int, default=0, help="Number of classes for conditional models (loaded from ckpt if available). Set > 0 for CFG.")
    parser.add_argument("--attention-type", type=str, choices=['full', 'swa'], default='full', help="Attention type (loaded from checkpoint if available).")
    parser.add_argument("--window-size", type=int, default=8, help="SWA window size (loaded from checkpoint if available).")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema", help="Which VAE weights to use (ema or mse) (loaded from ckpt if available).")
    # Sampling Args
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument("--num-sampling-steps", type=int, default=1000, help="Number of DDPM/DDIM sampling steps.")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of samples to generate.")
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="Scale for Classifier-Free Guidance. 1.0 means no guidance.")
    # Output Args
    parser.add_argument("--output-path", type=str, default="samples_unconditional.png", help="Path to save the output image grid.")

    args = parser.parse_args()
    main(args)

