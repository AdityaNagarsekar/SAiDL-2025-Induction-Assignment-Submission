import glob

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm # Use standard tqdm for terminal
import os
from PIL import Image
import numpy as np
import math
import argparse
import torch # Ensure torch is imported for type hints/checks

def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
        print("DDP process group destroyed.")

def create_npz_from_sample_folder(sample_dir, image_size, num_expected):
    samples = []
    files_found = sorted(glob.glob(f"{sample_dir}/*.png")) # Find all PNGs and sort
    print(f"Building NPZ: Found {len(files_found)} PNG files in {sample_dir}. Expecting up to {num_expected}.")

    # Limit to num_expected or files found, whichever is smaller
    num_to_process = min(len(files_found), num_expected)
    if num_to_process < num_expected:
         print(f"Warning: Found only {num_to_process} files, less than the expected {num_expected}.")
    if num_to_process == 0:
         print("Error: No image files found in the directory. Cannot create NPZ.")
         return None

    processed_count = 0
    for i in tqdm(range(num_to_process), desc="Building .npz file"):
        # Construct filename assuming leading zeros format used during saving
        # filename = f"{i:06d}.png" # Assuming 6 digit padding
        # Use glob results instead to handle potential gaps or different naming
        filepath = files_found[i]
        # sample_path = os.path.join(sample_dir, filename) # Original logic

        # if not os.path.exists(sample_path):
        #      print(f"Warning: Expected sample file {sample_path} not found. Skipping.")
        #      continue # Skip if a specific file in sequence is missing

        try:
            # Open image, ensure RGB, resize to target (important for FID consistency!)
            img = Image.open(filepath).convert('RGB')
            if img.size != (image_size, image_size):
                 img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
            sample_np = np.asarray(img) # Converts to (H, W, C) uint8
            samples.append(sample_np)
            processed_count += 1
        except Exception as e:
             print(f"Warning: Could not read/process file {filepath}: {e}")

    if not samples:
         print("Error: No valid samples could be processed to create npz file.")
         return None

    samples = np.stack(samples) # Stack along new dimension 0 -> (N, H, W, C)
    print(f"Successfully processed {processed_count} images. Final array shape: {samples.shape}")

    npz_path = f"{sample_dir}.npz" # Save NPZ adjacent to the folder
    try:
        # Use compression for potentially large files
        np.savez_compressed(npz_path, arr_0=samples) # FID script expects 'arr_0' key
        print(f"Saved compressed .npz file to {npz_path}")
        return npz_path
    except Exception as e:
         print(f"Error saving .npz file {npz_path}: {e}")
         return None


@torch.no_grad() # Disable gradients for inference
def main(args):
    # Setup Torch backend flags
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32 # Often same as matmul flag
    print(f"TF32 computation support: {args.tf32}")

    assert torch.cuda.is_available(), "Distributed sampling currently requires GPU(s)."
    torch.set_grad_enabled(False) # Ensure grads are off

    # --- DDP Setup ---
    # Identical to the training script DDP setup logic
    is_ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    if is_ddp:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        device = local_rank
        print(f"Initializing DDP for Sampling: RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}, DEVICE=cuda:{device}")
        try:
            dist.init_process_group(backend='nccl', init_method='env://')
            torch.cuda.set_device(device)
            dist.barrier()
            print(f"DDP Process group initialized for sampling. Rank {rank} on device cuda:{device}.")
        except Exception as e:
             print(f"Error initializing DDP process group on rank {rank}: {e}")
             raise
    else:
        rank = 0; world_size = 1; local_rank = 0; device = 0
        print("Single process sampling. DDP not initialized or world_size=1.")

    torch.cuda.set_device(device) # Set device again just in case
    print(f"Process Rank: {rank}, Local Rank: {local_rank}, Using Device: cuda:{device}")

    # --- Seeding ---
    seed = args.global_seed + rank # Rank-specific seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Process rank {rank} initialized with seed {seed}.")

    # --- Load Checkpoint (on CPU first) ---
    ckpt_path = args.ckpt
    if ckpt_path is None: raise ValueError("--ckpt argument is required.")
    if not os.path.isfile(ckpt_path): raise FileNotFoundError(f"Checkpoint {ckpt_path} not found.")

    if rank == 0: print(f"Loading checkpoint: {ckpt_path} (on CPU initially)")
    # Use weights_only=False as checkpoints contain args etc.
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
         raise RuntimeError(f"Failed to load checkpoint {ckpt_path} with weights_only=False: {e}")

    # --- Extract State Dict and Config ---
    # Same logic as single sample.py
    state_dict = None
    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        if rank == 0: print("Using EMA weights from checkpoint.")
        state_dict = checkpoint["ema"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        if rank == 0: print("Using 'model' weights from checkpoint.")
        state_dict = checkpoint["model"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()} # Strip DDP prefix
    elif isinstance(checkpoint, dict) and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in checkpoint.items()):
        if rank == 0: print("Using raw checkpoint weights (assuming checkpoint is the state_dict).")
        state_dict = checkpoint
    else:
        raise ValueError(f"Checkpoint format not recognized. Keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'N/A'}")

    # Load config from checkpoint args, fall back to command-line args
    ckpt_args = checkpoint.get("args", None) if isinstance(checkpoint, dict) else None
    if ckpt_args:
        try:
            model_name = ckpt_args.model
            image_size = ckpt_args.image_size
            attention_type = getattr(ckpt_args, 'attention_type', args.attention_type)
            window_size_ckpt = getattr(ckpt_args, 'window_size', args.window_size)
            num_classes_ckpt = getattr(ckpt_args, 'num_classes', 0)
            vae_ckpt = getattr(ckpt_args, 'vae', args.vae)
            if rank == 0: print(f"Model config loaded from checkpoint: {model_name}, size {image_size}, classes {num_classes_ckpt}, attn {attention_type}, win {window_size_ckpt}, vae {vae_ckpt}")
        except AttributeError as e:
             print(f"Warning: Attribute missing in checkpoint args ({e}). Falling back to command-line.")
             model_name = args.model or parser.get_default('model') # Use default if not provided
             image_size = args.image_size
             attention_type = args.attention_type
             window_size_ckpt = args.window_size
             num_classes_ckpt = args.num_classes # Use CLI arg for num_classes
             vae_ckpt = args.vae
             if model_name is None: raise ValueError("Need --model if 'model' not in ckpt args.")

    else:
        if rank == 0: print("Warning: Model args not found in checkpoint. Using command-line arguments.")
        model_name = args.model
        image_size = args.image_size
        attention_type = args.attention_type
        window_size_ckpt = args.window_size
        num_classes_ckpt = args.num_classes
        vae_ckpt = args.vae
        if model_name is None: raise ValueError("Need --model if args not found in checkpoint.")
        if rank == 0: print(f"Using command-line args for config: {model_name}, size {image_size}, classes {num_classes_ckpt}, attn {attention_type}, win {window_size_ckpt}, vae {vae_ckpt}")

    window_size_final = window_size_ckpt if attention_type == 'swa' else None

    # --- Create Model and Load Weights ---
    latent_size = image_size // 8
    model = DiT_models[model_name](
        input_size=latent_size,
        num_classes=num_classes_ckpt,
        window_size=window_size_final
    ).to(device) # Create model directly on target device

    load_result = model.load_state_dict(state_dict, strict=False)
    if rank == 0: print(f"Weight loading results (strict=False): {load_result}")
    model.eval() # Set to evaluation mode
    if rank == 0: print(f"Model '{model_name}' loaded to device {device} and set to eval mode.")

    # --- Create Diffusion Sampler ---
    diffusion = create_diffusion(str(args.num_sampling_steps))
    if rank == 0: print(f"Diffusion sampler created with {args.num_sampling_steps} steps.")

    # --- Load VAE ---
    vae_model_name = f"stabilityai/sd-vae-ft-{vae_ckpt}"
    try:
        if rank == 0: print(f"Loading VAE: {vae_model_name}")
        vae = AutoencoderKL.from_pretrained(vae_model_name).to(device)
        vae.eval()
        if rank == 0: print("VAE loaded successfully.")
    except Exception as e:
        print(f"Error: Could not load VAE '{vae_model_name}' on device {device}: {e}")
        cleanup()
        return

    # --- Prepare for Distributed Sampling ---
    total_samples_target = args.num_fid_samples
    # Calculate samples per process, handling potential uneven division
    samples_per_proc = total_samples_target // world_size
    remainder = total_samples_target % world_size
    # Distribute remainder among first 'remainder' ranks
    my_samples_count = samples_per_proc + 1 if rank < remainder else samples_per_proc

    if my_samples_count == 0:
        print(f"Rank {rank} has no samples to generate.")
    else:
        print(f"Rank {rank} will generate {my_samples_count} samples.")

    # Create unique sample directory for this run (if rank 0)
    # All ranks will save into this directory.
    # sample_folder_dir needs to be consistent across ranks. Rank 0 creates it.
    sample_folder_dir = args.sample_dir
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Sample output directory: {sample_folder_dir}")
    if is_ddp: dist.barrier() # Ensure directory exists before workers proceed

    # --- Sampling Loop (Each process generates its share) ---
    samples_saved_count = 0
    batch_size = args.per_proc_batch_size
    total_batches = math.ceil(my_samples_count / batch_size)

    if my_samples_count > 0:
         print(f"Rank {rank} starting sampling loop: {total_batches} batches of size {batch_size}")
         # Determine the global starting index for this rank's samples
         # Sum of samples from ranks 0 to rank-1
         start_index = rank * samples_per_proc + min(rank, remainder)

         for i in tqdm(range(total_batches), desc=f"Rank {rank} Sampling", disable=(rank!=0)):
             current_batch_size = min(batch_size, my_samples_count - (i * batch_size))
             if current_batch_size <= 0: break # Should not happen with ceil, but safety check

             # Generate noise for the current batch
             z = torch.randn(current_batch_size, 4, latent_size, latent_size, device=device)
             model_kwargs = dict() # Unconditional sampling assumed for FID

             # Use standard forward pass (no CFG typically for FID)
             sample_fn = model.forward

             # Perform diffusion sampling
             samples_latent = diffusion.p_sample_loop(
                 sample_fn, z.shape, z, clip_denoised=False,
                 model_kwargs=model_kwargs, progress=False, # Disable inner progress bar
                 device=device
             )

             # Decode batch with VAE
             try:
                 samples_pixels = vae.decode(samples_latent / 0.18215).sample
             except Exception as e:
                  print(f"Rank {rank} VAE decode failed on batch {i}: {e}. Skipping batch save.")
                  continue

             # Save individual images
             # Calculate global index for each image in the batch
             for j in range(current_batch_size):
                 img_index = start_index + i * batch_size + j
                 img_pixels = samples_pixels[j] # Get single image tensor (C, H, W)

                 # Convert to PIL image (normalize=True, value_range=(-1, 1) maps to [0, 1])
                 # Then scale to [0, 255] and convert to uint8
                 img_pixels = (img_pixels.clamp(-1, 1) + 1) / 2 # Map to [0, 1]
                 img_pixels = (img_pixels * 255).byte() # Map to [0, 255] uint8
                 img_pil = Image.fromarray(img_pixels.permute(1, 2, 0).cpu().numpy()) # HWC format

                 # Save image with leading zeros in filename
                 save_path = os.path.join(sample_folder_dir, f"{img_index:06d}.png")
                 try:
                     img_pil.save(save_path)
                     samples_saved_count += 1
                 except Exception as e:
                      print(f"Rank {rank} failed to save image {save_path}: {e}")

         print(f"Rank {rank} finished sampling loop. Saved {samples_saved_count} images.")

    # --- Wait for all processes to finish sampling ---
    if is_ddp:
        print(f"Rank {rank} entering barrier after sampling loop.")
        dist.barrier()
        print(f"Rank {rank} passed barrier.")

    # --- Create NPZ file (Rank 0 Only) ---
    npz_file_path = None
    if rank == 0:
        print(f"Rank 0 attempting to create NPZ file from {sample_folder_dir}...")
        npz_file_path = create_npz_from_sample_folder(sample_folder_dir, image_size, total_samples_target)
        if npz_file_path:
             print(f"Rank 0 successfully created NPZ file: {npz_file_path}")
        else:
             print("Rank 0 failed to create NPZ file.")

    # --- Final Cleanup ---
    if is_ddp:
        dist.barrier() # Ensure rank 0 finishes NPZ before cleanup
        cleanup()

    print(f"Process {rank} finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model & Checkpoint Args (similar to sample.py, prioritize ckpt args)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the DiT checkpoint (.pt file).")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default=None, help="DiT architecture (if not found in ckpt).")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256, help="Image size (if not found in ckpt).")
    parser.add_argument("--num-classes", type=int, default=0, help="Number of classes (if not found in ckpt).")
    parser.add_argument("--attention-type", type=str, choices=['full', 'swa'], default='full', help="Attention type (if not found in ckpt).")
    parser.add_argument("--window-size", type=int, default=8, help="SWA window size (if not found in ckpt).")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema", help="VAE weights (if not found in ckpt).")

    # Sampling Args
    parser.add_argument("--sample-dir", type=str, default="samples_fid", help="Directory to save individual samples before creating NPZ.")
    parser.add_argument("--per-proc-batch-size", type=int, default=16, help="Batch size per GPU during sampling.")
    parser.add_argument("--num-fid-samples", type=int, default=10000, help="Total number of samples to generate across all processes for FID.")
    parser.add_argument("--num-sampling-steps", type=int, default=1000, help="Number of DDPM/DDIM sampling steps.")
    parser.add_argument("--global-seed", type=int, default=0, help="Base random seed for sampling.")

    # System Args
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable TF32 precision.") # Uses --tf32 / --no-tf32

    # --- FIX: Parse the arguments ---
    args = parser.parse_args()
    # --------------------------------

    main(args) # Pass the parsed args to main

