# benchmark.py
"""
Main script to benchmark the DiT model with standard vs xformers attention.

Instantiates the models, runs a dummy sampling loop, and measures execution time.
"""
import torch
import time
import math # math might not be strictly needed here anymore, but good to keep if utils change
import numpy as np # numpy might not be strictly needed here anymore
from diffusers import DDPMScheduler # Or DDIMScheduler, etc.

# Import configuration parameters
from config import *

# Import model, scheduler, and utilities
from model import DiT
from scheduler import DummyScheduler
from utils import * # Import utility functions if needed directly (likely not)
# Import the availability flag from where it's defined
from modules.attention import XFORMERS_AVAILABLE

# --- Benchmarking Function ---

@torch.no_grad()
def sample_images(model, scheduler, num_images, num_steps, batch_size, device):
    """Simulates sampling images using a diffusers scheduler and measures the time."""
    model.eval()
    model.to(device)

    num_batches = (num_images + batch_size - 1) // batch_size
    # Latent shape based on model config
    latent_shape = (batch_size, IN_CHANNELS, IMG_SIZE, IMG_SIZE)

    total_time = 0.0

    for i in range(num_batches):
        current_batch_size = min(batch_size, num_images - i * batch_size)
        if current_batch_size == 0: break

        # Adjust batch size for the last potentially smaller batch
        if current_batch_size != latent_shape[0]:
             current_latent_shape = (current_batch_size, *latent_shape[1:])
        else:
             current_latent_shape = latent_shape

        print(f"  Batch {i+1}/{num_batches}, Size: {current_batch_size}")

        # --- Warmup Run (for the first batch only) ---
        if i == 0:
            print("    Warmup run...")
            latents_warmup = torch.randn(current_latent_shape, device=device)
            # Use a timestep within the typical training range (e.g., near the middle)
            t_warmup = torch.randint(1, scheduler.config.num_train_timesteps // 2, (current_batch_size,), device=device).long()
            _ = model(latents_warmup, t_warmup) # Run model once
            torch.cuda.synchronize()
            print("    Warmup done.")

        # --- Timed Run ---
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Start with random noise matching the batch size
        latents = torch.randn(current_latent_shape, device=device)

        # Set the number of inference steps in the scheduler
        scheduler.set_timesteps(num_steps)

        start_event.record()

        # Iterate over the scheduler's timesteps
        for t_step in scheduler.timesteps:
            # Prepare timestep tensor for the batch
            # Needs to be broadcastable to the batch size
            # Use .item() if t_step is a tensor, otherwise just use t_step if it's already int
            current_t = t_step.item() if isinstance(t_step, torch.Tensor) else t_step
            timestep_tensor = torch.full((current_batch_size,), current_t, device=device, dtype=torch.long)

            # Model prediction (predicts the noise added)
            noise_pred = model(latents, timestep_tensor)

            # Scheduler step: Compute the previous noisy sample
            # Arguments might vary slightly based on the specific scheduler,
            # but DDPMScheduler typically takes (model_output, timestep, sample)
            scheduler_output = scheduler.step(noise_pred, t_step, latents)

            # Update latents using the scheduler's output
            latents = scheduler_output.prev_sample

        end_event.record()
        torch.cuda.synchronize() # Wait for GPU operations to complete

        batch_time_ms = start_event.elapsed_time(end_event)
        total_time += batch_time_ms
        print(f"    Batch {i+1} time: {batch_time_ms / 1000.0:.4f} seconds")


    avg_time_per_batch = (total_time / num_batches) / 1000.0 if num_batches > 0 else 0.0
    total_time_sec = total_time / 1000.0
    print(f"  Finished sampling {num_images} images.")
    print(f"  Avg time per batch: {avg_time_per_batch:.4f} seconds")

    return total_time_sec

# --- Main Execution Block ---

if __name__ == "__main__":
    # Determine device
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        # Note: This benchmark is primarily meaningful on a GPU.
        # Running on CPU will be extremely slow and may not show xformers benefit (as it's CUDA-focused).
        DEVICE = torch.device("cpu")
        print("WARNING: CUDA not available. Running on CPU. Benchmarking results may not be representative.")
        # exit() # Optional: uncomment to prevent running on CPU if GPU is required.

    # --- Sanity Checks ---
    # Ensure hidden dimension is divisible by the number of heads for attention calculation
    if HIDDEN_DIM % NUM_HEADS != 0:
        raise ValueError(f"Configuration error: Hidden dimension ({HIDDEN_DIM}) "
                         f"must be divisible by number of heads ({NUM_HEADS}).")
    print(f"\nModel Configuration:")
    print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}, Patch Size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"  Channels: {IN_CHANNELS}, Hidden Dim: {HIDDEN_DIM}, Depth: {DEPTH}, Heads: {NUM_HEADS}")

    # --- Instantiate Models ---
    # Define common arguments for both model instantiations
    common_dit_args = {
        "img_size": IMG_SIZE,
        "patch_size": PATCH_SIZE,
        "in_chans": IN_CHANNELS,
        "hidden_size": HIDDEN_DIM,
        "depth": DEPTH,
        "num_heads": NUM_HEADS,
        "mlp_ratio": MLP_RATIO,
        "qkv_bias": True, # Bias usually enabled in QKV
        "attn_drop": 0.0, # No dropout during inference benchmark
        "proj_drop": 0.0, # No dropout during inference benchmark
    }

    # 1. Baseline Model (Standard PyTorch Attention)
    print("\nInstantiating Baseline DiT Model (Standard Attention)...")
    model_baseline = DiT(**common_dit_args, use_xformers=False)
    # Check model parameter count (optional)
    num_params_baseline = sum(p.numel() for p in model_baseline.parameters())
    print(f"Baseline model parameters: {num_params_baseline / 1e6:.2f} M")

    # 2. Optimized Model (xformers Attention, if available)
    model_xformers = None
    if XFORMERS_AVAILABLE and DEVICE.type == 'cuda': # Only benchmark xformers if available AND on CUDA
        print("\nInstantiating Optimized DiT Model (xformers Attention)...")
        model_xformers = DiT(**common_dit_args, use_xformers=True)
        # Optional: Copy weights if needed (e.g., for functional equivalence check)
        # model_xformers.load_state_dict(model_baseline.state_dict())
        num_params_xformers = sum(p.numel() for p in model_xformers.parameters())
        print(f"xformers model parameters: {num_params_xformers / 1e6:.2f} M")
        if num_params_baseline != num_params_xformers:
            print("WARNING: Parameter counts differ between models!")
    elif DEVICE.type != 'cuda' and XFORMERS_AVAILABLE:
        print("\nxformers is available but device is CPU. Skipping xformers benchmark.")
    else:
        print("\nxformers not available. Skipping optimized model benchmark.")


    # --- Instantiate Scheduler ---
    # Use the dummy scheduler for benchmarking
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,  # Max timesteps model was trained for
        beta_schedule="scaled_linear",  # Common schedule
        beta_start=0.00085,  # Common start value
        beta_end=0.012,  # Common end value
        clip_sample=False  # Don't clip sample range in scheduler
    )


    # --- Run Benchmarks ---
    baseline_time = -1.0 # Initialize with invalid time

    print("\n--- Benchmarking Baseline Model (Standard Attention) ---")
    try:
        baseline_time = sample_images(
            model=model_baseline,
            scheduler=scheduler,
            num_images=NUM_IMAGES_TO_SAMPLE,
            num_steps=NUM_SAMPLING_STEPS,
            batch_size=BENCHMARK_BATCH_SIZE,
            device=DEVICE
        )
        print(f"\nBaseline Total Sampling Time: {baseline_time:.4f} seconds")
        avg_baseline_time_per_image = baseline_time / NUM_IMAGES_TO_SAMPLE if NUM_IMAGES_TO_SAMPLE > 0 else 0.0
        print(f"Baseline Average Time per Image: {avg_baseline_time_per_image:.4f} seconds")
    except Exception as e:
        print(f"ERROR benchmarking baseline model: {e}")
        # Optional: re-raise exception if you want the script to stop
        # raise e

    xformers_time = -1.0 # Initialize with invalid time

    if model_xformers:
        print("\n--- Benchmarking Optimized Model (xformers Attention) ---")
        try:
            # Clear GPU cache before next benchmark (optional, might help consistency)
            if DEVICE.type == 'cuda':
                 torch.cuda.empty_cache()
                 print("Cleared CUDA cache before xformers benchmark.")

            xformers_time = sample_images(
                model=model_xformers,
                scheduler=scheduler,
                num_images=NUM_IMAGES_TO_SAMPLE,
                num_steps=NUM_SAMPLING_STEPS,
                batch_size=BENCHMARK_BATCH_SIZE,
                device=DEVICE
            )
            print(f"\nxformers Total Sampling Time: {xformers_time:.4f} seconds")
            avg_xformers_time_per_image = xformers_time / NUM_IMAGES_TO_SAMPLE if NUM_IMAGES_TO_SAMPLE > 0 else 0.0
            print(f"xformers Average Time per Image: {avg_xformers_time_per_image:.4f} seconds")
        except Exception as e:
             print(f"ERROR benchmarking xformers model: {e}")
             # Optional: re-raise exception
             # raise e

    # --- Calculate and Report Speedup ---
    print("\n--- Benchmark Results Summary ---")
    valid_baseline = baseline_time > 0
    valid_xformers = xformers_time > 0

    if valid_baseline:
        print(f"Baseline Time: {baseline_time:.4f} s ({avg_baseline_time_per_image:.4f} s/image)")
    else:
        print("Baseline benchmark did not run or failed.")

    if model_xformers: # Only report xformers if it was supposed to run
        if valid_xformers:
            print(f"xformers Time: {xformers_time:.4f} s ({avg_xformers_time_per_image:.4f} s/image)")
        else:
            print("xformers benchmark did not run or failed.")

        # Calculate speedup only if both benchmarks ran successfully
        if valid_baseline and valid_xformers:
            speedup = baseline_time / xformers_time
            percentage_improvement = (1 - (xformers_time / baseline_time)) * 100
            print(f"\nSpeedup (Baseline / xformers): {speedup:.2f}x")
            print(f"Time Reduction with xformers: {percentage_improvement:.2f}%")
            if speedup > 1.05: # Add a small threshold for meaningful speedup
                print("Result: xformers provides a significant speedup.")
            elif speedup > 1.0:
                print("Result: xformers provides a minor speedup.")
            else:
                print("Result: xformers does not provide a speedup (or is slower). "
                      "Check setup, parameters, or hardware compatibility.")
        elif valid_baseline:
            print("\nCould not calculate speedup (xformers benchmark failed or didn't run).")
        else:
            print("\nCould not calculate speedup (baseline benchmark failed or didn't run).")
    else:
        print("\nxformers benchmark was not run (xformers unavailable or running on CPU).")

    print("\nBenchmark complete.")
