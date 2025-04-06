# config.py
"""
Stores configuration parameters for the DiT model and benchmarking.
"""

# --- Model Parameters ---
# Adjust based on your hardware (e.g., GPU VRAM) and desired model scale.
IMG_SIZE = 64          # Input image spatial dimension (e.g., 64x64 pixels)
PATCH_SIZE = 4         # Size of square patches the image is divided into (e.g., 4x4 pixels)
IN_CHANNELS = 4        # Number of channels in the input latent space (e.g., from VAE)
HIDDEN_DIM = 768       # Dimensionality of the transformer's hidden layers
DEPTH = 12             # Number of transformer blocks (layers) in the model
NUM_HEADS = 12         # Number of attention heads in the multi-head self-attention layers
MLP_RATIO = 4.0        # Expansion ratio for the hidden dimension in the FeedForward network

# --- Benchmarking Parameters ---
NUM_IMAGES_TO_SAMPLE = 50  # Total number of 'images' (dummy latents) to generate
NUM_SAMPLING_STEPS = 50    # Number of steps in the simulated diffusion sampling loop
BENCHMARK_BATCH_SIZE = 50  # Number of images to process in parallel during benchmarking