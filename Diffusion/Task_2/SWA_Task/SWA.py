"""
DiT Unconditional Training and Evaluation Script
"""
import argparse
import os
import sys
import subprocess
import shutil
import glob
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
import numpy as np
import math
import time
import torch.optim.lr_scheduler # Import LR scheduler

# --- Configuration ---
WORKSPACE_DIR = os.path.abspath("dit_workspace")
REPO_URL = "https://github.com/facebookresearch/DiT.git"
REPO_DIR_NAME = "DiT" # Name of the directory created by git clone
REPO_PATH = os.path.join(WORKSPACE_DIR, REPO_DIR_NAME)

# --- Dataset Configuration ---
DATASET_NAME = "landscape" # <<< --- FIND A VALID KAGGLE DATASET IDENTIFIER HERE if using download
# Point this to your ORIGINAL, UNPROCESSED images (Needed for FID baseline)
LOCAL_DATASET_PATH = "/home/fractal/PycharmProjects/PythonProject/landscape_store"

#Point this to your PRE-PROCESSED (Resized/Augmented) images
PREPARED_DATASET_PATH = "/home/fractal/PycharmProjects/PythonProject/landscape"
# Example: PREPARED_DATASET_PATH = "/home/fractal/PycharmProjects/PythonProject/landscape_augmented_256"
STRUCTURED_DATA_DIR_NAME = "landscape_data_structured"
STRUCTURED_DATA_PATH = os.path.join(WORKSPACE_DIR, STRUCTURED_DATA_DIR_NAME) # Base dir for train/val folders
TRAIN_SPLIT_RATIO = 0.9 # 90% for training, 10% for validation
RESULTS_DIR_NAME = "results_unconditional"
RESULTS_PATH = os.path.join(WORKSPACE_DIR, RESULTS_DIR_NAME)
FID_SAMPLES_BASE_DIR = os.path.join(WORKSPACE_DIR, "samples_fid")
FID_SAMPLES_FULL_BEST_DIR = os.path.join(FID_SAMPLES_BASE_DIR, "full_attn_best")
FID_SAMPLES_SWA_BEST_DIR = os.path.join(FID_SAMPLES_BASE_DIR, "windowed_attn_w{}_best") # Use template
REAL_IMAGES_RESIZED_DIR = os.path.join(WORKSPACE_DIR, "real_images_fid_resized")
DUMMY_CLASS_NAME = "landscapes" # For ImageFolder structure

# Training parameters
MODEL_VARIANT = "DiT-S/4" #
IMAGE_SIZE = 256
GLOBAL_BATCH_SIZE = 64
NUM_WORKERS = 0
EPOCHS = 120 #
LOG_EVERY = 50
CKPT_EVERY = 500 # Keep interval checkpoints if desired, but best model is now val-based
VAL_EVERY = 1 # Validate every epoch
EARLY_STOPPING_PATIENCE = 50 # Stop if validation loss doesn't improve for 20 epochs (0 to disable)
SCHEDULER_DURATION_EPOCHS = 0 # Set > EPOCHS for slower decay, 0 uses EPOCHS

# ADD THESE HYPERPARAMETER CONSTANTS
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0

# Sampling parameters
NUM_GRID_SAMPLES = 16
NUM_FID_SAMPLES = 4000
PER_PROC_BATCH_SIZE_SAMPLE = 16

WARMUP_STEPS = 1000  # Example: Warmup for 1000 optimizer steps
GRAD_ACCUMULATION_STEPS = 4 # Example: Accumulate gradients over 4 micro-batches

# Paths for generated files
SAMPLE_GRID_FULL_BEST_PATH = os.path.join(WORKSPACE_DIR, "samples_unconditional_full_attn_BEST.png")
SAMPLE_GRID_SWA_BEST_PATH = os.path.join(WORKSPACE_DIR, "samples_unconditional_window{}_attn_BEST.png") # Use template

# Global variable to store downloaded dataset path
original_dataset_path = None

# --- Helper Functions ---

def print_header(text):
    """Prints a formatted header."""
    print("\n" + "=" * 80)
    print(f"=== {text.upper()} ")
    print("=" * 80)

def print_error(text):
    """Prints an error message."""
    print(f"\n!!! ERROR: {text}", file=sys.stderr)

def print_warning(text):
    """Prints a warning message."""
    print(f"\n!!! WARNING: {text}")

def print_info(text):
    """Prints an info message."""
    print(f"\n>>> {text}")

def run_command(command_list, cwd=None, description=""):
    """Runs a shell command using subprocess, captures output."""
    if description:
        print_info(f"Running: {description}...")
    print(f"Executing: {' '.join(command_list)}")
    try:
        # Use shell=True cautiously, but needed for complex commands like accelerate/python -m
        # Prefer list format when shell=False
        process = subprocess.run(command_list, capture_output=True, text=True, check=True, cwd=cwd)
        if process.stdout:
            print("--- Command Output ---")
            print(process.stdout)
            print("----------------------")
        if process.stderr:
            print("--- Command Error Stream ---")
            print(process.stderr, file=sys.stderr)
            print("--------------------------")
        print_info(f"{description or 'Command'} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"{description or 'Command'} failed with return code {e.returncode}")
        if e.stdout:
            print("--- Captured STDOUT ---")
            print(e.stdout)
            print("-----------------------")
        if e.stderr:
            print("--- Captured STDERR ---")
            print(e.stderr, file=sys.stderr) # Print stderr to stderr
            print("-----------------------")
        return False
    except Exception as e:
        print_error(f"An unexpected error occurred while running command: {e}")
        return False

def create_or_overwrite_file(filepath, content):
    """Writes content to a file, creating directories if needed."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(content)
        print_info(f"Successfully wrote/overwrote file: {filepath}")
        return True
    except Exception as e:
        print_error(f"Error writing file {filepath}: {e}")
        return False

def check_gpu():
    """Checks GPU availability using nvidia-smi."""
    print_header("Checking GPU")
    if not run_command(["nvidia-smi"], description="NVIDIA-SMI GPU Check"):
       print_warning("nvidia-smi command failed or no GPU detected. Training/Sampling will be very slow or fail.")

def setup_paths():
    """Validates required local dataset paths."""
    global original_dataset_path
    print_header("Validating Dataset Paths")

    valid_original = False
    valid_prepared = False

    # --- Validate Path to ORIGINAL data (for FID) ---
    if not LOCAL_DATASET_PATH:
        print_error("LOCAL_DATASET_PATH is not set. Path to original images is needed for FID.")
    else:
        print_info(f"Checking for ORIGINAL dataset path: {LOCAL_DATASET_PATH}")
        path_resolved = os.path.abspath(os.path.expanduser(LOCAL_DATASET_PATH))
        if os.path.isdir(path_resolved):
            try:
                if not os.listdir(path_resolved):
                     print_warning(f"Original dataset directory exists but is empty: {path_resolved}")
                     # Allow continuing, but FID will fail later if needed
                else:
                    print_info(f"Found ORIGINAL dataset directory: {path_resolved}")
                    original_dataset_path = path_resolved # Set global for resize_real_images_for_fid
                    valid_original = True
            except OSError as e:
                print_error(f"Error accessing ORIGINAL dataset directory {path_resolved}: {e}")
        else:
            print_error(f"Provided LOCAL_DATASET_PATH is not a valid directory: {path_resolved}")

    # --- Validate Path to PREPARED data (for Training) ---
    if not PREPARED_DATASET_PATH:
        print_error("PREPARED_DATASET_PATH is not set. Path to pre-processed images is needed for training.")
    else:
        print_info(f"Checking for PREPARED dataset path: {PREPARED_DATASET_PATH}")
        path_resolved = os.path.abspath(os.path.expanduser(PREPARED_DATASET_PATH))
        if os.path.isdir(path_resolved):
            try:
                if not os.listdir(path_resolved):
                     print_error(f"PREPARED dataset directory exists but is empty: {path_resolved}. Cannot train.")
                else:
                    print_info(f"Found PREPARED dataset directory: {path_resolved}")
                    valid_prepared = True
            except OSError as e:
                print_error(f"Error accessing PREPARED dataset directory {path_resolved}: {e}")
        else:
            print_error(f"Provided PREPARED_DATASET_PATH is not a valid directory: {path_resolved}")

    if not valid_prepared:
        print_error("Cannot proceed without a valid PREPARED_DATASET_PATH for training data.")
        return False

    if not valid_original:
        print_warning("Original dataset path not found or invalid. FID calculation step will fail.")
        # Allow script to continue for training, but warn about FID

    return True # Proceed if prepared data is valid

def install_packages():
    """Installs required pip packages."""
    print_header("Installing Python Packages")
    packages = [
        "diffusers", "timm", "accelerate", "einops",
        "torch", "torchvision", "pytorch-fid", "scipy",
        "ftfy", "regex", "pillow", "tqdm" # Added pillow/tqdm explicitly
    ]
    # Separate torch install command often needed for correct CUDA version
    # User should ideally run this manually matching their CUDA version
    print_warning("Skipping direct pip install from script.")
    print_info("Please ensure the following packages are installed (ideally in a virtual environment):")
    print(f"  pip install {' '.join(packages)}")
    print("  Ensure PyTorch is installed with the correct CUDA support for your system, e.g.:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    # Example check (optional, can be slow)
    # for pkg in packages:
    #     if not run_command([sys.executable, '-m', 'pip', 'show', pkg], description=f"Checking {pkg}"):
    #          if not run_command([sys.executable, '-m', 'pip', 'install', pkg, '--upgrade', '--quiet'], description=f"Installing {pkg}"):
    #              return False
    return True # Assume user handles installation

def clone_repo():
    """Clones the DiT repository."""
    print_header("Cloning DiT Repository")
    if os.path.isdir(REPO_PATH):
        print_info(f"Repository already exists at {REPO_PATH}. Skipping clone.")
        return True
    else:
        os.makedirs(WORKSPACE_DIR, exist_ok=True)
        # Clone into the workspace directory
        if run_command(["git", "clone", REPO_URL, REPO_PATH], cwd=WORKSPACE_DIR, description="Cloning DiT repository"):
            # List contents after cloning
            print_info(f"Listing contents of {REPO_PATH}:")
            run_command(["ls", "-l", REPO_PATH], description="Listing repo contents")
            return True
        else:
            return False

def define_models_py():
    """Writes the modified models.py content."""
    print_header("Defining models.py (with SWA Fix)")
    models_py_path = os.path.join(REPO_PATH, "models.py")
    # Content from Cell 6 (ensure it's correctly copied)
    models_py_content = """
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp # Attention removed from here
from torch.nn import functional as F # Added for padding

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                   Attention Layer (Masked Window Attention)                   #
#################################################################################

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., window_size=None):
        super().__init__()
        assert dim % num_heads == 0, f'dim ({dim}) should be divisible by num_heads ({num_heads})' # Corrected f-string
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.window_size = window_size
        self.attn_mask = None # Buffer for mask if needed

        if window_size is not None:
            if not isinstance(window_size, int) or window_size <= 0 or window_size % 2 == 0:
                 raise ValueError(f"window_size must be a positive odd integer, but got {window_size}") # Corrected f-string
            print(f"Attention: Initialized with Masked Window Attention, window_size={window_size}") # Corrected f-string
            print("           (Computes full QK^T then masks, No dilation, No RPB)")
        else:
            print("Attention: Initialized with Full Attention.")


    def _create_mask(self, N, device):
        if self.window_size is None: return None # No mask for full attention

        # Check if mask for this N is already computed and on the right device
        if self.attn_mask is not None and self.attn_mask.shape[-1] == N and self.attn_mask.device == device:
            return self.attn_mask

        # Create the attention mask
        # mask[i, j] is True if |i - j| <= window_size // 2
        half_window = self.window_size // 2
        indices = torch.arange(N, device=device)
        # (N, 1) - (1, N) -> (N, N) matrix where mat[i, j] = i - j
        relative_indices = indices.unsqueeze(1) - indices.unsqueeze(0)
        # Boolean mask: True where attention IS allowed
        # Shape: (N, N)
        mask = torch.abs(relative_indices) <= half_window

        # Expand mask to match attention matrix shape (B, H, N, N) for broadcasting
        # Add batch and head dimensions: (1, 1, N, N)
        mask = mask.unsqueeze(0).unsqueeze(0)

        # Store as simple cache (not buffer due to dynamic N)
        self.attn_mask = mask
        return mask

    def forward(self, x):
        B, N, C = x.shape
        H = self.num_heads
        D_head = C // H

        # Project to Q, K, V: (B, N, C) -> (B, N, 3*C)
        qkv = self.qkv(x)
        # Reshape and permute for multi-head attention
        # (B, N, 3*C) -> (B, N, 3, H, D_head) -> (3, B, H, N, D_head)
        qkv = qkv.view(B, N, 3, H, D_head).permute(2, 0, 3, 1, 4)
        # q, k, v: each (B, H, N, D_head)
        q, k, v = qkv.unbind(0)

        # Calculate scaled dot-product attention scores
        # (B, H, N, D) @ (B, H, D, N) -> (B, H, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # --- Apply Mask (if windowed attention) ---
        if self.window_size is not None:
            mask = self._create_mask(N, x.device) # Shape (1, 1, N, N)
            # Apply mask: set scores outside window to -inf before softmax.
            attn = attn.masked_fill(~mask.bool(), float('-inf'))

        # --- Softmax and Dropout ---
        attn = attn.softmax(dim=-1)
        attn = torch.nan_to_num(attn) # Replace NaN with 0
        attn = self.attn_drop(attn)

        # --- Compute weighted sum of values ---
        # (B, H, N, N) @ (B, H, N, D_head) -> (B, H, N, D_head)
        x = attn @ v

        # --- Transpose and reshape back ---
        # (B, H, N, D_head) -> (B, N, H, D_head) -> (B, N, C)
        x = x.transpose(1, 2).reshape(B, N, C)

        # --- Final Projection and Dropout ---
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, window_size=None, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Pass window_size to the modified Attention class
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, window_size=window_size, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # Apply modulation before norm1 and norm2, then self-attention/MLP, then add residual
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1, # Will be ignored if num_classes=0
        num_classes=1000,       # Set to 0 for unconditional
        learn_sigma=True,
        window_size=None        # Added window_size parameter
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes # Store num_classes

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Conditional models need a LabelEmbedder
        if self.num_classes > 0:
             self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        else:
             self.y_embedder = None # Unconditional case

        num_patches = self.x_embedder.num_patches
        # Positional embedding: learnable parameter initialized with sincos
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # DiT blocks, potentially with windowed attention
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, window_size=window_size) for _ in range(depth)
        ])
        # Final layer for output projection
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize weights similar to MAE
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize positional embedding with sincos
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (MAEv2 style)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedder if it exists
        if self.num_classes > 0 and self.y_embedder:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedder MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, y=None):
        x = self.x_embedder(x) + self.pos_embed  # (N, L, D)
        t = self.t_embedder(t)                   # (N, D)

        # Combine timestep embedding with class embedding if applicable
        if self.num_classes > 0 and self.y_embedder is not None:
            if y is None: # Handle case where y is needed but not provided (e.g., unconditional generation with conditional model)
                # This usually happens during inference/sampling with CFG where y is dropped.
                # Create a placeholder tensor of the 'unconditional' class index.
                # Assumes the unconditional index is self.num_classes
                y = torch.full((x.shape[0],), self.num_classes, device=x.device, dtype=torch.long)

            y = self.y_embedder(y, self.training)    # (N, D)
            c = t + y                             # (N, D) condition embedding
        else:
            # Unconditional model or y is intentionally None (should not happen if num_classes > 0 during training)
            c = t                                 # (N, D) use only timestep

        # Apply DiT blocks
        for block in self.blocks:
            x = block(x, c)                      # (N, L, D)

        # Apply final layer and unpatchify
        x = self.final_layer(x, c)                # (N, L, patch_size**2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x.shape[0] // 2
        x = torch.cat([x, x], dim=0)
        t = torch.cat([t, t], dim=0)
        y = torch.cat([y, y], dim=0) # y needs to contain actual labels and unconditional placeholders

        # Run model once in parallel for conditional and unconditional branches
        model_out = self.forward(x, t, y)

        # Split outputs and perform CFG combination
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, half, dim=0)
        half_rest = rest[:half]
        guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        model_out = torch.cat([guided_eps, half_rest], dim=1)
        return model_out


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# Positional embedding utils from MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here h corresponds to the first dim H, w to the second dim W
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################
# Define DiT model architecture variants

def DiT_XL_2(**kwargs): return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
def DiT_XL_4(**kwargs): return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)
def DiT_XL_8(**kwargs): return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs): return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)
def DiT_L_4(**kwargs): return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)
def DiT_L_8(**kwargs): return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs): return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)
def DiT_B_4(**kwargs): return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)
def DiT_B_8(**kwargs): return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs): return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)
def DiT_S_4(**kwargs): return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)
def DiT_S_8(**kwargs): return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


# Create a dictionary mapping model names to constructors
DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}

"""
    if create_or_overwrite_file(models_py_path, models_py_content):
        # Optional verification snippet check
        try:
            with open(models_py_path, "r") as f:
                content = f.read()
                if "# --- FIX: Slice unfolded windows" in content and "window_size=None" in content:
                    print_info("models.py verification snippet suggests SWA fix is present.")
                else:
                    print_warning("Could not verify SWA fix marker in models.py via simple string search.")
        except Exception as e:
            print_warning(f"Error reading back models.py for verification: {e}")
        return True
    else:
        return False

import torch.optim.lr_scheduler

def define_train_py():
    """Writes the modified train.py content with Warmup, Grad Accum, Validation, etc."""
    print_header("Defining train.py (Warmup, Grad Accum, Validation, Early Stop, Batch Save, Grad Clip)") # Updated header
    train_py_path = os.path.join(REPO_PATH, "train.py")

    # --- Start train_py_content ---
    train_py_content = f"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
import time
import argparse
import logging
import os
import math
import sys
import traceback
# +++ Import necessary schedulers +++
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
# --- End Import ---
from tqdm import tqdm

# Ensure models imports from the correct location relative to train.py
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

# Debug print function
def print_debug(rank, msg):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"[DEBUG {{timestamp}} Rank-{{rank}}] {{msg}}", flush=True)

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    # ... (EMA function remains the same) ...
    is_ddp = isinstance(model, DDP)
    model_params = OrderedDict(model.module.named_parameters() if is_ddp else model.named_parameters())
    ema_params = OrderedDict(ema_model.named_parameters())
    model_keys = set(model_params.keys())
    ema_keys = set(ema_params.keys())
    if model_keys != ema_keys:
         print(f"Warning: EMA key mismatch! Model keys: {{len(model_keys)}}, EMA keys: {{len(ema_keys)}}")
    for name, param in model_params.items():
        if name in ema_params:
             ema_params[name].mul_(decay).add_(param.data.detach(), alpha=1 - decay)


def requires_grad(model, flag=True):
    # ... (remains the same) ...
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    # ... (remains the same) ...
    if dist.is_available() and dist.is_initialized():
        print("[Cleanup] Debug: Calling dist.destroy_process_group()")
        dist.destroy_process_group()
        print("[Cleanup] Debug: dist.destroy_process_group() finished.")

def create_logger(logging_dir):
    # ... (remains the same) ...
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    is_rank_zero = (rank == 0)
    logger = logging.getLogger(f"{{__name__}}_rank{{rank}}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(f'[\033[34m%(asctime)s\033[0m Rank-{{rank}}] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if is_rank_zero and logging_dir:
        try:
            os.makedirs(logging_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(logging_dir, "log.txt"))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            print(f"Rank 0 logger setup with file logging in: {{logging_dir}}")
        except Exception as e:
            print(f"Error setting up file logger for Rank 0: {{e}}")
    logger.propagate = False
    print_debug(rank, "Logger instance created.")
    return logger

def center_crop_arr(pil_image, image_size):
    # ... (remains the same) ...
    try:
        pil_image = pil_image.convert('RGB')
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX)
        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC)
        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        cropped_arr = arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]
        if cropped_arr.shape[0] != image_size or cropped_arr.shape[1] != image_size:
             pil_image = pil_image.resize((image_size, image_size), resample=Image.Resampling.LANCZOS)
             return pil_image
        return Image.fromarray(cropped_arr)
    except Exception as e:
        print(f"Error during center_crop_arr: {{e}}. Returning black image.")
        return Image.new('RGB', (image_size, image_size), color = 'black')

# +++ VALIDATION FUNCTION (remains the same) +++
@torch.no_grad()
def validate_one_epoch(model, vae, diffusion, loader, device, rank, epoch, logger):
    # ... (validation logic is unchanged) ...
    model.eval(); vae.eval()
    total_val_loss = 0.0; val_steps = 0
    is_ddp = dist.is_available() and dist.is_initialized()
    pbar = tqdm(loader, desc=f"Epoch {{epoch}} Validation", disable=(rank != 0))
    for i, batch_data in enumerate(pbar):
        if not batch_data: continue
        try: x, _ = batch_data; x = x.to(device, non_blocking=True)
        except Exception as e: logger.warning(f"Skipping val batch {{i}}: {{e}}"); continue
        try:
            latent_x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (latent_x.shape[0],), device=device)
            model_kwargs = dict()
            loss_dict = diffusion.training_losses(model, latent_x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            if not torch.isfinite(loss): logger.warning(f"Non-finite val loss: {{loss.item()}}"); continue
            if is_ddp:
                 loss_tensor = loss.detach().clone(); dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG); reduced_loss_item = loss_tensor.item()
            else: reduced_loss_item = loss.item()
            total_val_loss += reduced_loss_item; val_steps += 1
            if rank == 0: avg_loss = total_val_loss / val_steps if val_steps > 0 else 0; pbar.set_postfix({{"AvgValLoss": f"{{avg_loss:.4f}}", "Steps": val_steps}})
        except Exception as e: logger.error(f"Val step {{i}} rank {{rank}} error: {{e}}\\n{{traceback.format_exc()}}")
    model.train()
    if val_steps == 0: logger.warning(f"Epoch {{epoch}} Validation: No steps completed."); return float('inf')
    avg_val_loss = total_val_loss / val_steps; return avg_val_loss

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    # ... (DDP Setup - world_size is defined here) ...
    rank_for_debug = int(os.environ.get('RANK', 0))
    print_debug(rank_for_debug, "Entering main function.")
    assert torch.cuda.is_available(), "Training requires GPU."
    is_ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    print_debug(rank_for_debug, f"DDP check: is_ddp = {{is_ddp}}")
    if is_ddp:
        rank = int(os.environ["RANK"]); world_size = int(os.environ['WORLD_SIZE']); local_rank = int(os.environ['LOCAL_RANK']); device = local_rank
        print_debug(rank_for_debug, f"Rank {{rank}} (local {{local_rank}}), World {{world_size}}. Initializing DDP...")
        try: dist.init_process_group(backend='nccl', init_method='env://'); torch.cuda.set_device(device); dist.barrier()
        except Exception as e: print_debug(rank_for_debug, f"!!! DDP Init failed: {{e}}"); raise
    else:
        rank = 0; world_size = 1; local_rank = 0; device = 0 # <<< world_size defined here for single process
        print_debug(rank_for_debug, f"Single process mode. Device: {{device}}")
    torch.cuda.set_device(device)


    # --- Batch size configuration ---
    # Moved the divisibility check here, after world_size is known
    if args.global_batch_size % (world_size * args.accumulation_steps) != 0:
         # Use logger if available, otherwise print
         msg = f"Global batch size ({{args.global_batch_size}}) must be divisible by world_size*accumulation_steps ({{world_size}}*{{args.accumulation_steps}}). Exiting."
         try: logger.error(msg);
         except NameError: print(f"!!! ERROR: {{msg}}")
         if is_ddp: cleanup() # Attempt cleanup if DDP was initialized
         sys.exit(1) # Exit script

    assert args.global_batch_size % world_size == 0, f"Global batch size {{args.global_batch_size}} must be divisible by world size {{world_size}}."
    per_proc_batch_size = int(args.global_batch_size // world_size)

    # Calculate micro_batch_size based on per_proc_batch_size
    # This was previously done in the logging print, do it explicitly here
    micro_batch_size = per_proc_batch_size // args.accumulation_steps
    effective_batch_size = args.global_batch_size # Effective size per optim step is the global size
    print_debug(rank_for_debug, f"Global BS: {{args.global_batch_size}}, Accum steps: {{args.accumulation_steps}}, WorldSize: {{world_size}}, PerProc BS: {{per_proc_batch_size}}, Micro-BS per proc: {{micro_batch_size}}")
    # --- Seeding ---
    seed = args.global_seed + rank; torch.manual_seed(seed); np.random.seed(seed)
    # --- Experiment Dirs & Logger ---
    experiment_dir = None; checkpoint_dir = None
    if rank == 0:
         # ... (Experiment dir creation logic remains the same) ...
         os.makedirs(args.results_dir, exist_ok=True)
         try: experiment_index = max([int(d.split('-')[0]) for d in os.listdir(args.results_dir) if d.split('-')[0].isdigit()] + [-1]) + 1
         except Exception: experiment_index = 0
         model_string_name = args.model.replace("/", "-"); attn_str = f"attn-{{args.attention_type}}" + (f"-w{{args.window_size}}" if args.attention_type == 'swa' else ""); uncond_str = "uncond"
         experiment_dir = os.path.join(args.results_dir, f"{{experiment_index:03d}}-{{model_string_name}}-{{attn_str}}-{{uncond_str}}")
         checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
         os.makedirs(experiment_dir, exist_ok=True); os.makedirs(checkpoint_dir, exist_ok=True)
    if is_ddp: dist.barrier()
    logger = create_logger(experiment_dir if rank == 0 else None)
    if rank == 0: # Log config
        logger.info("-------------------- Configuration --------------------")
        for arg_name, value in vars(args).items(): logger.info(f"{{arg_name}}: {{value}}")
        logger.info(f"World Size: {{world_size}}"); logger.info(f"Global Batch Size: {{args.global_batch_size}}"); logger.info(f"Accumulation Steps: {{args.accumulation_steps}}"); logger.info(f"Micro-batch Size Per Proc: {{micro_batch_size}}")
        logger.info(f"Experiment Directory: {{experiment_dir}}"); logger.info("------------------------------------------------------")

    # --- Best Validation Loss Tracking ---
    best_val_loss = float('inf'); best_val_epoch = 0; epochs_no_improve = 0; early_stop_triggered_local = False
    if rank == 0: logger.info(f"Initializing best validation loss tracker. Patience: {{args.patience}}")

    # --- Model Setup ---
    # ... (Model config, DiT instantiation, EMA creation remain the same) ...
    assert args.image_size % 8 == 0; latent_size = args.image_size // 8; window_size = args.window_size if args.attention_type == 'swa' else None
    model_config = {{'input_size': latent_size, 'num_classes': 0, 'window_size': window_size}}
    try: model = DiT_models[args.model](**model_config).to(device)
    except Exception as e: logger.error(f"Model Instantiation Error: {{e}}"); cleanup(); sys.exit(1)
    ema = deepcopy(model).to(device); requires_grad(ema, False)
    if is_ddp: # DDP Wrapping
        try: model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=args.find_unused_parameters)
        except Exception as e: logger.error(f"DDP Error: {{e}}"); cleanup(); sys.exit(1)

    # --- Diffusion, VAE, Optimizer ---
    diffusion = create_diffusion(timestep_respacing="")
    vae_model_name = f"stabilityai/sd-vae-ft-{{args.vae}}"
    try: vae = AutoencoderKL.from_pretrained(vae_model_name).to(device); requires_grad(vae, False); vae.eval()
    except Exception as e: logger.error(f"VAE Load Error: {{e}}"); cleanup(); sys.exit(1)
    if rank == 0: # Log param count
        num_params = sum(p.numel() for p in model.parameters()); num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total DiT Params: {{num_params:,}}, Trainable: {{num_params_trainable:,}}")
    # +++ Optimizer Definition (ADJUST FOR BETAS HERE if desired) +++
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999)) # Default betas
    # Example change: betas=(0.9, 0.98)
    # --- End Optimizer ---
    print_debug(rank_for_debug, "Optimizer created.")

    # --- Data Loading ---
    # +++ Data Augmentation Control (DISABLE/MODIFY HERE) +++
    train_transform_list = [
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ]
    # Example: Add ColorJitter
    # train_transform_list.insert(1, transforms.ColorJitter(brightness=0.1, contrast=0.1))
    train_transform = transforms.Compose(train_transform_list)
    # --- End Augmentation ---
    val_transform = transforms.Compose([ # Validation transform usually simpler
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    # ... (Dataset loading, path checks remain the same) ...
    train_data_path = args.data_path; train_dir = os.path.join(train_data_path, 'train'); val_dir = os.path.join(train_data_path, 'val')
    if not os.path.isdir(train_dir): logger.error(f"Train dir not found: '{{train_dir}}'"); cleanup(); sys.exit(1)
    has_val_data = os.path.isdir(val_dir)
    if not has_val_data: logger.warning(f"Val dir not found: '{{val_dir}}'. Val/EarlyStop disabled."); args.patience = 0
    try:
        train_dataset = ImageFolder(train_dir, transform=train_transform)
        val_dataset = ImageFolder(val_dir, transform=val_transform) if has_val_data else None
        train_len = len(train_dataset); val_len = len(val_dataset) if val_dataset else 0
        print_debug(rank_for_debug, f"Datasets loaded. Train: {{train_len}}, Val: {{val_len}}.")
        if train_len == 0: logger.error(f"Training dataset empty."); cleanup(); sys.exit(1)
        # Calculate steps based on *optimizer steps* per epoch
        steps_per_epoch_optim = math.ceil(train_len / args.global_batch_size) # Number of micro-batches / accum_steps
        effective_epochs_for_scheduler = args.scheduler_tmax_epochs if args.scheduler_tmax_epochs > 0 else args.epochs
        total_optim_steps_for_scheduler = steps_per_epoch_optim * effective_epochs_for_scheduler
        # Calculate interval checkpoint frequency based on optimizer steps
        # Calculate total *planned* optimizer steps
        total_planned_optim_steps = steps_per_epoch_optim * args.epochs
        ckpt_every = args.ckpt_every if args.ckpt_every > 0 else max(1, total_planned_optim_steps // 10) # ~10 checkpoints if 0
        print_debug(rank_for_debug, f"Micro-batches/epoch: {{math.ceil(train_len / per_proc_batch_size)}}. Optim Steps/epoch: {{steps_per_epoch_optim}}. Total Planned Optim Steps: {{total_planned_optim_steps}}. Scheduler T_max Optim Steps: {{total_optim_steps_for_scheduler}}. Interval Ckpt Freq: {{ckpt_every}} optim steps.")
    except Exception as e: logger.error(f"Dataset Load Error: {{e}}"); cleanup(); sys.exit(1)

    # --- LR Scheduler with Warmup ---
    try:
        # Main decay scheduler (Cosine) - T_max is total decay steps *after* warmup
        decay_steps = total_optim_steps_for_scheduler - args.warmup_steps
        if decay_steps <= 0:
             logger.warning(f"Warmup steps ({{args.warmup_steps}}) >= total scheduler steps ({{total_optim_steps_for_scheduler}}). Disabling decay.")
             # Option 1: Just use warmup (constant LR after)
             # main_scheduler = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=1) # Placeholder
             # Option 2: Use cosine over 1 step (effectively constant) - Safer?
             main_scheduler = CosineAnnealingLR(opt, T_max=1, eta_min=args.lr if args.warmup_steps > 0 else 1e-6) # Keep target LR if warmed up
             decay_steps = 1 # Avoid issues with SequentialLR
        else:
             main_scheduler = CosineAnnealingLR(opt, T_max=decay_steps, eta_min=1e-6)

        if args.warmup_steps > 0:
            # Linear warmup scheduler
            # Calculate start factor (avoiding division by zero if lr=0)
            start_lr = 1e-7 # Start warmup from a very small value
            start_factor = start_lr / args.lr if args.lr > 0 else 0.0
            warmup_scheduler = LinearLR(opt, start_factor=start_factor, end_factor=1.0, total_iters=args.warmup_steps)
            # Chain them together
            scheduler = SequentialLR(opt, schedulers=[warmup_scheduler, main_scheduler], milestones=[args.warmup_steps])
            print_debug(rank_for_debug, f"LR Scheduler: SequentialLR created. Warmup steps={{args.warmup_steps}}, Decay steps={{decay_steps}}.")
        else:
            # No warmup, use main scheduler directly
            scheduler = main_scheduler
            print_debug(rank_for_debug, f"LR Scheduler: CosineAnnealingLR created directly. T_max={{decay_steps}} steps.")

    except Exception as e: logger.error(f"Error creating LR scheduler: {{e}}\\n{{traceback.format_exc()}}"); cleanup(); sys.exit(1)
    # --- End LR Scheduler ---

    # --- Samplers ---
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.global_seed, drop_last=True) if is_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if is_ddp and has_val_data else None

    # --- DataLoaders ---
    try:
        # Use micro_batch_size for the DataLoader
        train_loader = DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = None
        if has_val_data:
             # Validation batch size can be different, maybe larger if memory allows? Or keep same?
             val_batch_size = micro_batch_size # Keep same for simplicity
             val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=False)
        print_debug(rank_for_debug, f"Loaders created. Train micro-BS={{micro_batch_size}}. Val micro-BS={{val_batch_size if has_val_data else 'N/A'}}.")
    except Exception as e: logger.error(f"Error creating DataLoaders: {{e}}\\n{{traceback.format_exc()}}"); cleanup(); sys.exit(1)

    # --- Training Prep ---
    update_ema(ema, model, decay=0) # Initialize EMA
    model.train()
    saved_batch_flag = False
    start_epoch = 0
    global_optim_step = 0 # Tracks optimizer steps completed by rank 0

    # --- Training Loop ---
    logger.info(f"Starting training from epoch {{start_epoch}} for up to {{args.epochs}} epochs...")
    training_should_continue = True
    epoch = 0 # Initialize loop var for final logging
    opt.zero_grad() # Ensure grads are zero at the beginning

    for epoch in range(start_epoch, args.epochs):
        if not training_should_continue: break

        epoch_start_time = time.time()
        model.train() # Set model to train mode at start of epoch
        if train_sampler: train_sampler.set_epoch(epoch)
        if rank == 0: logger.info(f"------ Beginning epoch {{epoch}} ------")

        running_train_loss = 0.0
        steps_in_epoch_logged = 0 # Track steps logged in current period
        micro_batches_processed_in_epoch = 0

        _batch_timer_start = time.time() # Timer for step duration logging

        # Iterate through micro-batches
        for i, batch_data in enumerate(train_loader):
            current_micro_batch_index = i + 1
            is_last_micro_batch_in_accum = (current_micro_batch_index % args.accumulation_steps == 0)

            if not batch_data: logger.warning(f"Skipping empty train micro-batch {{current_micro_batch_index}} in epoch {{epoch}}."); continue
            try: x, _ = batch_data
            except (TypeError, ValueError) as e: logger.error(f"Error unpacking micro-batch {{current_micro_batch_index}}: {{e}}"); continue

            try:
                x = x.to(device, non_blocking=True)
                # Save Batch Sample (Rank 0, First Actual Batch)
                if rank == 0 and not saved_batch_flag:
                    try:
                        save_path = "batch_sample_rank0.pt"; abs_save_path = os.path.abspath(save_path)
                        print_debug(rank, f"Saving batch sample (shape: {{x.shape}}) to {{abs_save_path}}...")
                        torch.save(x.cpu(), save_path); saved_batch_flag = True
                        print_debug(rank, f"Batch sample saved to {{abs_save_path}}.")
                    except Exception as save_e: logger.error(f"Failed to save batch sample: {{save_e}}")
            except Exception as e: logger.error(f"Error moving micro-batch {{current_micro_batch_index}} to device: {{e}}"); continue

            # --- Forward/Loss/Backward per micro-batch ---
            try:
                # If using DDP and gradient accumulation, need to handle model.no_sync()
                # Context manager to disable DDP gradient sync except on the last micro-batch
                sync_context = model.no_sync() if (is_ddp and not is_last_micro_batch_in_accum) else contextlib.nullcontext()

                with sync_context:
                    with torch.no_grad(): latent_x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                    t = torch.randint(0, diffusion.num_timesteps, (latent_x.shape[0],), device=device)
                    model_kwargs = dict()
                    loss_dict = diffusion.training_losses(model, latent_x, t, model_kwargs)
                    loss = loss_dict["loss"].mean()

                    if not torch.isfinite(loss):
                        logger.error(f"Non-finite train loss at micro-batch idx {{i}}, epoch {{epoch}}: {{loss.item()}}. Skipping accum step.");
                        # Need to decide how to handle this. Skip the whole accum step?
                        # For now, just skip backward for this micro-batch.
                        loss = None # Signal to skip backward/accum
                    else:
                         # Scale loss for accumulation BEFORE backward
                         scaled_loss = loss / args.accumulation_steps
                         scaled_loss.backward() # Accumulate gradients

            except Exception as e:
                 logger.error(f"Model forward/loss/backward failed micro-batch idx {{i}}: {{e}}")
                 if "CUDA out of memory" in str(e): logger.error("OOM! Reduce micro-batch size or accumulation steps?"); cleanup(); sys.exit(1)
                 logger.error(traceback.format_exc()); loss = None # Skip accum

            # --- Accumulate Loss for Logging (using non-scaled loss) ---
            if loss is not None: # Only accumulate if forward/backward succeeded
                 loss_item_local = loss.item() # Use the original mean loss for logging average
                 running_train_loss += loss_item_local
                 steps_in_epoch_logged += 1
            # --- End Forward/Loss/Backward ---

            # --- Optimizer Step (Conditional) ---
            if is_last_micro_batch_in_accum:
                if rank == 0: print_debug(rank, f"Performing optimizer step after micro-batch index {{i}}.")
                try:
                    # Gradient Clipping (Applied to accumulated gradients)
                    if args.grad_clip_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                        # Log grad norm here if desired, associated with the global_optim_step about to be completed

                    opt.step()
                    scheduler.step() # Step scheduler after optimizer
                    opt.zero_grad() # Zero gradients *after* stepping

                    # --- Logging and Checkpointing (Rank 0 - Tied to Optimizer Step) ---
                    if rank == 0:
                        global_optim_step += 1 # Increment global optimizer step count

                        # Log Training Progress periodically based on optimizer steps
                        if global_optim_step % args.log_every == 0:
                            current_time = time.time(); step_time = current_time - _batch_timer_start; _batch_timer_start = current_time
                            # Average loss over micro-batches since last log
                            avg_loss_log = running_train_loss / steps_in_epoch_logged if steps_in_epoch_logged > 0 else 0
                            steps_per_sec = args.log_every / step_time if step_time > 0 else 0 # Steps/sec is optim steps / time
                            current_lr = scheduler.get_last_lr()[0] # Get current LR from scheduler
                            logger.info(f"Epoch {{epoch}} OptimStep {{global_optim_step:07d}}: TrainLoss={{avg_loss_log:.4f}} | LR={{current_lr:.2e}} | OptimStepTime={{step_time:.2f}}s ({{steps_per_sec:.2f}} optim steps/sec)")
                            # Reset averaging stats
                            running_train_loss = 0.0; steps_in_epoch_logged = 0

                        # Save Interval Checkpoint based on optimizer steps
                        if args.ckpt_every > 0 and global_optim_step % ckpt_every == 0:
                             if checkpoint_dir:
                                ckpt_path = os.path.join(checkpoint_dir, f"ckpt_{{global_optim_step:07d}}_epoch{{epoch}}.pt")
                                try:
                                     model_state_dict = model.module.state_dict() if is_ddp else model.state_dict()
                                     checkpoint = {{"model": model_state_dict, "ema": ema.state_dict(), "opt": opt.state_dict(), "scheduler": scheduler.state_dict(), "args": args, "epoch": epoch, "train_step": global_optim_step}}
                                     torch.save(checkpoint, ckpt_path)
                                     logger.info(f"Saved interval checkpoint (Optim Step {{global_optim_step}}) to {{ckpt_path}}")
                                except Exception as e: logger.error(f"Error saving interval checkpoint: {{e}}")
                    # --- End Rank 0 Logging/Checkpointing ---

                except Exception as e: logger.error(f"Optimizer step/logging/checkpointing failed: {{e}}\\n{{traceback.format_exc()}}");

            # Update EMA model weights after each micro-batch's backward pass
            update_ema(ema, model, decay=args.ema_decay)
            micro_batches_processed_in_epoch += 1
            # --- End Optimizer Step Conditional ---
        # --- End Micro-Batch Loop ---

        # --- Validation, Best Model Checkpointing, Early Stopping (End of Epoch) ---
        if val_loader is not None and epoch % args.val_every == 0:
            print_debug(rank_for_debug, f"Starting validation for epoch {{epoch}}.")
            avg_val_loss = validate_one_epoch(model, vae, diffusion, val_loader, device, rank, epoch, logger)
            print_debug(rank_for_debug, f"Finished validation for epoch {{epoch}}. Avg Loss: {{avg_val_loss:.4f}}")

            if rank == 0:
                logger.info(f"------ Epoch {{epoch}} Validation Summary ------")
                logger.info(f"Micro-batches processed this epoch: {{micro_batches_processed_in_epoch}}")
                logger.info(f"Average Validation Loss: {{avg_val_loss:.4f}}")
                # Check for improvement and save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss; best_val_epoch = epoch; epochs_no_improve = 0
                    logger.info(f"*** New best validation loss: {{best_val_loss:.4f}} at epoch {{epoch}} ***")
                    if checkpoint_dir: # Save best model based on validation loss
                        best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
                        try:
                            model_state_dict = model.module.state_dict() if is_ddp else model.state_dict()
                            best_checkpoint = {{"model": model_state_dict, "ema": ema.state_dict(), "opt": opt.state_dict(), "scheduler": scheduler.state_dict(), "args": args, "epoch": epoch, "train_step": global_optim_step, "best_val_loss": best_val_loss}}
                            torch.save(best_checkpoint, best_ckpt_path)
                            logger.info(f"Saved best model checkpoint (Val Loss: {{best_val_loss:.4f}}) to {{best_ckpt_path}}")
                        except Exception as e: logger.error(f"Error saving best checkpoint: {{e}}")
                else: # No improvement
                    if best_val_loss != float('inf'): epochs_no_improve += 1 # Only increment if we already had a best loss
                    logger.info(f"Validation loss did not improve for {{epochs_no_improve}} epoch(s). Best was {{best_val_loss:.4f}} at epoch {{best_val_epoch}}.")
                # Early Stopping Check
                if args.patience > 0 and epochs_no_improve >= args.patience:
                    logger.warning(f"EARLY STOPPING triggered at epoch {{epoch}} after {{args.patience}} epochs without validation improvement.")
                    early_stop_triggered_local = True; training_should_continue = False
                logger.info(f"---------------------------------------")

        # --- Synchronize Early Stopping Signal (DDP) ---
        if is_ddp:
            stop_signal_tensor = torch.tensor(1.0 if early_stop_triggered_local else 0.0, device=device)
            dist.broadcast(stop_signal_tensor, src=0)
            if stop_signal_tensor.item() > 0.5: training_should_continue = False

        # --- Log Epoch Duration (Rank 0) ---
        if rank == 0:
            epoch_duration = time.time() - epoch_start_time
            logger.info(f"------ Epoch {{epoch}} Finished. Duration: {{epoch_duration:.2f}} seconds ------")
    # --- End Epoch Loop ---

    # --- Final Summary ---
    model.eval()
    if rank == 0:
        logger.info("="*40); logger.info("Training finished!")
        logger.info(f"Total optimizer steps completed: {{global_optim_step}}")
        logger.info(f"Best validation loss recorded: {{best_val_loss:.4f}} at epoch {{best_val_epoch}}")
        final_epoch_completed = epoch # 'epoch' holds the index of the last completed epoch
        if early_stop_triggered_local: logger.info(f"Training stopped early after completing epoch {{final_epoch_completed}}.")
        else: logger.info(f"Training completed planned {{args.epochs}} epochs (last epoch index: {{final_epoch_completed}}).")
        # Save final model state
        final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pt")
        if checkpoint_dir:
             logger.info(f"Saving final model state..."); model_state_dict = model.module.state_dict() if is_ddp else model.state_dict()
             last_val_loss = avg_val_loss if 'avg_val_loss' in locals() and has_val_data else float('inf')
             final_checkpoint = {{"model": model_state_dict, "ema": ema.state_dict(), "opt": opt.state_dict(), "scheduler": scheduler.state_dict(), "args": args, "epoch": final_epoch_completed, "train_step": global_optim_step, "final_val_loss": last_val_loss}}
             try: torch.save(final_checkpoint, final_checkpoint_path); logger.info(f"Saved final model state to {{final_checkpoint_path}}")
             except Exception as e: logger.error(f"Error saving final checkpoint: {{e}}")
        logger.info("="*40)

    # --- Cleanup ---
    if is_ddp: dist.barrier(); cleanup()
    print_debug(rank_for_debug, "Exiting main function.")

# --- Make sure if __name__ == "__main__": block follows and calls main(args) ---
if __name__ == "__main__":
    print("[DEBUG train.py __main__] Starting argument parsing...")
    parser = argparse.ArgumentParser()
    # +++ ADD/MODIFY ARGUMENTS +++
    parser.add_argument("--data-path", type=str, required=True, help="Path to the base data dir (containing 'train'/'val').")
    parser.add_argument("--results-dir", type=str, default="results_unconditional")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/4")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--attention-type", type=str, choices=['full', 'swa'], default='full')
    parser.add_argument("--window-size", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=500, help="Maximum training epochs.") # Keep higher default
    parser.add_argument("--global-batch-size", type=int, default=64, help="Total effective batch size across all GPUs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--wd", type=float, default=0, help="Weight decay.")
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100, help="Log progress every N *optimizer* steps.")
    parser.add_argument("--ckpt-every", type=int, default=0, help="Save interval checkpoint every N *optimizer* steps (approx). 0 calculates dynamically.")
    parser.add_argument("--find-unused-parameters", action='store_true', default=False)
    parser.add_argument("--val-every", type=int, default=1, help="Run validation every N epochs.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs). 0 disables.")
    parser.add_argument("--scheduler-tmax-epochs", type=int, default=0, help="Epochs for scheduler T_max override.")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Max grad norm for clipping. 0 disables.")
    # +++ NEW ARGUMENTS +++
    parser.add_argument("--warmup-steps", type=int, default=0, help="Number of linear warmup *optimizer* steps. 0 disables warmup.")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="Number of steps to accumulate gradients over.")

    args = parser.parse_args()
    print("[DEBUG train.py __main__] Arguments parsed.")

    # --- Argument Validation ---
    if args.attention_type == 'swa' and (args.window_size <= 0 or args.window_size % 2 == 0): print(f"Error: --window-size must be positive odd for 'swa'."); sys.exit(1)
    if args.image_size % 8 != 0: print("Error: --image-size must be divisible by 8."); sys.exit(1)
    if args.patience < 0: print("Warning: --patience set to 0."); args.patience = 0
    if args.grad_clip_norm < 0: print("Warning: --grad-clip-norm set to 0."); args.grad_clip_norm = 0
    if args.warmup_steps < 0: print("Warning: --warmup-steps set to 0."); args.warmup_steps = 0
    if args.accumulation_steps < 1: print("Warning: --accumulation-steps set to 1."); args.accumulation_steps = 1
    # if args.global_batch_size % (world_size * args.accumulation_steps) != 0: # Check divisibility for micro-batch calc
    #      print(f"Error: Global batch size ({{args.global_batch_size}}) must be divisible by world_size*accumulation_steps ({{world_size}}*{{args.accumulation_steps}}).")
    #      # Or adjust logic to handle remainder batches, but simpler to enforce divisibility
    #      sys.exit(1)

    print("[DEBUG train.py __main__] Argument validation passed.")
    print("[DEBUG train.py __main__] Calling main function...")
    # Need to import contextlib for the model.no_sync() part
    import contextlib # Add this import near the top with others
    main(args)
    print("[DEBUG train.py __main__] main function returned.")

"""
    # --- End train_py_content ---

    # Add contextlib import at the beginning of the string
    train_py_content = "import contextlib\n" + train_py_content
    return create_or_overwrite_file(train_py_path, train_py_content)

def define_sample_py():
    """Writes the modified sample.py content."""
    print_header("Defining sample.py (with weights_only=False)")
    sample_py_path = os.path.join(REPO_PATH, "sample.py")
    # Content from Cell 8
    sample_py_content = """
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

"""
    return create_or_overwrite_file(sample_py_path, sample_py_content)

def define_sample_ddp_py():
    """Writes the modified sample_ddp.py content."""
    print_header("Defining sample_ddp.py (with DDP Init, weights_only, parse_args fixes)")
    sample_ddp_py_path = os.path.join(REPO_PATH, "sample_ddp.py")
    # Content from Cell 9
    sample_ddp_py_content = """
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

"""
    # Need to import `glob` for the NPZ function in sample_ddp.py
    # Add `import glob` near the top of the generated file content
    if "import glob" not in sample_ddp_py_content:
         sample_ddp_py_content = "import glob\n" + sample_ddp_py_content

    return create_or_overwrite_file(sample_ddp_py_path, sample_ddp_py_content)

def restructure_data():
    """Copies and restructures the PRE-PROCESSED data into train/val splits."""
    print_header("Restructuring PRE-PROCESSED Data into Train/Val Splits")

    # Use the path to the already processed data as the source
    source_dir = os.path.abspath(os.path.expanduser(PREPARED_DATASET_PATH))

    if not source_dir or not os.path.isdir(source_dir):
        print_error("PREPARED_DATASET_PATH not found or invalid. Cannot restructure.")
        print_error(f"Checked path: {source_dir}")
        return False

    # Define target directories for train and validation within the structured path
    train_target_dir = os.path.join(STRUCTURED_DATA_PATH, 'train', DUMMY_CLASS_NAME)
    val_target_dir = os.path.join(STRUCTURED_DATA_PATH, 'val', DUMMY_CLASS_NAME)

    print_info(f"Source of pre-processed images: {source_dir}")
    print_info(f"Target Train directory: {train_target_dir}")
    print_info(f"Target Validation directory: {val_target_dir}")
    print_info(f"Train/Validation Split Ratio: {TRAIN_SPLIT_RATIO:.2f} / {1.0 - TRAIN_SPLIT_RATIO:.2f}")

    try:
        # Create directories, removing old ones first if they exist to ensure clean split
        if os.path.exists(STRUCTURED_DATA_PATH):
            print_warning(f"Removing existing structured data directory: {STRUCTURED_DATA_PATH}")
            shutil.rmtree(STRUCTURED_DATA_PATH)

        os.makedirs(train_target_dir, exist_ok=True)
        os.makedirs(val_target_dir, exist_ok=True)

        print(f"Scanning source directory '{os.path.basename(source_dir)}' for valid image files...")
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
        all_image_files = []
        # Scan the source directory (which contains the pre-processed images)
        for item_name in os.listdir(source_dir):
            source_item = os.path.join(source_dir, item_name)
            if os.path.isfile(source_item) and item_name.lower().endswith(valid_extensions):
                all_image_files.append(item_name)

        if not all_image_files:
            print_error("No valid image files found in the PREPARED source directory.")
            return False

        print(f"Found {len(all_image_files)} pre-processed image files.")

        # Shuffle the image list randomly
        import random
        random.shuffle(all_image_files)

        # Split the list
        split_index = int(len(all_image_files) * TRAIN_SPLIT_RATIO)
        train_files = all_image_files[:split_index]
        val_files = all_image_files[split_index:]

        print(f"Splitting into {len(train_files)} training files and {len(val_files)} validation files.")

        # Copy training files
        print("Copying training files...")
        copied_train = 0
        for item_name in tqdm(train_files, desc="Copying Train Images"):
            source_path = os.path.join(source_dir, item_name)
            target_path = os.path.join(train_target_dir, item_name)
            shutil.copy2(source_path, target_path)
            copied_train += 1

        # Copy validation files
        print("Copying validation files...")
        copied_val = 0
        for item_name in tqdm(val_files, desc="Copying Validation Images"):
            source_path = os.path.join(source_dir, item_name)
            target_path = os.path.join(val_target_dir, item_name)
            shutil.copy2(source_path, target_path)
            copied_val += 1

        print(f"\nFinished restructuring.")
        print(f"  Copied to Train: {copied_train} images")
        print(f"  Copied to Validation: {copied_val} images")
        print(f"Dataset ready for training. Base path: {STRUCTURED_DATA_PATH}")
        return True

    except Exception as e:
        print_error(f"Error during data restructure/split: {e}")
        import traceback
        print(traceback.format_exc())
        return False

# --- Function Definition ---

def train_model(attention_type, window_size=8, patience=20, scheduler_epochs=0,
                lr=1e-4, wd=0.0, grad_clip_norm=1.0,
                warmup_steps=0, accumulation_steps=1):  # New params
    """Runs the training command with specified parameters."""
    sched_ep_log = scheduler_epochs if scheduler_epochs > 0 else EPOCHS
    # Update header print
    print_header(
        f"Training Model - Attn: {attention_type.upper()} LR: {lr} WD: {wd} Clip: {grad_clip_norm} Patience: {patience} SchedEp: {sched_ep_log} Warmup: {warmup_steps} Accum: {accumulation_steps}")
    # --- Path Checks ---
    # Check if the base structured data path contains 'train' and 'val' subdirectories
    train_data_check_path = os.path.join(STRUCTURED_DATA_PATH, 'train')
    val_data_check_path = os.path.join(STRUCTURED_DATA_PATH, 'val')

    if not os.path.isdir(train_data_check_path):
         print_error(f"Structured training data path not found: {train_data_check_path}")
         print_error("Ensure data restructuring created the 'train' subfolder inside:")
         print_error(f"  {STRUCTURED_DATA_PATH}")
         return False # Stop if training data is missing

    original_patience = patience # Store original for logging
    if not os.path.isdir(val_data_check_path):
         # Only warn if validation data is missing, allow training to proceed
         print_warning(f"Structured validation data path not found: {val_data_check_path}")
         print_warning("Validation loop will be skipped by train.py.")
         print_warning("Early stopping and best model saving based on validation loss will not function.")
         # Set patience to 0 if val data missing, as early stopping won't work
         if patience > 0:
             patience = 0
             print_warning(f"Setting early stopping patience from {original_patience} to 0.")

    # --- Construct Command ---
    # Use accelerate launch for multi-GPU/mixed-precision support (even if single GPU)
    cmd = [
        "accelerate", "launch", "train.py",
        "--model", MODEL_VARIANT,               # Use configured model variant (Global Constant)
        "--attention-type", attention_type,
        "--data-path", STRUCTURED_DATA_PATH,    # Pass the base path containing train/val (Global Constant)
        "--results-dir", RESULTS_PATH,          # Global Constant
        "--image-size", str(IMAGE_SIZE),        # Global Constant
        "--global-batch-size", str(GLOBAL_BATCH_SIZE), # Global Constant
        "--num-workers", str(NUM_WORKERS),      # Global Constant
        "--epochs", str(EPOCHS),                # Actual training epochs limit (Global Constant)
        "--ckpt-every", str(CKPT_EVERY),        # Interval checkpoint frequency (Global Constant)
        "--log-every", str(LOG_EVERY),          # Logging frequency (Global Constant)
        "--vae", "ema",                         # Use EMA VAE weights (Hardcoded, could be constant)
        "--lr", str(lr),                        # Learning Rate (Argument)
        "--wd", str(wd),                        # Weight Decay (Argument)
        "--val-every", str(VAL_EVERY),          # Validation frequency (Global Constant)
        "--patience", str(patience),            # Early stopping patience (Argument, potentially modified)
        "--scheduler-tmax-epochs", str(scheduler_epochs), # Scheduler duration override (Argument)
        "--grad-clip-norm", str(grad_clip_norm), # Gradient Clipping Norm (Argument)
        "--warmup-steps", str(warmup_steps),
        "--accumulation-steps", str(accumulation_steps)
        # Add other fixed args if needed, using global constants or hardcoded values:
        # "--global-seed", str(GLOBAL_SEED)
        # "--ema-decay", str(EMA_DECAY)
        # "--find-unused-parameters"
    ]

    # Add window size only if using SWA
    if attention_type == "swa":
        # Basic validation for window size here too can be helpful
        if not isinstance(window_size, int) or window_size <= 0 or window_size % 2 == 0:
            print_error(f"Invalid window_size ({window_size}) passed to train_model for SWA. Must be positive odd integer.")
            return False
        cmd.extend(["--window-size", str(window_size)]) # Use window_size argument

    # --- Run Command ---
    # Ensure the command is run from the repository directory where train.py exists
    # Use a descriptive message for the run_command function
    description = f"Training {MODEL_VARIANT} with {attention_type} attention (LR={lr}, WD={wd}, Clip={grad_clip_norm})"
    success = run_command(cmd, cwd=REPO_PATH, description=description) # Use REPO_PATH global constant

    if not success:
        print_error(f"Training failed for {attention_type} attention. Check logs above for details.")
        # Depending on your workflow, you might want to raise an exception or exit here

    return success

def find_latest_experiment_dir(pattern):
    """Finds the most recently modified directory matching a pattern."""
    print_info(f"Searching for experiment directories matching: {pattern}")
    try:
        dirs = glob.glob(pattern)
        if not dirs:
            print_warning(f"No directories found matching pattern: {pattern}")
            return None
        # Sort by modification time, newest first
        dirs.sort(key=os.path.getmtime, reverse=True)
        latest_dir = dirs[0]
        print_info(f"Found latest directory: {latest_dir}")
        if not os.path.isdir(latest_dir):
             print_warning(f"Path found is not a directory: {latest_dir}")
             return None
        return latest_dir
    except Exception as e:
        print_error(f"Error finding experiment directory: {e}")
        return None

def get_best_checkpoint_path(exp_dir):
    """Finds the 'best_model.pt' within an experiment's checkpoint directory."""
    if exp_dir is None or not os.path.isdir(exp_dir):
        print_warning("Invalid experiment directory provided for finding best checkpoint.")
        return None

    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")

    if os.path.isfile(best_ckpt_path):
        print_info(f"Found best model checkpoint: {best_ckpt_path}")
        return best_ckpt_path
    else:
        print_warning(f"Best model checkpoint (best_model.pt) not found in: {checkpoint_dir}")
        # Fallback: Find the latest interval checkpoint if best doesn't exist?
        interval_ckpts = glob.glob(os.path.join(checkpoint_dir, "ckpt_*.pt"))
        if interval_ckpts:
             interval_ckpts.sort(key=os.path.getmtime, reverse=True)
             latest_interval_ckpt = interval_ckpts[0]
             print_warning(f"Using latest interval checkpoint as fallback: {latest_interval_ckpt}")
             return latest_interval_ckpt
        else:
             print_error(f"No checkpoints found in {checkpoint_dir}.")
             return None


def sample_grid(attention_type, output_path):
    """Generates a sample grid using the best model for the given attention type."""
    print_header(f"Generating Sample Grid - Attention: {attention_type.upper()}")

    # Find the latest experiment directory for this attention type
    attn_pattern = f"attn-{attention_type}"
    if attention_type == "swa": attn_pattern += "*" # Allow for window size suffix
    pattern = os.path.join(RESULTS_PATH, f"*{attn_pattern}*/") # Use results path
    latest_exp_dir = find_latest_experiment_dir(pattern)

    if latest_exp_dir:
        best_ckpt_path = get_best_checkpoint_path(latest_exp_dir)
        if best_ckpt_path:
            print_info(f"Running sample.py using checkpoint: {best_ckpt_path}")
            cmd = [
                sys.executable, # Use the same python interpreter that runs this script
                "sample.py",
                "--ckpt", best_ckpt_path,
                "--num-samples", str(NUM_GRID_SAMPLES),
                "--output-path", output_path
                # Add --image-size, --model etc. if needed and not reliably in checkpoint
                # However, sample.py is designed to read them from the checkpoint args
            ]
            if run_command(cmd, cwd=REPO_PATH, description=f"Generating {attention_type} sample grid"):
                if os.path.exists(output_path):
                    print_info(f"Sample grid saved to: {output_path}")
                    return True
                else:
                    print_error("sample.py ran but output file not found.")
                    return False
            else:
                return False # run_command failed
        else:
            print_error(f"Could not find a suitable checkpoint for {attention_type}. Skipping sample grid generation.")
            return False
    else:
        print_error(f"Could not find experiment directory for {attention_type}. Skipping sample grid generation.")
        return False

def generate_fid_samples(attention_type, sample_dir):
    """Generates samples for FID calculation using sample_ddp.py."""
    print_header(f"Generating FID Samples - Attention: {attention_type.upper()}")

    # Find the latest experiment directory for this attention type
    attn_pattern = f"attn-{attention_type}"
    if attention_type == "swa": attn_pattern += "*"
    pattern = os.path.join(RESULTS_PATH, f"*{attn_pattern}*/")
    latest_exp_dir = find_latest_experiment_dir(pattern)

    if latest_exp_dir:
        best_ckpt_path = get_best_checkpoint_path(latest_exp_dir)
        if best_ckpt_path:
            print_info(f"Running sample_ddp.py using checkpoint: {best_ckpt_path}")
            print_info(f"Output directory for samples: {sample_dir}")
            # Ensure the sample directory exists (handled by sample_ddp.py rank 0, but good practice)
            # os.makedirs(sample_dir, exist_ok=True)

            # Use accelerate launch for sample_ddp.py
            cmd = [
                "accelerate", "launch", "sample_ddp.py",
                "--ckpt", best_ckpt_path,
                "--sample-dir", sample_dir,
                "--num-fid-samples", str(NUM_FID_SAMPLES),
                "--per-proc-batch-size", str(PER_PROC_BATCH_SIZE_SAMPLE)
                # Add other args if necessary (image_size, model should come from ckpt)
            ]
            if run_command(cmd, cwd=REPO_PATH, description=f"Generating {attention_type} FID samples"):
                # Check if the NPZ file was created by rank 0 in sample_ddp.py
                expected_npz = f"{sample_dir}.npz"
                time.sleep(2) # Give filesystem a moment
                if os.path.exists(expected_npz):
                    print_info(f"FID samples and NPZ generated successfully in {sample_dir} and {expected_npz}")
                    return True
                else:
                     # Check if PNGs were generated even if NPZ failed
                     png_files = glob.glob(os.path.join(sample_dir, "*.png"))
                     if png_files:
                          print_warning(f"sample_ddp.py finished, PNG files found in {sample_dir}, but NPZ file {expected_npz} is missing. FID calculation might fail.")
                          return True # Allow FID calculation attempt
                     else:
                          print_error(f"sample_ddp.py finished, but no PNG files found in {sample_dir}. FID generation likely failed.")
                          return False
            else:
                return False # run_command failed
        else:
            print_error(f"Could not find suitable checkpoint for {attention_type}. Skipping FID sample generation.")
            return False
    else:
        print_error(f"Could not find experiment directory for {attention_type}. Skipping FID sample generation.")
        return False

def resize_real_images_for_fid():
    """Resizes the ORIGINAL dataset images for FID comparison."""
    print_header("Resizing ORIGINAL Images for FID Baseline")
    global original_dataset_path # Uses the path set by setup_paths

    # Check if the path to originals was actually set successfully
    if original_dataset_path is None or not os.path.isdir(original_dataset_path):
        print_error("Original dataset path (LOCAL_DATASET_PATH) not found or invalid.")
        print_error("Cannot resize original images for FID baseline.")
        return False # Indicate failure

    target_size = (IMAGE_SIZE, IMAGE_SIZE) # Match training size (Global Constant)
    target_dir = REAL_IMAGES_RESIZED_DIR   # Global Constant

    print_info(f"Source of ORIGINAL images: {original_dataset_path}")
    print_info(f"Target Directory for FID baseline: {target_dir}")
    print_info(f"Target Size: {target_size}")

    os.makedirs(target_dir, exist_ok=True)

    image_files = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
    try:
        for item_name in os.listdir(original_dataset_path):
            source_item = os.path.join(original_dataset_path, item_name)
            if os.path.isfile(source_item) and item_name.lower().endswith(valid_extensions):
                image_files.append(source_item)
    except Exception as e:
         print_error(f"Error listing files in {original_dataset_path}: {e}")
         return False

    print(f"Found {len(image_files)} potential original image files to resize.")
    if not image_files:
         print_error("No original image files found in the source directory.")
         return False

    copied_count = 0
    error_count = 0
    for img_path in tqdm(image_files, desc="Resizing original images"):
        try:
            # Use high-quality resampling
            resampling_filter = Image.Resampling.LANCZOS
            img = Image.open(img_path).convert('RGB') # Ensure RGB
            img_resized = img.resize(target_size, resampling_filter)

            base_name = os.path.basename(img_path)
            name, _ = os.path.splitext(base_name)
            target_filename = f"{name}.png" # Save as PNG for consistency
            target_path = os.path.join(target_dir, target_filename)

            img_resized.save(target_path)
            copied_count += 1
        except UnidentifiedImageError:
            print(f"\nWarning: Skipping file, cannot identify image file: {img_path}")
            error_count += 1
        except Exception as e:
            print(f"\nWarning: Failed to process/resize original image '{img_path}': {e}")
            error_count += 1

    print(f"\nFinished resizing original images.")
    print(f"  Processed successfully: {copied_count}")
    print(f"  Errors encountered/Skipped: {error_count}")
    if copied_count > 0:
        print(f"Resized original images saved to: {target_dir}")
        return True # Indicate success
    else:
        print_error("No original images were successfully resized for FID baseline.")
        return False # Indicate failure


def calculate_fid(generated_samples_path, name):
    """Calculates FID between resized real images and generated samples."""
    print_header(f"Calculating FID - {name}")

    real_path = REAL_IMAGES_RESIZED_DIR
    gen_path = generated_samples_path # This should be the directory containing generated PNGs

    print(f"Real images path: {real_path}")
    print(f"Generated images path: {gen_path}")

    # --- Crucial Checks ---
    if not os.path.isdir(real_path) or not os.listdir(real_path):
        print_error(f"Resized real image path not found or is empty: {real_path}")
        print_error("Please ensure the image resizing step ran successfully.")
        return False
    if not os.path.isdir(gen_path) or not os.listdir(gen_path):
        print_error(f"Generated samples path for '{name}' not found or is empty: {gen_path}")
        print_error("Please ensure the FID sample generation step ran successfully.")
        return False
    # --- End Checks ---

    # Use the pytorch_fid module directly
    # --num-workers 0 might be more stable on some systems
    cmd = [
        sys.executable, "-m", "pytorch_fid",
        real_path,
        gen_path,
        "--device", "cuda",
        "--num-workers", "0" # Changed to 0 based on notebook comment/potential issues
    ]

    return run_command(cmd, description=f"Calculating FID for {name}")


# --- Main Execution Logic ---

def main_script(args):
    """Orchestrates the entire workflow."""
    print_header("Starting DiT Unconditional Workflow")

    # 0. Initial Setup
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    print(f"Workspace directory: {WORKSPACE_DIR}")
    start_time = time.time()

    # 1. Check GPU
    check_gpu()

    # 2. Download Data
    if not setup_paths(): return

    # 3. Install Packages (User responsibility, but provide info)
    if not install_packages(): return

    # 4. Clone Repository
    if not clone_repo(): return

    # 5. Define/Write modified Python files
    if not define_models_py(): return
    if not define_train_py(): return # This generates train.py
    if not define_sample_py(): return
    if not define_sample_ddp_py(): return

    # 6. Restructure Data
    if not restructure_data(): return

    # --- Training ---
    # 7. Train Full Attention Model using Global Constants
    # --- Training ---
    print_info("Starting Full Attention Training...")
    train_success_full = train_model(
        attention_type="full",
        lr=LEARNING_RATE,
        wd=WEIGHT_DECAY,
        grad_clip_norm=GRAD_CLIP_NORM,
        patience=EARLY_STOPPING_PATIENCE,
        scheduler_epochs=SCHEDULER_DURATION_EPOCHS,
        warmup_steps=WARMUP_STEPS,  # Pass warmup
        accumulation_steps=GRAD_ACCUMULATION_STEPS  # Pass accumulation
    )
    if not train_success_full:
        print_warning("Full attention training failed. Skipping subsequent steps dependent on it.")
        # Decide if you want to exit or continue with SWA if possible
        # return # Option to stop entire script

    # 8. Train SWA Model using Global Constants and CLI window_size
    # Check if SWA is feasible first (requires successful full training?) - optional
    print_info("Starting SWA Training...")
    train_success_swa = train_model(
        attention_type="swa",
        window_size=args.window_size,
        lr=LEARNING_RATE,
        wd=WEIGHT_DECAY,
        grad_clip_norm=GRAD_CLIP_NORM,
        patience=EARLY_STOPPING_PATIENCE,
        scheduler_epochs=SCHEDULER_DURATION_EPOCHS,
        warmup_steps=WARMUP_STEPS,  # Pass warmup
        accumulation_steps=GRAD_ACCUMULATION_STEPS  # Pass accumulation
    )
    if not train_success_swa:
       print_warning("SWA training failed. Subsequent SWA steps might fail.")


    # --- Sampling & Evaluation ---
    # Need to change directory into the repo for sampling/FID scripts
    print_info(f"Changing working directory to: {REPO_PATH}")
    try:
        # Check if REPO_PATH exists before changing
        if not os.path.isdir(REPO_PATH):
             print_error(f"Repository path does not exist: {REPO_PATH}. Cannot change directory.")
             return
        os.chdir(REPO_PATH)
    except Exception as e:
        print_error(f"Failed to change directory to {REPO_PATH}: {e}")
        return

    # 9. Generate Sample Grid (Full Attention - Best Model)
    if train_success_full: # Only sample if training might have produced a checkpoint
        sample_grid(attention_type="full", output_path=SAMPLE_GRID_FULL_BEST_PATH)
    else:
        print_info("Skipping Full Attention sample grid due to training failure.")

    # 10. Generate Sample Grid (SWA - Best Model)
    if train_success_swa: # Only sample if training might have produced a checkpoint
        # Format the SWA output path correctly using the window size from args
        swa_grid_output_path = SAMPLE_GRID_SWA_BEST_PATH.format(args.window_size)
        sample_grid(attention_type="swa", output_path=swa_grid_output_path)
    else:
        print_info("Skipping SWA sample grid due to training failure.")


    # 11. Generate FID Samples (Full Attention - Best Model)
    if train_success_full:
        generate_fid_samples(attention_type="full", sample_dir=FID_SAMPLES_FULL_BEST_DIR)
    else:
         print_info("Skipping Full Attention FID sample generation due to training failure.")


    # 12. Generate FID Samples (SWA - Best Model)
    if train_success_swa:
        # Format the SWA FID sample dir correctly using the window size from args
        swa_fid_sample_dir = FID_SAMPLES_SWA_BEST_DIR.format(args.window_size)
        generate_fid_samples(attention_type="swa", sample_dir=swa_fid_sample_dir)
    else:
         print_info("Skipping SWA FID sample generation due to training failure.")


    # 13. Resize Real Images for FID Calculation
    # Change back to original directory temporarily if needed, or use absolute paths
    try:
        # Navigate back to the directory *containing* the workspace
        original_cwd = os.path.abspath(os.path.join(WORKSPACE_DIR, os.pardir))
        print_info(f"Changing back to directory: {original_cwd} for image resizing")
        os.chdir(original_cwd)
        resize_success = resize_real_images_for_fid() # Assumes this returns True/False
        print_info(f"Changing working directory back to: {REPO_PATH} for FID calculation")
        os.chdir(REPO_PATH) # Change back
        if not resize_success:
            print_warning("Real image resizing failed. FID calculation will likely fail.")
            # Decide whether to stop or continue
    except Exception as e:
         print_error(f"Error changing directories or resizing images: {e}")
         return


    # 14. Calculate FID Scores
    # --- Calculate FID for Full Attention ---
    if train_success_full and resize_success: # Check dependencies
        fid_npz_full = f"{FID_SAMPLES_FULL_BEST_DIR}.npz"
        path_to_use_full = None
        if os.path.exists(fid_npz_full):
            path_to_use_full = fid_npz_full
            print_info(f"Using NPZ file for Full Attention FID: {path_to_use_full}")
        elif os.path.isdir(FID_SAMPLES_FULL_BEST_DIR):
            path_to_use_full = FID_SAMPLES_FULL_BEST_DIR
            print_warning(f"NPZ file {fid_npz_full} not found. Using directory {path_to_use_full} for Full Attention FID.")
        else:
            print_warning(
                f"Neither NPZ nor directory found for Full Attention FID samples at {FID_SAMPLES_FULL_BEST_DIR}. Skipping FID calculation.")

        if path_to_use_full:
            calculate_fid(generated_samples_path=path_to_use_full, name="Full Attention")
    else:
        print_info("Skipping Full Attention FID calculation due to prior failures.")

    # --- Calculate FID for Windowed Attention ---
    if train_success_swa and resize_success: # Check dependencies
        # Construct the correct path for windowed FID samples using the template and args
        fid_samples_win_dir = FID_SAMPLES_SWA_BEST_DIR.format(args.window_size)
        fid_npz_win = f"{fid_samples_win_dir}.npz"  # Expected NPZ file path

        path_to_use_win = None
        if os.path.exists(fid_npz_win):
            path_to_use_win = fid_npz_win
            print_info(f"Using NPZ file for Windowed FID: {path_to_use_win}")
        elif os.path.isdir(fid_samples_win_dir):
            path_to_use_win = fid_samples_win_dir
            print_warning(f"NPZ file {fid_npz_win} not found. Using directory {path_to_use_win} for Windowed FID.")
        else:
            print_warning(
                f"Neither NPZ nor directory found for Windowed FID samples at {fid_samples_win_dir}. Skipping FID calculation.")

        if path_to_use_win:
            calculate_fid(generated_samples_path=path_to_use_win, name=f"Windowed Attention w={args.window_size}")
    else:
        print_info("Skipping SWA FID calculation due to prior failures.")


    # --- End of Workflow ---
    end_time = time.time()
    total_duration = end_time - start_time
    print_header("Workflow Complete")
    print(f"Total execution time: {total_duration / 60:.2f} minutes")
    print(f"Results, checkpoints, and samples are in: {WORKSPACE_DIR}")

# --- (The __main__ block remains the same as you provided) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DiT Unconditional Runner Script (Masked Window Attention)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=9, # Default odd window size
        help='Window size for the masked window attention mechanism (must be positive odd integer).'
    )
    cli_args = parser.parse_args()

    # Validate CLI window size
    if cli_args.window_size <= 0 or cli_args.window_size % 2 == 0:
        # Use print_error if defined, otherwise print to stderr
        error_func = print_error if 'print_error' in locals() else lambda msg: print(f"!!! ERROR: {msg}", file=sys.stderr)
        error_func(f"Invalid --window-size ({cli_args.window_size}). It must be a positive odd integer.")
        sys.exit(1)

    main_script(cli_args) # Pass args to main_script