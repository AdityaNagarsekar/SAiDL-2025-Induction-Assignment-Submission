# model.py
"""
Defines the main Diffusion Transformer (DiT) model architecture.
"""
import torch
import torch.nn as nn
import numpy as np

# Import building blocks from the modules directory
from modules.embedding import PatchEmbed, TimestepEmbedder
from modules.attention import StandardAttention, XFormersAttention, XFORMERS_AVAILABLE
from modules.transformer_block import DiTBlock
from modules.final_layer import FinalLayer
# Import utility functions
from utils import get_2d_sincos_pos_embed


class DiT(nn.Module):
    """
    Diffusion Transformer Model.

    Combines patch embedding, timestep embedding, a series of transformer blocks
    with AdaLN-Zero modulation, and a final layer to process sequences conditioned
    on timesteps. Capable of using standard PyTorch attention or optimized xformers attention.

    Args:
        img_size (int, optional): Input image size. Defaults to 32.
        patch_size (int, optional): Size of image patches. Defaults to 2.
        in_chans (int, optional): Number of input image channels. Defaults to 4.
        hidden_size (int, optional): Transformer hidden dimension. Defaults to 768.
        depth (int, optional): Number of transformer blocks. Defaults to 12.
        num_heads (int, optional): Number of attention heads. Defaults to 12.
        mlp_ratio (float, optional): Ratio for MLP hidden dimension. Defaults to 4.0.
        qkv_bias (bool, optional): If True, add bias to QKV projections. Defaults to True.
        attn_drop (float, optional): Dropout rate for attention weights. Defaults to 0.0.
        proj_drop (float, optional): Dropout rate for output projections. Defaults to 0.0.
        use_xformers (bool, optional): If True, try to use XFormersAttention.
            Falls back to StandardAttention if xformers is not available. Defaults to False.
    """
    def __init__(self,
                 img_size: int = 32,
                 patch_size: int = 2,
                 in_chans: int = 4,
                 hidden_size: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 use_xformers: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        # Output channels typically match input channels for noise prediction
        self.out_chans = in_chans
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth

        # --- Input Embeddings ---
        # 1. Patch Embedding: Converts image (B, C, H, W) to sequence (B, N, D)
        self.x_embedder = PatchEmbed(img_size, patch_size, in_chans, hidden_size)
        # 2. Timestep Embedding: Converts timestep scalar (B,) to vector (B, D)
        #    Timestep embedding dimension must match the transformer's hidden size
        #    as it acts as the conditioning input 'c'.
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.num_patches = self.x_embedder.num_patches

        # --- Positional Embedding ---
        # Learnable or fixed positional embeddings are added to patch embeddings.
        # DiT uses fixed sinusoidal embeddings, initialized here but not trainable.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        # --- Attention Mechanism Selection ---
        if use_xformers:
            if XFORMERS_AVAILABLE:
                print("INFO: Using xformers attention in DiT model.")
                AttentionClass = XFormersAttention
            else:
                print("WARNING: xformers requested but not available. Falling back to standard attention.")
                AttentionClass = StandardAttention
        else:
            print("INFO: Using standard PyTorch attention in DiT model.")
            AttentionClass = StandardAttention

        # --- Transformer Blocks ---
        # Stack of DiTBlocks. Each block takes the sequence `x` and conditioning `c` (timestep embedding).
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                cond_embed_dim=hidden_size, # Timestep embedding dim acts as conditioning dim
                attention_cls=AttentionClass # Pass the selected attention class
            )
            for _ in range(depth)])

        # --- Final Layer ---
        # Applies final modulation and projects to the output patch shape.
        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            patch_size=patch_size,
            out_channels=self.out_chans,
            cond_embed_dim=hidden_size # Conditioning comes from timestep embedding
        )

        # --- Initialize Weights ---
        self.initialize_weights()

    def initialize_weights(self):
        """Initializes model weights, including positional embeddings and layer weights."""
        # 1. Initialize Positional Embedding using sinusoidal helper function
        # Calculate grid size (assuming square image)
        grid_size = int(self.num_patches ** 0.5)
        pos_embed_numpy = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size)
        # Convert numpy array to torch tensor and assign to the parameter
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed_numpy).float().unsqueeze(0))

        # 2. Initialize transformer block linear layers (basic Xavier uniform)
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init) # Apply recursively to all submodules

        # 3. Special initializations:
        #    - Patch projection weights (already handled by _basic_init if Conv2d is Linear-like)
        #      Could use specific init like kaiming for conv layers if desired.
        #    - Timestep MLP weights (already handled by _basic_init)
        #    - AdaLN modulation layers' linear projections are already zero-initialized
        #      in the AdaLNModulation class __init__.
        #    - Final layer linear projection is already zero-initialized in the
        #      FinalLayer class __init__.

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transforms a sequence of flattened patches back into image space.

        Args:
            x (torch.Tensor): Input tensor of flattened patches.
                Shape: (B, N, P*P*C_out), where N = num_patches, P = patch_size.

        Returns:
            torch.Tensor: Reconstructed image tensor. Shape: (B, C_out, H, W).
        """
        B, N, L = x.shape
        P = self.patch_size
        C = self.out_chans
        H_patches = W_patches = int(N ** 0.5) # Number of patches along height/width
        if H_patches * W_patches != N:
             raise ValueError("Total number of patches must be a perfect square.")
        if L != P * P * C:
             raise ValueError("Input tensor last dimension doesn't match P*P*C.")

        # Reshape to (B, H_patches, W_patches, P, P, C)
        x = x.reshape(B, H_patches, W_patches, P, P, C)
        # Permute to (B, C, H_patches, P, W_patches, P)
        # Correct permutation: bring C forward, then group H and W dimensions
        x = torch.einsum('bhwpqc->bchpwq', x)
        # Reshape to (B, C, H_patches*P, W_patches*P) = (B, C, H, W)
        imgs = x.reshape(B, C, H_patches * P, W_patches * P)
        return imgs


    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DiT model.

        Args:
            x (torch.Tensor): Input latent tensor. Shape: (B, C_in, H, W).
            t (torch.Tensor): Diffusion timesteps. Shape: (B,).

        Returns:
            torch.Tensor: Output tensor (e.g., predicted noise). Shape: (B, C_out, H, W).
        """
        # 1. Embed Inputs
        # x: (B, C, H, W) -> (B, N, D)
        x = self.x_embedder(x)
        # Add positional embedding: (B, N, D) + (1, N, D) -> (B, N, D)
        x = x + self.pos_embed
        # t: (B,) -> (B, D)
        t = self.t_embedder(t) # Timestep embedding acts as conditioning `c`

        # 2. Apply Transformer Blocks
        # Each block takes `x` and `t` (conditioning)
        for block in self.blocks:
            x = block(x, t)

        # 3. Final Layer
        # Applies final LayerNorm, modulation, and projects to output shape
        # x: (B, N, D) -> (B, N, P*P*C_out)
        x = self.final_layer(x, t)

        # 4. Unpatchify to Image Space
        # x: (B, N, P*P*C_out) -> (B, C_out, H, W)
        x = self.unpatchify(x)

        return x