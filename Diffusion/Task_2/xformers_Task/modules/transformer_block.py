# modules/transformer_block.py
"""
Defines the core DiT (Diffusion Transformer) block, incorporating AdaLN-Zero
modulation and pluggable attention mechanisms.
"""
import torch
import torch.nn as nn
from typing import Type

# Import necessary components from other module files
from .feedforward import FeedForward
from .modulation import AdaLNModulation, modulate
# Note: Attention class is passed as an argument, not imported directly by default


class DiTBlock(nn.Module):
    """
    A single block of the Diffusion Transformer (DiT).

    This block applies adaptive layer normalization (AdaLN-Zero) based on
    conditioning inputs (e.g., timestep embeddings) before both the
    multi-head self-attention (MSA) and the feedforward network (MLP).
    It uses gated residual connections.

    Args:
        dim (int): Dimensionality of the input and output features.
        num_heads (int): Number of attention heads in the MSA layer.
        mlp_ratio (float, optional): Determines the hidden dimension of the MLP
            (hidden_dim = dim * mlp_ratio). Defaults to 4.0.
        qkv_bias (bool, optional): Whether to use bias in the QKV projection
            of the attention layer. Defaults to False.
        attn_drop (float, optional): Dropout rate for attention weights. Defaults to 0.0.
        proj_drop (float, optional): Dropout rate for projections (attention output and MLP output).
             Defaults to 0.0.
        cond_embed_dim (int, optional): Dimension of the conditioning embedding (e.g., timestep).
             If None, defaults to `dim`.
        attention_cls (Type[nn.Module]): The class to use for the attention mechanism
             (e.g., StandardAttention, XFormersAttention).
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = False,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 cond_embed_dim: int = None,
                 attention_cls: Type[nn.Module] = None): # Use passed attention class
        super().__init__()
        if attention_cls is None:
            raise ValueError("attention_cls must be provided to DiTBlock")
        self.dim = dim
        # Use provided conditioning embedding dim or default to main dimension
        cond_embed_dim = cond_embed_dim if cond_embed_dim is not None else dim

        # Layer Normalization layers
        # `elementwise_affine=False` because scale/shift are handled by AdaLN modulation
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        # Modulation layers to compute shift, scale, gate for Attention and MLP branches
        # These take the conditioning embedding `c` as input.
        self.adaLN_modulation_attn = AdaLNModulation(cond_embed_dim, dim)
        self.adaLN_modulation_mlp = AdaLNModulation(cond_embed_dim, dim)

        # Attention Layer (instantiated using the provided class)
        self.attn = attention_cls(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                  attn_drop=attn_drop, proj_drop=proj_drop)

        # FeedForward (MLP) Layer
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim=dim, hidden_dim=mlp_hidden_dim, dropout=proj_drop)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        Forward pass through the DiT block.

        Args:
            x (torch.Tensor): Input sequence tensor. Shape: (B, N, D).
            c (torch.Tensor): Conditioning embedding tensor (e.g., timestep embedding).
                              Shape: (B, cond_embed_dim).

        Returns:
            torch.Tensor: Output sequence tensor. Shape: (B, N, D).
        """
        # --- Attention Branch ---
        # 1. Compute modulation parameters (shift, scale, gate) from conditioning `c`
        params_attn = self.adaLN_modulation_attn(c) # Shape: (B, 1, 3*D)
        # Split the parameters along the last dimension
        shift_msa, scale_msa, gate_msa = params_attn.chunk(3, dim=-1) # Each: (B, 1, D)

        # 2. Apply AdaLN-Zero: Normalize, Modulate, then Attention
        residual = x
        x_norm = self.norm1(x)                 # Layer Norm (no affine)
        attn_input = modulate(x_norm, shift_msa, scale_msa) # Apply scale and shift
        attn_output = self.attn(attn_input)    # Multi-Head Self-Attention

        # 3. Apply gated residual connection
        x = residual + gate_msa * attn_output  # Gate controls contribution of attention output

        # --- MLP Branch ---
        # 1. Compute modulation parameters (shift, scale, gate) from conditioning `c`
        params_mlp = self.adaLN_modulation_mlp(c) # Shape: (B, 1, 3*D)
        shift_mlp, scale_mlp, gate_mlp = params_mlp.chunk(3, dim=-1) # Each: (B, 1, D)

        # 2. Apply AdaLN-Zero: Normalize, Modulate, then MLP
        residual = x
        x_norm = self.norm2(x)                 # Layer Norm (no affine)
        mlp_input = modulate(x_norm, shift_mlp, scale_mlp) # Apply scale and shift
        mlp_output = self.mlp(mlp_input)       # FeedForward Network

        # 3. Apply gated residual connection
        x = residual + gate_mlp * mlp_output   # Gate controls contribution of MLP output

        return x