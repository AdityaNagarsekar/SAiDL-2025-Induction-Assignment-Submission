# modules/final_layer.py
"""
Defines the final layer of the DiT model, which applies modulation and projects
the sequence back to the patchified output shape.
"""
import torch
import torch.nn as nn

# Import necessary components from other module files
from .modulation import AdaLNModulation, modulate

class FinalLayer(nn.Module):
    """
    The final layer of the Diffusion Transformer (DiT).

    It applies adaptive layer normalization (AdaLN-Zero) based on conditioning
    input and then projects the features to the dimensions required for reconstructing
    the output (e.g., noise prediction in the shape of input patches).

    Args:
        hidden_size (int): Dimensionality of the input features (transformer hidden dim).
        patch_size (int): Size of the square patches.
        out_channels (int): Number of output channels per patch (usually matches input channels).
        cond_embed_dim (int, optional): Dimension of the conditioning embedding.
             If None, defaults to `hidden_size`.
    """
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, cond_embed_dim: int = None):
        super().__init__()
        # Use provided conditioning embedding dim or default to hidden size
        cond_embed_dim = cond_embed_dim if cond_embed_dim is not None else hidden_size

        # Final Layer Normalization (no affine, handled by modulation)
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Linear layer to project hidden features to the flattened patch dimension
        # Output size per token: patch_size * patch_size * out_channels
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

        # Modulation layer to compute shift and scale (gate is often ignored in final layer)
        self.adaLN_modulation = AdaLNModulation(cond_embed_dim, hidden_size)

        # Initialize the final projection layer's weights and biases to zero,
        # following the DiT paper's practice. This ensures the model initially
        # outputs zeros, which can be beneficial for training stability.
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        Forward pass through the final layer.

        Args:
            x (torch.Tensor): Input sequence tensor from the last DiT block.
                              Shape: (B, N, hidden_size).
            c (torch.Tensor): Conditioning embedding tensor.
                              Shape: (B, cond_embed_dim).

        Returns:
            torch.Tensor: Output tensor representing flattened patches.
                          Shape: (B, N, patch_size * patch_size * out_channels).
        """
        # 1. Compute modulation parameters (shift, scale) from conditioning `c`
        # Note: The AdaLNModulation outputs 3 parameters (shift, scale, gate).
        # We typically only use shift and scale for the final layer modulation.
        params = self.adaLN_modulation(c) # Shape: (B, 1, 3 * hidden_size)
        shift, scale, _ = params.chunk(3, dim=-1) # Ignore gate, Shapes: (B, 1, hidden_size)

        # 2. Apply AdaLN-Zero: Normalize, then Modulate
        x_norm = self.norm_final(x)            # Layer Norm (no affine)
        x_modulated = modulate(x_norm, shift, scale) # Apply scale and shift

        # 3. Apply final linear projection
        x = self.linear(x_modulated)         # Project to output patch dimension

        return x