# modules/modulation.py
"""
Defines components for AdaLN-Zero modulation used in DiT blocks.
"""
import torch
import torch.nn as nn

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    """
    Applies affine transformation (scaling and shifting) to the input tensor.

    This is a core operation in AdaLN / AdaLN-Zero.

    Args:
        x (torch.Tensor): Input tensor. Shape: (B, N, D).
        shift (torch.Tensor): Shift tensor. Shape: (B, 1, D) for broadcasting.
        scale (torch.Tensor): Scale tensor. Shape: (B, 1, D) for broadcasting.

    Returns:
        torch.Tensor: Modulated tensor. Shape: (B, N, D).
    """
    # Element-wise multiplication for scaling (adding 1 ensures identity transform when scale=0)
    # Element-wise addition for shifting
    return x * (1 + scale) + shift


class AdaLNModulation(nn.Module):
    """
    Computes modulation parameters (shift, scale, gate) for AdaLN-Zero
    based on a conditioning embedding (e.g., timestep embedding).

    Uses a simple MLP with SiLU activation. The output layer's weights and biases
    are initialized to zero, ensuring that the initial modulation is an identity
    transform (scale=0, shift=0, gate=0), which is the 'Zero' part of AdaLN-Zero.

    Args:
        embedding_dim (int): Dimension of the conditioning input embedding (e.g., timestep embedding).
        target_dim (int): Dimension of the layer being modulated (e.g., transformer hidden dim).
                         The output will have 3 * target_dim channels for shift, scale, and gate.
    """
    def __init__(self, embedding_dim: int, target_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        # Linear layer projects conditioning embedding to 3 parameters per target dimension
        # (shift, scale, gate)
        self.linear = nn.Linear(embedding_dim, target_dim * 3, bias=True)

        # AdaLN-Zero initialization: Initialize weights and biases of the projection to zero.
        # This ensures that at the beginning of training, the modulation layers are identity
        # transformations, and the residual connections dominate.
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.linear.weight)

    def forward(self, c: torch.Tensor):
        """
        Computes modulation parameters from the conditioning embedding.

        Args:
            c (torch.Tensor): Conditioning embedding. Shape: (B, embedding_dim).

        Returns:
            torch.Tensor: Modulation parameters (shift, scale, gate concatenated).
                          Shape: (B, 1, target_dim * 3). The middle dimension (1)
                          is added for broadcasting compatibility with the target tensor `x`
                          in the `modulate` function.
        """
        # Apply SiLU activation to the conditioning embedding
        c_activated = self.silu(c)
        # Project to get the raw parameters
        params = self.linear(c_activated) # Shape (B, target_dim * 3)
        # Add a sequence dimension (dim=1) for broadcasting across sequence length N
        # Shape becomes (B, 1, target_dim * 3)
        return params.unsqueeze(1)