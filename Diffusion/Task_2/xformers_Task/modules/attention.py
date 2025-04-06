# modules/attention.py
"""
Defines different Multi-Head Self-Attention implementations.
Includes standard PyTorch attention and an optimized version using xformers.
"""
import torch
import torch.nn as nn

# --- xformers Check ---
# Attempt to import xformers for optimized attention.
# Set a flag indicating its availability.
try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
    print("INFO: xformers found. XFormersAttention is available.")
except ImportError:
    print("WARNING: xformers not found. Install it ('pip install xformers') "
          "to use optimized XFormersAttention. Falling back to standard attention if requested.")
    XFORMERS_AVAILABLE = False
    xops = None # Define xops as None if import fails


class StandardAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention implementation using PyTorch modules.

    Args:
        dim (int): Input and output dimension of the attention layer.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        qkv_bias (bool, optional): Whether to include bias in the QKV projection. Defaults to False.
        attn_drop (float, optional): Dropout probability for the attention scores. Defaults to 0.0.
        proj_drop (float, optional): Dropout probability for the output projection. Defaults to 0.0.
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5 # Scaling factor for dot products

        # Linear layer to project input to Q, K, V combined
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # Dropout for attention weights
        self.attn_drop = nn.Dropout(attn_drop)
        # Linear layer for the final output projection
        self.proj = nn.Linear(dim, dim)
        # Dropout for the final output
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor):
        """
        Applies the standard multi-head self-attention.

        Args:
            x (torch.Tensor): Input tensor. Shape: (B, N, C), where N is sequence length, C is dim.

        Returns:
            torch.Tensor: Output tensor. Shape: (B, N, C).
        """
        B, N, C = x.shape

        # 1. Project to QKV and reshape for multi-head attention
        # qkv(): -> (B, N, 3 * C)
        # reshape: -> (B, N, 3, num_heads, head_dim)
        # permute: -> (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Unbind along the first dimension (dim=0) to get Q, K, V
        # q, k, v: each (B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # 2. Compute scaled dot-product attention
        # (q @ k.transpose) -> (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Apply softmax to get attention weights
        attn = attn.softmax(dim=-1)
        # Apply attention dropout
        attn = self.attn_drop(attn)

        # 3. Apply attention weights to values
        # (attn @ v) -> (B, num_heads, N, head_dim)
        # transpose: -> (B, N, num_heads, head_dim)
        # reshape: -> (B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # 4. Apply final projection and dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class XFormersAttention(nn.Module):
    """
    Multi-Head Self-Attention implementation using the optimized `memory_efficient_attention`
    from the xformers library.

    Args:
        dim (int): Input and output dimension of the attention layer.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        qkv_bias (bool, optional): Whether to include bias in the QKV projection. Defaults to False.
        attn_drop (float, optional): Dropout probability (passed to xformers). Defaults to 0.0.
        proj_drop (float, optional): Dropout probability for the output projection. Defaults to 0.0.

    Raises:
        RuntimeError: If xformers is not installed or available.
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        if not XFORMERS_AVAILABLE:
            raise RuntimeError("xformers is not available, cannot instantiate XFormersAttention.")
        if dim % num_heads != 0:
             raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")

        self.num_heads = num_heads
        head_dim = dim // num_heads # Note: xformers might handle head_dim internally

        # Linear layer to project input to Q, K, V combined
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # Store dropout probability for use in the forward pass with xformers
        self.attn_drop_p = attn_drop
        # Linear layer for the final output projection
        self.proj = nn.Linear(dim, dim)
        # Dropout for the final output
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor):
        """
        Applies multi-head self-attention using xformers.

        Args:
            x (torch.Tensor): Input tensor. Shape: (B, N, C).

        Returns:
            torch.Tensor: Output tensor. Shape: (B, N, C).
        """
        B, N, C = x.shape

        # 1. Project to QKV and reshape for xformers
        # qkv(): -> (B, N, 3 * C)
        # reshape: -> (B, N, 3, num_heads, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        # Unbind along dimension 2 to get Q, K, V
        # q, k, v: each (B, N, num_heads, head_dim) - Note the shape difference from standard!
        q, k, v = qkv.unbind(2)

        # 2. Apply xformers memory-efficient attention
        # xformers expects query, key, value in shape (B, N, H, D)
        # Dropout is applied internally by xformers if p > 0.0 and model is in training mode
        attn_bias = None # No attention bias (like ALiBi) used here
        dropout_p = self.attn_drop_p if self.training else 0.0
        # Output shape: (B, N, num_heads, head_dim)
        x = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias, p=dropout_p)

        # 3. Reshape back to the standard (B, N, C) format
        x = x.reshape(B, N, C)

        # 4. Apply final projection and dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        return x