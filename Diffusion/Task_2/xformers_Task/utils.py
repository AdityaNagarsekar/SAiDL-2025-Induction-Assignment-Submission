# utils.py
"""
Utility functions, including helpers for generating positional embeddings.
Based on the original DiT repository and MAE implementations.
"""
import numpy as np
import math

# --- Positional Embedding Helpers ---

def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Generates 1D sinusoidal positional embeddings.

    Args:
        embed_dim (int): Output dimension for each position (should be even).
        pos (np.ndarray): A numpy array of positions to be encoded. Shape: (M,).

    Returns:
        np.ndarray: Sinusoidal positional embeddings. Shape: (M, embed_dim).
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")

    # Calculation uses float64 for precision, matching original implementations
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega, dtype=np.float64)  # Outer product (M, D/2)

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb.astype(np.float32) # Cast back to float32


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """
    Generates 2D sinusoidal positional embeddings from a grid.

    Args:
        embed_dim (int): The target embedding dimension (should be even).
        grid (np.ndarray): A 2D grid of coordinates. Shape expected: (2, H, W) or similar,
                           where grid[0] is H coordinates and grid[1] is W coordinates.

    Returns:
        np.ndarray: 2D sinusoidal positional embeddings. Shape: (H*W, embed_dim).
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    if grid.shape[0] != 2:
         raise ValueError(f"Grid must have shape (2, H, W), got {grid.shape}")

    # Use half of dimensions to encode grid_h and half for grid_w
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False, extra_tokens: int = 0) -> np.ndarray:
    """
    Creates 2D sinusoidal positional embeddings for a square grid.

    Args:
        embed_dim (int): Dimension of the positional embedding.
        grid_size (int): Height and width of the grid (e.g., sqrt(num_patches)).
        cls_token (bool, optional): Whether to reserve space for a class token
             (typically prepended). Defaults to False.
        extra_tokens (int, optional): Number of extra tokens (like class token) to
             account for, usually 1 if cls_token is True. Defaults to 0.

    Returns:
        np.ndarray: Positional embeddings. Shape: [grid_size*grid_size, embed_dim] or
                    [1+grid_size*grid_size, embed_dim] if cls_token is True.
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    # Create a meshgrid. grid[0] is Y coords (rows), grid[1] is X coords (cols)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0) # Shape: (2, grid_size, grid_size)

    # Reshape the grid for the embedding function
    grid = grid.reshape([2, 1, grid_size, grid_size]) # Not strictly needed reshape? Check if works without.
    # Actually, get_2d_sincos_pos_embed_from_grid handles flattening internally.
    # Let's pass the core (2, grid_size, grid_size) grid.
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid.reshape(2, -1)) # Pass (2, H*W)

    # Handle class token (if needed, though DiT typically doesn't use one)
    if cls_token and extra_tokens > 0:
        # Prepend zero embeddings for the class token(s)
        pos_embed = np.concatenate([np.zeros((extra_tokens, embed_dim)), pos_embed], axis=0)

    return pos_embed.astype(np.float32)