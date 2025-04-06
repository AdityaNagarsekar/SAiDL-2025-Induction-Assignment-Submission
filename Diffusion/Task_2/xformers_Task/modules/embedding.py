# modules/embedding.py
"""
Defines embedding modules: PatchEmbed for images and TimestepEmbedder for time.
"""
import torch
import torch.nn as nn
import math

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations using sinusoidal embeddings
    followed by an MLP.

    Args:
        hidden_size (int): The dimensionality of the output embedding vector.
        frequency_embedding_size (int, optional): The dimensionality of the
            intermediate sinusoidal embedding. Defaults to 256.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        # MLP to project the sinusoidal embedding to the target hidden size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(), # Swish activation function
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        """
        Creates sinusoidal timestep embeddings.

        Args:
            t (torch.Tensor): A 1-D tensor of N timesteps. Shape: (N,).
            dim (int): The dimension of the embedding.
            max_period (int, optional): Controls the range of frequencies. Defaults to 10000.

        Returns:
            torch.Tensor: Positionally embedded timesteps. Shape: (N, dim).
        """
        # Ensure the dimension is even for sin/cos pairs
        half = dim // 2
        # Calculate frequencies using log scale
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        # Calculate arguments for sin and cos functions
        args = t[:, None].float() * freqs[None]
        # Concatenate sin and cos embeddings
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # Handle odd dimensions by appending a zero column
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor):
        """
        Embeds the input timestep tensor.

        Args:
            t (torch.Tensor): Tensor of timesteps. Shape: (BatchSize,).

        Returns:
            torch.Tensor: Embedded timesteps. Shape: (BatchSize, hidden_size).
        """
        # Generate sinusoidal embeddings
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # Project through MLP
        t_emb = self.mlp(t_freq)
        return t_emb


class PatchEmbed(nn.Module):
    """
    Transforms a 2D image into a sequence of patch embeddings.

    Uses a convolution layer to project image patches into embedding vectors.

    Args:
        img_size (int, optional): Size of the input image (height and width). Defaults to 224.
        patch_size (int, optional): Size of each square patch. Defaults to 16.
        in_chans (int, optional): Number of input image channels. Defaults to 3.
        embed_dim (int, optional): Dimensionality of the output patch embeddings. Defaults to 768.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        if img_size % patch_size != 0:
             raise ValueError("Image dimensions must be divisible by patch size.")
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # Use a Conv2d layer to perform patching and embedding simultaneously
        # Kernel size and stride equal to patch_size achieves non-overlapping patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        """
        Applies patch embedding to the input image tensor.

        Args:
            x (torch.Tensor): Input image tensor. Shape: (B, C, H, W).

        Returns:
            torch.Tensor: Sequence of patch embeddings. Shape: (B, NumPatches, embed_dim).
        """
        B, C, H, W = x.shape
        # Ensure input image size matches the configured size
        if H != self.img_size or W != self.img_size:
            raise ValueError(f"Input image size ({H}x{W}) doesn't match model ({self.img_size}x{self.img_size}).")

        # Apply the convolution -> Output shape: (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        # Flatten the spatial dimensions (H/P, W/P) into a single sequence length (NumPatches)
        # Shape: (B, embed_dim, NumPatches)
        x = x.flatten(2)
        # Transpose to get the standard transformer input shape: (B, NumPatches, embed_dim)
        x = x.transpose(1, 2)
        return x