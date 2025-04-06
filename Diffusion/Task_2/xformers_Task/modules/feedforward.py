# modules/feedforward.py
"""
Defines the FeedForward network module used within the Transformer blocks.
"""
import torch.nn as nn

class FeedForward(nn.Module):
    """
    A standard two-layer FeedForward network with GELU activation and dropout.

    Args:
        dim (int): Input and output dimension of the network.
        hidden_dim (int): Dimension of the hidden layer.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  # Gaussian Error Linear Unit activation
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), # Project back to original dimension
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Applies the FeedForward network to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, SequenceLength, Dim).

        Returns:
            torch.Tensor: Output tensor of shape (Batch, SequenceLength, Dim).
        """
        return self.net(x)