
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

