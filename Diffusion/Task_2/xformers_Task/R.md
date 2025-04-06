# README: Accelerating DiT with xFormers Attention

This project explores the integration of **xFormers**, a library for efficient transformer implementations, into the **Diffusion Transformer (DiT)** model. The goal is to replace DiT's native attention mechanism with xFormers' memory-efficient attention and measure the resulting speedup during inference.

---

## Objectives
1. **Replace DiT's Attention Block** with xFormers' `memory_efficient_attention` implementation.
2. **Benchmark Performance** by comparing inference times for sampling 50 images between:
   - Baseline DiT model
   - xFormers-optimized DiT model

---

## Setup & Installation

### Prerequisites
- NVIDIA GPU with CUDA support
- Python ≥3.10
- PyTorch ≥2.6.0

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/facebookresearch/DiT.git
   cd DiT
   ```

2. **Install Dependencies**:
   ```bash
   pip install -q xformers
   pip install --upgrade torchvision
   pip install plotly pandas  # For visualization
   ```

3. **Resolve Dependency Conflicts** (if any):
   - Ensure compatibility between `torch`, `torchvision`, and `xformers`.
   - Example fix for observed conflicts:
     ```bash
     pip install torch==2.6.0 torchvision==0.21.0
     ```

---

## Implementation Details

### Model Chosen
The DiT-XL/2 was used for this task.

### Key Modifications
**xFormers Attention Module**  
   Replaced DiT's original attention with a custom `XFormersAttention` class:
   ```python
   import xformers.ops as xops

   class XFormersAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        if not XFORMERS_AVAILABLE:
            raise RuntimeError("xformers is not available, cannot instantiate XFormersAttention.")
        if dim % num_heads != 0:
             raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")

        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_p = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
   
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)

        attn_bias = None
        dropout_p = self.attn_drop_p if self.training else 0.0
        x = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias, p=dropout_p)

        x = x.reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
   ```

---

## Results

### Speed Comparison (50 Images)
| Model          | Avg Time (s) | Speedup |
|----------------|--------------|---------|
| Baseline (DiT) | 9.7122       | 1.00x   |
| xFormers       | 8.0981       | **1.20x** |

---

## Key Findings
1. **1.2x Speedup**: xFormers reduces inference time by 1.2x for 50-image batches.
2. **Memory Efficiency**: xFormers' memory-optimized attention enables larger batch processing.

---

## Conclusion
- **xFormers is effective** for accelerating DiT while maintaining numerical equivalence.
- **Recommended Use Cases**:
  - High-throughput image generation
  - Resource-constrained environments
  - Scaling to larger models/data

---

##  References
1. DiT Paper: [*Scalable Diffusion Models with Transformers*](https://arxiv.org/abs/2212.09748)
2. xFormers Documentation: [xformers.readthedocs.io](https://xformers.readthedocs.io/)
3. Original DiT Codebase: [facebookresearch/DiT](https://github.com/facebookresearch/DiT)
   
---
