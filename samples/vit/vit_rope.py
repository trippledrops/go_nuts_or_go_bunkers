import torch
import torch.nn as nn
from typing import Tuple

# ==========================================
# 1. 2D-RoPE Module (as defined previously)
# ==========================================
class VisionRotaryEmbedding2D(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 4 == 0, "Head dimension must be divisible by 4."
        self.head_dim = head_dim
        self.axis_dim = head_dim // 2
        self.base = base

    def _get_1d_freqs(self, coords: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        freq_dim = self.axis_dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.axis_dim, 2, dtype=torch.float32, device=device) / self.axis_dim))
        freqs = torch.outer(coords.float(), inv_freq)
        freqs_rep = torch.repeat_interleave(freqs, 2, dim=-1)
        return freqs_rep.cos(), freqs_rep.sin()

    def _get_2d_rotary_embed(self, hp: int, wp: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(hp, device=device),
            torch.arange(wp, device=device),
            indexing="ij"
        )
        grid_y, grid_x = grid_y.reshape(-1), grid_x.reshape(-1)
        cos_y, sin_y = self._get_1d_freqs(grid_y, device)
        cos_x, sin_x = self._get_1d_freqs(grid_x, device)
        cos_2d = torch.cat([cos_y, cos_x], dim=-1)
        sin_2d = torch.cat([sin_y, sin_x], dim=-1)
        return cos_2d.unsqueeze(0).unsqueeze(0), sin_2d.unsqueeze(0).unsqueeze(0)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def forward(self, q: torch.Tensor, k: torch.Tensor, hp: int, wp: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device = q.device
        N = q.shape[2]  
        num_patches = hp * wp
        has_cls_token = (N == num_patches + 1)

        cos, sin = self._get_2d_rotary_embed(hp, wp, device)

        if has_cls_token:
            cls_cos = torch.ones(1, 1, 1, self.head_dim, device=device)
            cls_sin = torch.zeros(1, 1, 1, self.head_dim, device=device)
            cos = torch.cat([cls_cos, cos], dim=2)  
            sin = torch.cat([cls_sin, sin], dim=2)  

        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)
        return q_rotated, k_rotated


# ==========================================
# 2. Multi-Head Attention with 2D-RoPE
# ==========================================
class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Projection matrices for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Initialize 2D-RoPE module
        self.rope_2d = VisionRotaryEmbedding2D(head_dim=self.head_dim)

    def forward(self, x: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
        B, N, D = x.shape
        
        # 1. Project to Q, K, V -> Shape: (B, N, 3 * D)
        qkv = self.qkv(x)
        
        # Reshape to (3, B, num_heads, N, head_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each has shape: (B, H, N, head_dim)

        # 2. APPLY 2D-RoPE to Query and Key dynamically using current grid sizes (hp, wp)
        q, k = self.rope_2d(q, k, hp, wp)

        # 3. Standard Self-Attention scale-dot product
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by Value (v is NOT rotated by RoPE)
        out = torch.matmul(attn_probs, v)  # Shape: (B, H, N, head_dim)
        
        # 4. Concatenate heads back and project
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        return out


# ==========================================
# 3. Simple Vision Transformer (ViT) Segment
# ==========================================
class ViTWith2DRoPE(nn.Module):
    def __init__(self, img_channels=3, patch_size=16, embed_dim=768, num_heads=12):
        super().__init__()
        self.patch_size = patch_size
        
        # Patch projection
        self.patch_embed = nn.Conv2d(
            in_channels=img_channels, out_channels=embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # CLS token parameter
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Attention layer with RoPE support
        self.attn = RoPEMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        
        # Calculate dynamic grid sizes based on input dimensions
        hp, wp = H // self.patch_size, W // self.patch_size
        
        # 1. Patch projection & Flatten: (B, C, H, W) -> (B, D, hp, wp) -> (B, N_patches, D)
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        # 2. Prepend CLS token: Shape becomes (B, 1 + N_patches, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_tokens = torch.cat((cls_tokens, patches), dim=1)
        
        # 3. Pass through multi-head attention (RoPE applies internally)
        # Note: No positional embeddings are added at the embedding layer anymore!
        out = self.norm(x_tokens)
        out = self.attn(out, hp=hp, wp=wp)
        
        return out

