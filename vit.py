import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. Vision Encoders & Tokenization (ViT Patch Embedding)
# ==========================================
class ViTPatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, embed_dim: int = 768, img_size: int = 224):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        
        # Learnable 1D Positional Embeddings (+1 for the CLS token)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Initialize parameters
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, C, H, W)
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Dimensions must be divisible by patch size"
        
        # Project patches: (B, C, H, W) -> (B, D, H/P, W/P) -> (B, D, N) -> (B, N, D)
        patches = self.proj(x).flatten(2).transpose(1, 2)
        
        # Prepend CLS token: (B, 1, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, patches), dim=1)  # (B, N + 1, D)
        
        # Add positional embedding
        x = x + self.pos_embed
        return x

# ==========================================
# 2. Multi-Head Self-Attention (MHSA)
# ==========================================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # scaling factor (1 / sqrt(d_k))
        
        # Use a single large matrix to project the input into Q, K, and V simultaneously with an output dimension of 3 * embed_dim, and then split them.
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)  # Linear mapping after attention
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input x shape: (B, N, D)  -> N sequence length (1 + H*W/P^2)
        B, N, D = x.shape
        
        # 1. calculate Q, K, V
        # (B, N, 3*D) -> (B, N, 3, num_heads, head_dim) -> permutation
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)
        
        # 2. Calculate Attention Score (Q @ K transpose)
        # (B, num_heads, N, head_dim) x (B, num_heads, head_dim, N) -> (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 3. weight multiplies V
        # (B, num_heads, N, N) x (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)
        x = attn @ v
        
        # 4. Concatenate multi head back
        # (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, D)
        x = x.transpose(1, 2).reshape(B, N, D)
        
        # 5. final linear output
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# ==========================================
# 3. Transformer Block
# ==========================================
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_drop=dropout, proj_drop=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP (Feed Forward Network)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm (Residual Connection)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ==========================================
# 4. Integrated Vision Transformer Model
# ==========================================
class VisionTransformer(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, embed_dim: int = 768, img_size: int = 224, num_heads: int = 12, depth: int = 1, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.patch_embed = ViTPatchEmbedding(
            in_channels=in_channels, 
            patch_size=patch_size, 
            embed_dim=embed_dim, 
            img_size=img_size
        )
        
        # Stack multiple Transformer blocks as desired
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Image to Patch Embeddings (+ Positional Encoding)
        x = self.patch_embed(x)  # Shape: (B, N + 1, D)
        
        # 2. Pass sequence through Transformer Blocks
        for block in self.blocks:
            x = block(x)
            
        # 3. Final normalization
        x = self.norm(x)
        return x

# ==========================================
# 5. Synthetic Dataset and Generator
# ==========================================
class SyntheticViTDataset(Dataset):
    def __init__(self, num_samples: int = 100, in_channels: int = 3, height: int = 224, width: int = 224, embed_dim: int = 768, patch_size: int = 16):
        self.num_samples = num_samples
        self.in_channels = in_channels
        self.height = height
        self.width = width
        
        # Calculate expected output sequence length: (H/P * W/P) + 1
        num_patches = (height // patch_size) * (width // patch_size)
        self.seq_len = num_patches + 1
        self.embed_dim = embed_dim
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Generate random synthetic image: (C, H, W)
        image = torch.randn(self.in_channels, self.height, self.width)
        # Generate random target embedding for MSE loss matching: (seq_len, embed_dim)
        target = torch.randn(self.seq_len, self.embed_dim)
        return image, target

def make_dataset(num_samples: int = 100, batch_size: int = 8) -> DataLoader:
    """Generates a DataLoader using the synthetic dataset."""
    dataset = SyntheticViTDataset(num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# ==========================================
# 6. Training Loop Function
# ==========================================
def train_loop(model: nn.Module, input_dataset: DataLoader, learning_rate: float, epochs: int):
    """Executes a standard training loop utilizing MSE Loss."""
    # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Define Loss Function (Mean Squared Error)
    criterion = nn.MSELoss()
    
    # Ensure model is in training mode
    model.train()
    
    print(f"Beginning training across {epochs} epochs...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (images, targets) in enumerate(input_dataset):
            # Zero-out existing gradients
            optimizer.zero_grad()
            
            # Forward pass through complete ViT network
            outputs = model(images)  # Shape: (B, seq_len, embed_dim)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass and Optimization
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(input_dataset)
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.6f}")


