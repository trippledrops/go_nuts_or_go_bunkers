import torch

from vit import VisionTransformer, make_dataset, train_loop
from vit_rope import ViTWith2DRoPE

def main():
    # Setup hyperparameters
    in_channels = 3
    patch_size = 16
    embed_dim = 768
    num_heads = 12  # Ensure embed_dim is divisible by num_heads (768 / 12 = 64)
    img_size = 224
    learning_rate = 1e-3
    epochs = 5
    num_samples = 40
    batch_size = 8
    
    # Instantiate the complete integrated Vision Transformer model
    model = VisionTransformer(
        in_channels=in_channels, 
        patch_size=patch_size, 
        embed_dim=embed_dim, 
        img_size=img_size,
        num_heads=num_heads,
        depth=1,         # 1 transformer block layer
        mlp_ratio=4.0,
        dropout=0.1
    )
    
    # Generate the synthetic data loaders
    dataloader = make_dataset(num_samples=num_samples, batch_size=batch_size)
    
    # Run the training loop
    train_loop(
        model=model, 
        input_dataset=dataloader, 
        learning_rate=learning_rate, 
        epochs=epochs
    )
    print("Training finished successfully!")
    ############ RoPE ##############
    vit = ViTWith2DRoPE(patch_size=16, embed_dim=128, num_heads=4)
    # Test Case A: Standard Square Image (224x224)
    img_square = torch.randn(2, 3, 224, 224)
    out_square = vit(img_square)
    # Patches: (224/16) * (224/16) = 196. Plus CLS token = 197 tokens.
    print(f"Square Image Output Shape: {out_square.shape}")  # Expected: [2, 197, 128]

    # Test Case B: Panoramic/Wide Image with Variable Aspect Ratio (112x448)
    img_wide = torch.randn(2, 3, 112, 448)
    out_wide = vit(img_wide)
    # Patches: (112/16) * (448/16) = 7 * 28 = 196. Plus CLS token = 197 tokens.
    print(f"Wide Image Output Shape:   {out_wide.shape}")    # Expected: [2, 197, 128]

if __name__ == "__main__":
    main()
