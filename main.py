import torch

from vit import VisionTransformer, make_dataset, train_loop

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


if __name__ == "__main__":
    main()

