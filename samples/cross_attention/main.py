import torch

from cross_attention import MultiHeadCrossAttention
import math


def main():
  cross_atten = MultiHeadCrossAttention(d_model = 256, num_heads = 8)
  text_embeds = torch.randn(4, 12, 256)
  image_embeds = torch.randn(4, 12, 256)
  context = cross_atten(text_embeds, image_embeds)
  print("context shape: ", context.shape)

if __name__ == "__main__":
    main()

