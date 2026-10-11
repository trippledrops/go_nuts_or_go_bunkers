import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class ClipInfoNCE(nn.Module):
  """
  Symmetric (CLIP-style) contrastive cross-entropy loss.
  """
  def __init__(
    self,
    init_temperature: float = 0.07,
    max_logit_scale: float = 100.0,
  ):
    super().__init__()
    self.log_temperature = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))
    self.max_log_scale = math.log(max_logit_scale)

  @property
  def logit_scale(self) -> torch.Tensor:
    return self.log_temperature.clamp(max=self.max_log_scale).exp()

  def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
    if image_embeds.shape != text_embeds.shape:
      raise ValueError(f"Shape mismatch: {image_embeds.shape} vs {text_embeds.shape}")
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    similarity_matrix = torch.matmul(image_embeds, text_embeds.t())
    # [B, B] similarity logits; row i = image i vs. all texts.
    logits_per_image = self.logit_scale * similarity_matrix
    logits_per_text = logits_per_image.t()

    labels = torch.arange(image_embeds.size(0), device = image_embeds.device) 

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)

    stats = retrieval_stats(logits_per_image.detach())
    print(
      f"i2t_acc {stats['i2t_acc']:.3f} t2i_acc {stats['t2i_acc']:.3f} | "
      f"pos_logit {stats['pos_logit']:6.2f} neg_logit {stats['neg_logit']:6.2f}"
    )
    print("\nlogits_per_image[:4, :4] from last step (diagonal = matches):")
    print(logits_per_image[:4, :4].detach())
    return (loss_i + loss_t) / 2.0

def make_pseudo_dataset(num_samples=256, latent_dim=16, image_dim=64,
                        text_dim=32, noise=0.1, seed=0) -> TensorDataset:
  """Paired "image"/"text" features generated from a shared latent z.
  image = z @ A + noise, text = z @ B + noise, so pair i is only recoverable
  via the shared z — a contrastive model must learn to align both views.
  """
  g = torch.Generator().manual_seed(seed)
  z = torch.randn(num_samples, latent_dim, generator=g)
  a = torch.randn(latent_dim, image_dim, generator=g)
  b = torch.randn(latent_dim, text_dim, generator=g)
  images = z @ a + noise * torch.randn(num_samples, image_dim, generator=g)
  texts = z @ b + noise * torch.randn(num_samples, text_dim, generator=g)
  return TensorDataset(images, texts)

def make_encoder(in_dim: int, embed_dim: int) -> nn.Module:
  return nn.Sequential(nn.Linear(in_dim, 128), nn.GELU(), nn.Linear(128, embed_dim))

@torch.no_grad()
def retrieval_stats(logits_per_image: torch.Tensor) -> dict[str, float]:
  labels = torch.arange(logits_per_image.size(0))
  off_diag_mask = ~torch.eye(logits_per_image.size(0), dtype=torch.bool)
  return {
      "i2t_acc": (logits_per_image.argmax(1) == labels).float().mean().item(),
      "t2i_acc": (logits_per_image.argmax(0) == labels).float().mean().item(),
      "pos_logit": logits_per_image.diag().mean().item(),
      "neg_logit": logits_per_image[off_diag_mask].mean().item(),
  }
