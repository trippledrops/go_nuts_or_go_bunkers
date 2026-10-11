import torch

#from model import LinearRegression, make_dataset, train
from clip_loss import ClipInfoNCE, make_pseudo_dataset, make_encoder, retrieval_stats
from torch.utils.data import DataLoader
import math


def main():
  torch.manual_seed(0)
  num_steps, batch_size, embed_dim = 5, 64, 32
  dataset = make_pseudo_dataset()
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  image_dim = dataset.tensors[0].shape[1]
  text_dim = dataset.tensors[1].shape[1]
  image_encoder = make_encoder(image_dim, embed_dim)
  text_encoder = make_encoder(text_dim, embed_dim)
  loss_fn = ClipInfoNCE()
  # print(type(loss_fn).__module__, [n for n, _ in loss_fn.named_parameters()], vars(loss_fn).keys())
  # τ goes in its own group with no weight decay.
  optimizer = torch.optim.AdamW([
      {"params": list(image_encoder.parameters()) + list(text_encoder.parameters()),
       "weight_decay": 0.1},
      {"params": loss_fn.parameters(), "weight_decay": 0.0},
  ], lr=1e-2)
  print(f"dataset: {len(dataset)} pairs | image_dim={image_dim} "
        f"text_dim={text_dim} embed_dim={embed_dim} batch={batch_size}")
  print(f"chance-level loss = log(B) = {math.log(batch_size):.4f}")
  print(f"init τ = {loss_fn.log_temperature.item():.4f} "
        f"(logit_scale = {loss_fn.logit_scale.item():.2f})\n")
  data_iter = iter(loader)
  for step in range(1, num_steps + 1):
    try:
      images, texts = next(data_iter)
    except StopIteration:
      data_iter = iter(loader)
      images, texts = next(data_iter)
    loss = loss_fn(image_encoder(images), text_encoder(texts))
    optimizer.zero_grad()
    loss.backward()
    tau_grad = loss_fn.log_temperature.grad.item()
    optimizer.step()
    #stats = retrieval_stats(logits_per_image.detach())
    print(
        f"step {step} | loss {loss.item():.4f} | "
        f"τ {loss_fn.log_temperature.item():.4f} (∂L/∂τ {tau_grad:+.4f}) | "
        f"scale {loss_fn.logit_scale.item():6.2f}"
    )



if __name__ == "__main__":
    main()

