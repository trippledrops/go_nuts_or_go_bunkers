import torch

from multimodal_interleave import MultimodalProjector, interleave_text_and_image, interleave_arbitrary_multimodal_batch, interleave_with_explicit_mapping
import math


def main():
  ######################## 1:1 ########################
  batch_size = 4
  text_seq_len = 10
  num_patches = 5
  image_embeds_dim = 1024
  text_embeds_dim = 4096
  image_token_id = 1500 # <image> token ID
  input_ids = torch.randint(100, 2000, (batch_size, text_seq_len))
  input_ids[0, 3] = image_token_id
  input_ids[1, 6] = image_token_id

  text_embeds = torch.randn(batch_size, text_seq_len, text_embeds_dim)
  image_embeds = torch.randn(batch_size, num_patches, image_embeds_dim)
  projector = MultimodalProjector(image_embeds_dim, text_embeds_dim)

  output = interleave_text_and_image(input_ids=input_ids, text_embeds=text_embeds, image_embeds=image_embeds, projector=projector, image_token_id=image_token_id)

  print(f"text shape: {text_embeds.shape}")
  print(f"image shape: {image_embeds.shape}")
  print(f"interleaved shape: {output.shape}")

  # validate data correctness
  assert torch.allclose(output[0, 0:3, :], text_embeds[0, 0:3, :]), "the front text is different"
  assert torch.allclose(output[0, 8:, :], text_embeds[0, 4:, :]), "the after text is different"

  #################### arbitrary ####################
  batch_size = 3
  orig_seq_len = 8
  num_patches = 4      # 4 patches / image
  image_embeds_dim = 1024
  text_embeds_dim = 4096
  img_token = 3000
  
  input_ids = torch.randint(100, 1000, (batch_size, orig_seq_len))
  # sample 0: 0 image
  # sample 1: 2 images at index 2 and 5
  input_ids[1, 2] = img_token
  input_ids[1, 5] = img_token
  # sample 2: 1 image at index 0
  input_ids[2, 0] = img_token
  
  total_imgs = (input_ids == img_token).sum().item() # 3 images
  
  text_embeds = torch.randn(batch_size, orig_seq_len, text_embeds_dim)
  image_embeds = torch.randn(total_imgs, num_patches, image_embeds_dim)
  proj = MultimodalProjector(image_embeds_dim, text_embeds_dim)
  
  out_embeds, out_mask = interleave_arbitrary_multimodal_batch(
      input_ids=input_ids,
      text_embeddings=text_embeds,
      attention_mask=None,
      image_embeddings=image_embeds,
      projector=proj,
      image_token_id=img_token
  )
  
  print(f"Out Embeddings Shape: {out_embeds.shape}")
  print(f"Out Attention Mask Shape: {out_mask.shape}")
  
  assert out_embeds.shape[1] == 14
  assert out_mask[0].sum().item() == 8
  assert out_mask[1].sum().item() == 14
  assert out_mask[2].sum().item() == 11

  ############ Explicit Image Indexing #############
  print("======= Explicit Image Indexing) =======\n")
    
  batch_size = 3
  seq_len = 8
  num_patches = 4
  image_embeds_dim = 1024
  text_embeds_dim = 4096
  image_token_id = 4500 # token ID
  
  input_ids = torch.randint(100, 1000, (batch_size, seq_len))
  # sample 0 : 0 images
  # sample 1: 2 images at index 2 and 5
  input_ids[1, 2] = image_token_id
  input_ids[1, 5] = image_token_id
  # sample 2: 2 image placeholders at index 1, 6, but we will only upload 1 later.
  input_ids[2, 1] = image_token_id
  input_ids[2, 6] = image_token_id
  
  image_sample_ids = torch.tensor([1, 1, 2], dtype=torch.long)
  total_loaded_images = len(image_sample_ids)
    
  text_embeddings = torch.randn(batch_size, seq_len, text_embeds_dim)
  image_embeddings = torch.randn(total_loaded_images, num_patches, image_embeds_dim)

  output_embeddings, output_attention_mask = interleave_with_explicit_mapping(
      input_ids=input_ids,
      text_embeddings=text_embeddings,
      attention_mask=None,
      image_embeddings=image_embeddings,
      image_sample_ids=image_sample_ids,
      projector=projector,
      image_token_id=image_token_id,
      num_patches=num_patches
  )
  
  # Verification
  print(f"Output Tensor Shape: {output_embeddings.shape}")
  # sample 0：len: 8 -> 8, pad 0
  # sample 1：len: 8 -> 8 - 2 + 2x4 = 14
  # sample 2：len: 8 -> 8 - 2 + 2x4 (but fill 0 as image missing) = 14
  # output_embeddings.shape = 14
  assert output_embeddings.shape == (batch_size, 14, text_embeds_dim), "dimension validation failed！"
  print("pass dimension alignment")

  assert torch.allclose(output_embeddings[0, :8, :], text_embeddings[0]), "sample 0 is wrong！"
  assert torch.all(output_embeddings[0, 8:, :] == 0), "sample 0 padding is wrong！"

  proj_img_embeds = projector(image_embeddings)
  assert torch.allclose(output_embeddings[1, 2:6, :], proj_img_embeds[0]), "sample 1 image 1 splice at wrong place！"

  missing_image_slot = output_embeddings[2, 9:13, :]
  assert torch.all(missing_image_slot == 0), "missing image is not filling 0！"
  print("success!")

if __name__ == "__main__":
    main()
