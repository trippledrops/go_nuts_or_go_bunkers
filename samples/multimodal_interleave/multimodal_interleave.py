from typing import Optional, Tuple
import torch
import torch.nn as nn
 
class MultimodalProjector(nn.Module):
  """
  project image embedding dim onto text embedding dim
  """
  def __init__(self, image_embed_dim: int, text_embed_dim: int):
    super().__init__()
    self.projector = nn.Sequential(
      nn.Linear(image_embed_dim, text_embed_dim),
      nn.GELU(),
      nn.Linear(text_embed_dim, text_embed_dim)
      )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.projector(x)

def interleave_text_and_image(
  input_ids: torch.Tensor,
  text_embeds: torch.Tensor,
  image_embeds: torch.Tensor,
  projector: nn.Module,
  image_token_id: int
) -> torch.Tensor:
  """
  input_ids: Original text token ID Tensor (B: batch_size, T: original text length)
  text_embeds: Tensors after text embedding layer (B, T, D: text latent layer dim)
  image_embeds: image embeddings for patches (B, N: number of patches, F: image latent dim)
  projector: projector Layer
  image_token_id: <image> placeholder ID (int) in the text
  """
  B, T, D = text_embeds.shape
  _, N, F = image_embeds.shape

  # 1. project
  projected_image_embeds = projector(image_embeds)

  # 2. find the location of <image> placeholder in each batch
  # image_mask shape: (B, T)
  image_mask = (input_ids == image_token_id)

  # 3. Calculate the new length of the new interleaved tensor
  # each <image> token is of length 1, the length will be (N - 1) longer after being replaced.
  # each batch might or might not include images. shape: (B, )
  num_image_per_sample = image_mask.sum(dim = -1)
  T_new = T + N - 1

  # init multimodal output
  device = text_embeds.device
  dtype = text_embeds.dtype
  interleaved_embeds = torch.zeros((B, T_new, D), device = device, dtype = dtype)

  # 4. sample wise interleave filling
  for i in range(B):
    # torch.where returns a tuple of tensors
    # look for the <image> placeholder in the current sample
    image_idx_list = torch.where(image_mask[i])[0]
    if image_idx_list.numel() == 0:
      interleaved_embeds[i, :T, :] = text_embeds[i]
      continue
    # find the location of the <image> token in the text
    img_idx = image_idx_list[0].item()
    # fill the part before image token
    if img_idx > 0:
      interleaved_embeds[i, :img_idx, : ]= text_embeds[i, :img_idx, :]
    interleaved_embeds[i, img_idx:img_idx + N, :] = projected_image_embeds[i]
    interleaved_embeds[i, img_idx + N:, :] = text_embeds[i, img_idx + 1:, :]
  return interleaved_embeds

def interleave_arbitrary_multimodal_batch(
    input_ids: torch.Tensor,                    # (B, T)
    text_embeddings: torch.Tensor,              # (B, T, D)
    attention_mask: Optional[torch.Tensor],     # (B, T), 1 for valid token, 0 for padding
    image_embeddings: Optional[torch.Tensor],     # (Total_Images, N, F)
    projector: nn.Module,
    image_token_id: int                         # <image> Token ID
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sequential Order: It assumes that the storage order of images in image_embeddings (Shape: [Total_Images, N, F]) is fully consistent with and in a strict one-to-one correspondence with the chronological order of encountering <image> tokens when scanning the text Token IDs from the first row to the last row of the batch, and from left to right.    
    Strict 1:1 Mapping: It assumes that the number of <image> placeholder tokens appearing in the text is exactly equal to the number of incoming global image embeddings (Total_Images)
    return:
        final_embeddings: (B, max_new_len, D)
        final_attention_mask: (B, max_new_len) Attention Mask after alignment
    """
    B, T, D = text_embeddings.shape
    device = text_embeddings.device
    dtype = text_embeddings.dtype
    
    # all valid if missing attention_mask
    if attention_mask is None:
        attention_mask = torch.ones((B, T), device=device, dtype=torch.long)
        
    # (Total_Images, N, F) -> (Total_Images, N, D)
    if image_embeddings is not None and image_embeddings.numel() > 0:
        proj_image_embeds = projector(image_embeddings)
        total_images, N, _ = proj_image_embeds.shape
    else:
        proj_image_embeds = None
        total_images, N = 0, 0

    new_sample_embeds_list = []
    new_sample_mask_list = []
    
    img_ptr = 0  # image pointer for globally processed image so far
    
    # Sample-wise assembly
    for i in range(B):
        # 1. filter padding in original text, only keep valid token
        valid_mask = attention_mask[i].bool()
        cur_input_ids = input_ids[i][valid_mask]
        cur_text_embeds = text_embeddings[i][valid_mask]
        
        # 2. find all <image> Tokens in current sample
        img_token_indices = torch.where(cur_input_ids == image_token_id)[0]
        num_images_in_sample = len(img_token_indices)
        
        # case A：no image, num_images_in_sample == 0
        if num_images_in_sample == 0:
            new_sample_embeds_list.append(cur_text_embeds)
            new_sample_mask_list.append(torch.ones(cur_text_embeds.shape[0], device=device, dtype=torch.long))
            continue
            
        # case B：1+ images
        cur_sample_pieces = []
        last_idx = 0
        
        for idx_tensor in img_token_indices:
            idx = idx_tensor.item()
            
            # copy the text before current <image>
            if idx > last_idx:
                cur_sample_pieces.append(cur_text_embeds[last_idx:idx])
                
            # insert image
            assert img_ptr < total_images, "total images are fewer than the # of <image> Tokens！"
            cur_sample_pieces.append(proj_image_embeds[img_ptr])
            img_ptr += 1
            
            # move to the token after current <image> Token
            last_idx = idx + 1
            
        # copy the text after the last <image>
        if last_idx < cur_text_embeds.shape[0]:
            cur_sample_pieces.append(cur_text_embeds[last_idx:])
            
        # splice current sample (dim=0)
        cur_assembled_embeds = torch.cat(cur_sample_pieces, dim=0)
        new_sample_embeds_list.append(cur_assembled_embeds)
        
        # mask the valid length to be 1 for current sample
        cur_mask = torch.ones(cur_assembled_embeds.shape[0], device=device, dtype=torch.long)
        new_sample_mask_list.append(cur_mask)

    # 3. calculate the max length of this batch, padding on the right.
    max_new_len = max(embed.shape[0] for embed in new_sample_embeds_list)
    
    # create final assembled result and attention mask.
    final_embeddings = text_embeddings.new_zeros((B, max_new_len, D))
    final_attention_mask = torch.zeros((B, max_new_len), device=device, dtype=torch.long)
    
    for i in range(B):
        cur_len = new_sample_embeds_list[i].shape[0]
        final_embeddings[i, :cur_len, :] = new_sample_embeds_list[i]
        final_attention_mask[i, :cur_len] = new_sample_mask_list[i]
        
    return final_embeddings, final_attention_mask

# Industry standard solution
# Explicit Image Indexing
# Assuming we pass the sample attribution labels of image_embeddings: image_sample_ids (Shape: [Total_Images])
# For example: image_sample_ids = torch.tensor([0, 1, 1]) means Image 0 goes to Sample 0, and Images 1 & 2 go to Sample 1.
def interleave_with_explicit_mapping(
    input_ids: torch.Tensor,              # (B, T)
    text_embeddings: torch.Tensor,        # (B, T, D)
    attention_mask: torch.Tensor,         # (B, T) - original text mask
    image_embeddings: torch.Tensor,         # (Total_Images, N, F)
    image_sample_ids: torch.Tensor,       # (Total_Images,) -> image to sample id in the batch 
    projector: nn.Module,
    image_token_id: int,                  # <image> Token ID
    num_patches: int                      # num of patches per image
) -> torch.Tensor:
    """
    Explicit Mapping
    """
    B, T, D = text_embeddings.shape
    N = num_patches
    device = text_embeddings.device
    dtype = text_embeddings.dtype
    
    if attention_mask is None:
      attention_mask = torch.ones((B, T), device=device, dtype=torch.long)
    # (Total_Images, N, F) -> (Total_Images, N, D)
    proj_image_embeds = projector(image_embeddings)
    total_images = proj_image_embeds.shape[0]
    
    # map image embeddings to sample id, isolate them to each container to avoid global mismatch
    image_by_sample = {i: [] for i in range(B)}
    for img_idx, sample_id in enumerate(image_sample_ids.tolist()):
        if sample_id < B:
            image_by_sample[sample_id].append(proj_image_embeds[img_idx])
            
    new_sample_embeds_list = []
    new_sample_mask_list = []  # 1D valid Mask for each sample
    for i in range(B):
        valid_mask = attention_mask[i].bool()
        cur_input_ids = input_ids[i][valid_mask]
        cur_text_embeds = text_embeddings[i][valid_mask]
        
        img_token_indices = torch.where(cur_input_ids == image_token_id)[0]
        num_images_in_sample = len(img_token_indices)
        
        if num_images_in_sample == 0:
            new_sample_embeds_list.append(cur_text_embeds)
            new_sample_mask_list.append(torch.ones(cur_text_embeds.shape[0], device=device, dtype=torch.long))
            continue
            
        cur_sample_pieces = []
        cur_mask_pieces = []
        last_idx = 0
        
        # find the image embeddings container for this sample
        cur_sample_images = image_by_sample[i]
        local_img_ptr = 0
        
        for idx_tensor in img_token_indices:
            idx = idx_tensor.item()
            
            # before image
            if idx > last_idx:
                cur_sample_pieces.append(cur_text_embeds[last_idx:idx])
                cur_mask_pieces.append(torch.ones(idx - last_idx, device=device, dtype=torch.long)) # Mask = 1
                
            if local_img_ptr < len(cur_sample_images):
                cur_sample_pieces.append(cur_sample_images[local_img_ptr])
                cur_mask_pieces.append(torch.zeros(N, device=device, dtype=torch.long)) # invalid part skip attention calculation, mark 0
                local_img_ptr += 1
            else:
                # fill 0 as Fallback if image embeddings is missing to avoid interrupting inference
                print(f"sample {i} is missing <image> token, filling 0。")
                cur_sample_pieces.append(torch.zeros((N, D), device=device, dtype=dtype))
                cur_mask_pieces.append(torch.zeros(N, device=device, dtype=torch.long))
                
            last_idx = idx + 1
            
        # after image
        if last_idx < cur_text_embeds.shape[0]:
            cur_sample_pieces.append(cur_text_embeds[last_idx:])
            cur_mask_pieces.append(torch.ones(cur_text_embeds.shape[0] - last_idx, device=device, dtype=torch.long))
            
            
        # splice
        new_sample_embeds_list.append(torch.cat(cur_sample_pieces, dim=0))
        new_sample_mask_list.append(torch.cat(cur_mask_pieces, dim=0))
        
    # dynamic Padding to max length in batch
    max_new_len = max(embed.shape[0] for embed in new_sample_embeds_list)
    final_embeddings = text_embeddings.new_zeros((B, max_new_len, D))
    final_attention_mask = torch.zeros((B, max_new_len), device=device, dtype=torch.long)
    
    for i in range(B):
        cur_len = new_sample_embeds_list[i].shape[0]
        final_embeddings[i, :cur_len, :] = new_sample_embeds_list[i]
        final_attention_mask[i, :cur_len] = new_sample_mask_list[i]
        
    return final_embeddings, final_attention_mask
