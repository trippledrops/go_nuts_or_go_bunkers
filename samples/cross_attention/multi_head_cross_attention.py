import math
import torch
import torch.nn as nn

class MultiHeadCrossAttention(nn.Module):
  def __init__(self, d_model, num_heads):
    """
    d_model: input and output embedding dim
    """
    super().__init__()
    assert d_model % num_heads == 0, "the dim of d_model has to be divided by num_heads"
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = d_model // num_heads

    # Linear projection Layer
    # query is from text, k, v are from image
    self.q_linear = nn.Linear(d_model, d_model)
    self.k_linear = nn.Linear(d_model, d_model)
    self.v_linear = nn.Linear(d_model, d_model)

    # Linear output Layer
    self.out_proj = nn.Linear(d_model, d_model)
    self.scale = math.sqrt(self.d_k)

  def forward(self, text_embeds, image_embeds, mask = None):
    """
    text_embeds: [batch_size, seq_len_q, d_model]
    image_embeds: [batch_size, seq_len_kv, d_model]
    mask: [batch_size, 1, seq_len_q, seq_len_kv]
    return: [batch_size, seq_len_q, d_model] same with text_embeds
    """
    batch_size = text_embeds.size(0)
    # linear projection
    Q = self.q_linear(text_embeds)
    K = self.k_linear(image_embeds)
    V = self.v_linear(image_embeds)

    # split heads and transpose
    Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
    # apply mask before softmax
    # padding mask, causal / lookahead mask / fine grained alignment (eg. only interact with region of interest, decoupled training to defer early interaction)
    if mask is not None:
      scores = scores.masked_fill(mask == 0, -1e9)
    attention_weights = torch.softmax(scores, dim=-1)
    context = torch.matmul(attention_weights, V)

    # concate multiheads back to original shape
    context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

    output = self.out_proj(context)
    return output


