import torch
import torch.nn as nn
from typing import Optional
from .embeddings import RotaryPositionalEmbedding
from .mha import MultiHeadAttention
from .ff import FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, rope: Optional[RotaryPositionalEmbedding] = None):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, rope)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,return_attention: bool = False):
        if return_attention:
            attn_out, attn_weights, raw_scores = self.self_attn(self.norm1(x), attention_mask, return_attention=True)
        else:
            attn_out = self.self_attn(self.norm1(x), attention_mask, return_attention=False)
            attn_weights = None
        x = x + attn_out
        x = x + self.feed_forward(self.norm2(x))
        if return_attention:
            return x, attn_weights, raw_scores
        return x


