import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from .embeddings import RotaryPositionalEmbedding


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1,rope: Optional[RotaryPositionalEmbedding] = None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model
        self.rope = rope
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,attention_mask: Optional[torch.Tensor] = None,return_attention: bool = False):
        """
        Args:
            x:              [batch, seq_len, d_model]
            attention_mask: [batch, seq_len]  1 = attend, 0 = ignore
            return_attention: return per-head attention weights
        Returns:
            output:       [batch, seq_len, d_model]
            attn_weights: [batch, num_heads, seq_len, seq_len]  (only if return_attention)
        """
        B, S, _ = x.shape
        #project and reshape → [B batch, num_heads, S seq len, head_dim]
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q, k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        raw_scores = scores.detach() #we never want gradients through this
        if attention_mask is not None:
            # [B, S] → [B, 1, 1, S] for broadcasting across heads and query positions
            mask = attention_mask[:, None, None, :]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_weights.nan_to_num(0.0) 
        attn_weights = self.attn_dropout(attn_weights)
        #weighted sum of values
        out = torch.matmul(attn_weights, v) # [B, heads, S, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        out = self.resid_dropout(self.out_proj(out))
        if return_attention:
            return out, attn_weights, raw_scores
        return out