import torch
import torch.nn as nn
import math


class RotaryPositionalEmbedding(nn.Module):
    """
    Applied in attention layer to query and key vectors
    Encodes relative positions naturally without learned parameters
    """
    def __init__(self, dim: int, max_seq_length: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.base = base
        # precompute inverse frequencies: 1 / (base ^ (2i / dim)) for i in [0, dim/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        #cache cos/sin values for efficiency
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
    
    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Compute and cache cos/sin values for given sequence length."""
        if (seq_len > self._seq_len_cached or self._cos_cached is None or self._cos_cached.device != device or self._cos_cached.dtype != dtype):
            self._seq_len_cached = seq_len
            # ceate position indices [0, 1, 2, ..., seq_len-1]
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            #compute frequencies: outer product of positions and inverse frequencies
            freqs = torch.outer(t, self.inv_freq)
            # concat to match head dimension
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary positional embedding to query and key tensors.
        
        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_heads, seq_len, head_dim]
        
        Returns:
            Tuple of (rotated_q, rotated_k) with same shape as inputs
        """
        seq_len = q.shape[2]
        self._update_cos_sin_cache(seq_len, q.device, q.dtype)
        cos = self._cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self._sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot

"""
 Args:
            dim: Dimension of each attention head (hidden_size // num_heads)
            max_seq_length: Maximum sequence length to support
            base: Base for frequency computation (10000 is standard) this parameter mean the frequency of the 
            positional encoding, higher base means slower frequency decay, which can help capture longer-range 
            dependencies. However, it may also make the model less sensitive to shorter-range patterns. 
            A smaller base will result in faster frequency decay, which can help capture fine-grained positional 
            information but may struggle with longer sequences. The optimal value often depends on the specific dataset 
            and task, and may require experimentation to find the best balance.
"""



#here the goal is to convert token IDs to dense embeddings with normalization and dropout
class PTXEmbedding(nn.Module):
    def __init__(self,vocab_size: int,hidden_size: int,dropout: float = 0.1,layer_norm_eps: float = 1e-12,padding_idx: int = 0,):
        super().__init__()  
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.token_embedding = nn.Embedding(vocab_size,hidden_size,padding_idx=padding_idx)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if self.token_embedding.padding_idx is not None:
            with torch.no_grad():
                #padding_idx=0 means row 0 of the matrix is frozen to zeros and receives no gradient
                # so padding tokens always map to the zero vector
                self.token_embedding.weight[self.token_embedding.padding_idx].fill_(0) 
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs [batch_size, seq_len]
        Returns:
            embeddings: Token embeddings [batch_size, seq_len, hidden_size]
        """
        embeddings = self.token_embedding(input_ids)
        embeddings = embeddings * math.sqrt(self.hidden_size)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

