import torch
import torch.nn as nn


class MLMHead(nn.Module):
    """Masked Language Model prediction head: hidden → Dense → GELU → LN → vocab logits"""
    def __init__(self, d_model: int, vocab_size: int, layer_norm_eps: float = 1e-12):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias #we added bias for weight tying 

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:   hidden_states [batch, seq_len, d_model]
        Returns: logits       [batch, seq_len, vocab_size]
        """
        x = self.layer_norm(self.act(self.dense(hidden_states)))
        return self.decoder(x)


class ITCHead(nn.Module):
    """
    Classes:
        0  GLOBAL_LOAD       6  ARITHMETIC        12 ATOMIC_REDUCE
        1  GLOBAL_STORE       7  LOGIC_BITWISE     13 SPECIAL
        2  SHARED_LOAD        8  COMPARISON
        3  SHARED_STORE       9  CONTROL_FLOW
        4  LOCAL_LOAD        10  SYNC_BARRIER
        5  LOCAL_STORE       11  CONVERSION_MOVE
    """
    def __init__(self, d_model: int, num_classes: int = 14, layer_norm_eps: float = 1e-12):
        super().__init__()
        self.num_classes = num_classes
        self.dense = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(self.act(self.dense(hidden_states)))
        return self.classifier(x)





