import math
import torch
import torch.nn as nn
from typing import Dict, Optional


class PretrainingObjectives:
    """
    Multi-task pretraining loss: MLM + ITC.
    Both losses use CrossEntropyLoss with ignore_index=-100 so that
    padding and non-target tokens are excluded automatically.
    Default weights: mlm=1.0, itc=1.0 (equal weighting, total ~2.0).
    """
    def __init__(self, label_smoothing: float = 0.0):
        self.ce_loss = nn.CrossEntropyLoss( ignore_index=-100, label_smoothing=label_smoothing,)

    def mlm_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))

    def itc_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Instruction Type Classification loss.
        Args:
            logits: [batch, seq_len, num_itc_classes]
            labels: [batch, seq_len] with values in [0..13] or -100 (ignore)
        """
        return self.ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))

    def compute_loss(self,mlm_logits: Optional[torch.Tensor] = None,mlm_labels: Optional[torch.Tensor] = None,itc_logits: Optional[torch.Tensor] = None,itc_labels: Optional[torch.Tensor] = None,weights: Optional[Dict[str, float]] = None,) -> Dict[str, torch.Tensor]:
        """
        Compute weighted multi-task loss: w_mlm * L_mlm + w_itc * L_itc.
        Args:
            mlm_logits/mlm_labels: MLM head outputs and targets
            itc_logits/itc_labels: ITC head outputs and targets
            weights: dict with 'mlm' and 'itc' keys (default both 1.0)
        Returns:
            Dict with 'mlm_loss', 'itc_loss', 'total_loss' keys
        """
        if weights is None:
            weights = {'mlm': 1.0, 'itc': 1.0}
        
        losses = {}
        total = None
        if mlm_logits is not None and mlm_labels is not None:
            mlm = self.mlm_loss(mlm_logits, mlm_labels)
            losses['mlm_loss'] = mlm
            w = weights.get('mlm', 1.0)
            total = w * mlm if total is None else total + w * mlm
        
        
        if itc_logits is not None and itc_labels is not None:
            itc = self.itc_loss(itc_logits, itc_labels)
            losses['itc_loss'] = itc
            w = weights.get('itc', 1.0)
            total = w * itc if total is None else total + w * itc
        
        if total is None:
            raise ValueError(
                "compute_loss called with no valid (logits, labels) pair. "
                "At least one of MLM or ITC must provide both logits and labels."
            )

        losses['total_loss'] = total
        return losses



@torch.no_grad()
def compute_mlm_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Accuracy on masked positions only.
    Args:
        logits: [batch, seq_len, vocab_size]
        labels: [batch, seq_len]  -100 for non-masked
    Returns:
        float accuracy in [0, 1]
    """
    mask = labels != -100
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    correct = (preds == labels) & mask
    return correct.sum().item() / mask.sum().item()


@torch.no_grad()
def compute_itc_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Accuracy on instruction tokens only (where label != -100).
    Args:
        logits: [batch, seq_len, num_itc_classes]
        labels: [batch, seq_len]  -100 for non-instruction tokens
    Returns:
        float accuracy in [0, 1]
    """
    mask = labels != -100
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    correct = (preds == labels) & mask
    return correct.sum().item() / mask.sum().item()


@torch.no_grad()
def compute_mlm_perplexity(loss: torch.Tensor | float) -> float:
    if isinstance(loss, torch.Tensor):
        loss = loss.item()
    return math.exp(loss) if loss < 100.0 else float('inf')