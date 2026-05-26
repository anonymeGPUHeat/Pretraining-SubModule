import torch
import torch.nn as nn
from typing import Optional, Dict
from .embeddings import PTXEmbedding, RotaryPositionalEmbedding
from .pretraining_heads import MLMHead, ITCHead
from .ptx_encoder import EncoderLayer

class PTXEncoder(nn.Module):
    def __init__(
        self,vocab_size: int = 8000,d_model: int = 768,num_layers: int = 6,num_heads: int = 8,
        d_ff: int = 3072,dropout: float = 0.1,max_seq_length: int = 2048,padding_idx: int = 0,layer_norm_eps: float = 1e-12,):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        #token embeddings (√d scaling + LN + dropout)
        self.embedding = PTXEmbedding( vocab_size=vocab_size, hidden_size=d_model, dropout=dropout, layer_norm_eps=layer_norm_eps, padding_idx=padding_idx,)
        #shared RoPE for all attention layers
        self.rope = RotaryPositionalEmbedding(dim=d_model // num_heads, max_seq_length=max_seq_length,)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout, self.rope)
            for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Embedding):
            return 
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


    def forward( self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, return_attentions: bool = False, return_all_hidden_states: bool = False,) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids:                [batch, seq_len]
            attention_mask:           [batch, seq_len]  1 = real, 0 = padding
            return_attentions:        collect per-layer attention weights
            return_all_hidden_states: collect hidden states after every layer
        Returns:
            Dict with:
                'last_hidden_state':  [batch, seq_len, d_model]
                'attentions':         list of [batch, heads, seq_len, seq_len]
                'raw_scores'        : list of [B, heads, S, S]  pre-softmax   (if return_attentions)
                'all_hidden_states':  list of [batch, seq_len, d_model]
        """
        x = self.embedding(input_ids)
        all_attentions = [] if return_attentions else None
        all_raw_scores = [] if return_attentions else None
        all_hidden_states = [x] if return_all_hidden_states else None
        for layer in self.layers:
            if return_attentions:
                x, attn_w, raw_s = layer(x, attention_mask, return_attention=True)
                all_attentions.append(attn_w)
                all_raw_scores.append(raw_s)
            else:
                x = layer(x, attention_mask, return_attention=False)
            if return_all_hidden_states:
                all_hidden_states.append(x)

        x = self.final_norm(x)

        outputs: Dict[str, object] = {'last_hidden_state': x}
        if return_attentions:
            outputs['attentions'] = all_attentions
            outputs['raw_scores'] = all_raw_scores
        if return_all_hidden_states:
            outputs['all_hidden_states'] = all_hidden_states
        return outputs

    def count_parameters(self) -> Dict[str, int]:
        counts = {
            'embedding': sum(p.numel() for p in self.embedding.parameters()),
            'encoder_layers': sum(
                sum(p.numel() for p in layer.parameters()) for layer in self.layers
            ),
            'final_norm': sum(p.numel() for p in self.final_norm.parameters()),
        }
        counts['total'] = sum(counts.values())
        return counts
    



class PTXTransformerForPretraining(nn.Module):
    """
    PTX Transformer for multi-task pre-training.
    Bidirectional encoder (BERT/RoBERTa style) with:
        - Token embeddings + RoPE (no segment / type embeddings)
        - Pre-LN encoder stack
        - MLM head with embedding weight tying
        - ITC head for per-token instruction type classification (14 classes)
    
    Loss = weight_mlm * MLM_loss + weight_itc * ITC_loss
    """

    def __init__( self, vocab_size: int = 8000, d_model: int = 768, num_layers: int = 6, num_heads: int = 8,
        d_ff: int = 3072, dropout: float = 0.1, max_seq_length: int = 2048, padding_idx: int = 0,
        num_itc_classes: int = 14, use_itc: bool = True,):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_itc_classes = num_itc_classes
        self.use_itc = use_itc
        self.encoder = PTXEncoder(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers,num_heads=num_heads,
            d_ff=d_ff,dropout=dropout,max_seq_length=max_seq_length,padding_idx=padding_idx,)
        self.mlm_head = MLMHead(d_model, vocab_size)
        self.itc_head = ITCHead(d_model, num_itc_classes) if use_itc else None
        #weight tying (MLM decoder shares embedding weights)
        self.mlm_head.decoder.weight = self.encoder.embedding.token_embedding.weight

    def forward( self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
        return_attentions: bool = False,) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids:        [batch, seq_len]
            attention_mask:   [batch, seq_len]  1 = real, 0 = padding
            return_attentions: return per-layer attention weights for heatmaps

        Returns:
            Dict with:
                'logits':        [batch, seq_len, vocab_size]   (MLM logits)
                'itc_logits':    [batch, seq_len, num_itc_classes]  (only if use_itc=True)
                'hidden_states': [batch, seq_len, d_model]
                'attentions':    list of attention weight tensors  (optional)
        """
        encoder_out = self.encoder(input_ids, attention_mask=attention_mask, return_attentions=return_attentions,)
        hidden_states = encoder_out['last_hidden_state']
        #sauvegarder weights before heads for analysis
        mlm_logits = self.mlm_head(hidden_states)
        outputs: Dict[str, object] = {
            'logits': mlm_logits,
            'hidden_states': hidden_states,
        }
        if self.use_itc and self.itc_head is not None:
            itc_logits = self.itc_head(hidden_states)
            outputs['itc_logits'] = itc_logits
        if return_attentions:
            outputs['attentions'] = encoder_out['attentions']
            outputs['raw_scores'] = encoder_out['raw_scores']
        return outputs

    @torch.no_grad()
    def get_attention_maps(self,input_ids: torch.Tensor,attention_mask: Optional[torch.Tensor] = None,
        layer_idx: int = -1,raw: bool = True, ) -> torch.Tensor:
        assert -(self.encoder.num_layers) <= layer_idx < self.encoder.num_layers, \
        f"layer_idx {layer_idx} out of range for {self.encoder.num_layers} layers"
        outputs = self.forward(input_ids, attention_mask, return_attentions=True)
        key = 'raw_scores' if raw else 'attentions'
        return outputs[key][layer_idx]

    def count_parameters(self) -> Dict[str, int]:
        encoder_counts = self.encoder.count_parameters()
        tied_weight = self.encoder.embedding.token_embedding.weight
        mlm_params = sum(p.numel() for p in self.mlm_head.parameters() if p is not tied_weight)
        itc_params = sum(p.numel() for p in self.itc_head.parameters()) if self.itc_head is not None else 0
        counts = {
            **encoder_counts,
            'mlm_head': mlm_params,
            'total': encoder_counts['total'] + mlm_params + itc_params,
        }
        if self.itc_head is not None:
            counts['itc_head'] = itc_params
        return counts