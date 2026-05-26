import torch
import random
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from ..preprocessing.itc_labels import detect_instruction_spans, ITC_IGNORE


# Masking granularity:
#   * For real instruction spans (opcode … semicolon), the masking decision
#     is taken *per instruction*: with probability ``mask_prob`` the whole
#     instruction is selected.  Within a selected instruction every eligible
#     token undergoes the standard 80 / 10 / 10 replacement strategy
#     (mask / random-replace / keep-original).
#   * For non-instruction spans (directives, labels, structural tokens), we
#     fall back to independent per-token masking so the model still learns
#     to reconstruct those too (they're short, so boundary concerns don't
#     apply).
#
# This ensures the model can never "cheat" by seeing half an instruction
# and trivially filling in a modifier,it must rely on the surrounding
# instruction context.

def apply_mlm_masking(token_ids: List[int],attention_mask: List[int],vocab_size: int,mask_token_id: int,pad_token_id: int = 0,mask_prob: float = 0.15,random_replace_prob: float = 0.10,keep_original_prob: float = 0.10,special_ids: Optional[set] = None,instruction_spans: Optional[List] = None,) -> Dict[str, List[int]]:
    if special_ids is None:
        special_ids = set()

    masked_ids = list(token_ids)
    labels = [-100] * len(token_ids)
    p_mask_token = 1.0 - random_replace_prob - keep_original_prob   # 0.80

    def _mask_position(i: int) -> None:
        if attention_mask[i] == 0:
            return
        if token_ids[i] in special_ids:
            return
        labels[i] = token_ids[i]
        r = random.random()
        if r < p_mask_token:
            masked_ids[i] = mask_token_id
        elif r < p_mask_token + random_replace_prob:
            masked_ids[i] = random.randint(5, vocab_size - 1)
        # else: keep original (labels[i] is still set so it counts in loss)

    if instruction_spans is not None:
        for start, end, is_instruction in instruction_spans:
            if is_instruction:
                if random.random() < mask_prob:
                    for i in range(start, end):
                        _mask_position(i)
            else:
                for i in range(start, end):
                    if random.random() < mask_prob:
                        _mask_position(i)
    else:
        for i in range(len(token_ids)):
            if random.random() < mask_prob:
                _mask_position(i)

    return {'input_ids': masked_ids, 'mlm_labels': labels}


def collate_with_mlm( batch: List[Dict[str, torch.Tensor]], vocab_size: int,mask_token_id: int,pad_token_id: int = 0,mask_prob: float = 0.15,special_ids: Optional[set] = None,tokenizer=None, use_itc: bool = True, masking_mode: str = 'instruction',) -> Dict[str, torch.Tensor]:
    """
    Collate function that:
      1. Detects instruction spans (shared pass for both ITC labels and masking).
      2. Generates ITC labels from the ORIGINAL (pre-masking) token IDs.
      3. Applies MLM masking according to ``masking_mode``.

    Args:
        masking_mode: One of
            - ``'token'``       – classic per-token random masking (no span awareness)
            - ``'instruction'`` – mask entire instructions (current default)
        use_itc: Whether to produce ITC labels (only meaningful when
                 masking_mode is ``'instruction'`` and a tokenizer is provided).

    ITC labels are a property of the token *position* 
    they don't change when a token is replaced by ``<mask>``.
    """
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    all_itc_labels = []

    for sample in batch:
        ids = sample['input_ids'].tolist()
        mask = sample['attention_mask'].tolist()
        itc_labels = None
        spans = None

        if masking_mode == 'instruction' and tokenizer is not None:
            # Detect instruction spans for instruction-level masking + ITC
            itc_labels_list, spans = detect_instruction_spans(
                token_ids=ids, tokenizer=tokenizer,
            )
            if use_itc:
                all_itc_labels.append(itc_labels_list)
        # else: spans stays None → apply_mlm_masking falls back to per-token

        mlm_result = apply_mlm_masking(token_ids=ids,attention_mask=mask,vocab_size=vocab_size,mask_token_id=mask_token_id, pad_token_id=pad_token_id,mask_prob=mask_prob, special_ids=special_ids, instruction_spans=spans,)
        all_input_ids.append(mlm_result['input_ids'])
        all_labels.append(mlm_result['mlm_labels'])
        all_attention_mask.append(sample['attention_mask'])

    result = {
        'input_ids': torch.tensor(all_input_ids, dtype=torch.long),
        'attention_mask': torch.stack(all_attention_mask),
        'mlm_labels': torch.tensor(all_labels, dtype=torch.long),
    }

    if all_itc_labels:
        result['itc_labels'] = torch.tensor(all_itc_labels, dtype=torch.long)

    return result


def create_pretraining_dataloader(dataset,tokenizer,batch_size: int = 16,mask_prob: float = 0.15,shuffle: bool = True,num_workers: int = 10,pin_memory: bool = True,seed: int = 42,use_itc: bool = True, masking_mode: str = 'instruction',) -> DataLoader:
    """
    Create a DataLoader with the appropriate collate function.

    Args:
        masking_mode: ``'token'`` for classic per-token MLM,
                      ``'instruction'`` for instruction-boundary-aware MLM.
        use_itc: Whether to generate ITC labels (only used when masking_mode='instruction').
    """
    mask_token_id = tokenizer.sp.PieceToId('<mask>')
    pad_token_id = tokenizer.pad_id
    vocab_size = tokenizer.vocab_size
    special_ids = {pad_token_id,tokenizer.unk_id,tokenizer.bos_id, tokenizer.eos_id, mask_token_id,}

    def collate_fn(batch):
        return collate_with_mlm(batch, vocab_size=vocab_size, mask_token_id=mask_token_id,pad_token_id=pad_token_id,mask_prob=mask_prob,special_ids=special_ids, tokenizer=tokenizer, use_itc=use_itc, masking_mode=masking_mode,)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=pin_memory,collate_fn=collate_fn,generator=generator if shuffle else None,drop_last=True,)
