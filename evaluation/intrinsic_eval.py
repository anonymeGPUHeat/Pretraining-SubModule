import sys
import torch
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

#to allow running as a module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.architecture.complete_model import PTXTransformerForPretraining
from model.tokenizer.tokenizer import PTXTokenizer
from model.preprocessing.dataset_builder import PTXDataset
from evaluation.eval_utils import (
    load_test_dataset, decode_chunks, extract_labeled_instructions,
    DEFAULT_DATA_DIR, DEFAULT_CACHE_DIR, DEFAULT_TOKENIZER,
)


def load_model(checkpoint_path: str,tokenizer_path: str,device: str = 'cuda',) -> Tuple[PTXTransformerForPretraining, PTXTokenizer, Dict]:
    """
    load a pretrained ptx transformer from a checkpoint file.
      - 'model_state_dict': the full model weights
      - 'config': dict with vocab_size, d_model, num_layers, num_heads, d_ff, etc.
    args:
        checkpoint_path: path to .pt checkpoint (best_model.pt, final_model.pt, etc.)
        tokenizer_path:  path to sentencepiece .model file
        device:          'cuda' or 'cpu'
    returns:
        (model, tokenizer, config_dict)
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    tokenizer = PTXTokenizer(tokenizer_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    state_dict = checkpoint['model_state_dict']
    use_itc = any(k.startswith('itc_head') for k in state_dict)
    model = PTXTransformerForPretraining(vocab_size=config.get('vocab_size', tokenizer.vocab_size),
        d_model=config.get('d_model', 768),num_layers=config.get('num_layers', 12),
        num_heads=config.get('num_heads', 8),d_ff=config.get('d_ff', 3072),
        dropout=0.0,  # no dropout during evaluation!!!
        max_seq_length=config.get('max_seq_length', 2048),padding_idx=tokenizer.pad_id,
        use_itc=use_itc,).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"loaded model from {checkpoint_path}")
    print(f"  config: {config}")
    print(f"  device: {device}")
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer, config


def _encode_ptx(tokenizer: PTXTokenizer,text: str,max_length: int = 1024,device: str = 'cpu',) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    encode a ptx text snippet into input_ids and attention_mask tensors.
    truncates to max_length and pads with pad_id.
    returns tensors of shape [1, max_length] ready for the model.
    """
    ids = tokenizer.encode(text)
    ids = ids[:max_length]
    actual_len = len(ids)
    padding = [tokenizer.pad_id] * (max_length - actual_len)
    ids = ids + padding
    mask = [1] * actual_len + [0] * (max_length - actual_len)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([mask], dtype=torch.long, device=device)
    return input_ids, attention_mask



#1: masked token recovery
def evaluate_masked_recovery( model: PTXTransformerForPretraining,tokenizer: PTXTokenizer, ptx_snippets: List[str], mask_ratio: float = 0.15,max_length: int = 2048, device: str = 'cuda',) -> Dict[str, float]:
    """
    for each ptx snippet, we:
      1. tokenize the text
      2. randomly mask ~15% of non-padding, non-special tokens
      3. feed the masked version to the model
      4. check if the model's top-1 prediction matches the original token
    args:
        model:        pretrained PTXTransformerForPretraining in eval mode
        tokenizer:    PTXTokenizer with sentencepiece backend
        ptx_snippets: list of raw ptx code strings
        mask_ratio:   fraction of tokens to mask (default 0.15 like bert)
        max_length:   max sequence length for encoding
        device:       torch device
    returns:
        dict with 'accuracy', 'correct', 'total', 'top5_accuracy'
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    mask_token_id = tokenizer.sp.PieceToId('<mask>')
    # special token ids that we never mask
    special_ids = {tokenizer.pad_id,tokenizer.unk_id,tokenizer.bos_id,
        tokenizer.eos_id, mask_token_id,}
    correct = 0
    correct_top5 = 0
    total = 0
    for snippet in ptx_snippets:
        input_ids, attention_mask = _encode_ptx(tokenizer, snippet, max_length, device)
        original_ids = input_ids.clone()
        seq_len = attention_mask.sum().item()  #number of real tokens
        #select positions to mask (skip special tokens)
        mask_positions = []
        for i in range(int(seq_len)):
            token_id = original_ids[0, i].item()
            if token_id not in special_ids:
                mask_positions.append(i)
        # randomly choose ~mask_ratio of eligible positions
        import random
        num_to_mask = max(1, int(len(mask_positions) * mask_ratio))
        chosen = random.sample(mask_positions, min(num_to_mask, len(mask_positions)))
        if not chosen:
            continue
        # apply masking (80% mask, 10% random, 10% keep — same as training)
        for pos in chosen:
            r = random.random()
            if r < 0.8:
                input_ids[0, pos] = mask_token_id
            elif r < 0.9:
                input_ids[0, pos] = random.randint(5, tokenizer.vocab_size - 1)
            # else keep original (10%)
        #forward pass
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs['logits']  # [1, seq_len, vocab_size]
        # check predictions at masked positions
        for pos in chosen:
            true_id = original_ids[0, pos].item()
            pred_id = logits[0, pos].argmax(dim=-1).item()
            top5_ids = logits[0, pos].topk(5).indices.tolist()
            total += 1
            if pred_id == true_id:
                correct += 1
            if true_id in top5_ids:
                correct_top5 += 1

    accuracy = correct / total if total > 0 else 0.0
    top5_accuracy = correct_top5 / total if total > 0 else 0.0
    print(f"\n--- masked token recovery ---")
    print(f"  snippets evaluated: {len(ptx_snippets)}")
    print(f"  masked positions:   {total}")
    print(f"  top-1 accuracy:     {accuracy:.4f} ({correct}/{total})")
    print(f"  top-5 accuracy:     {top5_accuracy:.4f} ({correct_top5}/{total})")
    return {
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'correct': correct,
        'total': total,
    }


# evaluation 2: address space separation
def evaluate_address_space_separation( model: PTXTransformerForPretraining, tokenizer: PTXTokenizer,
    test_dataset: Optional[PTXDataset] = None,
    max_length: int = 2048, device: str = 'cuda',) -> Dict[str, float]:
    """
    measure how well the encoder separates global vs shared memory operations.
    
    if test_dataset is provided, real memory instructions are extracted and
    auto-labeled by address space. otherwise falls back to synthetic snippets.
    
    lower cosine similarity = better separation between address spaces.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    global_snippets = []
    shared_snippets = []

    # try to extract real instructions from test data
    if test_dataset is not None:
        texts, labels = extract_labeled_instructions(
            test_dataset, tokenizer, task='address_space', max_per_class=30,
        )
        for text, label in zip(texts, labels):
            if label == 0:  # global
                global_snippets.append(text)
            elif label == 1:  # shared
                shared_snippets.append(text)

    # ensure we have enough samples (supplement with synthetics if needed)
    if len(global_snippets) < 4:
        global_snippets.extend([
            "ld.global.f32 %f1, [%rd1];",
            "ld.global.u32 %r1, [%rd2];",
            "ld.global.f64 %fd1, [%rd3];",
            "st.global.f32 [%rd4], %f2;",
            "ld.global.v4.f32 {%f3, %f4, %f5, %f6}, [%rd5];",
            "st.global.u32 [%rd6], %r2;",
            "ld.global.b32 %r3, [%rd7];",
            "ld.global.f32 %f7, [%rd8+16];",
        ])
    if len(shared_snippets) < 4:
        shared_snippets.extend([
            "ld.shared.f32 %f1, [%rd1];",
            "ld.shared.u32 %r1, [%rd2];",
            "ld.shared.f64 %fd1, [%rd3];",
            "st.shared.f32 [%rd4], %f2;",
            "ld.shared.v4.f32 {%f3, %f4, %f5, %f6}, [%rd5];",
            "st.shared.u32 [%rd6], %r2;",
            "ld.shared.b32 %r3, [%rd7];",
            "ld.shared.f32 %f7, [%rd8+16];",
        ])

    print(f"  using {len(global_snippets)} global + {len(shared_snippets)} shared snippets")

    def _get_embeddings(snippets: List[str]) -> np.ndarray:
        """extract mean-pooled hidden states from encoder for each snippet."""
        embeddings = []
        with torch.no_grad():
            for text in snippets:
                input_ids, attention_mask = _encode_ptx(tokenizer, text, max_length, device)
                outputs = model(input_ids, attention_mask=attention_mask)
                hidden = outputs['hidden_states']  # [1, seq_len, d_model]
                # mean pool over real tokens only (exclude padding)
                mask_expanded = attention_mask.unsqueeze(-1).float()  # [1, seq_len, 1]
                pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
                embeddings.append(pooled[0].cpu().numpy())
        return np.array(embeddings)

    global_embs = _get_embeddings(global_snippets)
    shared_embs = _get_embeddings(shared_snippets)

    # inter-class similarity: mean global embedding vs mean shared embedding
    global_mean = global_embs.mean(axis=0)
    shared_mean = shared_embs.mean(axis=0)
    inter_cosine = float( np.dot(global_mean, shared_mean)
        / (np.linalg.norm(global_mean) * np.linalg.norm(shared_mean) + 1e-8))

    # intra-class similarity: average pairwise cosine within each class
    def _avg_pairwise_cosine(embs: np.ndarray) -> float:
        n = len(embs)
        if n < 2:
            return 1.0
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                cos = np.dot(embs[i], embs[j]) / (
                    np.linalg.norm(embs[i]) * np.linalg.norm(embs[j]) + 1e-8
                )
                sims.append(cos)
        return float(np.mean(sims))
    global_intra = _avg_pairwise_cosine(global_embs)
    shared_intra = _avg_pairwise_cosine(shared_embs)
    print(f"\n--- address space separation ---")
    print(f"  inter-class cosine (global vs shared): {inter_cosine:.4f}")
    print(f"  intra-class cosine (global):           {global_intra:.4f}")
    print(f"  intra-class cosine (shared):           {shared_intra:.4f}")
    print(f"  separation gap:                        {global_intra + shared_intra - 2 * inter_cosine:.4f}")
    return {
        'inter_cosine': inter_cosine,
        'global_intra': global_intra,
        'shared_intra': shared_intra,
    }


# evaluation 3: instruction type separation (memory vs compute)
def evaluate_instruction_type_separation(  model: PTXTransformerForPretraining,tokenizer: PTXTokenizer,
    test_dataset: Optional[PTXDataset] = None,
    max_length: int = 2048,device: str = 'cuda',) -> Dict[str, float]:
    """
    measure how well the encoder separates memory ops from compute ops.
    
    if test_dataset is provided, real instructions are extracted and auto-labeled.
    otherwise falls back to synthetic snippets.
    
    returns cosine similarity between memory and compute centroids (lower = better).
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    memory_snippets = []
    compute_snippets = []

    # try to extract real instructions from test data
    if test_dataset is not None:
        texts, labels = extract_labeled_instructions(
            test_dataset, tokenizer, task='instruction_type', max_per_class=30,
        )
        for text, label in zip(texts, labels):
            if label == 1:  # memory
                memory_snippets.append(text)
            elif label == 0:  # compute
                compute_snippets.append(text)

    # supplement with synthetics if needed
    if len(memory_snippets) < 4:
        memory_snippets.extend([
            "ld.global.f32 %f1, [%rd1];",
            "st.global.f32 [%rd2], %f2;",
            "ld.shared.f32 %f3, [%rd3];",
            "st.shared.u32 [%rd4], %r1;",
            "ld.param.u64 %rd5, [<PARAM_0>];",
            "ld.global.v2.f32 {%f4, %f5}, [%rd6];",
        ])
    if len(compute_snippets) < 4:
        compute_snippets.extend([
            "add.f32 %f1, %f2, %f3;",
            "mul.f32 %f4, %f5, %f6;",
            "fma.rn.f32 %f7, %f8, %f9, %f10;",
            "mad.lo.s32 %r1, %r2, %r3, %r4;",
            "sub.f32 %f11, %f12, %f13;",
            "shr.u32 %r5, %r6, 2;",
        ])

    print(f"  using {len(memory_snippets)} memory + {len(compute_snippets)} compute snippets")

    def _get_mean_embedding(snippets: List[str]) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for text in snippets:
                input_ids, attn = _encode_ptx(tokenizer, text, max_length, device)
                outputs = model(input_ids, attention_mask=attn)
                hidden = outputs['hidden_states']
                mask_exp = attn.unsqueeze(-1).float()
                pooled = (hidden * mask_exp).sum(dim=1) / mask_exp.sum(dim=1)
                embeddings.append(pooled[0].cpu().numpy())
        return np.array(embeddings)

    mem_embs = _get_mean_embedding(memory_snippets)
    comp_embs = _get_mean_embedding(compute_snippets)

    mem_mean = mem_embs.mean(axis=0)
    comp_mean = comp_embs.mean(axis=0)
    cosine = float(
        np.dot(mem_mean, comp_mean)
        / (np.linalg.norm(mem_mean) * np.linalg.norm(comp_mean) + 1e-8)
    )
    print(f"\n--- instruction type separation (memory vs compute) ---")
    print(f"  cosine similarity: {cosine:.4f}  (lower = better)")
    print(f"  memory samples:    {len(memory_snippets)}")
    print(f"  compute samples:   {len(compute_snippets)}")
    return {'cosine_similarity': cosine}


# run all evaluations
def run_evaluation( checkpoint_path: str, tokenizer_path: str, data_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,max_seq_length: int = 2048,max_eval_samples: int = 100,device: str = 'cuda',) -> Dict[str, Dict]:
    """
    run all intrinsic evaluations on a pretrained checkpoint.

    always loads the test split from PTXDataset for evaluation.
    the test split is deterministically disjoint from train/val (seed=42).

    args:
        checkpoint_path:    path to saved .pt checkpoint
        tokenizer_path:     path to sentencepiece .model file
        data_dir:           directory with normalized PTX files (defaults to data/sprocessed/sprocessed)
        cache_dir:          cache directory for dataset (defaults to data/cache)
        max_seq_length:     max sequence length for evaluation
        max_eval_samples:   how many test samples to use for masked recovery
        device:             'cuda' or 'cpu'

    returns:
        dict of evaluation name -> results dict
    """
    model, tokenizer, config = load_model(checkpoint_path, tokenizer_path, device)
    device_str = str(next(model.parameters()).device)

    # load test dataset (always — use defaults if paths not provided)
    data_dir = data_dir or str(DEFAULT_DATA_DIR)
    cache_dir = cache_dir or str(DEFAULT_CACHE_DIR)

    test_dataset = load_test_dataset(
        tokenizer=tokenizer,
        data_dir=data_dir,
        cache_dir=cache_dir,
        max_seq_length=max_seq_length,
    )

    # decode test chunks for masked recovery
    ptx_snippets = decode_chunks(test_dataset, tokenizer, max_samples=max_eval_samples)
    print(f"using {len(ptx_snippets)} test snippets for masked recovery")

    results = {}
    results['masked_recovery'] = evaluate_masked_recovery(
        model, tokenizer, ptx_snippets,
        max_length=max_seq_length, device=device_str,
    )

    results['address_space'] = evaluate_address_space_separation(
        model, tokenizer, test_dataset=test_dataset, device=device_str,
    )
    results['instruction_type'] = evaluate_instruction_type_separation(
        model, tokenizer, test_dataset=test_dataset, device=device_str,
    )
    print(f"\n{'='*70}")
    print(f"intrinsic evaluation summary")
    print(f"{'='*70}")
    print(f"  masked recovery top-1:     {results['masked_recovery']['accuracy']:.4f}")
    print(f"  masked recovery top-5:     {results['masked_recovery']['top5_accuracy']:.4f}")
    print(f"  addr space inter-cosine:   {results['address_space']['inter_cosine']:.4f}")
    print(f"  mem vs compute cosine:     {results['instruction_type']['cosine_similarity']:.4f}")
    print(f"{'='*70}")

    return results



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='intrinsic evaluation for ptx transformer')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to model checkpoint (.pt)')
    parser.add_argument('--tokenizer', type=str, default=str(DEFAULT_TOKENIZER), help='path to sentencepiece .model file')
    parser.add_argument('--data-dir', type=str, default=str(DEFAULT_DATA_DIR), help='directory with normalized PTX files')
    parser.add_argument('--cache-dir', type=str, default=str(DEFAULT_CACHE_DIR), help='cache directory for dataset')
    parser.add_argument('--max-seq-length', type=int, default=2048, help='max sequence length for evaluation')
    parser.add_argument('--max-samples', type=int, default=100, help='max test samples to use for masked recovery')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    run_evaluation(checkpoint_path=args.checkpoint, tokenizer_path=args.tokenizer, data_dir=args.data_dir, cache_dir=args.cache_dir,
        max_seq_length=args.max_seq_length, max_eval_samples=args.max_samples, device=args.device,)






"""
intrinsic evaluation for ptx transformer pre-training quality.
evaluates the model on tasks that directly measure what mlm pre-training
should have taught the model, without any downstream fine-tuning:
  1. masked token recovery — can the model predict masked tokens?
  2. address space separation — does the model distinguish global vs shared memory?
  3. instruction type separation — does the model separate memory ops from compute ops?

all evaluations use:
  - normalized PTX format with real register names (%f1, %rd10, %r5, etc.)
    and normalized params (<PARAM_N>), kernels (<KERNEL_name>), BBs (<BB_N>)
  - test split from PTXDataset
  - PTXTransformerForPretraining model with sentencepiece tokenizer
"""