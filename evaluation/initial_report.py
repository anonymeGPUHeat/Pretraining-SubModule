import sys
import re
import random
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.architecture.complete_model import PTXTransformerForPretraining
from model.tokenizer.tokenizer import PTXTokenizer
from model.preprocessing.dataset_builder import PTXDataset
from model.preprocessing.itc_labels import (ITC_ID2NAME, ITC_IGNORE,detect_instruction_spans,)
from model.training.dataloader import apply_mlm_masking
from evaluation.eval_utils import ( load_test_dataset, get_non_overlap_range,DEFAULT_DATA_DIR, DEFAULT_CACHE_DIR, DEFAULT_TOKENIZER,)



# patterns for classifying subword pieces into semantic categories
_REG_PATTERN = re.compile(r'^%[a-z]{1,3}\d+$')       # %f1, %rd10, %r5, %p3
_IMM_PATTERN = re.compile(r'^-?\d+$|^0[xX]')          # 42, -1, 0xFF
_IMM_BUCKET  = re.compile(r'^<IMM_\d+')               # <IMM_6K>, <IMM_10K>
_ADDR_PATTERN = re.compile(r'^\[|^\+\d')              # [%rd1], +16
_SPECIAL_TOKENS = frozenset({
    '<pad>', '<unk>', '<s>', '</s>', '<mask>', '<cls>', '<sep>',
})

# common PTX opcodes (base opcode before any dot-suffix)
_PTX_OPCODES = frozenset({
    'ld', 'ldu', 'st', 'add', 'sub', 'mul', 'mad', 'fma', 'div', 'rem',
    'abs', 'neg', 'min', 'max', 'rcp', 'sqrt', 'rsqrt', 'sin', 'cos',
    'lg2', 'ex2', 'tanh', 'and', 'or', 'xor', 'not', 'shl', 'shr', 'shf',
    'setp', 'set', 'selp', 'slct', 'bra', 'ret', 'exit', 'call',
    'bar', 'barrier', 'membar', 'fence', 'cvt', 'cvta', 'mov',
    'atom', 'red', 'tex', 'wmma', 'mma', 'cp', 'prefetch', 'shfl', 'prmt',
    'bfe', 'bfi', 'bfind', 'brev', 'popc', 'clz', 'vote', 'match',
    'activemask', 'redux', 'nanosleep', 'trap', 'lop3',
})


def classify_token_type(piece: str) -> str:
    """
    classify a single tokenizer piece into one of:
      'opcode', 'register', 'immediate', 'address', 'modifier', 'structural', 'other'
    """
    if piece in _SPECIAL_TOKENS:
        return 'special'
    clean = piece.replace('\u2581', '').strip()
    if not clean:
        return 'other'
    if clean.startswith('<') and clean.endswith('>'):
        if _IMM_BUCKET.match(clean):
            return 'immediate'
        return 'structural'  
    if _REG_PATTERN.match(clean):
        return 'register'
    if clean.startswith('%') and len(clean) <= 5:
        return 'register'
    if _IMM_PATTERN.match(clean):
        return 'immediate'
    if clean in ('[', ']') or _ADDR_PATTERN.match(clean):
        return 'address'
    # opcodes — check if the whole piece (or first dot-segment) is a known opcode
    base = clean.split('.')[0].rstrip(';')
    if base in _PTX_OPCODES:
        return 'opcode'
    # type/space modifiers: .f32, .global, .shared, .rn, .ca, etc.
    if clean.startswith('.'):
        return 'modifier'
    if clean == ';' or clean.endswith(';'):
        return 'structural'

    return 'other'



def build_token_frequency_map(dataset: PTXDataset,max_chunks: int = 500, seed: int = 42,) -> Counter:
    """
    count token ID frequencies across a sample of the dataset.
    overlap-aware: for chunks that are not the first in their source file,
    the first ``overlap`` positions are skipped to avoid double-counting
    tokens shared with the previous chunk.
    """
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[:max_chunks]

    freq = Counter()
    for idx in indices:
        sample = dataset[idx]
        ids = sample['input_ids'].tolist()
        mask = sample['attention_mask'].tolist()
        start, end = get_non_overlap_range(dataset, idx)
        for pos in range(start, end):
            if mask[pos] == 1:
                freq[ids[pos]] += 1
    return freq


def assign_frequency_bucket(token_id: int, freq_map: Counter, total_tokens: int) -> str:
    """
    assign a token to a frequency bucket based on its corpus frequency.
      - 'high'   : top 10% most frequent tokens
      - 'medium' : next 40%
      - 'low'    : bottom 50% (rare tokens)
    """
    count = freq_map.get(token_id, 0)
    relative = count / total_tokens if total_tokens > 0 else 0
    if relative >= 1e-3:       # appears ≥ 0.1% of the time
        return 'high'
    elif relative >= 1e-4:     # ≥ 0.01%
        return 'medium'
    else:
        return 'low'


#ld r2 , [ rd32] : 6 


def evaluate_mlm_detailed(model: PTXTransformerForPretraining,tokenizer: PTXTokenizer, test_dataset: PTXDataset,freq_map: Counter,mask_prob: float = 0.15,max_chunks: int = 200,device: str = 'cuda', seed: int = 42,) -> Dict:
    """
    detailed MLM evaluation with:
      1. stratified accuracy by token type
      2. top-k accuracy (k=1,5)
      3. frequency-bucket difficulty analysis
    """
    print('MLM DETAILED EVALUATION\n')

    device_t = torch.device(device)
    mask_token_id = tokenizer.sp.PieceToId('<mask>')
    special_ids = {
        tokenizer.pad_id, tokenizer.unk_id, tokenizer.bos_id,
        tokenizer.eos_id, mask_token_id,
    }
    total_tokens_in_freq = sum(freq_map.values())
    # per-type accumulators
    type_correct  = defaultdict(int)
    type_correct5 = defaultdict(int)
    type_total    = defaultdict(int)
    # per-bucket accumulators
    bucket_correct  = defaultdict(int)
    bucket_correct5 = defaultdict(int)
    bucket_total    = defaultdict(int)
    # overall
    overall_correct = 0
    overall_correct5 = 0
    overall_total = 0
    rng = random.Random(seed)
    indices = list(range(len(test_dataset)))
    rng.shuffle(indices)
    indices = indices[:max_chunks]
    model.eval()
    for idx in indices:
        sample = test_dataset[idx]
        ids = sample['input_ids'].tolist()
        attn = sample['attention_mask'].tolist()
        # non-overlap range — skip positions duplicated from previous chunk
        eval_start, eval_end = get_non_overlap_range(test_dataset, idx)
        # detect instruction spans for instruction-boundary-aware masking
        _, spans = detect_instruction_spans(ids, tokenizer)
        mlm_result = apply_mlm_masking(token_ids=ids,attention_mask=attn,vocab_size=tokenizer.vocab_size,mask_token_id=mask_token_id,
        pad_token_id=tokenizer.pad_id,mask_prob=mask_prob, special_ids=special_ids,  instruction_spans=spans,)

        masked_ids = mlm_result['input_ids']
        mlm_labels = mlm_result['mlm_labels']

        input_t = torch.tensor([masked_ids], dtype=torch.long, device=device_t)
        attn_t = torch.tensor([attn], dtype=torch.long, device=device_t)

        with torch.no_grad():
            outputs = model(input_t, attention_mask=attn_t)
            logits = outputs['logits'][0]  # [seq_len, vocab_size]

        for pos in range(eval_start, min(eval_end, len(mlm_labels))):
            true_id = mlm_labels[pos]
            if true_id == -100:
                continue
            pred_id = logits[pos].argmax(dim=-1).item()
            top5 = logits[pos].topk(5).indices.tolist()
            # token type
            piece = tokenizer.sp.IdToPiece(true_id)
            ttype = classify_token_type(piece)
            # frequency bucket
            bucket = assign_frequency_bucket(true_id, freq_map, total_tokens_in_freq)
            is_correct = int(pred_id == true_id)
            is_top5 = int(true_id in top5)
            overall_correct += is_correct
            overall_correct5 += is_top5
            overall_total += 1
            type_correct[ttype] += is_correct
            type_correct5[ttype] += is_top5
            type_total[ttype] += 1
            bucket_correct[bucket] += is_correct
            bucket_correct5[bucket] += is_top5
            bucket_total[bucket] += 1

    acc = overall_correct / overall_total if overall_total else 0.0
    acc5 = overall_correct5 / overall_total if overall_total else 0.0
    type_results = {}
    print(f'\n stratified accuracy by token type :')
    print(f'  {"type":<14s} {"top-1":>8s} {"top-5":>8s} {"count":>8s}')
    print(f'  {"-"*14} {"-"*8} {"-"*8} {"-"*8}')
    for ttype in sorted(type_total.keys()):
        n = type_total[ttype]
        a1 = type_correct[ttype] / n if n else 0
        a5 = type_correct5[ttype] / n if n else 0
        type_results[ttype] = {'top1': a1, 'top5': a5, 'count': n}
        print(f'  {ttype:<14s} {a1:>8.2%} {a5:>8.2%} {n:>8d}')
    bucket_results = {}
    print(f'\nmasked position difficulty buckets: ')
    print(f'  {"bucket":<10s} {"top-1":>8s} {"top-5":>8s} {"count":>8s}')
    print(f'  {"-"*10} {"-"*8} {"-"*8} {"-"*8}')
    for bkt in ['high', 'medium', 'low']:
        n = bucket_total[bkt]
        a1 = bucket_correct[bkt] / n if n else 0
        a5 = bucket_correct5[bkt] / n if n else 0
        bucket_results[bkt] = {'top1': a1, 'top5': a5, 'count': n}
        print(f'  {bkt:<10s} {a1:>8.2%} {a5:>8.2%} {n:>8d}')

    print(f'\noverall')
    print(f'  total masked positions: {overall_total}')
    print(f'  top-1 accuracy:         {acc:.4f}')
    print(f'  top-5 accuracy:         {acc5:.4f}')

    return {
        'overall_top1': acc,
        'overall_top5': acc5,
        'total_masked': overall_total,
        'by_token_type': type_results,
        'by_frequency_bucket': bucket_results,
    }



def evaluate_itc_detailed(model: PTXTransformerForPretraining,tokenizer: PTXTokenizer,test_dataset: PTXDataset,max_chunks: int = 200,device: str = 'cuda',seed: int = 42, output_dir: Optional[str] = None,) -> Dict:
    """
    detailed ITC evaluation with:
      1. full 14-class confusion matrix
      2. per-class precision / recall / F1
      3. macro-F1, weighted-F1
      4. rare vs common class accuracy

    Returns empty dict if model has no ITC head.
    """
    # Skip if model has no ITC head
    if not getattr(model, 'use_itc', True) or model.itc_head is None:
        print('ITC DETAILED EVALUATION\n')
        print('  Skipped — model has no ITC head (training_mode is not mlm_itc)')
        return {}

    print('ITC DETAILED EVALUATION\n')

    device_t = torch.device(device)

    all_true: List[int] = []
    all_pred: List[int] = []

    rng = random.Random(seed)
    indices = list(range(len(test_dataset)))
    rng.shuffle(indices)
    indices = indices[:max_chunks]

    model.eval()
    for idx in indices:
        sample = test_dataset[idx]
        ids = sample['input_ids'].tolist()
        attn = sample['attention_mask'].tolist()
        # non-overlap range — skip positions duplicated from previous chunk
        eval_start, eval_end = get_non_overlap_range(test_dataset, idx)
        # generate ground-truth ITC labels (same method as training)
        itc_labels, _ = detect_instruction_spans(ids, tokenizer)
        input_t = torch.tensor([ids], dtype=torch.long, device=device_t)
        attn_t = torch.tensor([attn], dtype=torch.long, device=device_t)
        with torch.no_grad():
            outputs = model(input_t, attention_mask=attn_t)
            itc_logits = outputs['itc_logits'][0]  # [seq_len, 14]
        preds = itc_logits.argmax(dim=-1).cpu().tolist()
        for pos in range(eval_start, min(eval_end, len(itc_labels))):
            true_cls = itc_labels[pos]
            if true_cls == ITC_IGNORE:
                continue
            if attn[pos] == 0:
                continue
            all_true.append(true_cls)
            all_pred.append(preds[pos])

    n = len(all_true)
    if n == 0:
        print('  no valid ITC positions found in test data')
        return {}

    true_arr = np.array(all_true)
    pred_arr = np.array(all_pred)
    present_classes = sorted(set(all_true) | set(all_pred))
    num_present = len(present_classes)
    class_to_idx = {c: i for i, c in enumerate(present_classes)}
    cm = np.zeros((num_present, num_present), dtype=int)
    for t, p in zip(all_true, all_pred):
        cm[class_to_idx[t], class_to_idx[p]] += 1
    class_names = [ITC_ID2NAME.get(c, f'class_{c}') for c in present_classes]
    print(f'\nconfusion matrix ({n:,} positions)')
    header = '  {:>16s}'.format('')
    for name in class_names:
        short = name[:8]
        header += f' {short:>8s}'
    print(header)
    for i, name in enumerate(class_names):
        row = f'  {name:>16s}'
        for j in range(num_present):
            row += f' {cm[i, j]:>8d}'
        print(row)
    per_class: Dict[str, Dict] = {}
    print(f'\n--- per-class metrics ---')
    print(f'  {"class":<20s} {"prec":>8s} {"recall":>8s} {"F1":>8s} {"support":>8s}')
    print(f'  {"-"*20} {"-"*8} {"-"*8} {"-"*8} {"-"*8}')

    macro_f1_sum = 0.0
    weighted_f1_sum = 0.0
    total_support = 0

    for c in present_classes:
        name = ITC_ID2NAME.get(c, f'class_{c}')
        tp = int(((true_arr == c) & (pred_arr == c)).sum())
        fp = int(((true_arr != c) & (pred_arr == c)).sum())
        fn = int(((true_arr == c) & (pred_arr != c)).sum())
        support = int((true_arr == c).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[name] = {
            'precision': prec, 'recall': rec, 'f1': f1, 'support': support,
        }
        macro_f1_sum += f1
        weighted_f1_sum += f1 * support
        total_support += support

        print(f'  {name:<20s} {prec:>8.2%} {rec:>8.2%} {f1:>8.2%} {support:>8d}')

    macro_f1 = macro_f1_sum / len(present_classes) if present_classes else 0.0
    weighted_f1 = weighted_f1_sum / total_support if total_support else 0.0
    overall_acc = int((true_arr == pred_arr).sum()) / n

    print(f'\n  overall accuracy:  {overall_acc:.4f}')
    print(f'  macro F1:          {macro_f1:.4f}')
    print(f'  weighted F1:       {weighted_f1:.4f}')
    class_support = Counter()
    for c in all_true:
        class_support[c] += 1
    sorted_classes = sorted(class_support.items(), key=lambda x: x[1], reverse=True)
    cumulative = 0
    common_classes = set()
    for cls_id, count in sorted_classes:
        common_classes.add(cls_id)
        cumulative += count
        if cumulative >= total_support * 0.5:
            break
    rare_classes = set(class_support.keys()) - common_classes

    common_correct = sum(1 for t, p in zip(all_true, all_pred) if t in common_classes and t == p)
    common_total = sum(1 for t in all_true if t in common_classes)
    rare_correct = sum(1 for t, p in zip(all_true, all_pred) if t in rare_classes and t == p)
    rare_total = sum(1 for t in all_true if t in rare_classes)

    common_acc = common_correct / common_total if common_total else 0.0
    rare_acc = rare_correct / rare_total if rare_total else 0.0

    common_names = [ITC_ID2NAME.get(c, str(c)) for c in sorted(common_classes)]
    rare_names = [ITC_ID2NAME.get(c, str(c)) for c in sorted(rare_classes)]

    print(f'\n rare vs common class accuracy: ')
    print(f'  common classes ({common_total:,} tokens): {common_acc:.4f}')
    print(f'    classes: {", ".join(common_names)}')
    print(f'  rare classes   ({rare_total:,} tokens): {rare_acc:.4f}')
    print(f'    classes: {", ".join(rare_names)}')
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _plot_confusion_matrix(cm, class_names, out / 'itc_confusion_matrix.png')
        _plot_per_class_f1(per_class, out / 'itc_per_class_f1.png')

    return {
        'overall_accuracy': overall_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'total_positions': n,
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'common_accuracy': common_acc,
        'common_classes': common_names,
        'rare_accuracy': rare_acc,
        'rare_classes': rare_names,
    }


def _plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums != 0, out=np.zeros_like(cm, dtype=float))

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.9), max(8, len(class_names) * 0.7)))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,ax=ax, vmin=0, vmax=1,)
    ax.set_xlabel('predicted class')
    ax.set_ylabel('true class')
    ax.set_title('ITC confusion matrix (row-normalized)')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  saved confusion matrix to {save_path}')


def _plot_per_class_f1(per_class: Dict[str, Dict], save_path: Path) -> None:
    names = list(per_class.keys())
    f1s = [per_class[n]['f1'] for n in names]
    supports = [per_class[n]['support'] for n in names]
    order = np.argsort(f1s)
    names = [names[i] for i in order]
    f1s = [f1s[i] for i in order]
    supports = [supports[i] for i in order]
    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.4)))
    bars = ax.barh(range(len(names)), f1s, color='steelblue', edgecolor='white')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('F1 score')
    ax.set_title('ITC per-class F1')
    ax.set_xlim(0, 1.05)
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.3)
    for i, (f1, sup) in enumerate(zip(f1s, supports)):
        ax.text(f1 + 0.01, i, f'{f1:.2f} (n={sup})', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  saved per-class F1 chart to {save_path}')


def _plot_mlm_stratified(type_results: Dict[str, Dict], save_path: Path) -> None:
    names = sorted(type_results.keys(), key=lambda k: type_results[k]['count'], reverse=True)
    top1 = [type_results[n]['top1'] for n in names]
    top5 = [type_results[n]['top5'] for n in names]
    counts = [type_results[n]['count'] for n in names]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, top1, width, label='top-1', color='steelblue')
    ax.bar(x + width / 2, top5, width, label='top-5', color='coral')
    ax.set_xlabel('token type')
    ax.set_ylabel('accuracy')
    ax.set_title('MLM accuracy by token type')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}\n(n={c})' for n, c in zip(names, counts)], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  saved MLM stratified chart to {save_path}')


def _plot_mlm_buckets(bucket_results: Dict[str, Dict], save_path: Path) -> None:
    buckets = ['high', 'medium', 'low']
    top1 = [bucket_results.get(b, {}).get('top1', 0) for b in buckets]
    top5 = [bucket_results.get(b, {}).get('top5', 0) for b in buckets]
    counts = [bucket_results.get(b, {}).get('count', 0) for b in buckets]
    x = np.arange(len(buckets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, top1, width, label='top-1', color='steelblue')
    ax.bar(x + width / 2, top5, width, label='top-5', color='coral')
    ax.set_xlabel('token frequency bucket')
    ax.set_ylabel('accuracy')
    ax.set_title('MLM accuracy by token frequency (rare tokens = harder)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{b}\n(n={c})' for b, c in zip(buckets, counts)], fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  saved MLM bucket chart to {save_path}')


def load_model(checkpoint_path: str,tokenizer_path: str,device: str = 'cuda',) -> Tuple[PTXTransformerForPretraining, PTXTokenizer, Dict]:
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    tokenizer = PTXTokenizer(tokenizer_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    state_dict = checkpoint['model_state_dict']
    use_itc = any(k.startswith('itc_head') for k in state_dict)
    model = PTXTransformerForPretraining(vocab_size=config.get('vocab_size', tokenizer.vocab_size),
        d_model=config.get('d_model', 768),num_layers=config.get('num_layers', 12),num_heads=config.get('num_heads', 8),
        d_ff=config.get('d_ff', 3072),dropout=0.0,max_seq_length=config.get('max_seq_length', 2048),padding_idx=tokenizer.pad_id,
        use_itc=use_itc,).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f'loaded model from {checkpoint_path}  ({sum(p.numel() for p in model.parameters()):,} params)')
    return model, tokenizer, config


def generate_report(checkpoint_path: str,tokenizer_path: str,device: str = 'cuda',data_dir: Optional[str] = None,cache_dir: Optional[str] = None,
    max_seq_length: int = 2048,max_mlm_chunks: int = 200,max_itc_chunks: int = 200,output_dir: str = 'evaluation/report',seed: int = 42,) -> Dict:
    """
    generate the full initial diagnostic report.

    runs both MLM and ITC deep-dive evaluations on the test split,
    prints detailed metrics to stdout, saves plots and a JSON summary.
    """
    model, tokenizer, config = load_model(checkpoint_path, tokenizer_path, device)
    device_str = str(next(model.parameters()).device)
    data_dir = data_dir or str(DEFAULT_DATA_DIR)
    cache_dir = cache_dir or str(DEFAULT_CACHE_DIR)
    test_dataset = load_test_dataset(tokenizer=tokenizer,data_dir=data_dir,cache_dir=cache_dir,max_seq_length=max_seq_length,)
    print(f'test dataset: {len(test_dataset):,} chunks')
    print('\nbuilding token frequency map from test data ...')
    freq_map = build_token_frequency_map(test_dataset, max_chunks=min(500, len(test_dataset)), seed=seed)
    print(f'  unique tokens seen: {len(freq_map):,}')
    print(f'  total tokens counted: {sum(freq_map.values()):,}')
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results = {}
    mlm_results = evaluate_mlm_detailed(model, tokenizer, test_dataset, freq_map,max_chunks=max_mlm_chunks, device=device_str, seed=seed,)
    results['mlm'] = mlm_results
    if mlm_results.get('by_token_type'):
        _plot_mlm_stratified(mlm_results['by_token_type'], out_path / 'mlm_stratified_accuracy.png')
    if mlm_results.get('by_frequency_bucket'):
        _plot_mlm_buckets(mlm_results['by_frequency_bucket'], out_path / 'mlm_frequency_buckets.png')
    itc_results = evaluate_itc_detailed(
        model, tokenizer, test_dataset,
        max_chunks=max_itc_chunks, device=device_str, seed=seed,
        output_dir=output_dir,
    )
    results['itc'] = itc_results
    print('REPORT SUMMARY\n')
    print(f'  MLM top-1 accuracy:     {mlm_results.get("overall_top1", 0):.4f}')
    print(f'  MLM top-5 accuracy:     {mlm_results.get("overall_top5", 0):.4f}')
    print(f'  ITC overall accuracy:   {itc_results.get("overall_accuracy", 0):.4f}')
    print(f'  ITC macro F1:           {itc_results.get("macro_f1", 0):.4f}')
    print(f'  ITC weighted F1:        {itc_results.get("weighted_f1", 0):.4f}')
    print(f'  ITC common-class acc:   {itc_results.get("common_accuracy", 0):.4f}')
    print(f'  ITC rare-class acc:     {itc_results.get("rare_accuracy", 0):.4f}')

    json_path = out_path / 'initial_report.json'
    def _safe(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    safe_results = json.loads(json.dumps(results, default=_safe))
    with open(json_path, 'w') as f:
        json.dump(safe_results, f, indent=2)
    print(f'\n  saved JSON report to {json_path}')
    print(f'  saved plots to {out_path}/')

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='generate initial diagnostic report for PTX transformer pre-training',
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to model checkpoint (.pt)')
    parser.add_argument('--tokenizer', type=str, default=str(DEFAULT_TOKENIZER),
                        help='path to sentencepiece .model file')
    parser.add_argument('--data-dir', type=str, default=str(DEFAULT_DATA_DIR),
                        help='path to processed PTX directory')
    parser.add_argument('--cache-dir', type=str, default=str(DEFAULT_CACHE_DIR),
                        help='path to dataset cache directory')
    parser.add_argument('--max-seq-length', type=int, default=2048)
    parser.add_argument('--max-mlm-chunks', type=int, default=200,
                        help='max test chunks for MLM evaluation')
    parser.add_argument('--max-itc-chunks', type=int, default=200,
                        help='max test chunks for ITC evaluation')
    parser.add_argument('--output-dir', type=str, default='evaluation/report',
                        help='directory for plots and JSON output')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    generate_report(checkpoint_path=args.checkpoint,tokenizer_path=args.tokenizer,device=args.device,data_dir=args.data_dir,cache_dir=args.cache_dir,
    max_seq_length=args.max_seq_length,max_mlm_chunks=args.max_mlm_chunks,max_itc_chunks=args.max_itc_chunks,output_dir=args.output_dir,seed=args.seed,)
