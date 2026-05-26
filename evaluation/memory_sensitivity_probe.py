"""
Representation Sensitivity to Memory Access Patterns
=====================================================
Probe whether the pre-trained PTX encoder already separates
memory-heavy vs compute-heavy kernels in embedding space.

Pipeline
--------
1.  Load Nsight Compute profiling CSV  →  aggregate per-kernel metrics
    (mean across repeated runs of the same PTX / function pair).
2.  Split kernels into **high-hit-rate** vs **low-hit-rate** groups
    (median split on ``lts__t_sector_op_read_hit_rate.ratio``).
3.  For each kernel:
      raw PTX  →  normalise  →  tokenise  →  chunk (2 048 tokens)
      →  encode with pre-trained PTXEncoder  →  attention-weighted
      mean-pool  →  single embedding vector (d_model).
4.  Measure:
      • intra-group vs inter-group cosine similarity
      • linear separability (logistic regression, 80/20 stratified split)
5.  Produce console report + plots + JSON summary.

Usage
-----
    python -m evaluation.memory_sensitivity_probe \
        --checkpoint  <path/to/checkpoint.pt> \
        --tokenizer   data/tokenizer/ptx_tokenizer.model \
        --profiling   data/evaluation_data/runtimes_raw_0.txt \
        --ptx-dir     data/evaluation_data/ptx \
        --output-dir  evaluation/memory_probe_report
"""

import sys
import csv
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.architecture.complete_model import PTXTransformerForPretraining
from model.preprocessing.normalizer import PTXNormalizer
from model.tokenizer.tokenizer import PTXTokenizer

# ──────────────────────────────────────────────────────────────────────
# defaults
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILING = PROJECT_ROOT / 'data' / 'evaluation_data' / 'runtimes_raw_0.txt'
DEFAULT_PTX_DIR   = PROJECT_ROOT / 'data' / 'evaluation_data' / 'ptx'
DEFAULT_TOKENIZER = PROJECT_ROOT / 'data' / 'tokenizer' / 'ptx_tokenizer.model'


# ──────────────────────────────────────────────────────────────────────
# 1.  load & aggregate profiling data
# ──────────────────────────────────────────────────────────────────────

def load_profiling_data(
    csv_path: str,
    metric: str = 'lts__t_sector_op_read_hit_rate.ratio',
) -> Dict[str, float]:
    """
    Return ``{ptx_filename: mean_metric_value}`` aggregated over all
    run_ids for the same ``(ptx_path, function)`` pair.

    Rows with a non-empty ``error`` column are skipped.
    """
    acc: Dict[str, List[float]] = defaultdict(list)

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # skip rows with errors
            if row.get('error', '').strip():
                continue
            ptx_path = row.get('ptx_path', '').strip()
            if not ptx_path:
                continue
            val_str = row.get(metric, '').strip()
            if not val_str:
                continue
            try:
                val = float(val_str)
            except ValueError:
                continue
            # key on the filename only (hash.ptx)
            fname = ptx_path.rsplit('/', 1)[-1]
            acc[fname].append(val)

    return {fname: np.mean(vals) for fname, vals in acc.items()}


# ──────────────────────────────────────────────────────────────────────
# 2.  split into high / low groups
# ──────────────────────────────────────────────────────────────────────

def split_groups(
    metric_map: Dict[str, float],
    ptx_dir: str,
    threshold: Optional[float] = None,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], float]:
    """
    Split kernels into *high* (≥ threshold) and *low* (< threshold)
    groups.  If *threshold* is ``None`` the median is used.

    Only kernels whose PTX file actually exists on disk are kept.

    Returns ``(high_list, low_list, threshold)`` where each list
    element is ``(ptx_filename, metric_value)``.
    """
    ptx_path = Path(ptx_dir)
    available = {p.name for p in ptx_path.glob('*.ptx')}
    items = [(f, v) for f, v in metric_map.items() if f in available]

    if not items:
        raise RuntimeError('no PTX files from the profiling CSV were found on disk')

    values = [v for _, v in items]
    if threshold is None:
        threshold = float(np.median(values))

    # for hit rate: high = cache-friendly (≥ threshold), low = memory-intensive (< threshold)
    high = [(f, v) for f, v in items if v >= threshold]
    low  = [(f, v) for f, v in items if v <  threshold]
    return high, low, threshold


# ──────────────────────────────────────────────────────────────────────
# 3.  embed kernels
# ──────────────────────────────────────────────────────────────────────

def _chunk_ids(token_ids: List[int], max_len: int, pad_id: int) -> List[Tuple[List[int], List[int]]]:
    """
    Split *token_ids* into non-overlapping chunks of *max_len* tokens.
    The last chunk is right-padded.  Returns list of (ids, attn_mask).
    """
    chunks = []
    for start in range(0, len(token_ids), max_len):
        chunk = token_ids[start: start + max_len]
        attn  = [1] * len(chunk)
        pad_n = max_len - len(chunk)
        if pad_n > 0:
            chunk = chunk + [pad_id] * pad_n
            attn  = attn  + [0]      * pad_n
        chunks.append((chunk, attn))
    return chunks


@torch.no_grad()
def embed_kernels(
    ptx_filenames: List[str],
    ptx_dir: str,
    normalizer: PTXNormalizer,
    tokenizer: PTXTokenizer,
    encoder: torch.nn.Module,
    max_seq_length: int = 2048,
    device: str = 'cuda',
    batch_size: int = 8,
) -> np.ndarray:
    """
    Return an ``(N, d_model)`` matrix of kernel embeddings.

    For each file:
      normalise → tokenise → chunk to *max_seq_length*
      → forward through encoder → attention-weighted mean-pool
      → average across chunks.
    """
    device_t = torch.device(device)
    pad_id = tokenizer.pad_id
    embeddings = []

    encoder.eval()
    for i, fname in enumerate(ptx_filenames):
        filepath = Path(ptx_dir) / fname
        raw = filepath.read_text(encoding='utf-8', errors='replace')
        normed = normalizer.normalize_file(raw)
        ids = tokenizer.encode(normed)
        if len(ids) == 0:
            # degenerate file – use zero vector
            d_model = next(encoder.parameters()).shape[-1]
            embeddings.append(np.zeros(d_model))
            continue

        chunks = _chunk_ids(ids, max_seq_length, pad_id)

        # process chunks in mini-batches
        chunk_embeds = []
        for b_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[b_start: b_start + batch_size]
            ids_t  = torch.tensor([c[0] for c in batch_chunks],
                                  dtype=torch.long, device=device_t)
            attn_t = torch.tensor([c[1] for c in batch_chunks],
                                  dtype=torch.long, device=device_t)
            out = encoder(ids_t, attention_mask=attn_t)
            hidden = out['last_hidden_state']          # [B, S, D]
            mask = attn_t.unsqueeze(-1).float()        # [B, S, 1]
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            chunk_embeds.append(pooled.cpu().numpy())

        chunk_embeds = np.concatenate(chunk_embeds, axis=0)   # [n_chunks, D]
        kernel_emb = chunk_embeds.mean(axis=0)                 # [D]
        embeddings.append(kernel_emb)

        if (i + 1) % 50 == 0 or (i + 1) == len(ptx_filenames):
            print(f'  embedded {i + 1}/{len(ptx_filenames)} kernels', flush=True)

    return np.stack(embeddings, axis=0)


# ──────────────────────────────────────────────────────────────────────
# 4.  analysis helpers
# ──────────────────────────────────────────────────────────────────────

def _cosine_sim_matrix(A: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity matrix for (N, D)."""
    norms = np.linalg.norm(A, axis=1, keepdims=True).clip(min=1e-8)
    A_n = A / norms
    return A_n @ A_n.T


def compute_similarity_stats(
    high_emb: np.ndarray,
    low_emb: np.ndarray,
) -> Dict[str, float]:
    """
    Compute mean cosine similarity *within* each group and *across* groups.
    """
    # within high
    sim_hh = _cosine_sim_matrix(high_emb)
    n_h = len(high_emb)
    mask_hh = np.triu(np.ones((n_h, n_h), dtype=bool), k=1)
    within_high = float(sim_hh[mask_hh].mean()) if mask_hh.sum() > 0 else 0.0

    # within low
    sim_ll = _cosine_sim_matrix(low_emb)
    n_l = len(low_emb)
    mask_ll = np.triu(np.ones((n_l, n_l), dtype=bool), k=1)
    within_low = float(sim_ll[mask_ll].mean()) if mask_ll.sum() > 0 else 0.0

    # across groups
    norms_h = np.linalg.norm(high_emb, axis=1, keepdims=True).clip(min=1e-8)
    norms_l = np.linalg.norm(low_emb, axis=1, keepdims=True).clip(min=1e-8)
    cross = (high_emb / norms_h) @ (low_emb / norms_l).T
    across = float(cross.mean())

    within_avg = (within_high + within_low) / 2.0

    return {
        'within_high_mean_cos': within_high,
        'within_low_mean_cos': within_low,
        'within_avg_mean_cos': within_avg,
        'across_mean_cos': across,
        'gap': within_avg - across,
    }


def linear_separability(
    high_emb: np.ndarray,
    low_emb: np.ndarray,
    seed: int = 42,
    test_size: float = 0.2,
) -> Dict[str, float]:
    """
    80/20 stratified logistic regression.

    Returns accuracy, precision, recall, F1 on the held-out set.
    Uses manual stratified split + simple L2-regularised logistic regression
    (sklearn is not a dependency — we implement a lightweight version via
    numpy if sklearn is unavailable, but try sklearn first).
    """
    X = np.concatenate([high_emb, low_emb], axis=0)
    y = np.array([1] * len(high_emb) + [0] * len(low_emb))

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedShuffleSplit
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(sss.split(X, y))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LogisticRegression(max_iter=1000, random_state=seed, solver='lbfgs')
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        return {
            'accuracy': float(accuracy_score(y_test, preds)),
            'precision': float(precision_score(y_test, preds, zero_division=0)),
            'recall': float(recall_score(y_test, preds, zero_division=0)),
            'f1': float(f1_score(y_test, preds, zero_division=0)),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'method': 'sklearn_logistic_regression',
        }

    except ImportError:
        # fallback: manual stratified split + numpy-based logistic regression
        rng = np.random.RandomState(seed)
        idx_high = np.where(y == 1)[0]
        idx_low  = np.where(y == 0)[0]
        rng.shuffle(idx_high)
        rng.shuffle(idx_low)
        n_test_h = max(1, int(len(idx_high) * test_size))
        n_test_l = max(1, int(len(idx_low) * test_size))
        test_idx  = np.concatenate([idx_high[:n_test_h], idx_low[:n_test_l]])
        train_idx = np.concatenate([idx_high[n_test_h:], idx_low[n_test_l:]])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # z-score normalise columns
        mu = X_train.mean(axis=0)
        std = X_train.std(axis=0).clip(min=1e-8)
        Xtr = (X_train - mu) / std
        Xte = (X_test  - mu) / std

        # logistic regression via gradient descent
        d = Xtr.shape[1]
        w = np.zeros(d)
        b = 0.0
        lr = 0.1
        lam = 1e-4
        for _ in range(2000):
            z = Xtr @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
            grad_w = Xtr.T @ (p - y_train) / len(y_train) + lam * w
            grad_b = (p - y_train).mean()
            w -= lr * grad_w
            b -= lr * grad_b

        z_te = Xte @ w + b
        preds = (z_te >= 0).astype(int)
        acc = float((preds == y_test).mean())
        tp = int(((preds == 1) & (y_test == 1)).sum())
        fp = int(((preds == 1) & (y_test == 0)).sum())
        fn = int(((preds == 0) & (y_test == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'method': 'numpy_logistic_regression',
        }


# ──────────────────────────────────────────────────────────────────────
# 5.  plotting
# ──────────────────────────────────────────────────────────────────────

def _plot_similarity_bars(stats: Dict[str, float], save_path: Path) -> None:
    labels = ['Within\nHigh Hit Rate', 'Within\nLow Hit Rate', 'Across\nGroups']
    values = [
        stats['within_high_mean_cos'],
        stats['within_low_mean_cos'],
        stats['across_mean_cos'],
    ]
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor='white', width=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f'{v:.4f}', ha='center', va='bottom', fontsize=11)
    ax.set_ylabel('Mean cosine similarity')
    ax.set_title('Representation sensitivity to L2 cache hit rate')
    ax.set_ylim(0, max(values) * 1.15 + 0.01)
    ax.axhline(y=stats['across_mean_cos'], color='gray', ls='--', alpha=0.4)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  saved similarity bar chart → {save_path}')


def _plot_embedding_tsne(
    high_emb: np.ndarray,
    low_emb: np.ndarray,
    save_path: Path,
    seed: int = 42,
) -> None:
    """2-D t-SNE of kernel embeddings coloured by L2-miss group."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print('  [skip] t-SNE plot requires scikit-learn')
        return

    X = np.concatenate([high_emb, low_emb], axis=0)
    labels = np.array([1] * len(high_emb) + [0] * len(low_emb))

    perp = min(30, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=seed, init='pca',
                learning_rate='auto')
    coords = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, colour, name in [(1, '#2ecc71', 'High hit rate (cache-friendly)'),
                                 (0, '#e74c3c', 'Low hit rate (memory-intensive)')]:
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colour, label=name, alpha=0.6, s=30, edgecolors='white',
                   linewidths=0.3)
    ax.legend()
    ax.set_title('t-SNE of kernel embeddings (coloured by L2 cache hit rate)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  saved t-SNE plot → {save_path}')


# ──────────────────────────────────────────────────────────────────────
# 6.  main orchestrator
# ──────────────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: str,
    tokenizer_path: str,
    device: str = 'cuda',
) -> Tuple[torch.nn.Module, PTXTokenizer, Dict]:
    """Load checkpoint and return (encoder, tokenizer, config)."""
    device_t = torch.device(device if torch.cuda.is_available() else 'cpu')
    tokenizer = PTXTokenizer(tokenizer_path)

    checkpoint = torch.load(checkpoint_path, map_location=device_t, weights_only=False)
    config = checkpoint.get('config', {})
    state_dict = checkpoint['model_state_dict']
    use_itc = any(k.startswith('itc_head') for k in state_dict)

    model = PTXTransformerForPretraining(
        vocab_size=config.get('vocab_size', tokenizer.vocab_size),
        d_model=config.get('d_model', 768),
        num_layers=config.get('num_layers', 12),
        num_heads=config.get('num_heads', 8),
        d_ff=config.get('d_ff', 3072),
        dropout=0.0,
        max_seq_length=config.get('max_seq_length', 2048),
        padding_idx=tokenizer.pad_id,
        use_itc=use_itc,
    ).to(device_t)
    model.load_state_dict(state_dict)
    model.eval()

    encoder = model.encoder
    print(f'loaded encoder from {checkpoint_path}  '
          f'({sum(p.numel() for p in encoder.parameters()):,} params)')
    return encoder, tokenizer, config


def run_probe(checkpoint_path: str,tokenizer_path: str = str(DEFAULT_TOKENIZER),profiling_path: str = str(DEFAULT_PROFILING),ptx_dir: str = str(DEFAULT_PTX_DIR),metric: str = 'lts__t_sector_op_read_hit_rate.ratio',threshold: Optional[float] = None,max_kernels: Optional[int] = None,max_seq_length: int = 2048,output_dir: str = 'evaluation/memory_probe_report',device: str = 'cuda', seed: int = 42,) -> Dict:
    """
    Full probe pipeline.

    Parameters
    ----------
    checkpoint_path : path to model checkpoint (.pt)
    metric : Nsight Compute counter to split on
    threshold : split value (default: median)
    max_kernels : cap on kernels per group (for speed; None = all)
    """
    print('=' * 60)
    print(' REPRESENTATION SENSITIVITY TO MEMORY ACCESS PATTERNS')
    print('=' * 60)

    # ---- load model ----
    encoder, tokenizer, config = load_model(checkpoint_path, tokenizer_path, device)
    device_str = str(next(encoder.parameters()).device)
    seq_len = config.get('max_seq_length', max_seq_length)

    # ---- profiling data ----
    print(f'\nloading profiling data from {profiling_path} ...')
    metric_map = load_profiling_data(profiling_path, metric=metric)
    print(f'  unique kernels with metric "{metric}": {len(metric_map)}')

    # ---- split groups ----
    high, low, thr = split_groups(metric_map, ptx_dir, threshold)
    print(f'\nsplit threshold ({metric}): {thr:.4f}  ({thr*100:.1f}%)')
    print(f'  high-hit-rate kernels (cache-friendly):   {len(high)}')
    print(f'  low-hit-rate  kernels (memory-intensive): {len(low)}')

    # optional cap
    rng = random.Random(seed)
    if max_kernels is not None:
        rng.shuffle(high)
        rng.shuffle(low)
        high = high[:max_kernels]
        low  = low[:max_kernels]
        print(f'  (capped to {max_kernels} per group)')

    # ---- embed ----
    normalizer = PTXNormalizer(verbose=False)
    print(f'\nembedding {len(high)} high-hit-rate kernels ...')
    high_emb = embed_kernels(
        [f for f, _ in high], ptx_dir, normalizer, tokenizer,
        encoder, max_seq_length=seq_len, device=device_str,
    )
    print(f'embedding {len(low)} low-hit-rate kernels ...')
    low_emb = embed_kernels(
        [f for f, _ in low], ptx_dir, normalizer, tokenizer,
        encoder, max_seq_length=seq_len, device=device_str,
    )

    # ---- cosine similarity ----
    print('\n--- cosine similarity analysis ---')
    sim_stats = compute_similarity_stats(high_emb, low_emb)
    print(f'  within high-hit-rate (mean cos): {sim_stats["within_high_mean_cos"]:.4f}')
    print(f'  within low-hit-rate  (mean cos): {sim_stats["within_low_mean_cos"]:.4f}')
    print(f'  across groups       (mean cos): {sim_stats["across_mean_cos"]:.4f}')
    print(f'  gap (within_avg − across):      {sim_stats["gap"]:.4f}')

    if sim_stats['gap'] > 0.05:
        verdict_sim = 'PROMISING — encoder shows meaningful separation'
    elif sim_stats['gap'] > 0.01:
        verdict_sim = 'WEAK — some structure but limited'
    else:
        verdict_sim = 'NO SEPARATION — representations are not memory-sensitive'
    print(f'  → {verdict_sim}')

    # ---- linear separability ----
    print('\n--- linear separability (logistic regression, 80/20) ---')
    lr_stats = linear_separability(high_emb, low_emb, seed=seed)
    print(f'  accuracy:  {lr_stats["accuracy"]:.4f}')
    print(f'  precision: {lr_stats["precision"]:.4f}')
    print(f'  recall:    {lr_stats["recall"]:.4f}')
    print(f'  F1:        {lr_stats["f1"]:.4f}')
    print(f'  method:    {lr_stats["method"]}')
    print(f'  train/test: {lr_stats["train_size"]}/{lr_stats["test_size"]}')

    if lr_stats['accuracy'] > 0.75:
        verdict_lr = 'STRONG — encoder representations are linearly separable for memory behaviour'
    elif lr_stats['accuracy'] > 0.60:
        verdict_lr = 'MODERATE — partial linear separability'
    else:
        verdict_lr = 'WEAK — representations not linearly separable'
    print(f'  → {verdict_lr}')

    # ---- summary ----
    print('\n' + '=' * 60)
    print(' SUMMARY')
    print('=' * 60)
    print(f'  cosine gap:       {sim_stats["gap"]:.4f}  ({verdict_sim})')
    print(f'  linear probe acc: {lr_stats["accuracy"]:.4f}  ({verdict_lr})')

    if sim_stats['gap'] > 0.05 and lr_stats['accuracy'] > 0.75:
        overall = ('The encoder already captures memory-access semantics. '
                   'Fine-tuning should converge quickly with a conservative '
                   'encoder learning rate (e.g. 1e-5).')
    elif sim_stats['gap'] > 0.01 or lr_stats['accuracy'] > 0.60:
        overall = ('The encoder shows weak memory sensitivity. '
                   'Fine-tuning will need moderate epochs and a '
                   'standard learning rate on the encoder (e.g. 3e-5).')
    else:
        overall = ('The encoder learned PTX syntax but NOT memory semantics. '
                   'Fine-tuning will require more epochs and a larger '
                   'encoder learning rate (e.g. 5e-5 or higher).')
    print(f'\n  {overall}')

    # ---- save ----
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _plot_similarity_bars(sim_stats, out / 'cosine_similarity_bars.png')
    _plot_embedding_tsne(high_emb, low_emb, out / 'embedding_tsne.png', seed=seed)

    results = {
        'metric': metric,
        'threshold': thr,
        'n_high': len(high),
        'n_low': len(low),
        'similarity': sim_stats,
        'linear_probe': lr_stats,
        'verdict_similarity': verdict_sim,
        'verdict_linear': verdict_lr,
        'overall_recommendation': overall,
    }
    json_path = out / 'memory_sensitivity_probe.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\n  saved JSON → {json_path}')
    print(f'  saved plots → {out}/')
    return results


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(
        description='Probe encoder sensitivity to memory access patterns',
    )
    p.add_argument('--checkpoint', type=str, required=True,
                   help='path to model checkpoint (.pt)')
    p.add_argument('--tokenizer', type=str, default=str(DEFAULT_TOKENIZER),
                   help='path to sentencepiece .model file')
    p.add_argument('--profiling', type=str, default=str(DEFAULT_PROFILING),
                   help='path to Nsight Compute profiling CSV')
    p.add_argument('--ptx-dir', type=str, default=str(DEFAULT_PTX_DIR),
                   help='directory containing raw PTX files')
    p.add_argument('--metric', type=str, default='lts__t_sector_op_read_hit_rate.ratio',
                   help='metric column to split on (default: L2 read hit rate, 0.0-1.0)')
    # available hit-rate columns: lts__t_sector_op_read_hit_rate.ratio
    #                              lts__t_sector_op_write_hit_rate.ratio
    p.add_argument('--threshold', type=float, default=None,
                   help='split threshold (default: median)')
    p.add_argument('--max-kernels', type=int, default=None,
                   help='max kernels per group (for speed)')
    p.add_argument('--max-seq-length', type=int, default=2048)
    p.add_argument('--output-dir', type=str,
                   default='evaluation/memory_probe_report')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    run_probe(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        profiling_path=args.profiling,
        ptx_dir=args.ptx_dir,
        metric=args.metric,
        threshold=args.threshold,
        max_kernels=args.max_kernels,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )
