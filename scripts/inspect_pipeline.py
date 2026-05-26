#!/usr/bin/env python3
"""
Pipeline Inspector — walk a single PTX file through every component and
print detailed diagnostics at each stage.

Usage:
    python scripts/inspect_pipeline.py <ptx_file>
    python scripts/inspect_pipeline.py data/raw/00000_1_Conv2D_ReLU_BiasAdd.ptx
    python scripts/inspect_pipeline.py data/sprocessed/sprocessed/file_618.ptx --max-seq-length 512
"""

import sys, os, argparse, random, textwrap, math
from pathlib import Path
from collections import Counter

# ── project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

# ── imports from our codebase ─────────────────────────────────────────────────
from model.tokenizer.tokenizer import PTXTokenizer
from model.preprocessing.itc_labels import (
    detect_instruction_spans, ITC_ID2NAME, ITC_CLASSES, NUM_ITC_CLASSES, ITC_IGNORE,
)
from model.training.dataloader import apply_mlm_masking
from model.training.objectives import (
    PretrainingObjectives, compute_mlm_accuracy, compute_itc_accuracy, compute_mlm_perplexity,
)
from model.architecture.complete_model import PTXTransformerForPretraining


# ── pretty helpers ────────────────────────────────────────────────────────────
BLUE   = "\033[94m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"
SEP    = "─" * 80


def header(title: str):
    print(f"\n{BOLD}{BLUE}{'═' * 80}")
    print(f"  {title}")
    print(f"{'═' * 80}{RESET}\n")


def subheader(title: str):
    print(f"\n  {BOLD}{YELLOW}{title}{RESET}")
    print(f"  {SEP}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — READ FILE
# ══════════════════════════════════════════════════════════════════════════════

def step_read_file(path: Path):
    header("STEP 1 · READ PTX FILE")
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    chars = len(text)
    print(f"  File:       {path}")
    print(f"  Size:       {path.stat().st_size:,} bytes")
    print(f"  Characters: {chars:,}")
    print(f"  Lines:      {len(lines):,}")

    subheader("First 15 lines")
    for i, line in enumerate(lines[:15], 1):
        print(f"    {DIM}{i:4d}{RESET} │ {line}")
    if len(lines) > 15:
        print(f"    {DIM} ... ({len(lines) - 15} more lines){RESET}")
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — TOKENIZE
# ══════════════════════════════════════════════════════════════════════════════

def step_tokenize(text: str, tokenizer: PTXTokenizer):
    header("STEP 2 · TOKENIZATION")
    token_ids = tokenizer.encode(text)
    pieces = tokenizer.encode_as_pieces(text)
    assert len(token_ids) == len(pieces)
    n = len(token_ids)
    print(f"  Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"  Total tokens:    {n:,}")
    print(f"  Unique tokens:   {len(set(token_ids)):,}")
    print(f"  Compression:     {len(text) / n:.2f} chars/token")

    # ── Special token reference table ─────────────────────────────────
    subheader("Special token ID map")
    special_names = [
        ("<pad>",  tokenizer.pad_id),
        ("<unk>",  tokenizer.unk_id),
        ("<s>",    tokenizer.bos_id),
        ("</s>",   tokenizer.eos_id),
        ("<mask>", tokenizer.sp.PieceToId("<mask>")),
        ("<cls>",  tokenizer.sp.PieceToId("<cls>")),
        ("<sep>",  tokenizer.sp.PieceToId("<sep>")),
    ]
    for name, tid in special_names:
        print(f"    {name:10s}  →  ID {tid}")

    subheader("Token frequency (top 20)")
    freq = Counter(pieces)
    for piece, count in freq.most_common(20):
        bar = "█" * min(count, 40)
        print(f"    {piece:20s}  {count:5d}  {DIM}{bar}{RESET}")

    # ── ID distribution stats ─────────────────────────────────────────
    subheader("Token ID statistics")
    id_tensor = torch.tensor(token_ids, dtype=torch.float)
    print(f"    Min ID:    {int(id_tensor.min().item()):5d}  ({tokenizer.sp.IdToPiece(int(id_tensor.min().item()))})")
    print(f"    Max ID:    {int(id_tensor.max().item()):5d}  ({tokenizer.sp.IdToPiece(int(id_tensor.max().item()))})")
    print(f"    Mean ID:   {id_tensor.mean().item():.1f}")
    print(f"    Median ID: {int(id_tensor.median().item()):5d}")

    # ── Bidirectional token ↔ ID table ────────────────────────────────
    subheader("First 50 tokens — full translation table")
    print(f"    {'pos':>5s}  {'ID':>5s}  {'piece':20s}  {'decoded_text':30s}  {'bytes':>6s}")
    print(f"    {'─'*5}  {'─'*5}  {'─'*20}  {'─'*30}  {'─'*6}")
    for i in range(min(50, n)):
        tid = token_ids[i]
        piece = pieces[i]
        decoded = tokenizer.decode([tid])
        raw_bytes = piece.encode('utf-8')
        hex_str = raw_bytes.hex() if len(raw_bytes) <= 6 else raw_bytes[:6].hex() + '..'  
        print(f"    {i:5d}  {tid:5d}  {piece:20s}  {repr(decoded):30s}  {hex_str:>6s}")
    if n > 50:
        print(f"    {DIM}... ({n - 50} more tokens){RESET}")

    # ── Reconstruct text round-trip check ─────────────────────────────
    subheader("Round-trip decode check")
    reconstructed = tokenizer.decode(token_ids)
    match = text.strip() == reconstructed.strip()
    mark = f"{GREEN}MATCH{RESET}" if match else f"{YELLOW}DIFFERS{RESET}"
    print(f"    Original chars:      {len(text):,}")
    print(f"    Reconstructed chars: {len(reconstructed):,}")
    print(f"    Exact match:         [{mark}]")
    if not match:
        # Show first difference
        for i, (a, b) in enumerate(zip(text, reconstructed)):
            if a != b:
                print(f"    First diff at char {i}: orig={repr(a)} vs recon={repr(b)}")
                break

    return token_ids, pieces


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def step_chunking(token_ids, pieces, max_seq_length, overlap, tokenizer):
    header("STEP 3 · CHUNKING INTO SEQUENCES")
    n = len(token_ids)
    stride = max_seq_length - overlap

    # Boundary tokens for instruction-aligned splitting
    newline_piece = tokenizer.encode_as_pieces("\n")[0]
    boundary_tokens = frozenset({";", ":", newline_piece})

    chunks = []
    start = 0
    while start < n:
        end = min(start + max_seq_length, n)
        if end < n:
            ideal_next = start + stride
            lo = max(ideal_next - 64, start + stride // 2)
            hi = min(ideal_next + 64, n)
            best = ideal_next
            for i in range(lo, hi):
                if pieces[i] in boundary_tokens:
                    best = i + 1
                    break
            next_start = best
        else:
            next_start = n

        chunk_ids = token_ids[start:end]
        actual_len = len(chunk_ids)
        pad_len = max_seq_length - actual_len
        chunk_ids_padded = chunk_ids + [tokenizer.pad_id] * pad_len
        attn_mask = [1] * actual_len + [0] * pad_len

        chunks.append({
            "input_ids": chunk_ids_padded,
            "attention_mask": attn_mask,
            "actual_length": actual_len,
            "start_token": start,
        })
        start = next_start

    print(f"  Max sequence length:  {max_seq_length}")
    print(f"  Overlap:              {overlap}")
    print(f"  Stride:               {stride}")
    print(f"  Total tokens:         {n:,}")
    print(f"  Sequences produced:   {GREEN}{len(chunks)}{RESET}")

    subheader("Per-sequence summary")
    for i, c in enumerate(chunks):
        pct_pad = (max_seq_length - c["actual_length"]) / max_seq_length * 100
        print(f"    Seq {i:3d}:  tokens {c['start_token']:6d}–{c['start_token'] + c['actual_length'] - 1:6d}"
              f"   real={c['actual_length']:5d}   pad={max_seq_length - c['actual_length']:5d}"
              f"   ({pct_pad:5.1f}% padding)")

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — ITC LABELING
# ══════════════════════════════════════════════════════════════════════════════

def step_itc_labeling(chunks, tokenizer, show_tokens=30):
    header("STEP 4 · ITC INSTRUCTION-TYPE LABELING")
    all_labels = []
    all_spans  = []

    total_instr = 0
    total_ignore = 0
    global_class_counts = Counter()

    for idx, chunk in enumerate(chunks):
        ids = chunk["input_ids"]
        labels, spans = detect_instruction_spans(ids, tokenizer)
        all_labels.append(labels)
        all_spans.append(spans)

        instr_count = sum(1 for l in labels if l != ITC_IGNORE)
        ignore_count = sum(1 for l in labels if l == ITC_IGNORE)
        total_instr += instr_count
        total_ignore += ignore_count
        for l in labels:
            if l != ITC_IGNORE:
                global_class_counts[l] += 1

    print(f"  Total instruction tokens:  {GREEN}{total_instr:,}{RESET}")
    print(f"  Total ignored tokens:      {total_ignore:,}")
    print(f"  Instruction spans (seq 0): {len(all_spans[0])}")

    subheader("ITC class distribution (all sequences)")
    for cls_id in range(NUM_ITC_CLASSES):
        name = ITC_ID2NAME.get(cls_id, "?")
        count = global_class_counts.get(cls_id, 0)
        bar = "█" * min(count // max(total_instr // 40, 1), 40) if total_instr else ""
        print(f"    {cls_id:2d}  {name:20s}  {count:6d}  {DIM}{bar}{RESET}")

    # Show first N tokens of sequence 0
    subheader(f"First {show_tokens} tokens of sequence 0  (token → ITC label)")
    ids0 = chunks[0]["input_ids"]
    labels0 = all_labels[0]
    for i in range(min(show_tokens, len(ids0))):
        piece = tokenizer.sp.IdToPiece(ids0[i])
        lbl = labels0[i]
        lbl_name = ITC_ID2NAME.get(lbl, "IGNORE") if lbl != ITC_IGNORE else "—"
        color = GREEN if lbl != ITC_IGNORE else DIM
        print(f"    {DIM}[{i:4d}]{RESET}  {piece:20s}  →  {color}{lbl_name}{RESET}")

    return all_labels, all_spans


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — MLM MASKING
# ══════════════════════════════════════════════════════════════════════════════

def step_mlm_masking(chunks, all_spans, tokenizer, mask_prob, show_tokens=30):
    header("STEP 5 · MLM MASKING (instruction-boundary aware)")
    mask_id = tokenizer.sp.PieceToId("<mask>")
    special_ids = {tokenizer.pad_id, tokenizer.unk_id, tokenizer.bos_id, tokenizer.eos_id, mask_id}

    masked_chunks = []
    total_masked = 0
    total_real = 0

    for idx, chunk in enumerate(chunks):
        ids = chunk["input_ids"]
        attn = chunk["attention_mask"]
        spans = all_spans[idx] if all_spans else None

        result = apply_mlm_masking(
            token_ids=ids,
            attention_mask=attn,
            vocab_size=tokenizer.vocab_size,
            mask_token_id=mask_id,
            pad_token_id=tokenizer.pad_id,
            mask_prob=mask_prob,
            special_ids=special_ids,
            instruction_spans=spans,
        )
        n_masked = sum(1 for l in result["mlm_labels"] if l != -100)
        n_real = sum(attn)
        total_masked += n_masked
        total_real += n_real
        masked_chunks.append(result)

    pct = total_masked / max(total_real, 1) * 100
    print(f"  Mask probability:     {mask_prob}")
    print(f"  <mask> token ID:      {mask_id}")
    print(f"  Total real tokens:    {total_real:,}")
    print(f"  Total masked tokens:  {GREEN}{total_masked:,}{RESET}  ({pct:.1f}%)")

    # ── Masking strategy breakdown ────────────────────────────────
    subheader("Masking strategy breakdown (sequence 0)")
    mask_id = tokenizer.sp.PieceToId("<mask>")
    orig0 = chunks[0]["input_ids"]
    masked0 = masked_chunks[0]["input_ids"]
    labels0 = masked_chunks[0]["mlm_labels"]
    n_replaced_mask = 0
    n_replaced_random = 0
    n_kept_original = 0
    n_unselected = 0
    for i in range(len(orig0)):
        if labels0[i] == -100:
            n_unselected += 1
        elif masked0[i] == mask_id:
            n_replaced_mask += 1
        elif masked0[i] != orig0[i]:
            n_replaced_random += 1
        else:
            n_kept_original += 1
    n_selected = n_replaced_mask + n_replaced_random + n_kept_original
    print(f"    Selected for masking:  {n_selected:5d}")
    print(f"      → replaced with <mask>:   {n_replaced_mask:5d}  ({n_replaced_mask/max(n_selected,1)*100:5.1f}%)  (target: 80%)")
    print(f"      → replaced with random:   {n_replaced_random:5d}  ({n_replaced_random/max(n_selected,1)*100:5.1f}%)  (target: 10%)")
    print(f"      → kept original:          {n_kept_original:5d}  ({n_kept_original/max(n_selected,1)*100:5.1f}%)  (target: 10%)")
    print(f"    Not selected:          {n_unselected:5d}")

    subheader(f"First {show_tokens} tokens of sequence 0  (original → masked, with IDs)")
    print(f"    {'pos':>5s}  {'origID':>6s}  {'orig_piece':20s}  {'maskID':>6s}  {'mask_piece':20s}  {'label':>6s}  {'status'}")
    print(f"    {'─'*5}  {'─'*6}  {'─'*20}  {'─'*6}  {'─'*20}  {'─'*6}  {'─'*20}")
    for i in range(min(show_tokens, len(orig0))):
        oid = orig0[i]
        mid = masked0[i]
        orig_piece = tokenizer.sp.IdToPiece(oid)
        mask_piece = tokenizer.sp.IdToPiece(mid)
        lbl = labels0[i]
        if lbl == -100:
            status = f"{DIM}not selected{RESET}"
            lbl_str = "  —"
            color = DIM
        elif mid == mask_id:
            status = f"{RED}→ <mask>{RESET}"
            lbl_str = f"{lbl:5d}"
            color = RED
        elif mid != oid:
            status = f"{YELLOW}→ random{RESET}"
            lbl_str = f"{lbl:5d}"
            color = YELLOW
        else:
            status = f"{BLUE}→ keep{RESET}"
            lbl_str = f"{lbl:5d}"
            color = BLUE
        print(f"    {i:5d}  {oid:6d}  {orig_piece:20s}  {color}{mid:6d}  {mask_piece:20s}{RESET}  {lbl_str:>6s}  {status}")

    return masked_chunks


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — MODEL FORWARD PASS
# ══════════════════════════════════════════════════════════════════════════════

def step_forward_pass(masked_chunks, chunks, all_labels, tokenizer, d_model, num_layers, num_heads, max_seq_length):
    header("STEP 6 · MODEL FORWARD PASS (random weights — no training)")

    model = PTXTransformerForPretraining(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_model * 4,
        dropout=0.0,
        max_seq_length=max_seq_length,
        padding_idx=tokenizer.pad_id,
        num_itc_classes=NUM_ITC_CLASSES,
    ).eval()

    params = model.count_parameters()
    print(f"  Model config:")
    print(f"    d_model     = {d_model}")
    print(f"    num_layers  = {num_layers}")
    print(f"    num_heads   = {num_heads}")
    print(f"    d_ff        = {d_model * 4}")
    print(f"    head_dim    = {d_model // num_heads}")
    print(f"    vocab_size  = {tokenizer.vocab_size}")
    print(f"    itc_classes = {NUM_ITC_CLASSES}")
    print(f"    max_seq_len = {max_seq_length}")
    print(f"    padding_idx = {tokenizer.pad_id}")
    print(f"  Parameters:")
    for k, v in params.items():
        print(f"    {k:20s}  {v:>12,}")

    n_seqs = len(masked_chunks)
    print(f"\n  Sequences to process: {n_seqs}")

    # ── Encoder input detail ──────────────────────────────────────────
    subheader("ENCODER INPUT (sequence 0)")
    input_ids_0 = torch.tensor([masked_chunks[0]["input_ids"]], dtype=torch.long)
    attn_mask_0 = torch.tensor([chunks[0]["attention_mask"]], dtype=torch.long)
    real_len_0 = chunks[0]["actual_length"]
    print(f"    input_ids shape:      {list(input_ids_0.shape)}  (batch=1, seq={max_seq_length})")
    print(f"    attention_mask shape:  {list(attn_mask_0.shape)}")
    print(f"    Real tokens:           {real_len_0}")
    print(f"    Padding tokens:        {max_seq_length - real_len_0}")
    print(f"    Unique input IDs:      {input_ids_0.unique().numel()}")
    mask_id = tokenizer.sp.PieceToId("<mask>")
    n_mask_in_input = (input_ids_0 == mask_id).sum().item()
    print(f"    <mask> tokens in input: {n_mask_in_input}")

    print(f"\n    First 20 input tokens entering the encoder:")
    print(f"    {'pos':>5s}  {'ID':>5s}  {'piece':20s}  {'attn':>4s}  {'note'}")
    print(f"    {'─'*5}  {'─'*5}  {'─'*20}  {'─'*4}  {'─'*25}")
    for i in range(min(20, max_seq_length)):
        tid = input_ids_0[0, i].item()
        piece = tokenizer.sp.IdToPiece(tid)
        attn = attn_mask_0[0, i].item()
        note = ""
        if tid == mask_id:
            note = f"{RED}<mask> token{RESET}"
        elif tid == tokenizer.pad_id:
            note = f"{DIM}padding{RESET}"
        elif tid == tokenizer.unk_id:
            note = f"{YELLOW}unknown{RESET}"
        print(f"    {i:5d}  {tid:5d}  {piece:20s}  {attn:4d}  {note}")

    # ── Forward pass all sequences ────────────────────────────────────
    all_mlm_logits = []
    all_itc_logits = []
    all_hidden = []

    for idx in range(n_seqs):
        input_ids = torch.tensor([masked_chunks[idx]["input_ids"]], dtype=torch.long)
        attn_mask = torch.tensor([chunks[idx]["attention_mask"]], dtype=torch.long)

        with torch.no_grad():
            out = model(input_ids, attention_mask=attn_mask)

        all_mlm_logits.append(out["logits"])
        all_itc_logits.append(out["itc_logits"])
        all_hidden.append(out["hidden_states"])

    # ── Hidden states (encoder output) ────────────────────────────────
    subheader("ENCODER OUTPUT — HIDDEN STATES")
    h0 = all_hidden[0]  # [1, seq, d_model]
    print(f"    Shape:              {list(h0.shape)}  →  [batch, seq_len, d_model]")
    norms = h0[0].norm(dim=-1)  # [seq]
    print(f"    Per-token L2 norm:")
    print(f"      Min:   {norms.min().item():.4f}")
    print(f"      Max:   {norms.max().item():.4f}")
    print(f"      Mean:  {norms.mean().item():.4f}")
    print(f"      Std:   {norms.std().item():.4f}")
    # Padding vs real token norms
    real_norms = norms[:real_len_0]
    pad_norms = norms[real_len_0:] if real_len_0 < max_seq_length else torch.tensor([0.0])
    print(f"    Real token avg norm:    {real_norms.mean().item():.4f}")
    print(f"    Padding token avg norm: {pad_norms.mean().item():.4f}")

    print(f"\n    First 10 hidden state vectors (L2 norm + first 5 dims):")
    for i in range(min(10, max_seq_length)):
        vec = h0[0, i]
        piece = tokenizer.sp.IdToPiece(masked_chunks[0]["input_ids"][i])
        dims_str = ", ".join(f"{v:.3f}" for v in vec[:5].tolist())
        print(f"      [{i:4d}] {piece:15s}  norm={vec.norm().item():.4f}  dims=[{dims_str}, ...]")

    # ══════════════════════════════════════════════════════════════════
    #  MLM HEAD — detailed
    # ══════════════════════════════════════════════════════════════════
    subheader("MLM HEAD")
    print(f"    Architecture: hidden[{d_model}] → Dense[{d_model}] → GELU → LN → Linear[{tokenizer.vocab_size}]")
    print(f"    Weight tying: decoder.weight is encoder.embedding.weight")
    logits0 = all_mlm_logits[0]  # [1, seq, vocab]
    print(f"\n    Output logits:")
    print(f"      Shape:  {list(logits0.shape)}  →  [batch, seq_len, vocab_size]")
    print(f"      dtype:  {logits0.dtype}")
    print(f"      Min:    {logits0.min().item():.4f}")
    print(f"      Max:    {logits0.max().item():.4f}")
    print(f"      Mean:   {logits0.mean().item():.4f}")
    print(f"      Std:    {logits0.std().item():.4f}")

    # Softmax statistics
    probs0 = torch.softmax(logits0[0], dim=-1)  # [seq, vocab]
    entropy = -(probs0 * (probs0 + 1e-10).log()).sum(dim=-1)  # [seq]
    max_entropy = math.log(tokenizer.vocab_size)
    print(f"\n    Softmax probability statistics:")
    print(f"      Max prob (any position):     {probs0.max().item():.6f}")
    print(f"      Mean max prob per position:  {probs0.max(dim=-1).values.mean().item():.6f}")
    print(f"      Mean entropy:                {entropy[:real_len_0].mean().item():.4f}  (max possible: {max_entropy:.4f})")
    print(f"      Entropy ratio:               {entropy[:real_len_0].mean().item() / max_entropy:.4f}  (1.0 = uniform)")

    # Top-5 predictions for masked positions
    mlm_labels = masked_chunks[0]["mlm_labels"]
    masked_positions = [i for i, l in enumerate(mlm_labels) if l != -100]
    print(f"\n    Masked positions to predict: {len(masked_positions)}")

    show_n = min(15, len(masked_positions))
    if show_n > 0:
        print(f"\n    Top-5 MLM predictions (first {show_n} masked positions):")
        print(f"    {'pos':>5s}  {'true_id':>7s}  {'true_piece':15s}  {'rank':>4s}  {'top-1 (prob)':25s}  {'top-2':20s}  {'top-3':20s}  {'top-4':20s}  {'top-5':20s}")
        print(f"    {'─'*5}  {'─'*7}  {'─'*15}  {'─'*4}  {'─'*25}  {'─'*20}  {'─'*20}  {'─'*20}  {'─'*20}")
        for pos in masked_positions[:show_n]:
            true_id = mlm_labels[pos]
            true_piece = tokenizer.sp.IdToPiece(true_id)
            pos_probs = probs0[pos]
            top5 = pos_probs.topk(5)
            # Where does the true token rank?
            true_rank = (pos_probs > pos_probs[true_id]).sum().item() + 1
            preds = []
            for tid, p in zip(top5.indices, top5.values):
                piece = tokenizer.sp.IdToPiece(tid.item())
                mark = f"{GREEN}✓{RESET}" if tid.item() == true_id else " "
                preds.append(f"{piece:12s}{mark}({p.item():.4f})")
            print(f"    {pos:5d}  {true_id:7d}  {true_piece:15s}  {true_rank:4d}  {'  '.join(preds)}")

        # Quick accuracy summary
        top1_correct = sum(1 for pos in masked_positions if probs0[pos].argmax().item() == mlm_labels[pos])
        top5_correct = sum(1 for pos in masked_positions
                          if mlm_labels[pos] in probs0[pos].topk(5).indices.tolist())
        n_m = len(masked_positions)
        print(f"\n    MLM accuracy over all {n_m} masked positions:")
        print(f"      Top-1: {top1_correct}/{n_m} ({top1_correct/max(n_m,1)*100:.2f}%)")
        print(f"      Top-5: {top5_correct}/{n_m} ({top5_correct/max(n_m,1)*100:.2f}%)")
        print(f"      Chance: {1/tokenizer.vocab_size*100:.3f}% (top-1), {5/tokenizer.vocab_size*100:.3f}% (top-5)")

    # ══════════════════════════════════════════════════════════════════
    #  ITC HEAD — detailed
    # ══════════════════════════════════════════════════════════════════
    subheader("ITC HEAD")
    print(f"    Architecture: hidden[{d_model}] → Dense[{d_model}] → GELU → LN → Linear[{NUM_ITC_CLASSES}]")
    print(f"    Classes: {NUM_ITC_CLASSES} instruction types")
    itc0 = all_itc_logits[0]  # [1, seq, 14]
    print(f"\n    Output logits:")
    print(f"      Shape:  {list(itc0.shape)}  →  [batch, seq_len, num_itc_classes]")
    print(f"      dtype:  {itc0.dtype}")
    print(f"      Min:    {itc0.min().item():.4f}")
    print(f"      Max:    {itc0.max().item():.4f}")
    print(f"      Mean:   {itc0.mean().item():.4f}")
    print(f"      Std:    {itc0.std().item():.4f}")

    itc_probs0 = torch.softmax(itc0[0], dim=-1)  # [seq, 14]
    itc_labels0 = all_labels[0]
    instr_positions = [i for i, l in enumerate(itc_labels0) if l != ITC_IGNORE]

    # Per-class logit means for instruction tokens
    print(f"\n    Per-class mean logit (instruction tokens only):")
    print(f"    {'class':>4s}  {'name':20s}  {'mean_logit':>10s}  {'mean_prob':>10s}")
    print(f"    {'─'*4}  {'─'*20}  {'─'*10}  {'─'*10}")
    if instr_positions:
        instr_logits = itc0[0, instr_positions]  # [n_instr, 14]
        instr_probs = itc_probs0[instr_positions]  # [n_instr, 14]
        for cls_id in range(NUM_ITC_CLASSES):
            ml = instr_logits[:, cls_id].mean().item()
            mp = instr_probs[:, cls_id].mean().item()
            print(f"    {cls_id:4d}  {ITC_ID2NAME.get(cls_id, '?'):20s}  {ml:10.4f}  {mp:10.4f}")

    # Per-token predictions
    show_itc = min(20, len(instr_positions))
    if show_itc > 0:
        print(f"\n    ITC predictions for first {show_itc} instruction tokens:")
        print(f"    {'pos':>5s}  {'token':15s}  {'true':20s}  {'predicted':20s}  {'ok':>3s}  {'conf':>7s}  {'full logit vector (14 classes)'}")
        print(f"    {'─'*5}  {'─'*15}  {'─'*20}  {'─'*20}  {'─'*3}  {'─'*7}  {'─'*50}")
        correct = 0
        for pos in instr_positions[:show_itc]:
            true_cls = itc_labels0[pos]
            pred_cls = itc_probs0[pos].argmax().item()
            conf = itc_probs0[pos, pred_cls].item()
            is_correct = pred_cls == true_cls
            if is_correct:
                correct += 1
            mark = f"{GREEN}✓{RESET}" if is_correct else f"{RED}✗{RESET}"
            piece = tokenizer.sp.IdToPiece(masked_chunks[0]["input_ids"][pos])
            # Compact logit vector
            logit_vec = itc0[0, pos].tolist()
            logit_str = " ".join(f"{v:+.2f}" for v in logit_vec)
            print(f"    {pos:5d}  {piece:15s}  {ITC_ID2NAME.get(true_cls, '?'):20s}  "
                  f"{ITC_ID2NAME.get(pred_cls, '?'):20s}  {mark:>10s}  {conf:7.4f}  [{logit_str}]")

        print(f"\n    ITC accuracy on shown tokens: {correct}/{show_itc}")
        # Full accuracy over all instruction tokens
        all_correct = sum(1 for pos in instr_positions
                         if itc_probs0[pos].argmax().item() == itc_labels0[pos])
        print(f"    ITC accuracy over all {len(instr_positions)} instruction tokens: "
              f"{all_correct}/{len(instr_positions)} ({all_correct/max(len(instr_positions),1)*100:.2f}%)")
        print(f"    Chance: 1/{NUM_ITC_CLASSES} = {1/NUM_ITC_CLASSES*100:.2f}%")

    # ── Confusion sketch ──────────────────────────────────────────────
    if len(instr_positions) > 10:
        subheader("ITC CONFUSION MATRIX (sequence 0)")
        confusion = torch.zeros(NUM_ITC_CLASSES, NUM_ITC_CLASSES, dtype=torch.long)
        for pos in instr_positions:
            true_cls = itc_labels0[pos]
            pred_cls = itc_probs0[pos].argmax().item()
            confusion[true_cls, pred_cls] += 1
        # Print compact confusion matrix
        short_names = [ITC_ID2NAME.get(i, '?')[:8] for i in range(NUM_ITC_CLASSES)]
        print(f"    {'true/pred':>10s}  " + "  ".join(f"{s:>8s}" for s in short_names))
        print(f"    {'─'*10}  " + "  ".join("─" * 8 for _ in range(NUM_ITC_CLASSES)))
        for i in range(NUM_ITC_CLASSES):
            row = confusion[i]
            if row.sum() == 0:
                continue
            cells = []
            for j in range(NUM_ITC_CLASSES):
                v = row[j].item()
                if v == 0:
                    cells.append(f"{DIM}{'·':>8s}{RESET}")
                elif i == j:
                    cells.append(f"{GREEN}{v:>8d}{RESET}")
                else:
                    cells.append(f"{RED}{v:>8d}{RESET}")
            print(f"    {short_names[i]:>10s}  " + "  ".join(cells))

    return all_mlm_logits, all_itc_logits, all_hidden


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — LOSS & METRICS (untrained, sanity check)
# ══════════════════════════════════════════════════════════════════════════════

def step_loss_metrics(masked_chunks, all_labels, all_mlm_logits, all_itc_logits):
    header("STEP 7 · LOSS & METRICS (random weights — sanity check)")
    obj = PretrainingObjectives(label_smoothing=0.0)

    # Stack sequence 0 for quick check
    mlm_logits = all_mlm_logits[0]           # [1, S, V]
    mlm_labels = torch.tensor([masked_chunks[0]["mlm_labels"]], dtype=torch.long)
    itc_logits = all_itc_logits[0]           # [1, S, 14]
    itc_labels = torch.tensor([all_labels[0]], dtype=torch.long)

    loss_dict = obj.compute_loss(
        mlm_logits=mlm_logits, mlm_labels=mlm_labels,
        itc_logits=itc_logits, itc_labels=itc_labels,
    )

    mlm_acc = compute_mlm_accuracy(mlm_logits, mlm_labels)
    itc_acc = compute_itc_accuracy(itc_logits, itc_labels)
    ppl = compute_mlm_perplexity(loss_dict["mlm_loss"])

    print(f"  MLM loss:        {loss_dict['mlm_loss'].item():.4f}")
    print(f"  ITC loss:        {loss_dict['itc_loss'].item():.4f}")
    print(f"  Total loss:      {loss_dict['total_loss'].item():.4f}")
    print(f"  MLM accuracy:    {mlm_acc:.4f}  (chance ≈ {1/8000:.4f})")
    print(f"  ITC accuracy:    {itc_acc:.4f}  (chance ≈ {1/NUM_ITC_CLASSES:.4f})")
    print(f"  MLM perplexity:  {ppl:.2f}  (random ≈ {8000:.0f})")

    # Sanity checks
    subheader("Sanity checks")
    checks = [
        ("MLM loss > 0",           loss_dict["mlm_loss"].item() > 0),
        ("ITC loss > 0",           loss_dict["itc_loss"].item() > 0),
        ("Total ≈ MLM + ITC",      abs(loss_dict["total_loss"].item()
                                        - loss_dict["mlm_loss"].item()
                                        - loss_dict["itc_loss"].item()) < 1e-4),
        ("MLM accuracy < 5%",      mlm_acc < 0.05),
        ("ITC accuracy < 25%",     itc_acc < 0.25),
        ("Perplexity > 100",       ppl > 100),
    ]
    all_ok = True
    for desc, passed in checks:
        mark = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        if not passed:
            all_ok = False
        print(f"    [{mark}]  {desc}")

    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Walk a single PTX file through the full preprocessing + model pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/inspect_pipeline.py data/raw/00000_1_Conv2D_ReLU_BiasAdd.ptx
              python scripts/inspect_pipeline.py data/sprocessed/sprocessed/file_618.ptx --max-seq-length 512
        """),
    )
    parser.add_argument("ptx_file", type=Path, help="Path to a .ptx file")
    parser.add_argument("--tokenizer-path", type=Path,
                        default=PROJECT_ROOT / "data" / "tokenizer" / "ptx_tokenizer.model")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--d-model", type=int, default=128,
                        help="Hidden dim for the test model (small for speed)")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-tokens", type=int, default=30,
                        help="How many tokens to show in detailed views")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.ptx_file.exists():
        print(f"{RED}ERROR{RESET}: File not found: {args.ptx_file}")
        sys.exit(1)
    if not args.tokenizer_path.exists():
        print(f"{RED}ERROR{RESET}: Tokenizer not found: {args.tokenizer_path}")
        sys.exit(1)

    # ── Load tokenizer ────────────────────────────────────────────────────
    header("PIPELINE INSPECTOR")
    print(f"  PTX file:         {args.ptx_file}")
    print(f"  Tokenizer:        {args.tokenizer_path}")
    print(f"  Max seq length:   {args.max_seq_length}")
    print(f"  Overlap:          {args.overlap}")
    print(f"  Mask probability: {args.mask_prob}")
    print(f"  Model:            d={args.d_model}, L={args.num_layers}, H={args.num_heads}")
    print(f"  Seed:             {args.seed}")

    tokenizer = PTXTokenizer(str(args.tokenizer_path))

    # ── Run every stage ───────────────────────────────────────────────────
    text = step_read_file(args.ptx_file)

    token_ids, pieces = step_tokenize(text, tokenizer)

    chunks = step_chunking(token_ids, pieces, args.max_seq_length, args.overlap, tokenizer)

    all_labels, all_spans = step_itc_labeling(chunks, tokenizer, show_tokens=args.show_tokens)

    masked_chunks = step_mlm_masking(chunks, all_spans, tokenizer, args.mask_prob,
                                     show_tokens=args.show_tokens)

    all_mlm_logits, all_itc_logits, all_hidden = step_forward_pass(
        masked_chunks, chunks, all_labels, tokenizer,
        d_model=args.d_model, num_layers=args.num_layers,
        num_heads=args.num_heads, max_seq_length=args.max_seq_length,
    )

    all_ok = step_loss_metrics(masked_chunks, all_labels, all_mlm_logits, all_itc_logits)

    # ── Summary ───────────────────────────────────────────────────────────
    header("SUMMARY")
    print(f"  File:              {args.ptx_file.name}")
    print(f"  Tokens:            {len(token_ids):,}")
    print(f"  Sequences:         {len(chunks)}")
    n_masked = sum(1 for l in masked_chunks[0]['mlm_labels'] if l != -100)
    n_instr  = sum(1 for l in all_labels[0] if l != ITC_IGNORE)
    print(f"  Masked (seq 0):    {n_masked}")
    print(f"  Instr tok (seq 0): {n_instr}")
    print(f"  MLM logits shape:  {list(all_mlm_logits[0].shape)}")
    print(f"  ITC logits shape:  {list(all_itc_logits[0].shape)}")
    if all_ok:
        print(f"\n  {GREEN}{BOLD}All sanity checks passed ✓{RESET}")
    else:
        print(f"\n  {RED}{BOLD}Some sanity checks failed — review output above{RESET}")


if __name__ == "__main__":
    main()
