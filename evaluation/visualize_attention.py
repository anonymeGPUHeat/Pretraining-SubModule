import sys
import torch
import random
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.architecture.complete_model import PTXTransformerForPretraining
from model.tokenizer.tokenizer import PTXTokenizer
from model.preprocessing.dataset_builder import PTXDataset
from evaluation import eval_utils
import seaborn as sns
import re

def sanitize_filename(text: str, max_len: int = 40) -> str:
    # collapse whitespace (including tabs/newlines) to single underscore
    text = re.sub(r'\s+', '_', text)
    # keep only alphanumerics and underscores
    text = re.sub(r'[^A-Za-z0-9_]', '', text)
    # trim to max length and strip leading/trailing underscores
    text = text[:max_len].strip('_')
    return text or 'snippet'


def load_model_for_attention( checkpoint_path: str, tokenizer_path: str, device: str = 'cuda',):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    tokenizer = PTXTokenizer(tokenizer_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    state_dict = checkpoint['model_state_dict']
    use_itc = any(k.startswith('itc_head') for k in state_dict)
    model = PTXTransformerForPretraining( vocab_size=config.get('vocab_size', tokenizer.vocab_size), d_model=config.get('d_model', 768),
     num_layers=config.get('num_layers', 12), num_heads=config.get('num_heads', 8),
        d_ff=config.get('d_ff', 3072), dropout=0.0, max_seq_length=config.get('max_seq_length', 2048), padding_idx=tokenizer.pad_id,
        use_itc=use_itc,).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer


def visualize_attention_heatmap( model: PTXTransformerForPretraining, tokenizer: PTXTokenizer, ptx_code: str, layer_idx: int = -1, output_path: Optional[str] = None,) -> None:
    """
    create attention heatmap averaged over all heads for a ptx snippet.
    the heatmap has shape [seq_len, seq_len] where entry (i, j) shows
    how much token i attends to token j. bright cells = high attention.
    args:
        model:       pretrained PTXTransformerForPretraining (eval mode)
        tokenizer:   PTXTokenizer with sentencepiece backend
        ptx_code:    raw ptx text, e.g. "ld.global.f32 %f1, [%rd10];"
        layer_idx:   which layer to visualize (-1 = last layer)
        output_path: if given, save png; otherwise plt.show()
    """
    model.eval()
    device = next(model.parameters()).device
    tokens = tokenizer.encode_as_pieces(ptx_code)
    input_ids = torch.tensor([tokenizer.encode(ptx_code)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        attn_weights = model.get_attention_maps(input_ids, attention_mask=attention_mask, layer_idx=layer_idx,)

    attn_avg = attn_weights[0].mean(dim=0).cpu().numpy()
    seq_len = attn_avg.shape[0]
    tokens = tokens[:seq_len]

    plt.figure(figsize=(12, 10))

    sns.heatmap( attn_avg, xticklabels=tokens, yticklabels=tokens, cmap='viridis', cbar=True, square=True,)

    plt.title(f'attention heatmap — layer {layer_idx} (averaged over heads)', fontsize=13)
    plt.xlabel('key position (attended to)')
    plt.ylabel('query position (attending from)')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"saved attention heatmap to {output_path}")
    else:
        plt.show()

    plt.close()



def visualize_attention_per_head( model: PTXTransformerForPretraining, tokenizer: PTXTokenizer, ptx_code: str, layer_idx: int = -1, output_path: Optional[str] = None,) -> None:
    """
    create a grid of attention heatmaps, one per head.
    this reveals head specialization. common patterns in transformer models:
      - one head for local context (attending to neighbors)
      - one head for long-range dependencies
      - one head for positional patterns
    in a ptx encoder you might see:
      - a head that links opcodes to their operands
      - a head that links ld/st to their address space
      - a head that attends to the instruction boundary (semicolons)
    the grid is arranged as 2 rows × (num_heads/2) columns.
    args:
        model:       pretrained model
        tokenizer:   ptx tokenizer
        ptx_code:    raw ptx text
        layer_idx:   which layer (-1 = last)
        output_path: if given, save png
    """
    model.eval()
    device = next(model.parameters()).device
    tokens = tokenizer.encode_as_pieces(ptx_code)
    input_ids = torch.tensor([tokenizer.encode(ptx_code)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        attn_weights = model.get_attention_maps(input_ids, attention_mask=attention_mask, layer_idx=layer_idx,)
    # attn_weights shape: [1, num_heads, seq_len, seq_len]
    attn = attn_weights[0].cpu().numpy()  # [num_heads, seq_len, seq_len]
    num_heads = attn.shape[0]
    seq_len = attn.shape[1]
    tokens = tokens[:seq_len]
    cols = (num_heads + 1) // 2
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 8))
    axes = axes.flatten()

    for head_idx in range(num_heads):
        ax = axes[head_idx]
        im = ax.imshow(attn[head_idx], cmap='viridis', aspect='auto')
        ax.set_title(f'head {head_idx}', fontsize=10)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=6)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=6)
    for idx in range(num_heads, len(axes)):
        axes[idx].set_visible(False)
    plt.suptitle(f'per-head attention — layer {layer_idx}', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"saved per-head attention to {output_path}")
    else:
        plt.show()

    plt.close()



def run_attention_analysis(checkpoint_path: str,tokenizer_path: str,output_dir: str = 'evaluation/plots',device: str = 'cuda',data_dir: str = eval_utils.DEFAULT_DATA_DIR,cache_dir: str = eval_utils.DEFAULT_CACHE_DIR,max_seq_length: int = 2048,num_test_snippets: int = 3,) -> None:
    """
    run full attention analysis on representative ptx snippets.
    loads PTXDataset(split='test') and samples real multi-instruction snippets
    in addition to the built-in synthetic examples.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_for_attention(checkpoint_path, tokenizer_path, device)

    # load real snippets from test set
    test_dataset = eval_utils.load_test_dataset(
        tokenizer=tokenizer,
        data_dir=data_dir,
        cache_dir=cache_dir,
        max_seq_length=max_seq_length,
    )
    decoded = eval_utils.decode_chunks(test_dataset, tokenizer, max_samples=50)
    # pick multi-instruction snippets (truncated to ~100 tokens for readable heatmaps)
    real_snippets = []
    for text in decoded:
        lines = [l.strip() for l in text.split('\n') if l.strip() and not l.strip().startswith(('.', '//'))]
        if len(lines) >= 3:
            # take 3-5 instruction lines for a readable heatmap
            snippet = ' '.join(lines[:5])
            if len(snippet) < 300:
                real_snippets.append(snippet)
        if len(real_snippets) >= num_test_snippets:
            break

    # built-in synthetic snippets
    synthetic_snippets = [
        "ld.global.f32 %f1, [%rd10];",
        "fma.rn.f32 %f2, %f3, %f4, %f5;",
        "ld.shared.f32 %f6, [%rd1]; add.f32 %f7, %f6, %f8; st.shared.f32 [%rd2], %f7;",
    ]

    all_snippets = []
    for s in real_snippets:
        all_snippets.append(('test', s))
    for s in synthetic_snippets:
        all_snippets.append(('synthetic', s))

    print(f"visualizing {len(real_snippets)} test snippets + {len(synthetic_snippets)} synthetic snippets")

    for i, (source, snippet) in enumerate(all_snippets):
        clean_name = sanitize_filename(snippet)
        prefix = f'{source}_{i}'
        visualize_attention_heatmap(model, tokenizer, snippet, layer_idx=-1,
            output_path=str(output_path / f'attn_last_{prefix}_{clean_name}.png'))
        visualize_attention_heatmap(model, tokenizer, snippet, layer_idx=0,
            output_path=str(output_path / f'attn_first_{prefix}_{clean_name}.png'))
        visualize_attention_per_head(model, tokenizer, snippet, layer_idx=-1,
            output_path=str(output_path / f'attn_heads_{prefix}_{clean_name}.png'))

    print(f"\nattention analysis complete. plots saved to {output_path}/")




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='visualize ptx attention patterns')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to model checkpoint (.pt)')
    parser.add_argument('--tokenizer', type=str, default=eval_utils.DEFAULT_TOKENIZER, help='path to sentencepiece .model file')
    parser.add_argument('--output-dir', type=str, default='evaluation/plots', help='directory to save plots')
    parser.add_argument('--data-dir', type=str, default=eval_utils.DEFAULT_DATA_DIR, help='path to processed PTX directory')
    parser.add_argument('--cache-dir', type=str, default=eval_utils.DEFAULT_CACHE_DIR, help='path to dataset cache directory')
    parser.add_argument('--max-seq-length', type=int, default=2048)
    parser.add_argument('--num-test-snippets', type=int, default=15, help='number of real test snippets to visualize')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    run_attention_analysis(checkpoint_path=args.checkpoint,tokenizer_path=args.tokenizer,output_dir=args.output_dir,device=args.device,data_dir=args.data_dir,cache_dir=args.cache_dir,max_seq_length=args.max_seq_length,num_test_snippets=args.num_test_snippets,)