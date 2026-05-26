import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.architecture.complete_model import PTXTransformerForPretraining
from model.tokenizer.tokenizer import PTXTokenizer
from model.preprocessing.dataset_builder import PTXDataset
from evaluation import eval_utils
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def load_model_for_visualization(checkpoint_path: str,tokenizer_path: str,device: str = 'cuda',) -> Tuple[PTXTransformerForPretraining, PTXTokenizer]:
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    tokenizer = PTXTokenizer(tokenizer_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    state_dict = checkpoint['model_state_dict']
    use_itc = any(k.startswith('itc_head') for k in state_dict)
    model = PTXTransformerForPretraining(
        vocab_size=config.get('vocab_size', tokenizer.vocab_size), d_model=config.get('d_model', 768),
        num_layers=config.get('num_layers', 6), num_heads=config.get('num_heads', 8),
        d_ff=config.get('d_ff', 3072),dropout=0.0,  # no dropout at eval time
        max_seq_length=config.get('max_seq_length', 2048),padding_idx=tokenizer.pad_id,
        use_itc=use_itc,).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"loaded model from {checkpoint_path}")
    return model, tokenizer


def extract_embeddings( model: PTXTransformerForPretraining,tokenizer: PTXTokenizer,texts: List[str],max_length: int = 128, device: str = 'cpu',) -> np.ndarray:
    """
    extract mean-pooled embeddings for a list of ptx text snippets.
    each text is tokenized, padded to max_length, and passed through
    the encoder. the hidden states are mean-pooled over real tokens
    (ignoring padding) to get a single [d_model] vector per snippet.
    returns:
        numpy array of shape [n_texts, d_model]
    """
    embeddings = []
    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text)[:max_length]
            actual_len = len(ids)
            padding = [tokenizer.pad_id] * (max_length - actual_len)
            ids = ids + padding
            mask = [1] * actual_len + [0] * (max_length - actual_len)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            attention_mask = torch.tensor([mask], dtype=torch.long, device=device)
            outputs = model(input_ids, attention_mask=attention_mask)
            hidden = outputs['hidden_states']  # [1, seq_len, d_model]
            # mean pool over real tokens only
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            embeddings.append(pooled[0].cpu().numpy())

    return np.array(embeddings)



def visualize_tsne( embeddings: np.ndarray, labels: List[str], colors: Optional[List[str]] = None, title: str = 't-sne visualization of ptx embeddings', output_path: Optional[str] = None,) -> None:
    """
    plot embeddings reduced to 2d via t-sne.
    t-sne is a nonlinear technique that models pairwise similarities as
    probability distributions and minimizes kl divergence between the
    high-d and low-d distributions. it excels at revealing clusters.
    perplexity is clamped to (n_samples - 1) because t-sne requires
    perplexity < n_samples. for small datasets this matters.
    args:
        embeddings: [n_samples, d_model] — high-dimensional representations
        labels:     text label for each point (shown as annotation)
        colors:     one color per point (e.g., 'blue' for memory, 'red' for compute)
        title:      plot title
        output_path: if given, save png; otherwise plt.show()
    """
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        print(f"need at least 2 samples for t-sne, got {n_samples}")
        return

    # perplexity must be < n_samples
    perplexity = min(30, n_samples - 1)
    print(f"running t-sne on {n_samples} embeddings (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(12, 8))
    if colors:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                    c=colors, alpha=0.6, s=100)
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                    alpha=0.6, s=100)
    # annotate each point with its label
    for i, label in enumerate(labels):
        plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                     fontsize=8, alpha=0.7)

    plt.title(title, fontsize=14)
    plt.xlabel('t-sne component 1')
    plt.ylabel('t-sne component 2')
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"saved t-sne plot to {output_path}")
    else:
        plt.show()

    plt.close()



def visualize_pca( embeddings: np.ndarray, labels: List[str],title: str = 'pca visualization of ptx embeddings', output_path: Optional[str] = None,) -> None:
    """
    plot embeddings reduced to 2d via pca.
    pca finds the two orthogonal directions that capture the most variance
    in the data. the explained variance ratio tells us what fraction of
    total information is preserved (ideally > 50% combined).
    unlike t-sne, pca is linear and deterministic, so the axes have
    interpretable meaning and the result is reproducible.
    args:
        embeddings: [n_samples, d_model]
        labels:     text label for each point
        title:      plot title
        output_path: if given, save png; otherwise plt.show()
    """
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        print(f"need at least 2 samples for pca, got {n_samples}")
        return

    print(f"running pca on {n_samples} embeddings...")

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=100)

    for i, label in enumerate(labels):
        plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                     fontsize=8, alpha=0.7)
    var0 = pca.explained_variance_ratio_[0] * 100
    var1 = pca.explained_variance_ratio_[1] * 100
    plt.title(title, fontsize=14)
    plt.xlabel(f'pc1 ({var0:.1f}% variance)')
    plt.ylabel(f'pc2 ({var1:.1f}% variance)')
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"saved pca plot to {output_path}")
    else:
        plt.show()

    plt.close()




def visualize_opcode_clusters(checkpoint_path: str,tokenizer_path: str,
    output_dir: str = 'evaluation/plots', device: str = 'cuda',
    data_dir: str = eval_utils.DEFAULT_DATA_DIR,
    cache_dir: str = eval_utils.DEFAULT_CACHE_DIR,
    max_seq_length: int = 2048,) -> None:
    """
    visualize how different opcodes cluster in embedding space.
    loads PTXDataset(split='test') and extracts real memory/compute instructions.
    falls back to synthetic snippets if test data is insufficient.
    creates both t-sne and pca plots showing memory ops (blue) vs compute ops (red).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_for_visualization(checkpoint_path, tokenizer_path, device)
    device_str = str(next(model.parameters()).device)

    # try to extract real instructions from test dataset
    memory_texts = []
    compute_texts = []

    test_dataset = eval_utils.load_test_dataset(
        tokenizer=tokenizer,
        data_dir=data_dir,
        cache_dir=cache_dir,
        max_seq_length=max_seq_length,
    )
    texts, labels = eval_utils.extract_labeled_instructions(
        test_dataset, tokenizer, task='instruction_type', max_per_class=20,
    )
    for t, l in zip(texts, labels):
        if l == 1:
            memory_texts.append(t)
        elif l == 0:
            compute_texts.append(t)

    data_source = 'test_dataset'
    if len(memory_texts) < 4 or len(compute_texts) < 4:
        print(f"  insufficient test data ({len(memory_texts)} memory, {len(compute_texts)} compute), using synthetic")
        data_source = 'synthetic'
        memory_texts = [
            "ld.global.f32 %f1, [%rd1];",
            "ld.shared.f32 %f2, [%rd2];",
            "ld.const.f32 %f3, [%rd3];",
            "st.global.f32 [%rd4], %f4;",
            "st.shared.u32 [%rd5], %r1;",
            "ld.global.u32 %r2, [%rd6];",
            "ld.global.f64 %fd1, [%rd7];",
            "atom.global.add.f32 %f5, [%rd8], %f6;",
        ]
        compute_texts = [
            "add.f32 %f1, %f2, %f3;",
            "mul.f32 %f4, %f5, %f6;",
            "fma.rn.f32 %f7, %f8, %f9, %f10;",
            "mad.lo.s32 %r1, %r2, %r3, %r4;",
            "sub.f32 %f11, %f12, %f13;",
            "min.f32 %f14, %f15, %f16;",
            "max.s32 %r5, %r6, %r7;",
            "shr.u32 %r8, %r9, 2;",
        ]

    print(f"embedding {len(memory_texts)} memory + {len(compute_texts)} compute instructions (source: {data_source})")
    all_texts = memory_texts + compute_texts
    embeddings = extract_embeddings(model, tokenizer, all_texts, device=device_str)
    labels_list = [t.split()[0].rstrip(';') for t in all_texts]
    colors = ['blue'] * len(memory_texts) + ['red'] * len(compute_texts)

    visualize_tsne(embeddings, labels_list, colors,
        title=f't-sne: memory ops (blue) vs compute ops (red) [{data_source}]',
        output_path=str(output_path / 'opcode_clusters_tsne.png'))

    visualize_pca(embeddings, labels_list,
        title=f'pca: memory vs compute operations [{data_source}]',
        output_path=str(output_path / 'opcode_clusters_pca.png'))

    print(f"\nvisualization complete. plots saved to {output_path}/")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='visualize ptx embeddings')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to model checkpoint (.pt)')
    parser.add_argument('--tokenizer', type=str, default=eval_utils.DEFAULT_TOKENIZER, help='path to sentencepiece .model file')
    parser.add_argument('--output-dir', type=str, default='evaluation/plots', help='directory to save plots')
    parser.add_argument('--data-dir', type=str, default=eval_utils.DEFAULT_DATA_DIR, help='path to processed PTX directory')
    parser.add_argument('--cache-dir', type=str, default=eval_utils.DEFAULT_CACHE_DIR, help='path to dataset cache directory')
    parser.add_argument('--max-seq-length', type=int, default=2048)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    visualize_opcode_clusters(checkpoint_path=args.checkpoint,tokenizer_path=args.tokenizer,output_dir=args.output_dir,device=args.device,data_dir=args.data_dir,cache_dir=args.cache_dir,max_seq_length=args.max_seq_length,)