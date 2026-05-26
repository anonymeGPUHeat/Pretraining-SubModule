import torch
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
import json
import pickle
from typing import Dict, List, Optional, Tuple, Literal
import random


class PTXDataset(Dataset):
    """
    PTX Dataset for encoder pre-training on normalized PTX files.

    Features:
        - Chunks normalized PTX files into ``max_seq_length`` pieces with overlap
        - Chunk boundaries aligned to instruction boundaries (semicolons) so that
          no instruction is ever split across two chunks
        - Train / val / test splitting at file level (deterministic seed)
        - Efficient caching to disk (one cache per split configuration)
        - Per-chunk ordering metadata (``chunk_in_file``, ``total_chunks_in_file``)
          for later full-file reconstruction (e.g. attention visualisation)
        - Empty / corrupt files excluded from all counts
        - Generates proper attention masks for padding tokens

    Args:
        tokenizer:       PTXTokenizer instance.
        data_dir:        Directory containing normalized ``.ptx`` files.
        max_seq_length:  Maximum token length per chunk (default 2048).
        overlap:         Number of tokens to overlap between consecutive chunks.
        cache_dir:       Where to cache processed chunks (set ``None`` to disable).
        max_files:       Limit number of files — useful for quick testing.
        verbose:         Print progress bars and statistics.
        split:           ``'train'``, ``'val'``, ``'test'``, or ``'all'``.
        train_ratio:     Fraction of *non-empty* files for training (default 0.8).
        val_ratio:       Fraction of *non-empty* files for validation (default 0.1).
        seed:            Random seed for reproducible splitting.
    """

    def __init__(self,tokenizer,data_dir: str | Path,max_seq_length: int = 2048,overlap: int = 128,cache_dir: Optional[str | Path] = None,max_files: Optional[int] = None,verbose: bool = True,split: Literal['train', 'val', 'test', 'all'] = 'all',train_ratio: float = 0.8,val_ratio: float = 0.1, seed: int = 42,):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.overlap = overlap
        self.verbose = verbose
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.chunks, self.metadata = self._load_or_create_dataset(max_files)
        if self.verbose:
            self._print_statistics()
    def _cache_key(self) -> str:
        return f'seq{self.max_seq_length}_ovlp{self.overlap}_{self.split}'

    def _load_or_create_dataset( self, max_files: Optional[int],) -> Tuple[List[Dict], Dict]:
        if self.cache_dir:
            key = self._cache_key()
            cache_file = self.cache_dir / f'dataset_{key}.pkl'
            metadata_file = self.cache_dir / f'metadata_{key}.json'
            if cache_file.exists() and metadata_file.exists():
                if self.verbose:
                    print(f"Loading cached dataset from {cache_file}")
                with open(cache_file, 'rb') as f:
                    chunks = pickle.load(f)
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                if self.verbose:
                    print(f"Loaded {len(chunks):,} chunks from cache")
                return chunks, metadata

        if self.verbose:
            print(f"Creating dataset from {self.data_dir}")
        chunks, metadata = self._create_chunks(max_files)

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            key = self._cache_key()
            cache_file = self.cache_dir / f'dataset_{key}.pkl'
            metadata_file = self.cache_dir / f'metadata_{key}.json'
            if self.verbose:
                print(f"Saving chunks to cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(chunks, f)
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            if self.verbose:
                print(f"Cached {len(chunks):,} chunks")
        return chunks, metadata


    def _create_chunks(self, max_files: Optional[int],) -> Tuple[List[Dict], Dict]:
        stride = self.max_seq_length - self.overlap
        all_ptx = sorted(self.data_dir.glob('*.ptx'))
        ptx_files: List[Path] = []
        skipped_empty_discovery = 0
        for p in tqdm(all_ptx, desc="Filtering empty files", disable=not self.verbose):
            if p.stat().st_size == 0:
                skipped_empty_discovery += 1
                continue
            ptx_files.append(p)
        n_total = len(ptx_files)
        if self.verbose:
            print(f"Found {len(all_ptx):,} PTX files, "
                  f"{skipped_empty_discovery:,} empty (0-byte) removed → "
                  f"{n_total:,} usable files")

        rng = random.Random(self.seed)
        indices = list(range(n_total))
        rng.shuffle(indices)

        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        if self.split == 'train':
            selected = indices[:n_train]
        elif self.split == 'val':
            selected = indices[n_train:n_train + n_val]
        elif self.split == 'test':
            selected = indices[n_train + n_val:]
        else:  
            selected = indices
        split_files = [ptx_files[i] for i in selected]
        if self.verbose:
            print(f"Using {self.split.upper()} split: {len(split_files):,} / {n_total:,} files")
        if max_files:
            split_files = split_files[:max_files]
        newline_piece = self.tokenizer.encode_as_pieces("\n")[0]
        boundary_tokens = frozenset({';', ':', newline_piece})
        chunks: List[Dict] = []
        stats = {
            'total_files_found': len(all_ptx),
            'empty_files_removed': skipped_empty_discovery,
            'usable_files': n_total,
            'split_files': len(split_files),
            'processed_files': 0,
            'skipped_empty': 0,
            'skipped_errors': 0,
            'total_chunks': 0,
            'total_tokens': 0,
            'file_details': {},
        }

        if self.verbose:
            print(f"Processing {len(split_files):,} PTX files...")
            print(f"Max seq length: {self.max_seq_length}, "
                  f"Overlap: {self.overlap}, Stride: {stride}")

        for file_idx, file_path in enumerate(tqdm(split_files, desc="Chunking files", disable=not self.verbose)):
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                if not content.strip():
                    stats['skipped_empty'] += 1
                    continue

                token_ids = self.tokenizer.encode(content)
                if len(token_ids) == 0:
                    stats['skipped_empty'] += 1
                    continue

                token_pieces = self.tokenizer.encode_as_pieces(content)
                assert len(token_ids) == len(token_pieces), (
                    f"Token/piece mismatch in {file_path.name}: "
                    f"{len(token_ids)} vs {len(token_pieces)}")
                n_tokens = len(token_ids)
                stats['total_tokens'] += n_tokens
                file_chunks_list: List[Dict] = []
                start_idx = 0
                while start_idx < n_tokens:
                    end_idx = min(start_idx + self.max_seq_length, n_tokens)
                    if end_idx < n_tokens:
                        ideal_next = start_idx + stride
                        search_lo = max(ideal_next - 64, start_idx + stride // 2)
                        search_hi = min(ideal_next + 64, n_tokens)
                        best_boundary = ideal_next  

                        for i in range(search_lo, search_hi):
                            if token_pieces[i] in boundary_tokens:
                                best_boundary = i + 1  
                                break

                        next_start = best_boundary
                    else:
                        next_start = n_tokens

                    chunk_tokens = token_ids[start_idx:end_idx]
                    actual_length = len(chunk_tokens)
                    if actual_length < self.max_seq_length:
                        padding = [self.tokenizer.pad_id] * (self.max_seq_length - actual_length )
                        chunk_tokens = chunk_tokens + padding
                    attention_mask = ([1] * actual_length + [0] * (self.max_seq_length - actual_length) )
                    file_chunks_list.append({
                        'input_ids': chunk_tokens,
                        'attention_mask': attention_mask,
                        'file_idx': file_idx,
                        'chunk_in_file': len(file_chunks_list),
                        'actual_length': actual_length,
                        'start_token_idx': start_idx,
                    })

                    start_idx = next_start
                n_file_chunks = len(file_chunks_list)
                for c in file_chunks_list:
                    c['total_chunks_in_file'] = n_file_chunks
                chunks.extend(file_chunks_list)

                stats['file_details'][str(file_idx)] = {
                    'path': str(file_path.name),
                    'tokens': n_tokens,
                    'chunks': n_file_chunks,
                }
                stats['processed_files'] += 1
                stats['total_chunks'] += n_file_chunks

            except Exception as e:
                if self.verbose:
                    print(f"\nError processing {file_path.name}: "
                          f"{type(e).__name__}: {str(e)[:100]}")
                stats['skipped_errors'] += 1
                continue

        metadata = {
            'max_seq_length': self.max_seq_length,'overlap': self.overlap,'stride': stride,
            'vocab_size': self.tokenizer.vocab_size,'split': self.split,'statistics': stats,}
        return chunks, metadata


    def _print_statistics(self):
        stats = self.metadata['statistics']
        print("\nDATASET STATISTICS")
        print(f"  Split:                     {self.split.upper()}")
        print(f"  Total files found:         {stats.get('total_files_found', stats.get('total_files', '?')):,}")
        print(f"  Empty files removed:       {stats.get('empty_files_removed', 0):,}")
        print(f"  Usable files:              {stats.get('usable_files', '?'):,}")
        print(f"  Files in this split:       {stats.get('split_files', '?'):,}")
        print(f"  Files processed:           {stats['processed_files']:,}")
        print(f"  Files skipped (empty):     {stats['skipped_empty']:,}")
        print(f"  Files skipped (errors):    {stats['skipped_errors']:,}")
        print(f"  Total tokens:              {stats['total_tokens']:,}")
        print(f"  Total chunks:              {stats['total_chunks']:,}")
        print(f"  Avg chunks per file:       "
              f"{stats['total_chunks'] / max(stats['processed_files'], 1):.1f}")
        print(f"  Max sequence length:       {self.max_seq_length}")
        print(f"  Overlap:                   {self.overlap}")
        print(f"  Stride:                    {self.metadata['stride']}")


    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        return {
            'input_ids': torch.tensor(chunk['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(chunk['attention_mask'], dtype=torch.long),
        }

    def get_file_info(self, file_idx: int) -> Dict:
        return self.metadata['statistics']['file_details'].get(str(file_idx), {})

    def get_chunk_info(self, chunk_idx: int) -> Dict:
        if 0 <= chunk_idx < len(self.chunks):
            chunk = self.chunks[chunk_idx]
            return {
                'chunk_idx': chunk_idx,
                'file_idx': chunk['file_idx'],
                'chunk_in_file': chunk['chunk_in_file'],
                'total_chunks_in_file': chunk['total_chunks_in_file'],
                'actual_length': chunk['actual_length'],
                'start_token_idx': chunk['start_token_idx'],
                'is_padded': chunk['actual_length'] < self.max_seq_length,
            }
        return {}

    def get_file_chunks(self, file_idx: int) -> List[int]:
        return [
            i for i, c in enumerate(self.chunks)
            if c['file_idx'] == file_idx
        ]


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
    }


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tokenizer.tokenizer import PTXTokenizer

    project_root = Path(__file__).parent.parent.parent
    tokenizer_path = project_root / 'ptx_tokenizer.model'
    data_dir = project_root / 'data' / 'sprocessed' / 'sprocessed'
    cache_dir = project_root / 'data' / 'cache'

    print("Testing PTX Dataset Builder\n")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Data dir:  {data_dir}")
    print(f"Cache dir: {cache_dir}\n")

    if not tokenizer_path.exists():
        print(f"ERROR: Tokenizer not found at {tokenizer_path}")
        print("Please run tokenizer training first!")
        sys.exit(1)

    tokenizer = PTXTokenizer(str(tokenizer_path))
    print(f"Loaded tokenizer (vocab size: {tokenizer.vocab_size})")

    dataset = PTXDataset(tokenizer=tokenizer,data_dir=data_dir, max_seq_length=2048, overlap=128,cache_dir=cache_dir,max_files=100,verbose=True,)

    print(f"Total chunks: {len(dataset):,}")
    sample = dataset[0]
    print(f"\nSample chunk:")
    print(f"  input_ids shape:     {sample['input_ids'].shape}")
    print(f"  attention_mask shape: {sample['attention_mask'].shape}")
    print(f"  Real tokens:         {sample['attention_mask'].sum().item()}")
    print(f"  Padding tokens:      {(sample['attention_mask'] == 0).sum().item()}")
    print(f"  Chunk info:          {dataset.get_chunk_info(0)}")

    from torch.utils.data import DataLoader
    dataloader = DataLoader( dataset, batch_size=16,shuffle=True,collate_fn=collate_fn, num_workers=10,)
    batch = next(iter(dataloader))
    print(f"  Batch input_ids shape:     {batch['input_ids'].shape}")
    print(f"  Batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"\nCache saved to: {cache_dir}")