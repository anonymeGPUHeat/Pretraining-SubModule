import sys
import re
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.tokenizer.tokenizer import PTXTokenizer
from model.preprocessing.dataset_builder import PTXDataset
from model.preprocessing.itc_labels import (
    ITC_CLASSES, ITC_IGNORE, classify_instruction,
    _extract_opcode_from_text, _get_memory_address_space,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / 'data' / 'sprocessed' / 'sprocessed'
DEFAULT_CACHE_DIR = PROJECT_ROOT / 'data' / 'cache'
DEFAULT_TOKENIZER = PROJECT_ROOT / 'data' / 'tokenizer' / 'ptx_tokenizer.model'


def get_non_overlap_range(dataset: PTXDataset, chunk_idx: int) -> Tuple[int, int]:
    """
    return the (start, end) token-position range within a chunk that is NOT
    in the overlap zone with the previous chunk.

    PTXDataset chunks files with ``overlap`` tokens of context carried over
    between consecutive chunks of the SAME file.  for the first chunk of a
    file (chunk_in_file == 0) the full range [0, actual_length) is valid.
    for subsequent chunks the first ``overlap`` positions duplicate tokens
    already present in the previous chunk.

    use this when building frequency maps or evaluating to avoid
    double-counting tokens that appear in two adjacent chunks.
    """
    chunk = dataset.chunks[chunk_idx]
    actual_length = chunk['actual_length']
    chunk_in_file = chunk.get('chunk_in_file', 0)

    if chunk_in_file == 0:
        return 0, actual_length

    # for non-first chunks, skip the overlap zone
    overlap = dataset.overlap
    start = min(overlap, actual_length)
    return start, actual_length


def load_test_dataset( tokenizer: PTXTokenizer, data_dir: Optional[str] = None,cache_dir: Optional[str] = None,max_seq_length: int = 2048,overlap: int = 128,seed: int = 42,) -> PTXDataset:
    """
    load the test split of PTXDataset.
    uses the same deterministic split as training (seed=42) so the test
    files are guaranteed to be disjoint from train and val.
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    print(f"\nloading test dataset from {data_dir} ...")
    test_dataset = PTXDataset(tokenizer=tokenizer, data_dir=data_dir, max_seq_length=max_seq_length,overlap=overlap,cache_dir=cache_dir,split='test', verbose=True, seed=seed,)
    print(f"test dataset: {len(test_dataset):,} chunks")
    return test_dataset


def decode_chunks(test_dataset: PTXDataset,tokenizer: PTXTokenizer,max_samples: int = 100,seed: int = 42,) -> List[str]:
    """
    decode a random sample of test chunks back into PTX text strings.
    these are full multi-instruction sequences (up to max_seq_length tokens).
    """
    rng = random.Random(seed)
    n = min(max_samples, len(test_dataset))
    indices = rng.sample(range(len(test_dataset)), n)

    snippets = []
    for idx in indices:
        sample = test_dataset[idx]
        input_ids = sample['input_ids'].tolist()
        attention_mask = sample['attention_mask'].tolist()
        real_ids = [tid for tid, m in zip(input_ids, attention_mask) if m == 1]
        text = tokenizer.decode(real_ids)
        if text.strip():
            snippets.append(text.strip())
    return snippets


def extract_instruction_lines(text: str) -> List[str]:
    """
    split a PTX text block into individual instruction lines.
    filters out directives, labels, empty lines, and comments.
    returns only actual PTX instructions (lines containing opcodes).
    """
    lines = []
    for raw_line in text.split('\n'):
        line = raw_line.strip()
        if not line or line.startswith('//'):
            continue
        # skip directives, register declarations, labels, braces
        if line.startswith('.') and not line.startswith('@'):
            continue
        if line in ('{', '}', '(', ')'):
            continue
        # skip bare labels like <BB_3>:
        if line.endswith(':') and not line.startswith(('@', ' ', '\t')):
            continue
        lines.append(line)
    return lines



_ADDR_SPACE_MAP = {'global': 0, 'shared': 1, 'local': 2, 'const': 3, 'param': 2}


def _get_addr_space_label(instruction: str) -> Optional[int]:
    """classify a memory instruction's address space. returns None for non-memory."""
    opcode = _extract_opcode_from_text(instruction)
    if opcode not in ('ld', 'ldu', 'st', 'atom', 'red'):
        return None
    space = _get_memory_address_space(instruction)
    if not space:
        return None
    space_name = space.lstrip('.')  # '.global' → 'global'
    return _ADDR_SPACE_MAP.get(space_name)



_DTYPE_MAP = {'f32': 0, 'u32': 1, 's32': 2, 'f64': 3, 'b32': 4}
_DTYPE_PATTERN = re.compile(r'\.(f32|u32|s32|f64|b32|u64|s64|f16|b64|b16|u16|s16|u8|s8|b8)')


def _get_dtype_label(instruction: str) -> Optional[int]:
    """extract the data type from an instruction. returns None if not in _DTYPE_MAP."""
    m = _DTYPE_PATTERN.search(instruction)
    if not m:
        return None
    return _DTYPE_MAP.get(m.group(1))



def _get_instruction_type_label(instruction: str) -> Optional[int]:
    """
    binary classification: memory (1) vs compute (0).
    returns None for non-classifiable instructions (control flow, sync, etc.)
    """
    itc = classify_instruction(instruction)
    if itc == ITC_IGNORE:
        return None
    # memory ops: global/shared/local load/store + atomic
    if itc in (ITC_CLASSES['GLOBAL_LOAD'], ITC_CLASSES['GLOBAL_STORE'],ITC_CLASSES['SHARED_LOAD'], ITC_CLASSES['SHARED_STORE'],ITC_CLASSES['LOCAL_LOAD'], ITC_CLASSES['LOCAL_STORE'],ITC_CLASSES['ATOMIC_REDUCE'],):
        return 1  # memory
    # compute ops: arithmetic + logic/bitwise
    if itc in (ITC_CLASSES['ARITHMETIC'], ITC_CLASSES['LOGIC_BITWISE']):
        return 0  # compute
    return None  # skip comparison, control flow, sync, conversion, special


def extract_labeled_instructions(test_dataset: PTXDataset,tokenizer: PTXTokenizer,task: str = 'instruction_type',max_per_class: int = 50,seed: int = 42,) -> Tuple[List[str], List[int]]:
    """
    extract real PTX instructions from test dataset with auto-generated labels.

    args:
        test_dataset: PTXDataset(split='test')
        tokenizer:    PTXTokenizer
        task:         'instruction_type' | 'address_space' | 'data_type'
        max_per_class: max samples per class to collect
        seed:         random seed for reproducibility

    returns:
        (texts, labels) — lists of instruction strings and their integer labels
    """
    if task == 'instruction_type':
        label_fn = _get_instruction_type_label
        num_classes = 2
    elif task == 'address_space':
        label_fn = _get_addr_space_label
        num_classes = 4
    elif task == 'data_type':
        label_fn = _get_dtype_label
        num_classes = 5
    else:
        raise ValueError(f"unknown task: {task}")

    # collect instructions from test chunks
    class_buckets: Dict[int, List[str]] = {c: [] for c in range(num_classes)}
    all_full = lambda: all(len(v) >= max_per_class for v in class_buckets.values())

    rng = random.Random(seed)
    indices = list(range(len(test_dataset)))
    rng.shuffle(indices)

    for idx in indices:
        if all_full():
            break
        sample = test_dataset[idx]
        input_ids = sample['input_ids'].tolist()
        attention_mask = sample['attention_mask'].tolist()
        real_ids = [tid for tid, m in zip(input_ids, attention_mask) if m == 1]
        text = tokenizer.decode(real_ids)
        for line in extract_instruction_lines(text):
            label = label_fn(line)
            if label is not None and label in class_buckets:
                if len(class_buckets[label]) < max_per_class:
                    class_buckets[label].append(line)

    texts, labels = [], []
    for cls, lines in class_buckets.items():
        for line in lines:
            texts.append(line)
            labels.append(cls)

    # shuffle
    combined = list(zip(texts, labels))
    rng.shuffle(combined)
    if combined:
        texts, labels = zip(*combined)
        texts, labels = list(texts), list(labels)

    class_counts = {c: len(v) for c, v in class_buckets.items()}
    print(f"  extracted {len(texts)} labeled instructions for task '{task}'")
    print(f"  class distribution: {class_counts}")
    return texts, labels
