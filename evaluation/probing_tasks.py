import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.architecture.complete_model import PTXTransformerForPretraining
from model.tokenizer.tokenizer import PTXTokenizer
from model.preprocessing.dataset_builder import PTXDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from evaluation import eval_utils
from evaluation.eval_utils import extract_labeled_instructions

def load_encoder_for_probing( checkpoint_path: str, tokenizer_path: str, device: str = 'cuda',) -> Tuple[PTXTransformerForPretraining, PTXTokenizer, Dict]:
    """
    load pretrained model and freeze all parameters.
    freezing is critical for probing bc we want to measure what the encoder
    already knows, not train it further. the probe (logistic regression)
    is the only thing that gets optimized.
    returns:
        (model, tokenizer, config_dict)
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    tokenizer = PTXTokenizer(tokenizer_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    state_dict = checkpoint['model_state_dict']
    use_itc = any(k.startswith('itc_head') for k in state_dict)
    model = PTXTransformerForPretraining( vocab_size=config.get('vocab_size', tokenizer.vocab_size),
        d_model=config.get('d_model', 768), num_layers=config.get('num_layers', 12), num_heads=config.get('num_heads', 8),
        d_ff=config.get('d_ff', 3072), dropout=0.0, max_seq_length=config.get('max_seq_length', 2048),padding_idx=tokenizer.pad_id,
        use_itc=use_itc,).to(device)
    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    d_model = config.get('d_model', 768)
    print(f"loaded frozen encoder (d_model={d_model}) from {checkpoint_path}")
    return model, tokenizer, config



def _extract_representation( model: PTXTransformerForPretraining, tokenizer: PTXTokenizer, text: str,
    max_length: int = 2048,device: str = 'cpu',) -> np.ndarray:
    """
    encode a ptx snippet and extract mean-pooled hidden state.
    mean pooling over real tokens (attention_mask=1) gives a single
    fixed-size vector [d_model] that represents the entire snippet.
    this is what the linear probe classifies.
    """
    ids = tokenizer.encode(text)[:max_length]
    actual_len = len(ids)
    padding = [tokenizer.pad_id] * (max_length - actual_len)
    ids = ids + padding
    mask = [1] * actual_len + [0] * (max_length - actual_len)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([mask], dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        hidden = outputs['hidden_states']  # [1, seq_len, d_model]
        # mean pool over real tokens only
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)

    return pooled[0].cpu().numpy()


# probe 1: instruction type (memory vs compute)
def probe_instruction_type( model: PTXTransformerForPretraining, tokenizer: PTXTokenizer,
    test_dataset: Optional[PTXDataset] = None, device: str = 'cuda',) -> Dict:
    """
    binary classification: memory (1) vs compute (0).
    
    if test_dataset is provided, real instructions are extracted from the
    test split and auto-labeled. a 70/30 train/test split is applied.
    otherwise falls back to synthetic snippets.
    """
    print("\n--- probe: instruction type classification (memory vs compute) ---")
    device_str = str(device)

    if test_dataset is not None:
        texts, labels = extract_labeled_instructions(
            test_dataset, tokenizer, task='instruction_type', max_per_class=10000,
        )
        if len(texts) >= 20:
            # split 70/30 for train/test
            n_train = int(len(texts) * 0.7)
            train_texts, test_texts = texts[:n_train], texts[n_train:]
            train_labels, test_labels = labels[:n_train], labels[n_train:]

            train_X = np.array([_extract_representation(model, tokenizer, t, device=device_str) for t in train_texts])
            train_y = np.array(train_labels)
            test_X = np.array([_extract_representation(model, tokenizer, t, device=device_str) for t in test_texts])
            test_y = np.array(test_labels)

            probe = LogisticRegression(max_iter=1000, random_state=42)
            probe.fit(train_X, train_y)
            train_acc = accuracy_score(train_y, probe.predict(train_X))
            test_acc = accuracy_score(test_y, probe.predict(test_X))

            print(f"  data source: test dataset ({len(train_texts)} train, {len(test_texts)} test)")
            print(f"  train accuracy: {train_acc:.2%}")
            print(f"  test accuracy:  {test_acc:.2%}")
            print(f"\n  classification report (test):")
            print(classification_report(test_y, probe.predict(test_X), target_names=['compute', 'memory'], zero_division=0))
            return {'train_accuracy': train_acc, 'test_accuracy': test_acc, 'data_source': 'test_dataset'}
        else:
            print(f"  warning: only {len(texts)} instructions extracted, falling back to synthetic data")

    # fallback: synthetic snippets
    train_data = [
        # memory ops (label=1)
        ("ld.global.f32 %f1, [%rd1];", 1),
        ("st.global.f32 [%rd2], %f2;", 1),
        ("ld.shared.f32 %f3, [%rd3];", 1),
        ("st.shared.u32 [%rd4], %r1;", 1),
        ("ld.param.u64 %rd5, [<PARAM_0>];", 1),
        ("ld.global.u32 %r2, [%rd6];", 1),
        ("st.global.u32 [%rd7], %r3;", 1),
        ("ld.shared.f64 %fd1, [%rd8];", 1),
        ("ld.global.v2.f32 {%f4, %f5}, [%rd9];", 1),
        ("atom.global.add.f32 %f6, [%rd10], %f7;", 1),
        # compute ops (label=0)
        ("add.f32 %f1, %f2, %f3;", 0),
        ("mul.f32 %f4, %f5, %f6;", 0),
        ("fma.rn.f32 %f7, %f8, %f9, %f10;", 0),
        ("mad.lo.s32 %r1, %r2, %r3, %r4;", 0),
        ("sub.f32 %f11, %f12, %f13;", 0),
        ("shr.u32 %r5, %r6, 2;", 0),
        ("min.f32 %f14, %f15, %f16;", 0),
        ("max.s32 %r7, %r8, %r9;", 0),
        ("and.b32 %r10, %r11, %r12;", 0),
        ("or.b32 %r13, %r14, %r15;", 0),
    ]
    test_data = [
        ("ld.global.f32 %f20, [%rd20];", 1),
        ("st.shared.f32 [%rd21], %f21;", 1),
        ("ld.global.u32 %r20, [%rd22];", 1),
        ("ld.shared.b32 %r21, [%rd23];", 1),
        ("st.global.f64 [%rd24], %fd10;", 1),
        ("add.s32 %r30, %r31, %r32;", 0),
        ("mul.lo.s32 %r33, %r34, %r35;", 0),
        ("fma.rn.f64 %fd20, %fd21, %fd22, %fd23;", 0),
        ("sub.s32 %r36, %r37, %r38;", 0),
        ("shl.b32 %r39, %r40, 4;", 0),
    ]

    train_X = np.array([_extract_representation(model, tokenizer, t, device=device_str) for t, _ in train_data])
    train_y = np.array([l for _, l in train_data])
    test_X = np.array([_extract_representation(model, tokenizer, t, device=device_str) for t, _ in test_data])
    test_y = np.array([l for _, l in test_data])
    probe = LogisticRegression(max_iter=1000, random_state=42)
    probe.fit(train_X, train_y)
    train_acc = accuracy_score(train_y, probe.predict(train_X))
    test_acc = accuracy_score(test_y, probe.predict(test_X))

    print(f"  data source: synthetic snippets")
    print(f"  train accuracy: {train_acc:.2%}")
    print(f"  test accuracy:  {test_acc:.2%}")
    print(f"\n  classification report (test):")
    print(classification_report(test_y, probe.predict(test_X), target_names=['compute', 'memory'], zero_division=0))

    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'data_source': 'synthetic',
    }


# probe 2: address space classification
def probe_address_space(model: PTXTransformerForPretraining, tokenizer: PTXTokenizer,
    test_dataset: Optional[PTXDataset] = None, device: str = 'cuda',) -> Dict:
    """
    multi-class classification: which address space does this memory op use?
    labels: 0=global, 1=shared, 2=local, 3=const
    
    if test_dataset is provided, real memory instructions are auto-labeled.
    """
    print("\n--- probe: address space classification ---")
    addr_space_map = {'global': 0, 'shared': 1, 'local': 2, 'const': 3}
    device_str = str(device)

    if test_dataset is not None:
        texts, labels = extract_labeled_instructions(
            test_dataset, tokenizer, task='address_space', max_per_class=10000,
        )
        if len(texts) >= 16:
            n_train = int(len(texts) * 0.7)
            train_texts, test_texts = texts[:n_train], texts[n_train:]
            train_labels, test_labels = labels[:n_train], labels[n_train:]

            train_X = np.array([_extract_representation(model, tokenizer, t, device=device_str) for t in train_texts])
            train_y = np.array(train_labels)
            test_X = np.array([_extract_representation(model, tokenizer, t, device=device_str) for t in test_texts])
            test_y = np.array(test_labels)

            probe = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')
            probe.fit(train_X, train_y)
            test_acc = accuracy_score(test_y, probe.predict(test_X))

            present_classes = sorted(set(train_labels) | set(test_labels))
            target_names = [k for k, v in addr_space_map.items() if v in present_classes]

            print(f"  data source: test dataset ({len(train_texts)} train, {len(test_texts)} test)")
            print(f"  test accuracy: {test_acc:.2%}")
            print(f"\n  classification report (test):")
            print(classification_report(test_y, probe.predict(test_X), target_names=target_names, zero_division=0))
            return {'test_accuracy': test_acc, 'address_space_map': addr_space_map, 'data_source': 'test_dataset'}
        else:
            print(f"  warning: only {len(texts)} instructions, falling back to synthetic data")

    # fallback: synthetic snippets
    train_data = [
        ("ld.global.f32 %f1, [%rd1];", 0),
        ("st.global.f32 [%rd2], %f2;", 0),
        ("ld.global.u32 %r1, [%rd3];", 0),
        ("ld.global.f64 %fd1, [%rd4];", 0),
        ("st.global.u32 [%rd5], %r2;", 0),
        ("ld.shared.f32 %f3, [%rd6];", 1),
        ("st.shared.f32 [%rd7], %f4;", 1),
        ("ld.shared.u32 %r3, [%rd8];", 1),
        ("ld.shared.f64 %fd2, [%rd9];", 1),
        ("st.shared.u32 [%rd10], %r4;", 1),
        ("ld.local.f32 %f5, [%rd11];", 2),
        ("st.local.f32 [%rd12], %f6;", 2),
        ("ld.local.u32 %r5, [%rd13];", 2),
        ("ld.local.b32 %r6, [%rd14];", 2),
        ("st.local.u32 [%rd15], %r7;", 2),
        ("ld.const.f32 %f7, [%rd16];", 3),
        ("ld.const.u32 %r8, [%rd17];", 3),
        ("ld.const.f64 %fd3, [%rd18];", 3),
        ("ld.const.b32 %r9, [%rd19];", 3),
        ("ld.const.f32 %f8, [%rd20];", 3),
    ]
    test_data = [
        ("ld.global.f32 %f10, [%rd30];", 0),
        ("st.global.f32 [%rd31], %f11;", 0),
        ("ld.shared.f32 %f12, [%rd32];", 1),
        ("st.shared.u32 [%rd33], %r10;", 1),
        ("ld.local.f32 %f13, [%rd34];", 2),
        ("st.local.u32 [%rd35], %r11;", 2),
        ("ld.const.f32 %f14, [%rd36];", 3),
        ("ld.const.u32 %r12, [%rd37];", 3),
    ]
    train_X = np.array([_extract_representation(model, tokenizer, t, device=device_str) for t, _ in train_data])
    train_y = np.array([l for _, l in train_data])
    test_X = np.array([_extract_representation(model, tokenizer, t, device=device_str) for t, _ in test_data])
    test_y = np.array([l for _, l in test_data])
    probe = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')
    probe.fit(train_X, train_y)
    test_acc = accuracy_score(test_y, probe.predict(test_X))
    print(f"  data source: synthetic snippets")
    print(f"  test accuracy: {test_acc:.2%}")
    print(f"\n  classification report (test):")
    print(classification_report(test_y, probe.predict(test_X), target_names=list(addr_space_map.keys()), zero_division=0))
    return {'test_accuracy': test_acc, 'address_space_map': addr_space_map, 'data_source': 'synthetic'}


# probe 3: data type classification
def probe_data_type(model: PTXTransformerForPretraining, tokenizer: PTXTokenizer,
    test_dataset: Optional[PTXDataset] = None, device: str = 'cuda',) -> Dict:
    """
    multi-class classification: what data type does this instruction use?
    labels: 0=f32, 1=u32, 2=s32, 3=f64, 4=b32

    if test_dataset is provided, real instructions are auto-labeled by data type.
    """
    print("\n--- probe: data type classification ---")
    type_map = {'f32': 0, 'u32': 1, 's32': 2, 'f64': 3, 'b32': 4}
    device_str = str(device)

    if test_dataset is not None:
        texts, labels = extract_labeled_instructions(
            test_dataset, tokenizer, task='data_type', max_per_class=10000,
        )
        if len(texts) >= 20:
            n_train = int(len(texts) * 0.7)
            train_texts, test_texts = texts[:n_train], texts[n_train:]
            train_labels, test_labels = labels[:n_train], labels[n_train:]

            train_X = np.array([_extract_representation(model, tokenizer, t, device=device_str) for t in train_texts])
            train_y = np.array(train_labels)
            test_X = np.array([_extract_representation(model, tokenizer, t, device=device_str) for t in test_texts])
            test_y = np.array(test_labels)

            probe = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')
            probe.fit(train_X, train_y)
            test_acc = accuracy_score(test_y, probe.predict(test_X))

            present_classes = sorted(set(train_labels) | set(test_labels))
            target_names = [k for k, v in type_map.items() if v in present_classes]

            print(f"  data source: test dataset ({len(train_texts)} train, {len(test_texts)} test)")
            print(f"  test accuracy: {test_acc:.2%}")
            print(f"  types tested:  {target_names}")
            print(f"\n  classification report (test):")
            print(classification_report(test_y, probe.predict(test_X), target_names=target_names, zero_division=0))
            return {'test_accuracy': test_acc, 'num_types': len(present_classes), 'type_map': type_map, 'data_source': 'test_dataset'}
        else:
            print(f"  warning: only {len(texts)} instructions, falling back to synthetic data")

    # fallback: synthetic snippets
    train_data = [
        # f32
        ("add.f32 %f1, %f2, %f3;", 0),
        ("mul.f32 %f4, %f5, %f6;", 0),
        ("ld.global.f32 %f7, [%rd10];", 0),
        ("fma.rn.f32 %f8, %f9, %f10, %f11;", 0),
        # u32
        ("add.u32 %r1, %r2, %r3;", 1),
        ("shr.u32 %r4, %r5, 2;", 1),
        ("ld.global.u32 %r6, [%rd20];", 1),
        ("and.u32 %r7, %r8, 0xFF;", 1),
        # s32
        ("add.s32 %r10, %r11, %r12;", 2),
        ("mad.lo.s32 %r13, %r14, %r15, %r16;", 2),
        ("mul.lo.s32 %r17, %r18, %r19;", 2),
        ("sub.s32 %r20, %r21, %r22;", 2),
        # f64
        ("add.f64 %fd1, %fd2, %fd3;", 3),
        ("mul.f64 %fd4, %fd5, %fd6;", 3),
        ("ld.global.f64 %fd7, [%rd30];", 3),
        ("fma.rn.f64 %fd8, %fd9, %fd10, %fd11;", 3),
        # b32
        ("and.b32 %r30, %r31, %r32;", 4),
        ("or.b32 %r33, %r34, %r35;", 4),
        ("xor.b32 %r36, %r37, %r38;", 4),
        ("not.b32 %r39, %r40;", 4),
    ]

    test_data = [
        ("sub.f32 %f20, %f21, %f22;", 0),
        ("st.global.f32 [%rd40], %f23;", 0),
        ("mul.u32 %r50, %r51, %r52;", 1),
        ("ld.shared.u32 %r53, [%rd41];", 1),
        ("add.s32 %r60, %r61, 1;", 2),
        ("sub.s32 %r62, %r63, %r64;", 2),
        ("sub.f64 %fd20, %fd21, %fd22;", 3),
        ("shl.b32 %r70, %r71, 4;", 4),
    ]

    device_str = str(device)
    train_X = np.array([_extract_representation(model, tokenizer, t, device=device_str) for t, _ in train_data])
    train_y = np.array([l for _, l in train_data])
    test_X = np.array([_extract_representation(model, tokenizer, t, device=device_str) for t, _ in test_data])
    test_y = np.array([l for _, l in test_data])

    probe = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')
    probe.fit(train_X, train_y)

    test_acc = accuracy_score(test_y, probe.predict(test_X))
    print(f"  data source: synthetic snippets")
    print(f"  test accuracy: {test_acc:.2%}")
    print(f"  types tested:  {list(type_map.keys())}")
    print(f"\n  classification report (test):")
    print(classification_report(test_y, probe.predict(test_X), target_names=list(type_map.keys()), zero_division=0))

    return {
        'test_accuracy': test_acc,
        'num_types': len(type_map),
        'type_map': type_map,
        'data_source': 'synthetic',
    }



def run_all_probing_tasks(checkpoint_path: str,tokenizer_path: str,device: str = 'cuda',data_dir: str = eval_utils.DEFAULT_DATA_DIR,cache_dir: str = eval_utils.DEFAULT_CACHE_DIR,max_seq_length: int = 2048,) -> Dict:
    """
    run all probing tasks on a pretrained checkpoint.
    loads PTXDataset(split='test') and passes it to every probe.
    probing order:
      1. instruction type (binary, easiest)
      2. address space    (4-class)
      3. data type        (5-class, finest)
    """
    model, tokenizer, config = load_encoder_for_probing(checkpoint_path, tokenizer_path, device)
    device_str = str(next(model.parameters()).device)
    test_dataset = eval_utils.load_test_dataset(tokenizer=tokenizer, data_dir=data_dir, cache_dir=cache_dir, max_seq_length=max_seq_length,)
    print(f"loaded test dataset: {len(test_dataset)} chunks")
    results = {}
    results['instruction_type'] = probe_instruction_type(model, tokenizer, test_dataset=test_dataset, device=device_str)
    results['address_space'] = probe_address_space(model, tokenizer, test_dataset=test_dataset, device=device_str)
    results['data_type'] = probe_data_type(model, tokenizer, test_dataset=test_dataset, device=device_str)
    print(f"\n{'='*50}")
    print(f"probing tasks summary")
    print(f"{'='*50}")
    for name, res in results.items():
        acc = res.get('test_accuracy', 'n/a')
        src = res.get('data_source', 'unknown')
        if isinstance(acc, float):
            print(f"  {name:20s} test accuracy: {acc:.2%}  (data: {src})")
        else:
            print(f"  {name:20s} {acc}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='probing tasks for ptx transformer')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to model checkpoint (.pt)')
    parser.add_argument('--tokenizer', type=str, default=eval_utils.DEFAULT_TOKENIZER, help='path to sentencepiece .model file')
    parser.add_argument('--data-dir', type=str, default=eval_utils.DEFAULT_DATA_DIR, help='path to processed PTX directory')
    parser.add_argument('--cache-dir', type=str, default=eval_utils.DEFAULT_CACHE_DIR, help='path to dataset cache directory')
    parser.add_argument('--max-seq-length', type=int, default=2048)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    run_all_probing_tasks(checkpoint_path=args.checkpoint, tokenizer_path=args.tokenizer,device=args.device, data_dir=args.data_dir, cache_dir=args.cache_dir,max_seq_length=args.max_seq_length,)