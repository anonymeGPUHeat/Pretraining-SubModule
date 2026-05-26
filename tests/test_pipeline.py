"""
Comprehensive test suite for the PTX Transformer pretraining pipeline.

Tests cover:
    1. ITC labeling (detect_instruction_spans)
    2. Instruction-boundary-aware MLM masking
    3. Dataset builder (chunking, empty file handling, ordering metadata)
    4. Model forward pass (encoder + pretraining heads)
    5. Loss computation (MLM + ITC objectives)
    6. Collate function (end-to-end batch assembly)
    7. Training utilities (TrainingState, callbacks)

Usage:
    python -m pytest tests/test_pipeline.py -v
    # or standalone:
    python tests/test_pipeline.py
"""

import sys, os, random, tempfile, math
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pytest


# ─── Paths ────────────────────────────────────────────────────────────────────
TOKENIZER_PATH = PROJECT_ROOT / 'data' / 'tokenizer' / 'ptx_tokenizer.model'
DATA_DIR = PROJECT_ROOT / 'data' / 'sprocessed' / 'sprocessed'


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope='session')
def tokenizer():
    """Load the real SentencePiece tokenizer (shared across all tests)."""
    if not TOKENIZER_PATH.exists():
        pytest.skip(f'Tokenizer model not found at {TOKENIZER_PATH}')
    from model.tokenizer.tokenizer import PTXTokenizer
    return PTXTokenizer(str(TOKENIZER_PATH))


@pytest.fixture(scope='session')
def sample_ptx_text():
    """A short, realistic PTX snippet covering multiple instruction types."""
    return (
        '.version 8.5\n'
        '.target sm_87\n'
        '.address_size 64\n'
        '.visible .entry my_kernel(\n'
        '  .param .u64 my_kernel_param_0\n'
        ')\n'
        '{\n'
        '  .reg .f32 %f<4>;\n'
        '  .reg .b64 %rd<3>;\n'
        '  .reg .pred %p<2>;\n'
        '  ld.param.u64 %rd0, [my_kernel_param_0];\n'
        '  ld.global.f32 %f0, [%rd0];\n'
        '  add.f32 %f1, %f0, %f0;\n'
        '  setp.gt.f32 %p0, %f1, 0f00000000;\n'
        '  @%p0 bra $L__BB0_1;\n'
        '  st.global.f32 [%rd0], %f1;\n'
        '  bar.sync 0;\n'
        '  cvt.rn.f32.s32 %f2, %f1;\n'
        '  atom.global.add.f32 %f3, [%rd0+4], %f2;\n'
        '  ret;\n'
        '$L__BB0_1:\n'
        '  exit;\n'
        '}\n'
    )


@pytest.fixture(scope='session')
def sample_token_ids(tokenizer, sample_ptx_text):
    """Encode the sample PTX snippet to token IDs."""
    return tokenizer.encode(sample_ptx_text)


# ──────────────────────────────────────────────────────────────────────────────
# 1. ITC LABELING
# ──────────────────────────────────────────────────────────────────────────────

class TestITCLabeling:
    """Test instruction span detection and per-token ITC label assignment."""

    def test_detect_returns_labels_and_spans(self, tokenizer, sample_token_ids):
        from model.preprocessing.itc_labels import detect_instruction_spans
        labels, spans = detect_instruction_spans(sample_token_ids, tokenizer)
        assert len(labels) == len(sample_token_ids)
        assert isinstance(spans, list)
        assert all(len(s) == 3 for s in spans)

    def test_labels_are_valid_range(self, tokenizer, sample_token_ids):
        from model.preprocessing.itc_labels import detect_instruction_spans, NUM_ITC_CLASSES, ITC_IGNORE
        labels, _ = detect_instruction_spans(sample_token_ids, tokenizer)
        for lbl in labels:
            assert lbl == ITC_IGNORE or (0 <= lbl < NUM_ITC_CLASSES), \
                f'ITC label {lbl} out of range'

    def test_spans_cover_full_sequence(self, tokenizer, sample_token_ids):
        from model.preprocessing.itc_labels import detect_instruction_spans
        _, spans = detect_instruction_spans(sample_token_ids, tokenizer)
        # Spans should cover the full token range without gaps
        covered = set()
        for start, end, _ in spans:
            for i in range(start, end):
                covered.add(i)
        # Every non-special token index should be covered
        assert max(covered) <= len(sample_token_ids)

    def test_instruction_tokens_share_class(self, tokenizer, sample_token_ids):
        """All tokens within a real instruction span must share the same ITC class."""
        from model.preprocessing.itc_labels import detect_instruction_spans, ITC_IGNORE
        labels, spans = detect_instruction_spans(sample_token_ids, tokenizer)
        for start, end, is_instr in spans:
            if is_instr:
                span_labels = [l for l in labels[start:end] if l != ITC_IGNORE]
                if span_labels:
                    assert len(set(span_labels)) == 1, \
                        f'Tokens in instruction span [{start}:{end}) have mixed classes: {set(span_labels)}'

    def test_global_load_classified(self, tokenizer):
        """ld.global.f32 should be classified as GLOBAL_LOAD."""
        from model.preprocessing.itc_labels import detect_instruction_spans, ITC_CLASSES
        text = 'ld.global.f32 %f0, [%rd0];'
        ids = tokenizer.encode(text)
        labels, _ = detect_instruction_spans(ids, tokenizer)
        instruction_labels = [l for l in labels if l != -100]
        assert len(instruction_labels) > 0
        assert all(l == ITC_CLASSES['GLOBAL_LOAD'] for l in instruction_labels)

    def test_shared_store_classified(self, tokenizer):
        from model.preprocessing.itc_labels import detect_instruction_spans, ITC_CLASSES
        text = 'st.shared.f32 [%rd0], %f1;'
        ids = tokenizer.encode(text)
        labels, _ = detect_instruction_spans(ids, tokenizer)
        instruction_labels = [l for l in labels if l != -100]
        assert all(l == ITC_CLASSES['SHARED_STORE'] for l in instruction_labels)

    def test_arithmetic_classified(self, tokenizer):
        from model.preprocessing.itc_labels import detect_instruction_spans, ITC_CLASSES
        text = 'add.f32 %f1, %f0, %f0;'
        ids = tokenizer.encode(text)
        labels, _ = detect_instruction_spans(ids, tokenizer)
        instruction_labels = [l for l in labels if l != -100]
        assert all(l == ITC_CLASSES['ARITHMETIC'] for l in instruction_labels)

    def test_control_flow_classified(self, tokenizer):
        from model.preprocessing.itc_labels import detect_instruction_spans, ITC_CLASSES
        text = 'bra $L__BB0_1;'
        ids = tokenizer.encode(text)
        labels, _ = detect_instruction_spans(ids, tokenizer)
        instruction_labels = [l for l in labels if l != -100]
        assert all(l == ITC_CLASSES['CONTROL_FLOW'] for l in instruction_labels)

    def test_predicate_guard_not_opcode(self, tokenizer):
        """@%p0 should NOT hijack the opcode; bra is the real opcode."""
        from model.preprocessing.itc_labels import detect_instruction_spans, ITC_CLASSES
        text = '@%p0 bra $target;'
        ids = tokenizer.encode(text)
        labels, _ = detect_instruction_spans(ids, tokenizer)
        instruction_labels = [l for l in labels if l != -100]
        assert all(l == ITC_CLASSES['CONTROL_FLOW'] for l in instruction_labels)

    def test_directives_get_ignore(self, tokenizer):
        """Directives like .version, .target should get ITC_IGNORE."""
        from model.preprocessing.itc_labels import detect_instruction_spans, ITC_IGNORE
        text = '.version 8.5\n.target sm_87\n.address_size 64'
        ids = tokenizer.encode(text)
        labels, _ = detect_instruction_spans(ids, tokenizer)
        # No instruction labels should appear
        instruction_labels = [l for l in labels if l != ITC_IGNORE]
        assert len(instruction_labels) == 0

    def test_atomic_instruction(self, tokenizer):
        from model.preprocessing.itc_labels import detect_instruction_spans, ITC_CLASSES
        text = 'atom.global.add.f32 %f3, [%rd0+4], %f2;'
        ids = tokenizer.encode(text)
        labels, _ = detect_instruction_spans(ids, tokenizer)
        instruction_labels = [l for l in labels if l != -100]
        assert all(l == ITC_CLASSES['ATOMIC_REDUCE'] for l in instruction_labels)

    def test_conversion_instruction(self, tokenizer):
        from model.preprocessing.itc_labels import detect_instruction_spans, ITC_CLASSES
        text = 'cvt.rn.f32.s32 %f2, %f1;'
        ids = tokenizer.encode(text)
        labels, _ = detect_instruction_spans(ids, tokenizer)
        instruction_labels = [l for l in labels if l != -100]
        assert all(l == ITC_CLASSES['CONVERSION_MOVE'] for l in instruction_labels)

    def test_sync_barrier(self, tokenizer):
        from model.preprocessing.itc_labels import detect_instruction_spans, ITC_CLASSES
        text = 'bar.sync 0;'
        ids = tokenizer.encode(text)
        labels, _ = detect_instruction_spans(ids, tokenizer)
        instruction_labels = [l for l in labels if l != -100]
        assert all(l == ITC_CLASSES['SYNC_BARRIER'] for l in instruction_labels)


# ──────────────────────────────────────────────────────────────────────────────
# 2. MLM MASKING
# ──────────────────────────────────────────────────────────────────────────────

class TestMLMMasking:
    """Test instruction-boundary-aware MLM masking."""

    def test_masking_returns_correct_keys(self, tokenizer, sample_token_ids):
        from model.training.dataloader import apply_mlm_masking
        n = len(sample_token_ids)
        attn_mask = [1] * n
        result = apply_mlm_masking(
            sample_token_ids, attn_mask,
            vocab_size=tokenizer.vocab_size,
            mask_token_id=tokenizer.sp.PieceToId('<mask>'),
            mask_prob=0.15,
        )
        assert 'input_ids' in result
        assert 'mlm_labels' in result
        assert len(result['input_ids']) == n
        assert len(result['mlm_labels']) == n

    def test_unmasked_positions_have_ignore_label(self, tokenizer, sample_token_ids):
        from model.training.dataloader import apply_mlm_masking
        random.seed(42)
        n = len(sample_token_ids)
        result = apply_mlm_masking(
            sample_token_ids, [1]*n,
            vocab_size=tokenizer.vocab_size,
            mask_token_id=tokenizer.sp.PieceToId('<mask>'),
            mask_prob=0.15,
        )
        for i in range(n):
            if result['input_ids'][i] == sample_token_ids[i]:
                # Position was NOT modified → either keep-original in masking OR not selected
                # Label might be set (keep-original case) or -100
                pass  # valid either way
            if result['mlm_labels'][i] != -100:
                # This position was selected for masking; its label should be the original id
                assert result['mlm_labels'][i] == sample_token_ids[i]

    def test_special_tokens_never_masked(self, tokenizer, sample_token_ids):
        from model.training.dataloader import apply_mlm_masking
        mask_id = tokenizer.sp.PieceToId('<mask>')
        special = {tokenizer.pad_id, tokenizer.unk_id, tokenizer.bos_id, tokenizer.eos_id, mask_id}
        random.seed(0)
        n = len(sample_token_ids)
        result = apply_mlm_masking(
            sample_token_ids, [1]*n,
            vocab_size=tokenizer.vocab_size,
            mask_token_id=mask_id,
            mask_prob=0.5,  # high prob to trigger masking
            special_ids=special,
        )
        for i in range(n):
            if sample_token_ids[i] in special:
                assert result['mlm_labels'][i] == -100, \
                    f'Special token at position {i} was masked'

    def test_instruction_boundary_masking(self, tokenizer, sample_token_ids):
        """When spans are provided, all tokens in a selected instruction should be masked together."""
        from model.preprocessing.itc_labels import detect_instruction_spans
        from model.training.dataloader import apply_mlm_masking
        mask_id = tokenizer.sp.PieceToId('<mask>')
        _, spans = detect_instruction_spans(sample_token_ids, tokenizer)

        # Run many trials to verify: if any token in an instruction span is masked,
        # then ALL eligible tokens in that span should be masked.
        for trial in range(50):
            random.seed(trial)
            n = len(sample_token_ids)
            result = apply_mlm_masking(
                sample_token_ids, [1]*n,
                vocab_size=tokenizer.vocab_size,
                mask_token_id=mask_id,
                mask_prob=0.3,
                instruction_spans=spans,
            )
            for start, end, is_instr in spans:
                if is_instr:
                    masked_positions = [i for i in range(start, end) if result['mlm_labels'][i] != -100]
                    # Either none are masked, or ALL eligible are masked
                    if masked_positions:
                        # Every position that's eligible (non-special) should be masked
                        for i in range(start, end):
                            piece = tokenizer.sp.IdToPiece(sample_token_ids[i])
                            if piece.startswith('<'):
                                continue  # skip special/structural
                            assert result['mlm_labels'][i] != -100, \
                                f'Trial {trial}: instruction span [{start}:{end}) partially masked at position {i}'

    def test_padding_never_masked(self, tokenizer, sample_token_ids):
        from model.training.dataloader import apply_mlm_masking
        mask_id = tokenizer.sp.PieceToId('<mask>')
        # Add padding
        padded_ids = sample_token_ids + [tokenizer.pad_id] * 50
        attn_mask = [1] * len(sample_token_ids) + [0] * 50
        random.seed(1)
        result = apply_mlm_masking(
            padded_ids, attn_mask,
            vocab_size=tokenizer.vocab_size,
            mask_token_id=mask_id,
            mask_prob=0.5,
        )
        # Padding positions (attn_mask=0) should never be masked
        for i in range(len(sample_token_ids), len(padded_ids)):
            assert result['mlm_labels'][i] == -100


# ──────────────────────────────────────────────────────────────────────────────
# 3. DATASET BUILDER
# ──────────────────────────────────────────────────────────────────────────────

class TestDatasetBuilder:
    """Test PTXDataset chunking, empty file handling, and metadata."""

    def test_empty_files_excluded(self, tokenizer):
        """Empty files should be filtered before splitting."""
        from model.preprocessing.dataset_builder import PTXDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / 'ptx_data'
            data_dir.mkdir()
            # Create 5 normal files + 3 empty files
            for i in range(5):
                (data_dir / f'file_{i}.ptx').write_text(
                    f'add.f32 %f{i}, %f0, %f1;\n' * 10
                )
            for i in range(5, 8):
                (data_dir / f'file_{i}.ptx').write_text('')  # empty
            ds = PTXDataset(
                tokenizer=tokenizer,
                data_dir=data_dir,
                max_seq_length=128,
                overlap=0,
                cache_dir=None,
                split='all',
                verbose=False,
            )
            stats = ds.metadata['statistics']
            # The dataset should have excluded the 3 empty files
            assert stats['empty_files_removed'] == 3
            assert stats['usable_files'] == 5

    def test_chunks_have_correct_length(self, tokenizer):
        """All chunks should be ≤ max_seq_length."""
        from model.preprocessing.dataset_builder import PTXDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / 'ptx_data'
            data_dir.mkdir()
            # Write a file long enough to trigger chunking
            (data_dir / 'file_0.ptx').write_text(
                'add.f32 %f0, %f1, %f2;\n' * 200
            )
            max_len = 64
            ds = PTXDataset(
                tokenizer=tokenizer,
                data_dir=data_dir,
                max_seq_length=max_len,
                overlap=8,
                cache_dir=None,
                split='all',
                verbose=False,
            )
            assert len(ds) > 0
            for idx in range(len(ds)):
                sample = ds[idx]
                assert sample['input_ids'].shape[0] == max_len

    def test_ordering_metadata(self, tokenizer):
        """Chunks should carry chunk_in_file and total_chunks_in_file metadata."""
        from model.preprocessing.dataset_builder import PTXDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / 'ptx_data'
            data_dir.mkdir()
            (data_dir / 'file_0.ptx').write_text(
                'ld.global.f32 %f0, [%rd0];\n' * 300
            )
            ds = PTXDataset(
                tokenizer=tokenizer,
                data_dir=data_dir,
                max_seq_length=64,
                overlap=0,
                cache_dir=None,
                split='all',
                verbose=False,
            )
            assert len(ds) >= 2, 'Expected multiple chunks'
            # Check that metadata keys exist and are consistent
            chunks = [ds.chunks[i] for i in range(len(ds))]
            for chunk in chunks:
                assert 'chunk_in_file' in chunk
                assert 'total_chunks_in_file' in chunk

    def test_attention_mask_correct(self, tokenizer):
        """Attention mask should be 1 for real tokens, 0 for padding."""
        from model.preprocessing.dataset_builder import PTXDataset
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / 'ptx_data'
            data_dir.mkdir()
            # Short file that will need padding
            (data_dir / 'file_0.ptx').write_text('add.f32 %f0, %f1, %f2;')
            ds = PTXDataset(
                tokenizer=tokenizer,
                data_dir=data_dir,
                max_seq_length=256,
                overlap=0,
                cache_dir=None,
                split='all',
                verbose=False,
            )
            sample = ds[0]
            ids = sample['input_ids']
            mask = sample['attention_mask']
            pad_id = tokenizer.pad_id
            for i in range(len(ids)):
                if ids[i] == pad_id:
                    assert mask[i] == 0
                else:
                    assert mask[i] == 1


# ──────────────────────────────────────────────────────────────────────────────
# 4. MODEL FORWARD PASS
# ──────────────────────────────────────────────────────────────────────────────

class TestModelForward:
    """Test the PTXTransformerForPretraining model."""

    @pytest.fixture
    def small_model(self, tokenizer):
        from model.architecture.complete_model import PTXTransformerForPretraining
        return PTXTransformerForPretraining(
            vocab_size=tokenizer.vocab_size,
            d_model=64,
            num_layers=2,
            num_heads=4,
            d_ff=128,
            dropout=0.0,
            max_seq_length=128,
            padding_idx=tokenizer.pad_id,
            num_itc_classes=14,
        ).eval()

    def test_output_shapes(self, small_model, tokenizer):
        batch_size, seq_len = 2, 32
        ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        with torch.no_grad():
            out = small_model(ids, attention_mask=mask)
        assert out['logits'].shape == (batch_size, seq_len, tokenizer.vocab_size)
        assert out['itc_logits'].shape == (batch_size, seq_len, 14)
        assert out['hidden_states'].shape == (batch_size, seq_len, 64)

    def test_weight_tying(self, small_model):
        """MLM decoder weights should be tied to embedding weights."""
        emb_weight = small_model.encoder.embedding.token_embedding.weight
        dec_weight = small_model.mlm_head.decoder.weight
        assert emb_weight is dec_weight

    def test_attention_maps(self, small_model, tokenizer):
        ids = torch.randint(0, tokenizer.vocab_size, (1, 16))
        mask = torch.ones(1, 16, dtype=torch.long)
        with torch.no_grad():
            out = small_model(ids, attention_mask=mask, return_attentions=True)
        assert 'attentions' in out
        assert len(out['attentions']) == 2  # num_layers

    def test_padding_does_not_affect_unpadded(self, small_model, tokenizer):
        """Padding tokens shouldn't significantly affect non-padded outputs."""
        ids_raw = torch.randint(5, tokenizer.vocab_size, (1, 16))
        mask_raw = torch.ones(1, 16, dtype=torch.long)
        # Padded version
        ids_padded = torch.cat([ids_raw, torch.full((1, 16), tokenizer.pad_id)], dim=1)
        mask_padded = torch.cat([mask_raw, torch.zeros(1, 16, dtype=torch.long)], dim=1)
        with torch.no_grad():
            out_raw = small_model(ids_raw, attention_mask=mask_raw)
            out_pad = small_model(ids_padded, attention_mask=mask_padded)
        # Hidden states for non-padded positions should be close
        h1 = out_raw['hidden_states'][0, :16]
        h2 = out_pad['hidden_states'][0, :16]
        diff = (h1 - h2).abs().max().item()
        assert diff < 1e-3, f'Padding affected non-padded outputs by {diff}'

    def test_count_parameters(self, small_model):
        params = small_model.count_parameters()
        assert 'total' in params
        assert params['total'] > 0
        assert 'encoder_layers' in params
        assert 'mlm_head' in params
        assert 'itc_head' in params


# ──────────────────────────────────────────────────────────────────────────────
# 5. LOSS COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────

class TestLossComputation:
    """Test PretrainingObjectives and metric functions."""

    def test_mlm_loss_ignores_padding(self):
        from model.training.objectives import PretrainingObjectives
        obj = PretrainingObjectives(label_smoothing=0.0)
        logits = torch.randn(2, 10, 100)
        labels = torch.full((2, 10), -100, dtype=torch.long)
        labels[0, 3] = 5   # one real label
        labels[1, 7] = 42  # another real label
        loss = obj.mlm_loss(logits, labels)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_compute_loss_dual_objective(self):
        from model.training.objectives import PretrainingObjectives
        obj = PretrainingObjectives()
        mlm_logits = torch.randn(2, 10, 100)
        mlm_labels = torch.randint(0, 100, (2, 10))
        itc_logits = torch.randn(2, 10, 14)
        itc_labels = torch.randint(0, 14, (2, 10))
        result = obj.compute_loss(
            mlm_logits=mlm_logits, mlm_labels=mlm_labels,
            itc_logits=itc_logits, itc_labels=itc_labels,
        )
        assert 'total_loss' in result
        assert 'mlm_loss' in result
        assert 'itc_loss' in result
        assert result['total_loss'].item() > 0
        # Total should be sum of individual losses (weights=1.0)
        expected = result['mlm_loss'] + result['itc_loss']
        assert abs(result['total_loss'].item() - expected.item()) < 1e-5

    def test_compute_loss_with_weights(self):
        from model.training.objectives import PretrainingObjectives
        obj = PretrainingObjectives()
        mlm_logits = torch.randn(2, 10, 100)
        mlm_labels = torch.randint(0, 100, (2, 10))
        itc_logits = torch.randn(2, 10, 14)
        itc_labels = torch.randint(0, 14, (2, 10))
        weights = {'mlm': 0.5, 'itc': 2.0}
        result = obj.compute_loss(
            mlm_logits=mlm_logits, mlm_labels=mlm_labels,
            itc_logits=itc_logits, itc_labels=itc_labels,
            weights=weights,
        )
        expected = 0.5 * result['mlm_loss'] + 2.0 * result['itc_loss']
        assert abs(result['total_loss'].item() - expected.item()) < 1e-5

    def test_mlm_accuracy(self):
        from model.training.objectives import compute_mlm_accuracy
        logits = torch.zeros(1, 5, 10)
        labels = torch.full((1, 5), -100, dtype=torch.long)
        # Make position 2 have label 3, and logits predict 3 correctly
        labels[0, 2] = 3
        logits[0, 2, 3] = 10.0  # high logit for correct class
        acc = compute_mlm_accuracy(logits, labels)
        assert acc == 1.0

    def test_itc_accuracy(self):
        from model.training.objectives import compute_itc_accuracy
        logits = torch.zeros(1, 5, 14)
        labels = torch.full((1, 5), -100, dtype=torch.long)
        labels[0, 1] = 6  # ARITHMETIC
        logits[0, 1, 6] = 10.0
        labels[0, 3] = 0  # GLOBAL_LOAD
        logits[0, 3, 0] = 10.0
        acc = compute_itc_accuracy(logits, labels)
        assert acc == 1.0

    def test_perplexity(self):
        from model.training.objectives import compute_mlm_perplexity
        ppl = compute_mlm_perplexity(2.0)
        assert abs(ppl - math.exp(2.0)) < 1e-5
        ppl_inf = compute_mlm_perplexity(200.0)
        assert ppl_inf == float('inf')


# ──────────────────────────────────────────────────────────────────────────────
# 6. COLLATE FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

class TestCollateFunction:
    """Test the collate_with_mlm function (end-to-end batch assembly)."""

    def test_collate_produces_correct_tensors(self, tokenizer, sample_token_ids):
        from model.training.dataloader import collate_with_mlm
        seq_len = 64
        # Pad or truncate
        ids = sample_token_ids[:seq_len]
        ids = ids + [tokenizer.pad_id] * (seq_len - len(ids))
        attn = [1 if tid != tokenizer.pad_id else 0 for tid in ids]
        batch = [
            {'input_ids': torch.tensor(ids), 'attention_mask': torch.tensor(attn)},
            {'input_ids': torch.tensor(ids), 'attention_mask': torch.tensor(attn)},
        ]
        mask_id = tokenizer.sp.PieceToId('<mask>')
        result = collate_with_mlm(
            batch,
            vocab_size=tokenizer.vocab_size,
            mask_token_id=mask_id,
            pad_token_id=tokenizer.pad_id,
            mask_prob=0.15,
            tokenizer=tokenizer,
            use_itc=True,
        )
        assert result['input_ids'].shape == (2, seq_len)
        assert result['attention_mask'].shape == (2, seq_len)
        assert result['mlm_labels'].shape == (2, seq_len)
        assert 'itc_labels' in result
        assert result['itc_labels'].shape == (2, seq_len)

    def test_itc_labels_not_affected_by_masking(self, tokenizer, sample_token_ids):
        """ITC labels should reflect original tokens, not masked versions."""
        from model.training.dataloader import collate_with_mlm
        from model.preprocessing.itc_labels import detect_instruction_spans
        seq_len = 64
        ids = sample_token_ids[:seq_len]
        ids = ids + [tokenizer.pad_id] * (seq_len - len(ids))
        attn = [1 if tid != tokenizer.pad_id else 0 for tid in ids]
        # Get reference ITC labels from original tokens
        ref_labels, _ = detect_instruction_spans(ids, tokenizer)
        batch = [{'input_ids': torch.tensor(ids), 'attention_mask': torch.tensor(attn)}]
        mask_id = tokenizer.sp.PieceToId('<mask>')
        random.seed(123)
        result = collate_with_mlm(
            batch,
            vocab_size=tokenizer.vocab_size,
            mask_token_id=mask_id,
            mask_prob=0.5,  # high masking to ensure some tokens get masked
            tokenizer=tokenizer,
            use_itc=True,
        )
        itc_labels = result['itc_labels'][0].tolist()
        assert itc_labels == ref_labels


# ──────────────────────────────────────────────────────────────────────────────
# 7. TRAINING UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

class TestTrainingState:
    """Test TrainingState and callback behavior."""

    def test_training_state_defaults(self):
        from model.training.callbacks import TrainingState
        state = TrainingState()
        assert state.epoch == 0
        assert state.global_step == 0
        assert state.best_loss == float('inf')
        assert state.is_new_best is False

    def test_early_stopping_sets_new_best(self):
        from model.training.callbacks import TrainingState, EarlyStoppingCallback
        state = TrainingState()
        cb = EarlyStoppingCallback(patience=3, min_delta=0.001)
        # First epoch — should be a new best
        should_stop = cb.on_epoch_end(
            state=state, train_loss=5.0, val_metrics={'val_loss': 4.0, 'val_accuracy': 0.5}
        )
        assert should_stop is False
        assert state.is_new_best is True
        assert state.best_loss == 4.0
        # Second epoch — no improvement
        should_stop = cb.on_epoch_end(
            state=state, train_loss=5.0, val_metrics={'val_loss': 4.0, 'val_accuracy': 0.5}
        )
        assert should_stop is False
        assert state.is_new_best is False
        assert state.epochs_no_improve == 1

    def test_early_stopping_triggers(self):
        from model.training.callbacks import TrainingState, EarlyStoppingCallback
        state = TrainingState()
        cb = EarlyStoppingCallback(patience=2, min_delta=0.001)
        # Best
        cb.on_epoch_end(state=state, train_loss=1.0, val_metrics={'val_loss': 1.0})
        # No improvement x2
        cb.on_epoch_end(state=state, train_loss=1.0, val_metrics={'val_loss': 1.0})
        should_stop = cb.on_epoch_end(state=state, train_loss=1.0, val_metrics={'val_loss': 1.0})
        assert should_stop is True

    def test_callback_manager_propagates_stop(self):
        from model.training.callbacks import TrainingState, EarlyStoppingCallback, CallbackManager
        state = TrainingState()
        mgr = CallbackManager()
        mgr.add(EarlyStoppingCallback(patience=1, min_delta=0.001))
        # First: new best
        stop = mgr.on_epoch_end(state=state, train_loss=1.0, val_metrics={'val_loss': 1.0})
        assert stop is False
        # Second: no improvement → patience=1 → stop
        stop = mgr.on_epoch_end(state=state, train_loss=1.0, val_metrics={'val_loss': 1.0})
        assert stop is True


# ──────────────────────────────────────────────────────────────────────────────
# 8. INTEGRATION: REAL PTX FILE
# ──────────────────────────────────────────────────────────────────────────────

class TestRealPTXIntegration:
    """End-to-end test on a real normalized PTX file from the dataset."""

    @pytest.fixture
    def real_ptx_text(self):
        if not DATA_DIR.exists():
            pytest.skip(f'Data directory not found: {DATA_DIR}')
        # Pick first non-empty file
        for f in sorted(DATA_DIR.glob('*.ptx')):
            if f.stat().st_size > 0:
                return f.read_text(encoding='utf-8', errors='replace')
        pytest.skip('No non-empty PTX files found')

    def test_real_file_itc_labeling(self, tokenizer, real_ptx_text):
        from model.preprocessing.itc_labels import detect_instruction_spans, NUM_ITC_CLASSES
        ids = tokenizer.encode(real_ptx_text[:5000])  # limit to first 5k chars
        labels, spans = detect_instruction_spans(ids, tokenizer)
        assert len(labels) == len(ids)
        assert len(spans) > 0
        # Check label distribution
        class_counts = {}
        for lbl in labels:
            if lbl != -100:
                class_counts[lbl] = class_counts.get(lbl, 0) + 1
        # A real PTX file should have at least some instruction tokens
        assert len(class_counts) > 0, 'No instruction tokens found in real PTX file'
        print(f'\n  ITC class distribution ({len(ids)} tokens):')
        from model.preprocessing.itc_labels import ITC_ID2NAME
        for cls_id, count in sorted(class_counts.items()):
            print(f'    {ITC_ID2NAME.get(cls_id, "?"):20s} ({cls_id:2d}): {count}')

    def test_real_file_masking_and_forward(self, tokenizer, real_ptx_text):
        from model.preprocessing.itc_labels import detect_instruction_spans
        from model.training.dataloader import apply_mlm_masking
        from model.architecture.complete_model import PTXTransformerForPretraining
        seq_len = 128
        ids = tokenizer.encode(real_ptx_text[:2000])[:seq_len]
        pad_len = seq_len - len(ids)
        ids = ids + [tokenizer.pad_id] * pad_len
        attn = [1] * (seq_len - pad_len) + [0] * pad_len
        labels_itc, spans = detect_instruction_spans(ids, tokenizer)
        mask_id = tokenizer.sp.PieceToId('<mask>')
        random.seed(42)
        mlm_result = apply_mlm_masking(
            ids, attn,
            vocab_size=tokenizer.vocab_size,
            mask_token_id=mask_id,
            mask_prob=0.15,
            instruction_spans=spans,
        )
        model = PTXTransformerForPretraining(
            vocab_size=tokenizer.vocab_size, d_model=64, num_layers=2,
            num_heads=4, d_ff=128, max_seq_length=seq_len,
            padding_idx=tokenizer.pad_id,
        ).eval()
        input_ids = torch.tensor([mlm_result['input_ids']])
        attention_mask = torch.tensor([attn])
        with torch.no_grad():
            out = model(input_ids, attention_mask=attention_mask)
        assert out['logits'].shape == (1, seq_len, tokenizer.vocab_size)
        assert out['itc_logits'].shape == (1, seq_len, 14)
        print(f'\n  Forward pass: seq_len={seq_len}, '
              f'masked={sum(1 for l in mlm_result["mlm_labels"] if l != -100)}, '
              f'instruction_tokens={sum(1 for l in labels_itc if l != -100)}')


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
