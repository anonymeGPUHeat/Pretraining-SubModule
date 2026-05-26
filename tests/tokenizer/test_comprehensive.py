import sentencepiece as spm
import numpy as np
from pathlib import Path
from typing import List, Dict,  Optional
import sys
import subprocess
import json


class PTXTokenizerTester:
    def __init__(self, tokenizer_model: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(tokenizer_model)
        self.model_path = tokenizer_model
        self.vocab_size = self.sp.GetPieceSize()
        self.results = {}
        
    def run_all_tests(self, ptx_files: Optional[List[Path]] = None, verbose: bool = True, 
                     sample_size: int = 100, max_seq_length: int = 2048):
        print("PTX TOKENIZER COMPREHENSIVE TEST SUITE")
        print(f"Model: {self.model_path}")
        print(f"Vocabulary size: {self.vocab_size}")
        
        self.results['number_tokenization'] = self.test_number_constant_tokenization(verbose)
        self.results['token_distribution'] = self.test_token_distribution_by_category(verbose)
        self.results['instruction_tokenization'] = self.test_complete_instruction_tokenization(verbose)
        if ptx_files and len(ptx_files) > 0:
            self.results['sequence_lengths'] = self.test_sequence_length_distribution(
                ptx_files, sample_size=sample_size, max_seq_length=max_seq_length, verbose=verbose)
        else:
            print("\nWARNING: No PTX files provided, skipping sequence length distribution test")
            self.results['sequence_lengths'] = None
        self.results['out_of_distribution'] = self.test_out_of_distribution_patterns(verbose)
        self.results['contextual_consistency'] = self.test_contextual_consistency(verbose)
        self._print_summary()
        return self.results
    
    def test_number_constant_tokenization(self, verbose: bool = True) -> Dict:
        """
        Validates that numeric constants are properly tokenized:
        - Frequent constants (0-100) should be single tokens
        - Medium numbers (100-10K) should be 2-3 tokens max
        - Hex constants should be preserved or split into 2 parts max
        - Memory offsets should be distinct
        """
        
        test_cases = [
            ('0x3f800000', 'Float32 constant', 3),  
            ('0xDEADBEEF', 'Hex constant', 3),
            ('0x7fffffff', 'Max int constant', 3),
            ('0x00000000', 'Zero constant', 3),
            ('0x400', 'Small hex', 2),
            ('0', 'Zero', 1),
            ('1', 'One', 1),
            ('4', 'Four', 1),
            ('8', 'Eight', 1),
            ('16', 'Sixteen', 1),
            ('32', 'Thirty-two', 2),
            ('64', 'Sixty-four', 2),
            ('128', 'One-twenty-eight', 2),
            ('256', 'Two-fifty-six', 2),
            ('1024', 'One-thousand', 3),
            ('4096', 'Four-thousand', 3),
            ('65536', 'Sixty-five-thousand', 3),
            ('1048576', 'One million', 4),
            ('4294967295', 'Max uint32', 5),
            ('[%rd0+1024]', 'Offset with common number', 6),
            ('[%rd0+0x400]', 'Offset with hex', 6),
            ('[%rd5+128]', 'Small offset', 6),
            ('<IMM_6K>', 'Normalized immediate 6K', 1),
            ('<IMM_10K>', 'Normalized immediate 10K', 1),
            ('<IMM_100K>', 'Normalized immediate 100K', 1),
            ('<IMM_1M>', 'Normalized immediate 1M', 1),
            ('<IMM_LARGE>', 'Normalized large immediate', 1),
            ('<IMM_NEG_6K>', 'Normalized negative immediate', 1),
        ]
        
        results = {
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': []
        }
        
        for text, description, max_expected_tokens in test_cases:
            pieces = self.sp.EncodeAsPieces(text)
            ids = self.sp.EncodeAsIds(text)
            num_tokens = len(pieces)
            status = 'PASS'
            if num_tokens == 1:
                status = 'EXCELLENT'
                results['passed'] += 1
            elif num_tokens <= max_expected_tokens:
                status = 'PASS'
                results['passed'] += 1
            elif num_tokens <= max_expected_tokens + 2:
                status = 'WARNING'
                results['warnings'] += 1
            else:
                status = 'FAIL'
                results['failed'] += 1
            
            result_detail = {
                'text': text,
                'description': description,
                'num_tokens': num_tokens,
                'max_expected': max_expected_tokens,
                'pieces': pieces,
                'status': status
            }
            results['details'].append(result_detail)
            
            if verbose:
                print(f"{status:15} | '{text:20}' → {num_tokens} tokens (max: {max_expected_tokens})")
                if status in ['❌ FAIL', '⚠ WARNING']:
                    print(f"                  Pieces: {pieces}")
        
        total = len(test_cases)
        pass_rate = (results['passed'] / total) * 100
        
        if verbose:
            print(f"Results: {results['passed']}/{total} passed, "
                  f"{results['warnings']} warnings, {results['failed']} failed")
            print(f"Pass rate: {pass_rate:.1f}%")
            
            if pass_rate >= 90:
                print("EXCELLENT: Numbers and constants are well tokenized")
            elif pass_rate >= 70:
                print("GOOD: Acceptable tokenization, but some improvements possible")
            elif pass_rate >= 50:
                print("WARNING: Many numbers are over-tokenized")
            else:
                print("CRITICAL: Severe over-tokenization of numbers")
        
        results['pass_rate'] = pass_rate
        results['status'] = 'PASS' if pass_rate >= 70 else ('WARNING' if pass_rate >= 50 else 'FAIL')
        
        return results
    
    def test_token_distribution_by_category(self, verbose: bool = True) -> Dict:
        """
        Analyzes vocabulary balance across different token types:
        - Opcodes (ld, st, add, mul, fma, etc.)
        - Modifiers (.global, .shared, .ca, .f32, etc.)
        - Registers (%r, %rd, %f, %p patterns)
        - Symbols (<KERNEL_, <BB_, etc.)
        - Punctuation (; , [ ] ( ) + -)
        - Numbers (0-9, hex)
        - Subwords (BPE fragments)
        """
        categories = {
            'special': [],           # <pad>, <unk>, <s>, </s>, <mask>, <cls>, <sep>
            'normalized_symbols': [], # <KERNEL_, <BB_, <PARAM_, <IMM_...>
            'opcodes': [],           # ld, st, add, mul, fma, etc.
            'modifiers': [],         # .global, .shared, .ca, .f32, .v2, etc.
            'special_registers': [], # %tid.x, %ntid.x, %ctaid.x, %laneid, etc.
            'register_fragments': [], # %r, %rd, %f, %p, %pd, etc.
            'punctuation': [],       # ; , [ ] ( ) { } + - * / =
            'hex_numbers': [],       # 0x..., 0X..., 0d..., 0D..., 0f..., 0F...
            'decimal_numbers': [],   # pure digits
            'subwords': [],          # BPE subword fragments
            'whitespace': [],        # space, tab, newline tokens
            'other': []
        }
        
        # Common PTX opcodes
        common_opcodes = {
            'ld', 'st', 'add', 'sub', 'mul', 'mad', 'fma', 'div', 'rem',
            'and', 'or', 'xor', 'not', 'neg', 'abs', 'min', 'max',
            'mov', 'cvt', 'setp', 'selp', 'bra', 'ret', 'call', 'exit',
            'bar', 'atom', 'red', 'vote', 'shfl', 'tex', 'suld', 'sust',
            'fence', 'membar', 'wmma', 'mma', 'cp'
        }
        
        # Common modifiers
        common_modifiers = {
            '.global', '.shared', '.local', '.param', '.const',
            '.ca', '.cg', '.cs', '.lu', '.cv', '.nc',
            '.v2', '.v4', '.f32', '.f64', '.u32', '.s32', '.u64', '.s64',
            '.b8', '.b16', '.b32', '.b64', '.pred',
            '.lo', '.hi', '.wide', '.sat', '.ftz',
            '.rn', '.rz', '.rm', '.rp',
            '.sync', '.async', '.arrive', '.release', '.acquire',
            '.entry', '.func', '.visible', '.extern', '.weak',
            '.reg', '.align', '.maxnreg', '.reqntid', '.maxntid', '.minnctapersm'
        }
        
        # Special registers
        special_regs = {
            '%tid.x', '%tid.y', '%tid.z',
            '%ntid.x', '%ntid.y', '%ntid.z',
            '%ctaid.x', '%ctaid.y', '%ctaid.z',
            '%nctaid.x', '%nctaid.y', '%nctaid.z',
            '%laneid', '%warpid', '%clock', '%clock64'
        }
        
        for i in range(self.vocab_size):
            piece = self.sp.IdToPiece(i)
            piece_clean = piece.replace('▁', '')
            if piece in ['<pad>', '<unk>', '<s>', '</s>', '<mask>', '<cls>', '<sep>']:
                categories['special'].append((i, piece))
            elif piece.startswith('<') and piece.endswith('>'):
                categories['normalized_symbols'].append((i, piece))
            elif piece in special_regs or piece_clean in special_regs:
                categories['special_registers'].append((i, piece))

            elif piece_clean in common_opcodes or piece in common_opcodes:
                categories['opcodes'].append((i, piece))
            
            elif piece in common_modifiers or piece_clean in common_modifiers:
                categories['modifiers'].append((i, piece))
            elif piece_clean.startswith('%') or piece.startswith('%'):
                categories['register_fragments'].append((i, piece))

            elif piece_clean in ';,[](){}+-*=/:<>!&|~' or len(piece_clean) == 1 and not piece_clean.isalnum():
                categories['punctuation'].append((i, piece))
            
            elif piece_clean.startswith(('0x', '0X', '0d', '0D', '0f', '0F')):
                categories['hex_numbers'].append((i, piece))

            elif piece_clean.isdigit():
                categories['decimal_numbers'].append((i, piece))

            elif piece in ['▁', ' ', '\t', '\n'] or piece_clean in ['', ' ', '\t', '\n']:
                categories['whitespace'].append((i, piece))

            else:
                categories['subwords'].append((i, piece))
        
        # Calculate statistics
        total_tokens = self.vocab_size
        results = {
            'total_vocab': total_tokens,
            'categories': {},
            'status': 'UNKNOWN'
        }
        
        if verbose:
            print(f"Total vocabulary size: {total_tokens}\n")
        
        for cat_name, tokens in categories.items():
            count = len(tokens)
            percentage = (count / total_tokens) * 100
            results['categories'][cat_name] = {
                'count': count,
                'percentage': percentage,
                'samples': [piece for _, piece in tokens[:5]]  # First 5 samples
            }
            
            if verbose:
                print(f"{cat_name.upper():25} {count:5} tokens ({percentage:5.1f}%)")
                if count > 0 and count <= 20:
                    for id, piece in tokens[:10]:
                        print(f"  [{id:5}] {repr(piece)}")
                elif count > 0:
                    for id, piece in tokens[:5]:
                        print(f"  [{id:5}] {repr(piece)}")
                    if count > 5:
                        print(f"  ... and {count - 5} more")
                print()
        

        opcodes_count = results['categories']['opcodes']['count']
        modifiers_count = results['categories']['modifiers']['count']
        subwords_pct = results['categories']['subwords']['percentage']
        special_regs_count = results['categories']['special_registers']['count']
        normalized_count = results['categories']['normalized_symbols']['count']
        
        warnings = []
        errors = []
        
        if opcodes_count < 50:
            errors.append(f"Too few opcodes ({opcodes_count}), instructions may be fragmented")
        elif opcodes_count < 100:
            warnings.append(f"Low opcode count ({opcodes_count}), consider adding more")
        
        if modifiers_count < 30:
            errors.append(f"Too few modifiers ({modifiers_count}), modifiers may be fragmented")
        elif modifiers_count < 50:
            warnings.append(f"Low modifier count ({modifiers_count})")
        
        if subwords_pct > 90:
            errors.append(f"Subwords dominate vocabulary ({subwords_pct:.1f}%), vocab may be too small")
        elif subwords_pct > 85:
            warnings.append(f"High subword percentage ({subwords_pct:.1f}%)")
        
        if special_regs_count < 5:
            warnings.append(f"Few special registers preserved ({special_regs_count})")
        

            if errors:
                print("\nERRORS:")
                for err in errors:
                    print(f"  - {err}")
            
            if warnings:
                print("\nWARNINGS:")
                for warn in warnings:
                    print(f"  - {warn}")
            
            if not errors and not warnings:
                print("\nEXCELLENT: Vocabulary is well-balanced")
                results['status'] = 'EXCELLENT'
            elif not errors:
                print("\nGOOD: Vocabulary is acceptable with minor issues")
                results['status'] = 'GOOD'
            elif len(errors) <= 2:
                print("\nWARNING: Vocabulary needs improvement")
                results['status'] = 'WARNING'
            else:
                print("\nFAIL: Vocabulary is poorly balanced")
                results['status'] = 'FAIL'
        
        results['warnings'] = warnings
        results['errors'] = errors
        
        return results
    
    def test_complete_instruction_tokenization(self, verbose: bool = True) -> Dict:
        """
        Tests tokenization of complete PTX instructions.
        Target metrics:
        - Simple instructions (mov, add): 5-8 tokens
        - Memory instructions (ld, st): 7-12 tokens
        - Complex instructions (mad, fma): 8-15 tokens
        - Atomic/Shuffle: 10-20 tokens
        - Alert if > 25 tokens for a single instruction
        """
        test_instructions = [
            # Simple instructions
            ('mov.u32 %r5, %ntid.x;', 'Simple move', 5, 8),
            ('add.s32 %r1, %r2, %r3;', 'Simple add', 5, 8),
            ('mul.lo.s32 %r4, %r5, %r6;', 'Simple multiply', 5, 9),
            ('bar.sync 0;', 'Barrier', 3, 6),
            ('ret;', 'Return', 2, 3),
            
            # Memory instructions
            ('ld.global.ca.f32 %f0, [%rd0+1024];', 'Load with offset', 7, 12),
            ('ld.global.f32 %f1, [%rd1];', 'Simple load', 6, 10),
            ('st.global.f32 [%rd0+128], %f0;', 'Store with offset', 7, 12),
            ('st.shared.u32 [%r0], %r1;', 'Shared store', 6, 10),
            ('ld.shared.v4.f32 {%f0, %f1, %f2, %f3}, [%r0];', 'Vector load', 10, 18),
            
            # Complex arithmetic
            ('mad.lo.s32 %r8, %r6, %r5, %r7;', 'Multiply-add', 8, 15),
            ('fma.rn.f32 %f0, %f1, %f2, %f3;', 'Fused multiply-add', 8, 15),
            ('setp.ge.u64 %p1, %rd3, %rd33;', 'Predicate set', 7, 12),
            ('selp.f32 %f0, %f1, %f2, %p0;', 'Select with predicate', 8, 13),
            
            # Atomic operations
            ('atom.global.add.u32 %r1, [%rd0], 1;', 'Atomic add', 10, 15),
            ('atom.shared.cas.b32 %r0, [%r1], %r2, %r3;', 'Atomic CAS', 10, 16),
            ('red.global.add.u32 [%rd0], %r1;', 'Reduction', 8, 13),
            
            # Warp-level operations
            ('shfl.sync.up.b32 %r1|%p0, %r2, 1, 0, -1;', 'Shuffle', 10, 20),
            ('vote.sync.all.pred %p0|%p1, %p2, -1;', 'Vote', 10, 16),
            ('wmma.load.a.sync.aligned.m16n16k16.f16 {%r0, %r1, %r2, %r3}, [%rd0];', 'WMMA load', 15, 25),
            
            # Control flow
            ('@%p0 bra BB1;', 'Predicated branch', 4, 8),
            ('bra.uni BB2;', 'Uniform branch', 3, 6),
            ('call.uni func_name, (%r0);', 'Function call', 6, 10),
        ]
        
        results = {
            'passed': 0,
            'warnings': 0,
            'failed': 0,
            'details': [],
            'stats': {
                'min_tokens': float('inf'),
                'max_tokens': 0,
                'total_tokens': 0,
                'avg_tokens': 0
            }
        }
        
        for instruction, description, min_expected, max_expected in test_instructions:
            pieces = self.sp.EncodeAsPieces(instruction)
            num_tokens = len(pieces)
            
            # Update stats
            results['stats']['min_tokens'] = min(results['stats']['min_tokens'], num_tokens)
            results['stats']['max_tokens'] = max(results['stats']['max_tokens'], num_tokens)
            results['stats']['total_tokens'] += num_tokens
            
            # Determine status
            if min_expected <= num_tokens <= max_expected:
                status = 'PASS'
                results['passed'] += 1
            elif num_tokens < min_expected:
                status = 'TOO FEW'
                results['warnings'] += 1
            elif num_tokens <= 25:
                status = 'HIGH'
                results['warnings'] += 1
            else:
                status = 'FAIL'
                results['failed'] += 1
            
            result_detail = {
                'instruction': instruction,
                'description': description,
                'num_tokens': num_tokens,
                'expected_range': (min_expected, max_expected),
                'pieces': pieces,
                'status': status
            }
            results['details'].append(result_detail)
            
            if verbose:
                print(f"{status:12} | {description:25} | {num_tokens:2} tokens (expected: {min_expected}-{max_expected})")
                print(f"             {instruction}")
                if status != 'PASS':
                    print(f"             Pieces: {pieces}")
                print()
        

        total = len(test_instructions)
        results['stats']['avg_tokens'] = results['stats']['total_tokens'] / total if total > 0 else 0
        pass_rate = (results['passed'] / total) * 100
        
        if verbose:
            print(f"Total instructions tested: {total}")
            print(f"Passed: {results['passed']}, Warnings: {results['warnings']}, Failed: {results['failed']}")
            print(f"Pass rate: {pass_rate:.1f}%")
            print(f"Token count: min={results['stats']['min_tokens']}, "
                  f"max={results['stats']['max_tokens']}, "
                  f"avg={results['stats']['avg_tokens']:.1f}")
            
            if pass_rate >= 85 and results['stats']['avg_tokens'] <= 12:
                print("\nEXCELLENT: Instructions are efficiently tokenized")
                results['status'] = 'EXCELLENT'
            elif pass_rate >= 70:
                print("\nGOOD: Acceptable instruction tokenization")
                results['status'] = 'GOOD'
            elif pass_rate >= 50:
                print("\nWARNING: Many instructions are poorly tokenized")
                results['status'] = 'WARNING'
            else:
                print("\nFAIL: Severe instruction tokenization issues")
                results['status'] = 'FAIL'
        
        results['pass_rate'] = pass_rate
        
        return results
    
    def test_sequence_length_distribution(self, ptx_files: List[Path], 
                                         sample_size: int = 100000, 
                                         max_seq_length: int = 2048,  verbose: bool = True) -> Dict:
        """
        Analyzes token sequence lengths across PTX files to determine:
        - Mean, median, P95, P99, max sequence lengths
        - Percentage of files that fit within max_seq_length
        - Whether truncation will be a significant issue
        
        Target metrics:
        - Median < 1500 tokens (most kernels fit)
        - P95 < max_seq_length (95% of kernels fit without truncation)
        """
        if verbose:
            print(f"Max sequence length: {max_seq_length}")
            print(f"Sampling {min(sample_size, len(ptx_files))} files from {len(ptx_files)} total")
        
        import random
        sample_files = random.sample(ptx_files, min(sample_size, len(ptx_files))) if len(ptx_files) > sample_size else ptx_files
        
        seq_lengths = []
        truncation_needed = 0
        errors = 0
        
        if verbose:
            print("\nProcessing files...")
        
        for i, ptx_file in enumerate(sample_files):
            try:
                with open(ptx_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if not content.strip():
                    continue
                
                ids = self.sp.EncodeAsIds(content)
                seq_len = len(ids)
                seq_lengths.append(seq_len)
                
                if seq_len > max_seq_length:
                    truncation_needed += 1
                
                if verbose and (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(sample_files)} files...")
            
            except Exception as e:
                errors += 1
                if verbose and errors <= 5:
                    print(f"  Error processing {ptx_file.name}: {e}")
        
        if not seq_lengths:
            return {
                'status': 'ERROR',
                'message': 'No valid files processed'
            }

        seq_lengths = np.array(seq_lengths)
        stats = {
            'count': len(seq_lengths),
            'mean': float(np.mean(seq_lengths)),
            'median': float(np.median(seq_lengths)),
            'std': float(np.std(seq_lengths)),
            'min': int(np.min(seq_lengths)),
            'max': int(np.max(seq_lengths)),
            'p25': float(np.percentile(seq_lengths, 25)),
            'p50': float(np.percentile(seq_lengths, 50)),
            'p75': float(np.percentile(seq_lengths, 75)),
            'p90': float(np.percentile(seq_lengths, 90)),
            'p95': float(np.percentile(seq_lengths, 95)),
            'p99': float(np.percentile(seq_lengths, 99)),
        }
        
        fit_rate = ((len(seq_lengths) - truncation_needed) / len(seq_lengths)) * 100
        
        results = {
            'stats': stats,
            'max_seq_length': max_seq_length,
            'truncation_needed': truncation_needed,
            'truncation_rate': (truncation_needed / len(seq_lengths)) * 100,
            'fit_rate': fit_rate,
            'errors': errors,
            'status': 'UNKNOWN'
        }
        
        if verbose:
            print("SEQUENCE LENGTH STATISTICS")
            print(f"Files processed: {stats['count']}")
            print(f"Errors: {errors}")
            print()
            print(f"{'Statistic':<15} {'Value':>10}")
            print("-" * 27)
            print(f"{'Mean':<15} {stats['mean']:>10.1f}")
            print(f"{'Median':<15} {stats['median']:>10.1f}")
            print(f"{'Std Dev':<15} {stats['std']:>10.1f}")
            print(f"{'Min':<15} {stats['min']:>10}")
            print(f"{'Max':<15} {stats['max']:>10}")
            print()
            print(f"{'Percentile':<15} {'Tokens':>10}")
            print("-" * 27)
            print(f"{'P25':<15} {stats['p25']:>10.1f}")
            print(f"{'P50 (Median)':<15} {stats['p50']:>10.1f}")
            print(f"{'P75':<15} {stats['p75']:>10.1f}")
            print(f"{'P90':<15} {stats['p90']:>10.1f}")
            print(f"{'P95':<15} {stats['p95']:>10.1f}")
            print(f"{'P99':<15} {stats['p99']:>10.1f}")
            print()
            print("=" * 80)
            print("TRUNCATION ANALYSIS")
            print("=" * 80)
            print(f"Max sequence length: {max_seq_length}")
            print(f"Files requiring truncation: {truncation_needed}/{stats['count']} ({results['truncation_rate']:.1f}%)")
            print(f"Files that fit: {stats['count'] - truncation_needed}/{stats['count']} ({fit_rate:.1f}%)")
            print()
            
            # Evaluation
            if stats['p95'] < max_seq_length and stats['median'] < max_seq_length * 0.75:
                print("EXCELLENT: Most files fit comfortably within max_seq_length")
                results['status'] = 'EXCELLENT'
            elif stats['p95'] < max_seq_length:
                print("GOOD: 95% of files fit within max_seq_length")
                results['status'] = 'GOOD'
            elif stats['median'] < max_seq_length:
                print("WARNING: Median fits but significant truncation (>5%) needed")
                results['status'] = 'WARNING'
                print(f"\nRECOMMENDATION: Consider one of:")
                print(f"  1. Increase max_seq_length to {int(stats['p95'] * 1.1)}")
                print(f"  2. Reduce vocabulary size for better compression")
                print(f"  3. Accept truncation for long files")
            else:
                print("CRITICAL: More than half of files require truncation")
                results['status'] = 'CRITICAL'
                print(f"\nACTION REQUIRED: One of:")
                print(f"  1. Increase max_seq_length to at least {int(stats['p95'] * 1.1)}")
                print(f"  2. Significantly reduce vocabulary or change tokenization strategy")
        
        return results
    
    def test_out_of_distribution_patterns(self, verbose: bool = True) -> Dict:
        """
        Tests tokenizer robustness on rare but valid PTX patterns:
        - Vector operations with different modifiers
        - Rare memory operations
        - Modern PTX features (async copy, warp-level ops)
        - System-level operations
        
        Target: UNK rate < 20% on rare patterns (< 10% is excellent)
        """
        rare_patterns = [
            # Vector operations with various modifiers
            ('ld.global.nc.v4.f32 {%f0, %f1, %f2, %f3}, [%rd0];', 'Non-cached vector load'),
            ('st.global.wb.v2.u64 [%rd0], {%rd1, %rd2};', 'Write-back vector store'),
            ('ld.local.ca.v2.f64 {%fd0, %fd1}, [%r0];', 'Local cached vector load'),
            
            # Reduction operations
            ('red.global.add.f64 [%rd0], %fd1;', 'Float64 reduction'),
            ('red.shared.min.s32 [%r0], %r1;', 'Shared min reduction'),
            ('red.global.max.f32 [%rd0], %f0;', 'Global max reduction'),
            
            # Warp-level primitives
            ('match.sync.all.b32 %r1|%p0, %r2, 0xffffffff;', 'Warp match'),
            ('vote.sync.any.pred %p0|%p1, %p2, 0xffffffff;', 'Warp vote any'),
            ('activemask.b32 %r0;', 'Active mask'),
            
            # Memory fences
            ('fence.sc.sys;', 'System fence'),
            ('fence.acq_rel.gpu;', 'GPU acquire-release fence'),
            ('membar.gl;', 'Global memory barrier'),
            
            # Async operations
            ('cp.async.ca.shared.global [%r0], [%rd1], 16;', 'Async copy'),
            ('cp.async.commit_group;', 'Async commit'),
            ('cp.async.wait_group 0;', 'Async wait'),
            
            # Tensor core operations
            ('wmma.load.a.sync.aligned.m16n16k16.f16 {%r0, %r1, %r2, %r3}, [%rd0];', 'WMMA load A'),
            ('wmma.mma.sync.aligned.m16n16k16.f16.f16 {%r0, %r1, %r2, %r3}, {%r4, %r5, %r6, %r7}, {%r8, %r9, %r10, %r11}, {%r0, %r1, %r2, %r3};', 'WMMA MMA'),
            ('mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%f0, %f1, %f2, %f3}, {%r0, %r1}, {%r2}, {%f0, %f1, %f2, %f3};', 'MMA TF32'),
            
            # Surface and texture operations
            ('suld.b.1d.v4.b32 {%r0, %r1, %r2, %r3}, [%r4, {%r5}];', 'Surface load'),
            ('sust.b.2d.v2.b32 [%r0, {%r1, %r2}], {%r3, %r4};', 'Surface store'),
            
            # Prefetch operations  
            ('prefetch.global.L2 [%rd0];', 'L2 prefetch'),
            ('prefetchu.L1 [%rd0+1024];', 'L1 uniform prefetch'),
            
            # Miscellaneous rare operations
            ('brkpt;', 'Breakpoint'),
            ('trap;', 'Trap'),
            ('pmevent %r0, 0;', 'Performance event'),
        ]
        
        results = {
            'passed': 0,
            'warnings': 0,
            'failed': 0,
            'details': [],
            'total_unk': 0,
            'total_tokens': 0
        }
        
        for pattern, description in rare_patterns:
            pieces = self.sp.EncodeAsPieces(pattern)
            ids = self.sp.EncodeAsIds(pattern)
            
            unk_count = sum(1 for id in ids if id == self.sp.unk_id())
            unk_rate = (unk_count / len(ids)) * 100 if len(ids) > 0 else 0
            
            results['total_unk'] += unk_count
            results['total_tokens'] += len(ids)
            
            # Determine status
            if unk_rate == 0:
                status = 'EXCELLENT'
                results['passed'] += 1
            elif unk_rate < 10:
                status = 'GOOD'
                results['passed'] += 1
            elif unk_rate < 20:
                status = 'ACCEPTABLE'
                results['warnings'] += 1
            else:
                status = 'HIGH UNK'
                results['failed'] += 1
            
            result_detail = {
                'pattern': pattern,
                'description': description,
                'num_tokens': len(ids),
                'unk_count': unk_count,
                'unk_rate': unk_rate,
                'pieces': pieces,
                'status': status
            }
            results['details'].append(result_detail)
            
            if verbose:
                print(f"{status:15} | {description:30} | UNK: {unk_count}/{len(ids)} ({unk_rate:.1f}%)")
                print(f"                {pattern}")
                if unk_rate > 10:
                    print(f"                Pieces: {pieces[:10]}{'...' if len(pieces) > 10 else ''}")
                print()
        
        total = len(rare_patterns)
        pass_rate = (results['passed'] / total) * 100
        overall_unk_rate = (results['total_unk'] / results['total_tokens']) * 100 if results['total_tokens'] > 0 else 0
        
        if verbose:
            print("=" * 80)
            print("SUMMARY")
            print("=" * 80)
            print(f"Patterns tested: {total}")
            print(f"Excellent/Good: {results['passed']}, Acceptable: {results['warnings']}, High UNK: {results['failed']}")
            print(f"Overall UNK rate: {overall_unk_rate:.2f}%")
            print(f"Pass rate: {pass_rate:.1f}%")
            print()
            
            if overall_unk_rate < 5:
                print("EXCELLENT: Tokenizer handles rare patterns very well")
                results['status'] = 'EXCELLENT'
            elif overall_unk_rate < 10:
                print("GOOD: Tokenizer is robust to rare patterns")
                results['status'] = 'GOOD'
            elif overall_unk_rate < 20:
                print("ACCEPTABLE: Some issues with rare patterns")
                results['status'] = 'ACCEPTABLE'
            else:
                print("POOR: High UNK rate on rare patterns")
                results['status'] = 'POOR'
                print("\nRECOMMENDATION: Train on more diverse PTX samples")
        
        results['pass_rate'] = pass_rate
        results['overall_unk_rate'] = overall_unk_rate
        
        return results
    
    def test_contextual_consistency(self, verbose: bool = True) -> Dict:
        """
        Tests whether the same token in different contexts is tokenized consistently.
        This is important for the model to learn stable representations.
        
        Tests:
        - Same opcode with different operands
        - Same register in different positions
        - Same modifier in different instructions
        """
        test_groups = [
            # Same opcode in different contexts
            {
                'name': 'ld.global.ca.f32 opcode',
                'contexts': [
                    'ld.global.ca.f32 %f0, [%rd0];',
                    'ld.global.ca.f32 %f10, [%rd5+1024];',
                    'ld.global.ca.f32 %f99, [%rd100];',
                    'ld.global.ca.f32 %f1, [%rd1+0x400];',
                ],
                'target': 'ld.global.ca.f32',
                'expected_position': 0,
                'expected_tokens': 4
            },
            {
                'name': 'add.f32 opcode',
                'contexts': [
                    'add.f32 %f0, %f1, %f2;',
                    'add.f32 %f10, %f20, %f30;',
                    'add.f32 %f100, %f200, %f300;',
                ],
                'target': 'add.f32',
                'expected_position': 0,
                'expected_tokens': 2
            },
            {
                'name': 'setp.lt.f32 opcode',
                'contexts': [
                    'setp.lt.f32 %p0, %f0, %f1;',
                    'setp.lt.f32 %p1, %f2, %f3;',
                    '@%p2 setp.lt.f32 %p3, %f4, %f5;',
                ],
                'target': 'setp.lt.f32',
                'expected_position': 0,
                'expected_tokens': 3
            },
            # Same register type in different positions
            {
                'name': '%rd0 register',
                'contexts': [
                    'ld.global.f32 %f0, [%rd0];',
                    'ld.global.f32 %f0, [%rd0+1024];',
                    'st.global.f32 [%rd0], %f0;',
                    'add.u64 %rd0, %rd0, %rd1;',
                ],
                'target': '%rd0',
                'expected_position': None,  # Can appear in different positions
                'expected_tokens': 2
            },
            # Same modifier in different instructions
            {
                'name': '.global modifier',
                'contexts': [
                    'ld.global.f32 %f0, [%rd0];',
                    'st.global.u32 [%rd0], %r0;',
                    'atom.global.add.u32 %r0, [%rd0], 1;',
                ],
                'target': '.global',
                'expected_position': None,
                'expected_tokens': 1
            },
        ]
        
        results = {
            'passed': 0,
            'failed': 0,
            'details': []
        }
        
        for group in test_groups:
            name = group['name']
            contexts = group['contexts']
            target = group['target']
            expected_tokens = group['expected_tokens']
            
            # Tokenize all contexts
            all_pieces = []
            all_ids = []
            for ctx in contexts:
                pieces = self.sp.EncodeAsPieces(ctx)
                ids = self.sp.EncodeAsIds(ctx)
                all_pieces.append(pieces)
                all_ids.append(ids)
            
            # Extract target tokens from each context
            target_sequences = []
            for pieces in all_pieces:
                # Try to find the target sequence
                target_found = []
                for i, piece in enumerate(pieces):
                    if target.startswith(piece.replace('▁', '')):
                        # Found potential match, collect next tokens
                        sequence = pieces[i:min(i + expected_tokens + 2, len(pieces))]
                        target_found.append(tuple(sequence[:expected_tokens]))
                        break
                if target_found:
                    target_sequences.append(target_found[0])
            
            # Check consistency
            if len(target_sequences) > 1:
                # Check if all sequences are identical or at least first few tokens
                first_sequence = target_sequences[0]
                consistent = all(seq[:2] == first_sequence[:2] for seq in target_sequences)
                
                if consistent:
                    status = 'CONSISTENT'
                    results['passed'] += 1
                else:
                    status = 'INCONSISTENT'
                    results['failed'] += 1
            else:
                status = 'UNABLE TO TEST'
                results['failed'] += 1
                consistent = False
            
            result_detail = {
                'name': name,
                'target': target,
                'contexts': contexts,
                'tokenizations': [str(pieces[:expected_tokens + 2]) for pieces in all_pieces],
                'target_sequences': [str(seq) for seq in target_sequences] if target_sequences else [],
                'consistent': consistent,
                'status': status
            }
            results['details'].append(result_detail)
            
            if verbose:
                print(f"\n{status} | Testing: {name}")
                print(f"Target: '{target}'")
                print()
                for i, ctx in enumerate(contexts):
                    print(f"  Context {i+1}: {ctx}")
                    print(f"    Tokens: {all_pieces[i][:expected_tokens + 3]}")
                    if i < len(target_sequences):
                        print(f"    Target sequence: {target_sequences[i]}")
                print()
                if not consistent and len(target_sequences) > 1:
                    print(f"WARNING: Target is tokenized differently across contexts")
                    print(f"  This may affect model's ability to learn consistent representations")
        
        # Summary
        total = len(test_groups)
        pass_rate = (results['passed'] / total) * 100 if total > 0 else 0
        
        if verbose:
            print(f"Consistency tests: {total}")
            print(f"Consistent: {results['passed']}, Inconsistent: {results['failed']}")
            print(f"Consistency rate: {pass_rate:.1f}%")
            print()
            
            if pass_rate >= 90:
                print("EXCELLENT: Tokenization is highly consistent")
                results['status'] = 'EXCELLENT'
            elif pass_rate >= 70:
                print("GOOD: Mostly consistent tokenization")
                results['status'] = 'GOOD'
            elif pass_rate >= 50:
                print("WARNING: Some inconsistencies detected")
                results['status'] = 'WARNING'
            else:
                print("POOR: Significant inconsistencies")
                results['status'] = 'POOR'
                print("\nThis may affect model training quality")
        
        results['pass_rate'] = pass_rate
        
        return results
    
    def _print_summary(self):
        """Print overall test summary"""
        print("\n" + "=" * 80)
        print("OVERALL TEST SUMMARY")
        print("=" * 80)
        
        test_names = {
            'number_tokenization': 'Number/Constant Tokenization',
            'token_distribution': 'Token Distribution by Category',
            'instruction_tokenization': 'Complete Instruction Tokenization',
            'sequence_lengths': 'Sequence Length Distribution',
            'out_of_distribution': 'Out-of-Distribution Patterns',
            'contextual_consistency': 'Contextual Consistency'
        }
        
        for key, name in test_names.items():
            if key in self.results and self.results[key] is not None:
                result = self.results[key]
                status = result.get('status', 'UNKNOWN')
                if status in ['EXCELLENT', 'PASS']:
                    status_str = f"{status}"
                elif status in ['GOOD', 'ACCEPTABLE']:
                    status_str = f"{status}"
                elif status in ['WARNING']:
                    status_str = f"{status}"
                else:
                    status_str = f"{status}"
                
                print(f"{name:60} {status_str}")
            else:
                print(f"{name:60}  SKIPPED")
        
        
        critical_ok = True
        important_ok = True
        
        for key in ['number_tokenization', 'token_distribution']:
            if key in self.results:
                status = self.results[key].get('status', 'UNKNOWN')
                if status in ['FAIL', 'CRITICAL', 'ERROR']:
                    critical_ok = False
        
        for key in ['instruction_tokenization', 'sequence_lengths']:
            if key in self.results and self.results[key] is not None:
                status = self.results[key].get('status', 'UNKNOWN')
                if status in ['FAIL', 'CRITICAL', 'ERROR']:
                    important_ok = False
        
        print()
        if critical_ok and important_ok:
            print("FINAL VERDICT: Tokenizer is production-ready!")
        elif critical_ok:
            print("FINAL VERDICT: Tokenizer passes critical tests, review important test warnings")
        else:
            print("FINAL VERDICT: Critical issues detected, tokenizer needs retraining")
            print("\nRecommendations:")
            print("- Review failed critical tests")
            print("- Consider increasing vocabulary size")
            print("- Add more user-defined symbols")
            print("- Ensure training data is diverse and representative")
    
    def save_results(self, output_file: Path):
        """Save test results to JSON file"""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy types to native Python types for JSON serialization
        results_serializable = json.loads(
            json.dumps(self.results, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else str(x))
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Comprehensive PTX Tokenizer Test Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('model', type=str,
                       help='Path to trained tokenizer model (.model file)')
    parser.add_argument('--ptx-dir', type=str, default=None,
                       help='Directory containing PTX files for sequence length testing')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='Number of PTX files to sample for sequence length test (default: 100)')
    parser.add_argument('--max-seq-length', type=int, default=2048,
                       help='Maximum sequence length for truncation analysis (default: 2048)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file for test results (JSON format)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode (less verbose output)')
    
    args = parser.parse_args()
    
    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)
    
    # Get PTX files if directory provided
    ptx_files = None
    if args.ptx_dir:
        ptx_dir = Path(args.ptx_dir)
        if not ptx_dir.exists():
            print(f"WARNING: PTX directory not found: {ptx_dir}")
            print("Skipping sequence length distribution test")
        else:
            # Use subprocess to find files (handles large directories better)
            try:
                cmd = ['find', str(ptx_dir), '-type', 'f', '-name', '*.ptx']
                result = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().splitlines()
                ptx_files = [Path(line.strip()) for line in result if line.strip()]
                print(f"Found {len(ptx_files)} PTX files in {ptx_dir}")
            except Exception as e:
                # Fallback to glob
                ptx_files = list(ptx_dir.rglob('*.ptx'))
                print(f"Found {len(ptx_files)} PTX files in {ptx_dir}")
    
    # Run tests
    tester = PTXTokenizerTester(str(model_path))
    results = tester.run_all_tests(
        ptx_files=ptx_files,
        verbose=not args.quiet,
        sample_size=args.sample_size,
        max_seq_length=args.max_seq_length
    )
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        tester.save_results(output_path)
    else:
        # Default output location
        default_output = model_path.parent / 'comprehensive_test_results.json'
        tester.save_results(default_output)


if __name__ == '__main__':
    main()
