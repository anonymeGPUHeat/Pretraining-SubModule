import sentencepiece as spm
import numpy as np
from pathlib import Path
import random
from typing import List
import subprocess


def test_vocabulary_coverage(tokenizer_model: str, ptx_files: List[Path], sample_size: int = 100):
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_model)   
    print("TOKENIZER COVERAGE & QUALITY TEST\n")
    print(f"Model: {tokenizer_model}")
    print(f"Vocabulary size: {sp.GetPieceSize()}")
    print(f"Found {len(ptx_files)} total PTX files")
    print(f"Sampling {min(sample_size, len(ptx_files))} files for analysis")
    sample_files = random.sample(ptx_files, min(sample_size, len(ptx_files)))
    total_tokens = 0
    total_chars = 0
    unk_count = 0
    token_lengths = []
    tokens_per_file = []
    roundtrip_errors = 0
    total_files = 0
    skipped_empty = 0
    skipped_errors = 0
    error_log = []  
    
    print("\nProcessing files...")
    for i, file_path in enumerate(sample_files):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            #if not content.strip():
            #    skipped_empty += 1
            #    continue

            if len(content) == 0:  
                skipped_empty += 1
                continue

            tokens = sp.EncodeAsPieces(content)
            ids = sp.EncodeAsIds(content)
            decoded = sp.DecodeIds(ids)
            
            total_tokens += len(tokens)
            total_chars += len(content)
            tokens_per_file.append(len(ids))
            
            unk_count += ids.count(sp.unk_id())
            token_lengths.extend([len(t) for t in tokens])
            
            if decoded.strip() != content.strip():
                roundtrip_errors += 1
            
            total_files += 1
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(sample_files)} files...")
        
        except MemoryError as e:
            error_msg = f"MEMORY ERROR: {file_path.name} - File too large"
            print(f"  {error_msg}")
            error_log.append({'file': file_path.name, 'error': 'MemoryError', 'details': str(e)})
            skipped_errors += 1
            continue
        
        except UnicodeDecodeError as e:
            error_msg = f"ENCODING ERROR: {file_path.name} - {str(e)[:100]}"
            print(f"  {error_msg}")
            error_log.append({'file': file_path.name, 'error': 'UnicodeDecodeError', 'details': str(e)})
            skipped_errors += 1
            continue
        
        except Exception as e:
            error_msg = f"ERROR: {file_path.name} - {type(e).__name__}: {str(e)[:100]}"
            print(f"  {error_msg}")
            error_log.append({'file': file_path.name, 'error': type(e).__name__, 'details': str(e)})
            skipped_errors += 1
            continue
    compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0
    unk_percentage = (unk_count / total_tokens * 100) if total_tokens > 0 else 0
    avg_token_length = np.mean(token_lengths) if token_lengths else 0
    avg_tokens_per_file = np.mean(tokens_per_file) if tokens_per_file else 0
    median_tokens_per_file = np.median(tokens_per_file) if tokens_per_file else 0
    roundtrip_error_rate = (roundtrip_errors / total_files * 100) if total_files > 0 else 0
    vocab_coverage = 100 - unk_percentage
    if error_log:
        error_log_file = Path('tokenizer_coverage_errors.log')
        with open(error_log_file, 'w', encoding='utf-8') as f:
            f.write("TOKENIZER COVERAGE TEST - ERROR LOG\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total errors: {len(error_log)}\n\n")
            error_types = {}
            for err in error_log:
                err_type = err['error']
                if err_type not in error_types:
                    error_types[err_type] = []
                error_types[err_type].append(err)
            
            for err_type, errors in error_types.items():
                f.write(f"\n{err_type} ({len(errors)} files):\n")
                f.write("-" * 80 + "\n")
                for err in errors:
                    f.write(f"  File: {err['file']}\n")
                    f.write(f"  Details: {err['details']}\n\n")
        
        print(f"\nError log saved to: {error_log_file.absolute()}")
    
    print("COVERAGE STATISTICS")
    print(f"Files in sample:             {len(sample_files):,}")
    print(f"Files processed:             {total_files:,}")
    print(f"Files skipped (empty):       {skipped_empty:,}")
    print(f"Files skipped (errors):      {skipped_errors:,}")
    
    if error_log:
        print(f"\nError breakdown:")
        error_types = {}
        for err in error_log:
            err_type = err['error']
            error_types[err_type] = error_types.get(err_type, 0) + 1
        for err_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {err_type:25} {count:>6,} files")
    
    print(f"\nTotal characters:            {total_chars:,}")
    print(f"Total tokens:                {total_tokens:,}")
    print(f"Compression ratio:           {compression_ratio:.2f} chars/token")
    print(f"Avg token length:            {avg_token_length:.2f} chars")
    print("QUALITY METRICS")
    # Metric 1: Unknown rate
    print(f"\n1. UNKNOWN TOKEN RATE: {unk_percentage:.2f}%")
    print(f"   Unknown tokens: {unk_count:,} / {total_tokens:,}")
    if unk_percentage < 1.0:
        print(f"GOOD: <1% unknown tokens")
        unk_score = "GOOD"
    elif unk_percentage < 5.0:
        print(f"ACCEPTABLE: <5% unknown tokens")
        unk_score = "ACCEPTABLE"
    else:
        print(f"BAD: >5% unknown tokens - increase vocab size or add more symbols")
        unk_score = "BAD"
    print(f"\n2. TOKENS PER FILE:")
    print(f"   Average:  {avg_tokens_per_file:.0f}")
    print(f"   Median:   {median_tokens_per_file:.0f}")
    print(f"   Min/Max:  {min(tokens_per_file) if tokens_per_file else 0} / {max(tokens_per_file) if tokens_per_file else 0}")
    if 500 <= avg_tokens_per_file <= 2000:
        print(f"GOOD: 500-2000 tokens per file")
        tokens_score = "GOOD"
    elif 2000 < avg_tokens_per_file <= 5000:
        print(f" ACCEPTABLE: 2000-5000 tokens per file")
        tokens_score = "ACCEPTABLE"
    else:
        print(f"BAD: >5000 tokens per file - vocab may be too small")
        tokens_score = "BAD"
    
    # Metric 3: Vocabulary coverage
    print(f"\n3. VOCABULARY COVERAGE: {vocab_coverage:.2f}%")
    if vocab_coverage > 99.0:
        print(f"GOOD: >99% coverage")
        coverage_score = "GOOD"
    elif vocab_coverage > 95.0:
        print(f"ACCEPTABLE: >95% coverage")
        coverage_score = "ACCEPTABLE"
    else:
        print(f"BAD: <95% coverage - increase vocab size")
        coverage_score = "BAD"
    
    # Metric 4: Roundtrip errors
    print(f"\n4. ROUNDTRIP ACCURACY: {100 - roundtrip_error_rate:.2f}%")
    print(f"   Errors: {roundtrip_errors} / {total_files}")
    if roundtrip_error_rate == 0:
        print(f"GOOD: 0% error rate")
        roundtrip_score = "GOOD"
    elif roundtrip_error_rate < 1.0:
        print(f"ACCEPTABLE: <1% error rate")
        roundtrip_score = "ACCEPTABLE"
    else:
        print(f"BAD: >1% error rate - check tokenizer settings")
        roundtrip_score = "BAD"
    print("\n" + "=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)
    scores = [unk_score, tokens_score, coverage_score, roundtrip_score]
    good_count = scores.count("GOOD")
    acceptable_count = scores.count("ACCEPTABLE")
    bad_count = scores.count("BAD")
    print(f"GOOD:       {good_count}/4 metrics")
    print(f"ACCEPTABLE: {acceptable_count}/4 metrics")
    print(f"BAD:        {bad_count}/4 metrics")
    if bad_count == 0 and good_count >= 3:
        print("\nEXCELLENT TOKENIZER - Ready for training!")
    elif bad_count == 0:
        print("\nGOOD TOKENIZER - Can be used, but room for improvement")
    elif bad_count == 1:
        print("\nACCEPTABLE TOKENIZER - Consider retraining with adjustments")
    else:
        print("\nPOOR TOKENIZER - Retrain with larger vocab or better symbols")
    
    
    return {
        'compression_ratio': compression_ratio,
        'unk_percentage': unk_percentage,
        'vocab_coverage': vocab_coverage,
        'avg_token_length': avg_token_length,
        'avg_tokens_per_file': avg_tokens_per_file,
        'roundtrip_error_rate': roundtrip_error_rate,
        'scores': {
            'unknown_rate': unk_score,
            'tokens_per_file': tokens_score,
            'vocab_coverage': coverage_score,
            'roundtrip': roundtrip_score
        }
    }


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python coverage.py <tokenizer_model> [ptx_files_dir] [sample_size]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_dir = sys.argv[2] if len(sys.argv) > 2 else '~/processed'
    sample_size = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    test_dir = Path(test_dir).expanduser()
    
    #ptx_files = list(test_dir.glob('*.ptx'))

    cmd = ['find', str(test_dir), '-maxdepth', '1', '-type', 'f', '-name', '*.ptx']
    result = subprocess.check_output(cmd).decode().splitlines()
    ptx_files = [Path(line.strip()) for line in result if line.strip()]
    
    if not ptx_files:
        print(f"ERROR: No PTX files found in {test_dir}")
        sys.exit(1)
    
    print(f"Found {len(ptx_files)} PTX files in {test_dir}")
    print(f"Requested sample size: {sample_size}")
    
    if len(ptx_files) < sample_size:
        print(f"Note: Only {len(ptx_files)} files available (less than requested {sample_size})")
    
    results = test_vocabulary_coverage(model_path, ptx_files, sample_size)



"""
PTX Tokenizer Coverage and Quality Metrics Test
Metrics:
- Unknown rate: <1% GOOD, <5% ACCEPTABLE, >5% BAD
- Tokens per file: 500-2000 GOOD, 2000-5000 ACCEPTABLE, >5000 BAD
- Vocab coverage: >99% GOOD, >95% ACCEPTABLE, <95% BAD
- Roundtrip errors: 0% GOOD, <1% ACCEPTABLE, >1% BAD
"""