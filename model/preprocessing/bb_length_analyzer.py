import argparse
from asyncio import subprocess
import cmd
import cmd
import re
import subprocess
from pathlib import Path
from typing import List, Dict
import sys
from unittest import result
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from tokenizer.tokenizer import PTXTokenizer


class BBLengthAnalyzer:
    def __init__(self, tokenizer: PTXTokenizer, verbose: bool = False):
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.bb_lengths = []  
        self.file_stats = []  
        
    def analyze_file(self, file_path: Path) -> Dict:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                return None
            bb_blocks = self._extract_bb_blocks(content)
            
            if not bb_blocks:
                if self.verbose:
                    print(f"  No basic blocks found in {file_path.name}")
                return None
            
            bb_lengths_in_file = [block['token_count'] for block in bb_blocks]
            
            if self.verbose:
                print(f"\n  Found {len(bb_blocks)} basic blocks in {file_path.name}:")
                for i, block in enumerate(bb_blocks[:10]):  
                    print(f"    {block['label']:12} - {block['token_count']:>5} tokens "
                          f"({block['char_count']:>5} chars)")
                if len(bb_blocks) > 10:
                    print(f"    ... and {len(bb_blocks) - 10} more")
                print()
            
            total_tokens = sum(bb_lengths_in_file)
            file_stats = {
                'file': file_path.name,
                'total_tokens': total_tokens,
                'num_bbs': len(bb_blocks),
                'bb_lengths': bb_lengths_in_file,
                'min': min(bb_lengths_in_file),
                'max': max(bb_lengths_in_file),
                'mean': np.mean(bb_lengths_in_file),
                'median': np.median(bb_lengths_in_file),
                'std': np.std(bb_lengths_in_file),
                'p95': np.percentile(bb_lengths_in_file, 95),
                'p99': np.percentile(bb_lengths_in_file, 99),
            }
            self.bb_lengths.extend(bb_lengths_in_file)
            self.file_stats.append(file_stats)
            
            if self.verbose:
                print(f"  Summary: {len(bb_blocks)} BBs, "
                      f"min={file_stats['min']}, max={file_stats['max']}, "
                      f"mean={file_stats['mean']:.1f}, median={file_stats['median']:.1f}\n")
            
            return file_stats
            
        except Exception as e:
            print(f"Error analyzing {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_bb_blocks(self, content: str) -> List[Dict]:
        bb_pattern = re.compile(r'<BB_(\d+)>\s*:')
        matches = list(bb_pattern.finditer(content))
        
        if not matches:
            return []
        
        blocks = []
        for i, match in enumerate(matches):
            label = f"<BB_{match.group(1)}>:"
            start_pos = match.start()
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(content)
            block_text = content[start_pos:end_pos]
            token_ids = self.tokenizer.encode(block_text)
            
            blocks.append({
                'label': label,
                'token_count': len(token_ids),
                'char_count': len(block_text),
                'start_pos': start_pos,
                'end_pos': end_pos,
            })
        
        return blocks
    
    def analyze_directory(self, dir_path: Path, max_files: int = None) -> None:
        cmd = ['find', str(dir_path), '-maxdepth', '1', '-type', 'f', '-name', '*.ptx']
        result = subprocess.check_output(cmd).decode().splitlines()
        ptx_files = sorted(Path(line.strip()) for line in result if line.strip())
        print(f"Found {len(ptx_files):,} PTX files via find") 
        
        if max_files:
            ptx_files = ptx_files[:max_files]
        
        total_files = len(ptx_files)
        print(f"Found {total_files} PTX files in {dir_path}")
        print(f"Analyzing basic block lengths...\n")
        
        for i, ptx_file in enumerate(ptx_files, 1):
            if not self.verbose and i % 100 == 0:
                print(f"  Processed {i}/{total_files} files...")
            
            self.analyze_file(ptx_file)
        
        print(f"\nCompleted analysis of {total_files} files")
    
    def print_summary(self) -> None:
        if not self.bb_lengths:
            print("No basic blocks found in analyzed files!")
            return
        print("BASIC BLOCK LENGTH ANALYSIS SUMMARY\n")
        print(f"\nOverall Statistics:")
        print(f"  Total files analyzed:    {len(self.file_stats)}")
        print(f"  Total basic blocks:      {len(self.bb_lengths):,}")
        print(f"  Min BB length:           {min(self.bb_lengths)} tokens")
        print(f"  Max BB length:           {max(self.bb_lengths):,} tokens")
        print(f"  Mean BB length:          {np.mean(self.bb_lengths):.2f} tokens")
        print(f"  Median BB length:        {np.median(self.bb_lengths):.2f} tokens")
        print(f"  Std deviation:           {np.std(self.bb_lengths):.2f} tokens")
        print(f"\nPercentile Analysis:")
        percentiles = [50, 75, 90, 95, 99, 99.9]
        for p in percentiles:
            value = np.percentile(self.bb_lengths, p)
            print(f"  {p:5.1f}th percentile:     {value:>8.1f} tokens")
        print(f"\nDistribution Breakdown:")
        ranges = [
            (0, 50, "Tiny (0-50 tokens)"),
            (51, 100, "Small (51-100 tokens)"),
            (101, 200, "Medium (101-200 tokens)"),
            (201, 500, "Large (201-500 tokens)"),
            (501, 1000, "Very Large (501-1000 tokens)"),
            (1001, 2000, "Huge (1001-2000 tokens)"),
            (2001, float('inf'), "Massive (>2000 tokens)"),
        ]
        
        for min_val, max_val, label in ranges:
            count = sum(1 for length in self.bb_lengths if min_val <= length <= max_val)
            percentage = (count / len(self.bb_lengths)) * 100
            print(f"  {label:30} {count:>7,} ({percentage:>5.2f}%)")
        print("RECOMMENDATIONS FOR MAX_SEQUENCE_LENGTH\n")
        p95 = np.percentile(self.bb_lengths, 95)
        p99 = np.percentile(self.bb_lengths, 99)
        max_len = max(self.bb_lengths)
        print(f"\nBased on the analysis:")
        print(f"  - Conservative (covers 95%): {int(p95)} tokens")
        print(f"  - Standard (covers 99%):     {int(p99)} tokens")
        print(f"  - Maximum (covers 100%):     {int(max_len)} tokens")
        print(f"\nRecommended max_sequence_length values:")
        if p95 <= 512:
            print(f"512 tokens   - Covers {sum(1 for l in self.bb_lengths if l <= 512)/len(self.bb_lengths)*100:.1f}% of BBs (efficient)")
        if p99 <= 1024:
            print(f"1024 tokens  - Covers {sum(1 for l in self.bb_lengths if l <= 1024)/len(self.bb_lengths)*100:.1f}% of BBs (balanced)")
        if p99 <= 2048:
            print(f"2048 tokens  - Covers {sum(1 for l in self.bb_lengths if l <= 2048)/len(self.bb_lengths)*100:.1f}% of BBs (safe)")
        
        print(f"4096 tokens  - Covers {sum(1 for l in self.bb_lengths if l <= 4096)/len(self.bb_lengths)*100:.1f}% of BBs (very safe)")
        
        print(f"\nNote: Very long BBs (>{int(p99)} tokens) are rare ({(1 - 0.99)*100:.1f}%) and can be:")
        print(f"Truncated during training")
        print(f"Split into multiple sequences")
        print(f"Handled with sliding window attention")
        print("TOP 10 LONGEST BASIC BLOCKS")
        top_files = sorted(self.file_stats, key=lambda x: x['max'], reverse=True)[:10]
        
        for i, stats in enumerate(top_files, 1):
            print(f"{i:2}. {stats['file']:50} {stats['max']:>6,} tokens (mean: {stats['mean']:>6.1f})")
    
    def save_detailed_report(self, output_file: Path) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("BASIC BLOCK LENGTH ANALYSIS - DETAILED REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total files analyzed: {len(self.file_stats)}\n")
            f.write(f"Total basic blocks: {len(self.bb_lengths):,}\n\n")
            f.write("Overall Statistics:\n")
            f.write(f"  Min:     {min(self.bb_lengths)} tokens\n")
            f.write(f"  Max:     {max(self.bb_lengths):,} tokens\n")
            f.write(f"  Mean:    {np.mean(self.bb_lengths):.2f} tokens\n")
            f.write(f"  Median:  {np.median(self.bb_lengths):.2f} tokens\n")
            f.write(f"  Std Dev: {np.std(self.bb_lengths):.2f} tokens\n\n")
            f.write("Percentile Analysis:\n")
            percentiles = [50, 75, 90, 95, 99, 99.5, 99.9]
            for p in percentiles:
                value = np.percentile(self.bb_lengths, p)
                f.write(f"  {p:5.1f}th: {value:>8.1f} tokens\n")
            f.write("\n" + "="*80 + "\n")
            f.write("PER-FILE STATISTICS\n")
            f.write("="*80 + "\n\n")
            sorted_files = sorted(self.file_stats, key=lambda x: x['max'], reverse=True)
            f.write(f"{'File':<50} {'BBs':>6} {'Min':>6} {'Max':>6} {'Mean':>7} {'Median':>7} {'P95':>7} {'P99':>7}\n")
            f.write("-"*110 + "\n")
            for stats in sorted_files:
                f.write(f"{stats['file']:<50} "
                       f"{stats['num_bbs']:>6} "
                       f"{stats['min']:>6} "
                       f"{stats['max']:>6} "
                       f"{stats['mean']:>7.1f} "
                       f"{stats['median']:>7.1f} "
                       f"{stats['p95']:>7.1f} "
                       f"{stats['p99']:>7.1f}\n")
        print(f"\nDetailed report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze basic block lengths in PTX files to determine optimal max_sequence_length',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single file
  python bb_length_analyzer.py --model data/tokenizer/ptx_tokenizer.model --file data/processed/sample.ptx
  
  # Analyze all files in processed/ directory
  python bb_length_analyzer.py --model data/tokenizer/ptx_tokenizer.model --dir data/processed
  
  # Analyze with verbose output and save detailed report
  python bb_length_analyzer.py --model data/tokenizer/ptx_tokenizer.model --dir data/processed --verbose --save report.txt
        """
    )
    
    parser.add_argument('--model', '-m', type=str, required=True,
                       help='Path to trained tokenizer model file')
    parser.add_argument('--file', '-f', type=str, default=None,
                       help='Path to single PTX file to analyze')
    parser.add_argument('--dir', '-d', type=str, default=None,
                       help='Path to directory containing PTX files')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to analyze (default: all)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print verbose output for each file')
    parser.add_argument('--save', '-s', type=str, default=None,
                       help='Save detailed report to file')
    
    args = parser.parse_args()
    if not args.file and not args.dir:
        parser.error("Must specify either --file or --dir")
    
    if args.file and args.dir:
        parser.error("Cannot specify both --file and --dir")

    project_root = Path(__file__).parent.parent.parent
    
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    
    if not model_path.exists():
        print(f"ERROR: Tokenizer model not found: {model_path}")
        sys.exit(1)
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = PTXTokenizer(str(model_path))
    print(f"Tokenizer loaded (vocab size: {tokenizer.vocab_size})\n")
    analyzer = BBLengthAnalyzer(tokenizer, verbose=args.verbose)
    if args.file:
        file_path = Path(args.file)
        if not file_path.is_absolute():
            file_path = project_root / file_path
        
        if not file_path.exists():
            print(f"ERROR: File not found: {file_path}")
            sys.exit(1)
        
        print(f"Analyzing single file: {file_path.name}")
        analyzer.analyze_file(file_path)
    
    else:  
        dir_path = Path(args.dir)
        if not dir_path.is_absolute():
            dir_path = project_root / dir_path
        
        if not dir_path.exists():
            print(f"ERROR: Directory not found: {dir_path}")
            sys.exit(1)
        
        analyzer.analyze_directory(dir_path, max_files=args.max_files)
    
    analyzer.print_summary()
    if args.save:
        save_path = Path(args.save)
        if not save_path.is_absolute():
            save_path = project_root / save_path
        
        analyzer.save_detailed_report(save_path)


if __name__ == '__main__':
    main()

