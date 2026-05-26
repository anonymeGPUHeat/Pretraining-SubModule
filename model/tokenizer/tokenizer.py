import sentencepiece as spm
from pathlib import Path
from typing import List, Optional
import argparse
import sys
import subprocess
sys.path.insert(0, str(Path(__file__).parent))
from predefined_symbols import create_user_symbols


class PTXTokenizer:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>" 
    EOS_TOKEN = "</s>"
    MASK_TOKEN = "<mask>" #because it can be useful for masked language modeling pretraining, but we won't use it for now
    CLS_TOKEN = "<cls>"
    SEP_TOKEN = "<sep>"
    def __init__(self, model_path: Optional[str] = None):
        self.sp = None
        self.model_path = model_path
        if model_path and Path(model_path).exists():
            self.load(model_path)
    
    def load(self, model_path: str) -> None:
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.model_path = model_path
        print(f"Loaded tokenizer from {model_path}")
        print(f"Vocabulary size: {self.vocab_size}")
    
    @property
    def vocab_size(self) -> int:
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded. Train or load a model first.")
        return self.sp.GetPieceSize()
    
    @property
    def pad_id(self) -> int:
        return self.sp.PieceToId(self.PAD_TOKEN)
    
    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()
    
    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()
    
    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()
    
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded.")
        ids = self.sp.EncodeAsIds(text)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids
    
    def encode_as_pieces(self, text: str) -> List[str]:
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded.")
        return self.sp.EncodeAsPieces(text)
    
    def decode(self, ids: List[int]) -> str:
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded.")
        return self.sp.DecodeIds(ids)
    
    def decode_pieces(self, pieces: List[str]) -> str:
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded.")
        return self.sp.DecodePieces(pieces)
    
    def id_to_piece(self, id: int) -> str:
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded.")
        return self.sp.IdToPiece(id)
    
    def piece_to_id(self, piece: str) -> int:
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded.")
        return self.sp.PieceToId(piece)


def prepare_training_data( input_dir: Path, output_file: Path,max_files: Optional[int] = None,verbose: bool = False) -> int:
    #ptx_files = sorted(input_dir.glob("*.ptx")) 
    cmd = ['find', str(input_dir), '-maxdepth', '1', '-type', 'f', '-name', '*.ptx']
    result = subprocess.check_output(cmd).decode().splitlines()
    ptx_files = sorted(Path(line.strip()) for line in result if line.strip())
    print(f"Found {len(ptx_files):,} PTX files via find")
    if max_files:
        ptx_files = ptx_files[:max_files]
    total_files = len(ptx_files)
    print(f"Found {total_files} PTX files in {input_dir}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for i, ptx_file in enumerate(ptx_files):
            try:
                with open(ptx_file, 'r', encoding='utf-8', errors='ignore') as in_f:
                    content = in_f.read()
                    out_f.write(content)
                    out_f.write('\n')
                if verbose and (i + 1) % 5000 == 0:
                    print(f"Processed {i + 1}/{total_files} files...")  
            except Exception as e:
                print(f"Error reading {ptx_file}: {e}")
                continue
    print(f"Training data written to {output_file}")
    return total_files


def train_tokenizer(training_data_file: Path,output_dir: Path,vocab_size: int = 8000, model_type: str = "bpe",
    character_coverage: float = 1.0, #since PTX is ASCII, we can set it to 1.0 to cover all characters
    num_threads: int =30 , #a machine of 32 cores 
    max_sentence_length: int = 1000, #PTX files can have very long lines, this is the highest i recorded
    shuffle_input_sentence: bool = True,
    seed: int = 42) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = output_dir / "ptx_tokenizer"
    user_symbols = create_user_symbols()
    seen = set()
    unique_symbols = []
    for s in user_symbols:
        if s not in seen:
            seen.add(s)
            unique_symbols.append(s)
    print(f"User-defined symbols: {len(unique_symbols)}")
    
    #create user_defined_symbols file for reference
    symbols_file = output_dir / "user_defined_symbols.txt"
    with open(symbols_file, 'w', encoding='utf-8') as f:
        for symbol in unique_symbols:
            f.write(f"{symbol}\n")
    
    # Training arguments
    train_args = [
        f"--input={training_data_file}",
        f"--model_prefix={model_prefix}",
        f"--vocab_size={vocab_size}",
        f"--model_type={model_type}",
        f"--character_coverage={character_coverage}",
        f"--num_threads={num_threads}",
        f"--max_sentence_length={max_sentence_length}",
        f"--shuffle_input_sentence={str(shuffle_input_sentence).lower()}",
        f"--seed_sentencepiece_size={seed}",
        # Special tokens
        "--pad_id=0","--unk_id=1", "--bos_id=2","--eos_id=3","--pad_piece=<pad>","--unk_piece=<unk>","--bos_piece=<s>","--eos_piece=</s>",
        f"--user_defined_symbols={','.join(unique_symbols)}",
        "--input_sentence_size=100000000",  
        "--train_extremely_large_corpus=true",
        "--split_by_whitespace=true",
        "--split_by_number=false",    
        # Byte fallback for unknown characters
        "--byte_fallback=true", # 
        '--treat_whitespace_as_suffix=false',
        '--allow_whitespace_only_pieces=true',
        '--normalization_rule_name=identity',
        '--remove_extra_whitespaces=false','--add_dummy_prefix=false'
    ]
    
    print(f"Training SentencePiece tokenizer...")
    print(f"  Model type: {model_type}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Output: {model_prefix}")
    spm.SentencePieceTrainer.Train(" ".join(train_args))
    model_file = str(model_prefix) + ".model"
    vocab_file = str(model_prefix) + ".vocab"
    print(f"\nTraining complete!")
    print(f"  Model: {model_file}")
    print(f"  Vocab: {vocab_file}")
    return model_file


def analyze_vocabulary(tokenizer: PTXTokenizer, output_file: Optional[Path] = None) -> None:
    print("\nVOCABULARY ANALYSIS")
    vocab_size = tokenizer.vocab_size
    print(f"Total vocabulary size: {vocab_size}")
    #categorize tokens
    categories = {
        'special': [],      # <pad>, <unk>, etc.
        'normalized': [],   # <KERNEL_xxx>, <PARAM_N>, etc.
        'ptx_keywords': [], # .entry, .func, etc.
        'registers': [],    # %tid.x, etc.
        'operators': [],    # +, -, etc.
        'subwords': [],     # BPE subwords
    }
    for i in range(vocab_size):
        piece = tokenizer.id_to_piece(i)
        if piece.startswith('<') and piece.endswith('>'):
            if piece in ['<pad>', '<unk>', '<s>', '</s>', '<mask>']:
                categories['special'].append((i, piece))
            else:
                categories['normalized'].append((i, piece))
        elif piece.startswith('.'):
            categories['ptx_keywords'].append((i, piece))
        elif piece.startswith('%'):
            categories['registers'].append((i, piece))
        elif piece in ['+', '-', '*', '/', '=', ',', ';', ':', '[', ']', '{', '}', '(', ')']:
            categories['operators'].append((i, piece))
        else:
            categories['subwords'].append((i, piece))
    
    for cat, tokens in categories.items():
        print(f"\n{cat.upper()}: {len(tokens)} tokens")
        if len(tokens) <= 20:
            for id, piece in tokens:
                print(f"  [{id}] {repr(piece)}")
        else:
            for id, piece in tokens[:10]:
                print(f"  [{id}] {repr(piece)}")
            print(f"  ... and {len(tokens) - 10} more")
    
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"PTX Tokenizer Vocabulary Analysis\n")
            f.write(f"=" * 60 + "\n\n")
            f.write(f"Total vocabulary size: {vocab_size}\n\n")
            
            for cat, tokens in categories.items():
                f.write(f"\n{cat.upper()}: {len(tokens)} tokens\n")
                f.write("-" * 40 + "\n")
                for id, piece in tokens:
                    f.write(f"[{id}] {repr(piece)}\n")
        
        print(f"\nVocabulary analysis written to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='PTX Tokenizer Training and Testing')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    train_parser = subparsers.add_parser('train', help='Train tokenizer')
    train_parser.add_argument('--input', '-i', type=str, default='~/processed',
                              help='Input directory with processed PTX files')
    train_parser.add_argument('--output', '-o', type=str, default='~/tokenizer_16k',
                              help='Output directory for tokenizer files')
    train_parser.add_argument('--vocab-size', '-v', type=int, default=16000,
                              help='Vocabulary size (default: 16000)')
    train_parser.add_argument('--model-type', '-m', type=str, default='bpe',
                              choices=['bpe', 'unigram'],
                              help='Model type (default: bpe)')
    train_parser.add_argument('--max-files', type=int, default=None,
                              help='Maximum number of files to use (default: all)')
    train_parser.add_argument('--threads', '-t', type=int, default=8,
                              help='Number of training threads')
    train_parser.add_argument('--verbose', action='store_true',
                              help='Verbose output')
    test_parser = subparsers.add_parser('test', help='Test tokenizer')
    test_parser.add_argument('--model', '-m', type=str, required=True,
                             help='Path to trained model file')
    test_parser.add_argument('--text', '-t', type=str, default=None,
                             help='Text to tokenize (optional)')
    test_parser.add_argument('--analyze', '-a', action='store_true',
                             help='Analyze vocabulary')
    test_parser.add_argument('--save', '-s', type=str, default=None,
                             help='Save test results to file')
    encode_parser = subparsers.add_parser('encode', help='Encode text or file')
    encode_parser.add_argument('--model', '-m', type=str, required=True,
                               help='Path to trained model file')
    encode_parser.add_argument('--text', '-t', type=str, default=None,
                               help='Text to encode')
    encode_parser.add_argument('--file', '-f', type=str, default=None,
                               help='File to encode')
    encode_parser.add_argument('--output', '-o', type=str, default=None,
                               help='Output file for encoded IDs')
    encode_parser.add_argument('--save-pieces', type=str, default=None,
                               help='Save token pieces to file')
    encode_parser.add_argument('--save-detailed', action='store_true',
                               help='Save detailed tokenization (input, pieces, IDs, mapping)')
    
    args = parser.parse_args()
    project_root = Path(__file__).parent.parent.parent
    
    if args.command == 'train':
        input_dir = Path(args.input).expanduser()
        output_dir = Path(args.output).expanduser()
        if not input_dir.is_absolute():
            input_dir = project_root / input_dir
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
        
        print(f"PTX Tokenizer Training")
        print(f"=" * 60)
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Vocabulary size: {args.vocab_size}")
        print(f"Model type: {args.model_type}")
        print(f"=" * 60)
        training_data_file = output_dir / "training_data.txt"
        num_files = prepare_training_data( input_dir,  training_data_file,  max_files=args.max_files,verbose=args.verbose)
        
        if num_files == 0:
            print("ERROR: No PTX files found!")
            return
        
        model_file = train_tokenizer( training_data_file, output_dir, vocab_size=args.vocab_size, model_type=args.model_type, num_threads=args.threads )
        tokenizer = PTXTokenizer(model_file)
        analyze_vocabulary(tokenizer, output_dir / "vocab_analysis.txt")
        
        print(f"\nTraining complete!")
        print(f"  Model: {model_file}")
        print(f"  Files processed: {num_files}")
        
    elif args.command == 'test':
        model_path = Path(args.model)

        if not model_path.is_absolute():
            if model_path.exists():
                model_path = model_path.resolve()
            else:
                model_path = project_root / model_path
        
        if not model_path.exists():
            print(f"ERROR: Model file not found: {model_path}")
            print(f"Searched in:")
            print(f"  - {Path(args.model).resolve()}")
            print(f"  - {project_root / args.model}")
            return
        
        tokenizer = PTXTokenizer(str(model_path))
        save_path = None
        if args.save:
            save_path = Path(args.save)
            if not save_path.is_absolute():
                save_path = save_path.resolve()
        
        if args.analyze:
            analyze_vocabulary(tokenizer)
    
    elif args.command == 'encode':
        model_path = Path(args.model)
        if not model_path.is_absolute():
            if model_path.exists():
                model_path = model_path.resolve()
            else:
                model_path = project_root / model_path
        
        if not model_path.exists():
            print(f"ERROR: Model file not found: {model_path}")
            return
        
        tokenizer = PTXTokenizer(str(model_path))
        
        if args.text:
            ids = tokenizer.encode(args.text)
            pieces = tokenizer.encode_as_pieces(args.text)
            print(f"Text: {args.text}")
            print(f"Pieces ({len(pieces)}): {pieces}")
            print(f"IDs ({len(ids)}): {ids}")
            print("\nPiece -> ID mapping:")
            for i, (piece, id) in enumerate(zip(pieces, ids)):
                print(f"  [{i}] '{piece}' -> {id}")
        
        elif args.file:
            file_path = Path(args.file)
            if not file_path.is_absolute():
                if file_path.exists():
                    file_path = file_path.resolve()
                else:
                    file_path = project_root / file_path
            
            if not file_path.exists():
                print(f"ERROR: Input file not found: {file_path}")
                return
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            ids = tokenizer.encode(content, add_bos=True, add_eos=True)
            pieces = tokenizer.encode_as_pieces(content)
            
            print(f"File: {file_path}")
            print(f"Tokens: {len(ids)}")
            print(f"First 10 pieces: {pieces[:10]}")
            print(f"First 10 IDs: {ids[:10]}")
            if args.output:
                output_path = Path(args.output)
                if not output_path.is_absolute():
                    output_path = output_path.resolve()
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(' '.join(map(str, ids)))
                print(f"\nIDs written to {output_path}")
            if args.save_pieces:
                pieces_path = Path(args.save_pieces)
                if not pieces_path.is_absolute():
                    pieces_path = pieces_path.resolve()
                
                pieces_path.parent.mkdir(parents=True, exist_ok=True)
                with open(pieces_path, 'w', encoding='utf-8') as f:
                    for piece in pieces:
                        f.write(f"{piece}\n")
                print(f"Pieces written to {pieces_path}")
            if args.save_detailed:
                base_path = Path(args.output) if args.output else file_path.parent / (file_path.stem + "_tokenized")
                if not base_path.is_absolute():
                    base_path = base_path.resolve()
                
                detailed_path = base_path.parent / f"{base_path.stem}_detailed.txt"
                detailed_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(detailed_path, 'w', encoding='utf-8') as f:
                    f.write("DETAILED TOKENIZATION REPORT\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"Input file: {file_path}\n")
                    f.write(f"Total tokens: {len(ids)}\n")
                    f.write(f"Total pieces: {len(pieces)}\n")
                    f.write(f"Vocabulary size: {tokenizer.vocab_size}\n\n")
                    
                    f.write("=" * 80 + "\n")
                    f.write("ORIGINAL TEXT\n")
                    f.write("=" * 80 + "\n")
                    f.write(content[:1000]) 
                    if len(content) > 1000:
                        f.write(f"\n... (showing first 1000 of {len(content)} characters)\n")
                    f.write("\n\n")
                    
                    f.write("=" * 80 + "\n")
                    f.write("TOKEN PIECES\n")
                    f.write("=" * 80 + "\n")
                    for i, piece in enumerate(pieces[:100]): 
                        f.write(f"[{i}] {repr(piece)}\n")
                    if len(pieces) > 100:
                        f.write(f"... (showing first 100 of {len(pieces)} pieces)\n")
                    f.write("\n\n")
                    
                    f.write("=" * 80 + "\n")
                    f.write("TOKEN IDs\n")
                    f.write("=" * 80 + "\n")
                    for i in range(0, min(100, len(ids)), 20):
                        chunk = ids[i:i+20]
                        f.write(f"[{i:4d}-{i+len(chunk)-1:4d}] {' '.join(map(str, chunk))}\n")
                    if len(ids) > 100:
                        f.write(f"... (showing first 100 of {len(ids)} IDs)\n")
                    f.write("\n\n")
                    
                    f.write("=" * 80 + "\n")
                    f.write("PIECE -> ID MAPPING (first 100)\n")
                    f.write("=" * 80 + "\n")
                    for i, (piece, id) in enumerate(zip(pieces[:100], ids[:100])):
                        f.write(f"[{i:4d}] '{piece}' -> {id}\n")
                    if len(pieces) > 100:
                        f.write(f"... (showing first 100 of {len(pieces)} mappings)\n")
                
                print(f"Detailed report written to {detailed_path}")
        else:
            print("Specify --text or --file to encode")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

