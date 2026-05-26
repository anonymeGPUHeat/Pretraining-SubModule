import re
from typing import List, Dict, Tuple


ITC_CLASSES = {
    'GLOBAL_LOAD':     0,
    'GLOBAL_STORE':    1,
    'SHARED_LOAD':     2,
    'SHARED_STORE':    3,
    'LOCAL_LOAD':      4,
    'LOCAL_STORE':     5,
    'ARITHMETIC':      6,
    'LOGIC_BITWISE':   7,
    'COMPARISON':      8,
    'CONTROL_FLOW':    9,
    'SYNC_BARRIER':   10,
    'CONVERSION_MOVE':11,
    'ATOMIC_REDUCE':  12,
    'SPECIAL':        13,
}

NUM_ITC_CLASSES = len(ITC_CLASSES)   
ITC_IGNORE = -100

ITC_ID2NAME = {v: k for k, v in ITC_CLASSES.items()}



#built from the complete PTX ISA opcode table provided in the PTX spec.
# each opcode is mapped to exactly one ITC class.

# memory opcodes need context (address space modifier) to classify
_MEMORY_OPCODES = {'ld', 'st', 'ldu', 'ldmatrix', 'stmatrix'}

_OPCODE_TO_ITC: Dict[str, int] = {}

#arithmetic
for op in [
    'add', 'addc', 'sub', 'subc', 'mul', 'mul24', 'mad', 'mad24', 'madc',
    'fma', 'div', 'rem', 'abs', 'neg', 'min', 'max',
    'rcp', 'sqrt', 'rsqrt', 'sin', 'cos', 'lg2', 'ex2', 'tanh',
    'sad', 'dp2a', 'dp4a',
    # vector arithmetic
    'vadd', 'vadd2', 'vadd4', 'vsub', 'vsub2', 'vsub4',
    'vmad', 'vmax', 'vmax2', 'vmax4', 'vmin', 'vmin2', 'vmin4',
    'vavrg2', 'vavrg4', 'vabsdiff', 'vabsdiff2', 'vabsdiff4',
    'vset', 'vset2', 'vset4', 'vshl', 'vshr',
    'multimem',
]:
    _OPCODE_TO_ITC[op] = ITC_CLASSES['ARITHMETIC']

#Logic / Bitwise
for op in [
    'and', 'or', 'xor', 'not', 'cnot',
    'shl', 'shr', 'shf',
    'bfe', 'bfi', 'bfind', 'brev', 'bmsk',
    'popc', 'clz', 'lop3',
    'szext',
    'copysign',
]:
    _OPCODE_TO_ITC[op] = ITC_CLASSES['LOGIC_BITWISE']

#Comparison / Predicate 
for op in [
    'setp', 'set', 'selp', 'slct',
    'vote', 'match',
    'testp', 'isspacep', 'istypep',
]:
    _OPCODE_TO_ITC[op] = ITC_CLASSES['COMPARISON']

# control flow 
for op in [
    'bra', 'brx', 'ret', 'exit', 'call',
    'brkpt', 'trap',
    'alloca',
    'applypriority',
    'discard',
    'nanosleep',
    'pmevent',
    'setmaxnreg',
    'stackrestore', 'stacksave',
    'griddepcontrol',
    'clusterlaunchcontrol',
    'elect',
    'getctarank',
]:
    _OPCODE_TO_ITC[op] = ITC_CLASSES['CONTROL_FLOW']

# sync / barrier 
for op in [
    'bar', 'barrier',
    'membar', 'fence',
    'mbarrier',
    'activemask',
]:
    _OPCODE_TO_ITC[op] = ITC_CLASSES['SYNC_BARRIER']

# conversion / Data Movement 
for op in [
    'cvt', 'cvta', 'mov', 'movmatrix',
    'shfl', 'prmt',
    'redux',
    'fns',
    'mapa',
    'createpolicy',
]:
    _OPCODE_TO_ITC[op] = ITC_CLASSES['CONVERSION_MOVE']

#Atomic / Reduction
for op in [
    'atom', 'red',
    'sured',
]:
    _OPCODE_TO_ITC[op] = ITC_CLASSES['ATOMIC_REDUCE']

# Special (texture, surface, tensor core, async copy, etc.) 
for op in [
    'tex', 'tld4', 'txq',
    'suld', 'sust', 'suq',
    'wmma', 'mma', 'wgmma',
    'cp',
    'tcgen05',
    'tensormap',
    'prefetch', 'prefetchu',
    'ldmatrix', 'stmatrix',
]:
    _OPCODE_TO_ITC[op] = ITC_CLASSES['SPECIAL']


#Address-space patterns for memory instructions

_GLOBAL_SPACE = {'.global'}
_SHARED_SPACE = {'.shared'}
_LOCAL_SPACE  = {'.local', '.param', '.const'}


# Directive patterns (these tokens get IGNORE)

_DIRECTIVE_PREFIXES = (
    '.version', '.target', '.address_size',
    '.visible', '.entry', '.func', '.extern', '.weak',
    '.reg', '.align', '.maxnreg', '.reqntid', '.maxntid', '.minnctapersm',
    '.reqnctapercluster',
    '.global .align',  # global variable declarations
    '.const', '.shared', '.local',  # space declarations (not instructions)
    '.param',  # parameter declarations
    '.loc',    # debug info
    '.file',   # debug info
    '.section',
    '.b8', '.b16', '.b32', '.b64',  # type declarations in directives
)

_PREDICATE_PATTERN = re.compile(r'^@!?%p\d+$')


def _extract_opcode_from_text(text: str) -> str:
    """
    Extract the base opcode from instruction text
    Given 'ld.global.ca.f32', returns 'ld'
    Given '@%p0 bra', returns 'bra'
    """
    text = text.strip()
    if text.startswith('@'):
        parts = text.split(None, 1)
        if len(parts) > 1:
            text = parts[1]
        else:
            return ''
    first_word = text.split(None, 1)[0] if text else ''
    first_word = first_word.rstrip(';')
    base = first_word.split('.')[0]
    return base


def _get_memory_address_space(instruction_text: str) -> str:
    #check for address space modifiers after the opcode
    for space in ['.global', '.shared', '.local', '.param', '.const']:
        if space in instruction_text:
            return space
    return ''


def classify_instruction(instruction_text: str) -> int:
    """
    Classify a single PTX instruction statement into an ITC class.
    Args:
        instruction_text: Full instruction text (e.g. 'ld.global.ca.f32 %f0, [%rd0];')
    
    Returns:
        ITC class index (0-13) or ITC_IGNORE (-100) for non-instructions
    """
    text = instruction_text.strip()
    if not text or text.startswith('//'):
        return ITC_IGNORE
    for prefix in _DIRECTIVE_PREFIXES:
        if text.startswith(prefix):
            return ITC_IGNORE
    if text.endswith(':') and not text.startswith(('@', ' ', '\t')):
        return ITC_IGNORE
    if text.startswith('.reg'):
        return ITC_IGNORE
    if text in ('{', '}', '(', ')'):
        return ITC_IGNORE
    opcode = _extract_opcode_from_text(text)
    if not opcode:
        return ITC_IGNORE
    if opcode in ('ld', 'ldu'):
        space = _get_memory_address_space(text)
        if space in _GLOBAL_SPACE:
            return ITC_CLASSES['GLOBAL_LOAD']
        elif space in _SHARED_SPACE:
            return ITC_CLASSES['SHARED_LOAD']
        elif space in _LOCAL_SPACE:
            return ITC_CLASSES['LOCAL_LOAD']
        else:
            return ITC_CLASSES['GLOBAL_LOAD']
    
    if opcode == 'st':
        space = _get_memory_address_space(text)
        if space in _GLOBAL_SPACE:
            return ITC_CLASSES['GLOBAL_STORE']
        elif space in _SHARED_SPACE:
            return ITC_CLASSES['SHARED_STORE']
        elif space in _LOCAL_SPACE:
            return ITC_CLASSES['LOCAL_STORE']
        else:
            return ITC_CLASSES['GLOBAL_STORE']

    if opcode in _OPCODE_TO_ITC:
        return _OPCODE_TO_ITC[opcode]
    if text.startswith('<') and text.endswith('>'):
        return ITC_IGNORE
    return ITC_IGNORE


def generate_itc_labels_for_text(text: str,tokenizer,token_ids: List[int],) -> List[int]:
    """
    Generate per-token ITC labels for a tokenized PTX text.
    Strategy:
      1. Split the raw text into instruction statements (semicolon-delimited lines).
      2. Classify each statement's opcode.
      3. Map token positions back to statements using character offsets.
      4. Each token inherits the ITC label of the statement it belongs to.
    This is the main entry point used by the dataset builder
    Args:
        text:      Raw PTX text (same text that was tokenized)
        tokenizer: PTXTokenizer instance (has sp.EncodeAsPieces)
        token_ids: Token ID list from tokenizer.encode(text)
    Returns:
        List[int] of length len(token_ids), with values in [0..13] or -100.
    """
    pieces = tokenizer.sp.EncodeAsPieces(text)
    assert len(pieces) == len(token_ids), \
        f"Piece/ID mismatch: {len(pieces)} vs {len(token_ids)}"
    lines = text.split('\n')
    statements: List[Tuple[int, int, int]] = []
    char_pos = 0
    for line in lines:
        line_start = char_pos
        line_end = char_pos + len(line)
        stripped = line.strip()
        
        if stripped:
            itc_class = classify_instruction(stripped)
            statements.append((line_start, line_end, itc_class))
        else:
            statements.append((line_start, line_end, ITC_IGNORE))
        
        char_pos = line_end + 1  # +1 for the '\n'
    labels = [ITC_IGNORE] * len(token_ids)
    current_char = 0
    token_char_starts = []
    
    for piece in pieces:
        token_char_starts.append(current_char)
        clean = piece.replace('▁', ' ')
        current_char += len(clean)
    
    for tok_idx, char_start in enumerate(token_char_starts):
        piece = pieces[tok_idx]
        if piece in ('<pad>', '<unk>', '<s>', '</s>', '<mask>', '<cls>', '<sep>'):
            labels[tok_idx] = ITC_IGNORE
            continue
        for stmt_start, stmt_end, itc_class in statements:
            if stmt_start <= char_start < stmt_end:
                labels[tok_idx] = itc_class
                break
    
    return labels


#Shared constants 
_SPECIAL_TOKENS = frozenset({
    '<pad>', '<unk>', '<s>', '</s>', '<mask>', '<cls>', '<sep>',
})


def _is_structural_token(piece: str) -> bool:
    return piece.startswith('<') 


def _clean_piece(piece: str) -> str:
    return piece.replace('\u2581', '').strip()




def detect_instruction_spans(token_ids: List[int], tokenizer,) -> Tuple[List[int], List[Tuple[int, int, bool]]]:
    """
    Returns:
        labels  – ``List[int]``  of length ``len(token_ids)``.
                  Values in ``[0..13]`` for instruction tokens, ``-100`` otherwise.
        spans   – ``List[Tuple[int, int, bool]]``  where each element is
                  ``(start_idx, end_idx_exclusive, is_real_instruction)``.
                  ``is_real_instruction`` is True when the span contains a
                  recognized PTX opcode (ITC class ≠ -100).
    """
    pieces = [tokenizer.sp.IdToPiece(tid) for tid in token_ids]
    n = len(pieces)
    labels = [ITC_IGNORE] * n
    spans: List[Tuple[int, int, bool]] = []

    inst_start = 0
    current_opcode: str | None = None
    current_class: int = ITC_IGNORE

    def _finalise_span(end_exclusive: int) -> None:
        is_instruction = current_class != ITC_IGNORE
        if is_instruction:
            for j in range(inst_start, end_exclusive):
                p = pieces[j]
                if p in _SPECIAL_TOKENS or _is_structural_token(p):
                    continue
                labels[j] = current_class
        spans.append((inst_start, end_exclusive, is_instruction))

    for i, piece in enumerate(pieces):
        clean = _clean_piece(piece)
        if piece in _SPECIAL_TOKENS or _is_structural_token(piece):
            continue
        if clean in ('{', '}', '(', ')'):
            continue
        if clean.endswith(';') or clean == ';':
            _finalise_span(i + 1)
            inst_start = i + 1
            current_opcode = None
            current_class = ITC_IGNORE
            continue
        if clean.startswith('@'):
            continue
        if (current_opcode is None and clean and clean[0].isalpha() and not clean.startswith('.') and not clean.startswith('%')):
            base_opcode = clean.split('.')[0]
            current_opcode = base_opcode
            if base_opcode in ('ld', 'ldu'):
                current_class = ITC_CLASSES['GLOBAL_LOAD']          
                for mod in clean.split('.')[1:]:
                    if mod == 'global':
                        current_class = ITC_CLASSES['GLOBAL_LOAD'];   break
                    elif mod == 'shared':
                        current_class = ITC_CLASSES['SHARED_LOAD'];   break
                    elif mod in ('local', 'param', 'const'):
                        current_class = ITC_CLASSES['LOCAL_LOAD'];    break
            elif base_opcode == 'st':
                current_class = ITC_CLASSES['GLOBAL_STORE']      
                for mod in clean.split('.')[1:]:
                    if mod == 'global':
                        current_class = ITC_CLASSES['GLOBAL_STORE'];  break
                    elif mod == 'shared':
                        current_class = ITC_CLASSES['SHARED_STORE'];  break
                    elif mod in ('local', 'param', 'const'):
                        current_class = ITC_CLASSES['LOCAL_STORE'];   break

            elif base_opcode in _OPCODE_TO_ITC:
                current_class = _OPCODE_TO_ITC[base_opcode]
            else:
                current_class = ITC_IGNORE
            continue

        if current_opcode in ('ld', 'ldu', 'st') and clean.startswith('.'):
            mod = clean.rstrip(',;')
            if mod == '.global':
                current_class = (ITC_CLASSES['GLOBAL_LOAD']
                                 if current_opcode in ('ld', 'ldu')
                                 else ITC_CLASSES['GLOBAL_STORE'])
            elif mod == '.shared':
                current_class = (ITC_CLASSES['SHARED_LOAD']
                                 if current_opcode in ('ld', 'ldu')
                                 else ITC_CLASSES['SHARED_STORE'])
            elif mod in ('.local', '.param', '.const'):
                current_class = (ITC_CLASSES['LOCAL_LOAD']
                                 if current_opcode in ('ld', 'ldu')
                                 else ITC_CLASSES['LOCAL_STORE'])

    if inst_start < n:
        _finalise_span(n)

    return labels, spans


def generate_itc_labels_fast( token_ids: List[int],tokenizer,) -> List[int]:
    labels, _spans = detect_instruction_spans(token_ids, tokenizer)
    return labels




"""
Instruction Type Classification (ITC) Label Generator for PTX tokens.

Assigns per-token instruction type labels based on the PTX opcode that starts
each instruction statement. Every token in an instruction inherits the label 
of its opcode. Non-instruction tokens (directives, labels, padding) get the
IGNORE label (-100) so they are excluded from loss computation.

Classes (13 + IGNORE):
    0  GLOBAL_LOAD       - ld.global, ldu (global memory reads)
    1  GLOBAL_STORE       - st.global (global memory writes)
    2  SHARED_LOAD        - ld.shared (shared memory reads) 
    3  SHARED_STORE       - st.shared (shared memory writes)
    4  LOCAL_LOAD         - ld.local, ld.param (local/param reads)
    5  LOCAL_STORE        - st.local (local memory writes)
    6  ARITHMETIC         - add, sub, mul, mad, fma, div, rem, neg, abs, min, max, etc.
    7  LOGIC_BITWISE      - and, or, xor, not, shl, shr, bfe, bfi, popc, clz, etc.
    8  COMPARISON         - setp, set, selp, slct, vote
    9  CONTROL_FLOW       - bra, ret, exit, call, brx
    10 SYNC_BARRIER       - bar, barrier, membar, fence, bar.sync
    11 CONVERSION_MOVE    - cvt, mov, shfl, prmt
    12 ATOMIC_REDUCE      - atom, red
    13 SPECIAL            - tex, suld, sust, wmma, mma, cp, prefetch, etc.

The label is assigned at the *token level*: all tokens belonging to the same
instruction statement share the same ITC class.  Tokens in directives (.version,
.target, .reg, .param declarations, .visible, .entry, etc.) and structural
tokens (<BB_*, <KERNEL_*, labels) receive -100 (IGNORE).
"""