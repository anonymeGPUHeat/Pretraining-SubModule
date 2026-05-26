import re
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
@dataclass
class Instruction:
    """
    Structured representation of a single PTX instruction
    Preserves all semantic components for tokenization
    i must make sure to represent every possible state
    """
    opcode: str
    address_space: Optional[str] = None
    data_type: Optional[str] = None
    modifiers: List[str] = field(default_factory=list)
    predicate: Optional[str] = None  # e.g., "@%p1"
    operands: List[str] = field(default_factory=list)
    source_line: int = 0  # Line number in original PTX file
    raw_text: str = ""  
    
    def is_memory_op(self) -> bool:
        memory_ops = {'ld', 'st', 'ldu', 'atom', 'red', 'prefetch', 'prefetchu'}
        return self.opcode in memory_ops
    
    def is_predicated(self) -> bool:
        return self.predicate is not None


@dataclass
class Label:
    """Basic block or branch target label"""
    name: str
    source_line: int
    raw_text: str


@dataclass
class Directive:
    """PTX directive (.reg, .param, .global, etc.)"""
    directive_type: str  # e.g., "reg", "param", "global"
    content: str
    source_line: int
    raw_text: str


class PTXParser:
    """
    Parse PTX files into structured instruction sequences
    Design principles:
    - Preserve instruction order exactly
    - Track source line numbers for reversibility
    - Separate semantic components (opcode, address space, type, modifiers)
    - Distinguish predicated from non-predicated instructions
    """
    def __init__(self):
        # PTX instruction pattern
        # Format: [@predicate] opcode[.modifier]* operands;
        self.instruction_pattern = re.compile(
            r'^(?:@(%\w+)\s+)?'  
            r'(\w+(?:\.\w+)*)'  # Opcode with modifiers
            r'\s+'
            r'([^;]+);'  # Operands
        )
        # Label pattern: $L__BB0_3:
        self.label_pattern = re.compile(r'^([\$\w]+):\s*$')
        # Directive pattern: .reg .u32 %r<10>; like headers and params declarations
        self.directive_pattern = re.compile(r'^\s*\.(\w+)\s+(.+);')
        self.address_spaces = {
            'global', 'shared', 'local', 'const', 'param', 'generic'
        }
        self.data_types = {
            's8', 's16', 's32', 's64',
            'u8', 'u16', 'u32', 'u64',
            'f16', 'f32', 'f64',
            'b8', 'b16', 'b32', 'b64',
            'pred'
        }
    
    def parse_file(self, ptx_path: Path) -> List:
        """
        Parse a PTX file into structured elements.
        
        Returns:
            List of parsed elements (Instruction, Label, Directive)
        """
        elements = []
        with open(ptx_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line_num, line in enumerate(lines, start=1):
            line = line.rstrip()
            if not line or line.strip().startswith('//'):
                continue
            label_match = self.label_pattern.match(line.strip())
            if label_match:
                elements.append(Label(name=label_match.group(1),
                    source_line=line_num, raw_text=line))
                continue
            directive_match = self.directive_pattern.match(line)
            if directive_match:
                elements.append(Directive(directive_type=directive_match.group(1),
                    content=directive_match.group(2), source_line=line_num,raw_text=line ))
                continue
            instruction = self._parse_instruction(line, line_num)
            if instruction:
                elements.append(instruction)
        return elements
    
    def _parse_instruction(self, line: str, line_num: int) -> Optional[Instruction]:
        line = line.strip()
        match = self.instruction_pattern.match(line)
        if not match:
            return None
        predicate = match.group(1)  # e.g., "%p1" or None
        opcode_with_modifiers = match.group(2)  # e.g., "ld.global.f32"
        operands_str = match.group(3)  # e.g., "%f12, [%rd27]"
        parts = opcode_with_modifiers.split('.')
        opcode = parts[0]
        modifiers = parts[1:]
        address_space = None
        data_type = None
        other_modifiers = []
        for mod in modifiers:
            if mod in self.address_spaces:
                address_space = mod
            elif mod in self.data_types:
                data_type = mod
            else:
                other_modifiers.append(mod)
        operands = [op.strip() for op in operands_str.split(',')]
        return Instruction(
            opcode=opcode, address_space=address_space,data_type=data_type,
            modifiers=other_modifiers, predicate=f"@{predicate}" if predicate else None,
            operands=operands,source_line=line_num, raw_text=line)
    
    def get_instructions_only(self, elements: List) -> List[Instruction]:
        return [elem for elem in elements if isinstance(elem, Instruction)]
    
    def get_labels(self, elements: List) -> List[Label]:
        return [elem for elem in elements if isinstance(elem, Label)]
    
    def verify_parse(self, elements: List, original_path: Path) -> bool:
        instructions = self.get_instructions_only(elements)
        line_numbers = [inst.source_line for inst in instructions]
        if line_numbers != sorted(line_numbers):
            return False
        for elem in elements:
            if not elem.raw_text:
                return False
        return True


def test_parser():
    parser = PTXParser()
    '''test_instructions = [
        "ld.global.u32  %r1, [%rd24];",
        "@%p1 ld.global.u32 %r1, [%rd24];",
        "ld.global.f32 %f12, [%rd27];",
        "fma.rn.f32 %f29, %f13, %f12, %f29;",
        "add.s32 %r16, %r1, 1;",
        "setp.eq.s32 %p3, %r16, 100;",
        "@%p3 bra $L__BB0_3;",
        "atom.shared.add.u32 	%r2300, [%r846], %r180;"
    ]
    print("Testing PTX Parser:")
    for inst_str in test_instructions:
        inst = parser._parse_instruction(inst_str, line_num=1)
        if inst:
            print(f"\nInput: {inst_str}")
            print(f"  Opcode: {inst.opcode}")
            print(f"  Address space: {inst.address_space}")
            print(f"  Data type: {inst.data_type}")
            print(f"  Modifiers: {inst.modifiers}")
            print(f"  Predicate: {inst.predicate}")
            print(f"  Operands: {inst.operands}")
            print(f"  Is memory op: {inst.is_memory_op()}")
            print(f"  Is predicated: {inst.is_predicated()}")'''
    sample_ptx_path = Path("./data/raw/00002_stride_loop_matmul_base.ptx")
    elements = parser.parse_file(sample_ptx_path)
    print(f"\nParsed {len(elements)} elements from {sample_ptx_path}")
    for elem in elements[10:]: 
        print(f"{elem.source_line}: {elem.raw_text}")
        inst = parser._parse_instruction(elem.raw_text, line_num=elem.source_line)
        if inst:
            print(f"\nInput: {elem.raw_text}")
            print(f"  Opcode: {inst.opcode}")
            print(f"  Address space: {inst.address_space}")
            print(f"  Data type: {inst.data_type}")
            print(f"  Modifiers: {inst.modifiers}")
            print(f"  Predicate: {inst.predicate}")
            print(f"  Operands: {inst.operands}")
            print(f"  Is memory op: {inst.is_memory_op()}")
            print(f"  Is predicated: {inst.is_predicated()}")
        print("-----\n\n")


    

if __name__ == "__main__":
    test_parser()
