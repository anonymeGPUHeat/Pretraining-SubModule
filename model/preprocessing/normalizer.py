import re
from pathlib import Path
from typing import Dict, Tuple, List
from dataclasses import dataclass, field
import argparse


@dataclass
class NormalizationContext:
    param_map: Dict[str, str] = field(default_factory=dict)
    func_map: Dict[str, str] = field(default_factory=dict)
    kernel_map: Dict[str, str] = field(default_factory=dict)
    bb_map: Dict[str, str] = field(default_factory=dict)
    vtable_map: Dict[str, str] = field(default_factory=dict)
    str_map: Dict[str, str] = field(default_factory=dict)
    table_map: Dict[str, str] = field(default_factory=dict)
    param_counter: int = 0
    bb_counter: int = 0
    vtable_counter: int = 0
    str_counter: int = 0
    table_counter: int = 0


class PTXNormalizer:
    REGISTER_PATTERNS = {
        'FP64_REG': re.compile(r'%fd(\d+)'),   
        'FP32_REG': re.compile(r'%f(\d+)'),
        'INT64_REG': re.compile(r'%rd(\d+)'),   
        'INT32_REG': re.compile(r'%r(\d+)'),        
        'INT16_REG': re.compile(r'%rs(\d+)'),   
    }

    REGISTER_TOKENS = {
        'FP32_REG': '<FP32_REG>',
        'FP64_REG': '<FP64_REG>',
        'INT32_REG': '<INT32_REG>',
        'INT64_REG': '<INT64_REG>',
        'INT16_REG': '<INT16_REG>',
    }

    SPECIAL_REGISTERS = {
        "%tid", "%ntid", "%laneid", "%warpid", "%nwarpid", "%ctaid", "%nctaid",
        "%smid", "%nsmid", "%gridid", "%is_explicit_cluster", "%clusterid",
        "%nclusterid", "%cluster_ctaid", "%cluster_nctaid", "%cluster_ctarank", "%cluster_nctarank",
        "%lanemask_eq", "%lanemask_le", "%lanemask_lt", "%lanemask_ge", "%lanemask_gt",
        "%clock", "%clock_hi", "%clock64",
        "%pm0", "%pm1", "%pm2", "%pm3", "%pm4", "%pm5", "%pm6", "%pm7",
        "%pm0_64", "%pm1_64", "%pm2_64", "%pm3_64", "%pm4_64", "%pm5_64", "%pm6_64", "%pm7_64",
        *[f"%envreg{i}" for i in range(32)],
        "%globaltimer", "%globaltimer_lo", "%globaltimer_hi",
        "%reserved_smem_offset_begin", "%reserved_smem_offset_end", "%reserved_smem_offset_cap",
        "%reserved_smem_offset0", "%reserved_smem_offset1",
        "%total_smem_size", "%aggr_smem_size", "%dynamic_smem_size", "%current_graph_exec",
    }
    REG_DECL_PATTERN = re.compile(r'\.reg\s+\.\w+\s+%\w+<\d+>;')
    INTERNAL_GLOBAL_PATTERN = re.compile(r'^\.global\s+.*_INTERNAL_.*$', re.MULTILINE)
    MANGLED_NAME_PATTERN = re.compile(r'_Z[A-Za-z0-9_]+')
    #pattern for basic block labels: $L__BB0_2, $L__BB1_15, etc.
    #captures the inner number after the underscore (group 1)
    #include optional colon to avoid stray digits
    BB_LABEL_PATTERN = re.compile(r'\$L__BB\d+_(\d+)(:)?')
    #pattern to match parameter names in declarations
    PARAM_DECL_PATTERN = re.compile(r'\.param\s+\.\w+\s+([A-Za-z_][A-Za-z0-9_]*_param_(\d+))')
    #pattern to match hex float constants (IEEE 754)
    HEX_FLOAT_PATTERN = re.compile(r'0[fFdD][0-9A-Fa-f]+')
    VTABLE_PATTERN = re.compile(r'(_ZTV[A-Za-z0-9_]+)')
    STRING_PATTERN = re.compile(r'(\$str(?:\$?(\d+))?)')
    TABLE_DECL_PATTERN = re.compile(r'\.(global|const)\s+\.align\s+\d+\s+\.\w+\s+([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]')
    ARRAY_INIT_PATTERN = re.compile(r'(\.(global|const)\s+\.align\s+\d+\s+\.\w+\s+[A-Za-z_$][A-Za-z0-9_$]*\[\d+\])\s*=\s*\{[^}]*\}\s*;')
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        escaped = [re.escape(s) for s in sorted(self.SPECIAL_REGISTERS, key=len, reverse=True)]
        combined_pattern = '|'.join(escaped)
        self._special_reg_pattern = re.compile(rf'({combined_pattern})(?:\.[xyzw])?')
    
    def demangle_name(self, mangled: str) -> str:
        if not mangled.startswith('_Z'):
            return mangled
            
        name = mangled[2:]  
        if name.startswith('N'):
            return self._demangle_nested(name[1:])
        if name.startswith('NK'):
            return self._demangle_nested(name[2:])
        return self._extract_simple_name(name)
    
    def _demangle_nested(self, name: str) -> str:
        parts = []
        i = 0
        while i < len(name):
            length_match = re.match(r'(\d+)', name[i:])
            if length_match:
                length = int(length_match.group(1))
                i += len(length_match.group(1))
                if i + length <= len(name):
                    part = name[i:i+length]
                    if not part.startswith('_INTERNAL'):
                        parts.append(part)
                    i += length
                else:
                    break
            elif name[i] == 'E': 
                break
            else:
                break
        
        return '_'.join(parts) if parts else name
    
    def _extract_simple_name(self, name: str) -> str:
        """Extract name from simple mangled format: length + name + encoding."""
        match = re.match(r'(\d+)', name)
        if match:
            length = int(match.group(1))
            start = len(match.group(1))
            if start + length <= len(name):
                return name[start:start+length]
        return name
    
    def normalize_kernel_name(self, mangled: str) -> str:
        name = self.demangle_name(mangled)
        return f'<KERNEL_{name}>'
    
    def normalize_func_name(self, mangled: str) -> str:
        name = self.demangle_name(mangled)
        return f'<FUNC_{name}>'
    
    def bucket_immediate(self, value: int) -> str:
        """
        Bucket large immediate values.
        Values between -5000 and 5000 are kept as-is.
        Larger values are bucketed.
        """
        abs_val = abs(value)
        if abs_val <= 5000:
            return str(value)
        elif abs_val < 6000:
            return '<IMM_6K>' if value > 0 else '<IMM_NEG_6K>'
        elif abs_val < 10000:
            return '<IMM_10K>' if value > 0 else '<IMM_NEG_10K>'
        elif abs_val < 100000:
            return '<IMM_100K>' if value > 0 else '<IMM_NEG_100K>'
        elif abs_val < 1000000:
            return '<IMM_1M>' if value > 0 else '<IMM_NEG_1M>'
        else:
            return '<IMM_LARGE>' if value > 0 else '<IMM_NEG_LARGE>'
    
    def normalize_file(self, content: str) -> str:
        ctx = NormalizationContext()
        #1.remove INTERNAL globals
        content = self._remove_internal_globals(content)
        #2. normalize globals (VTables, strings, large tables)
        content = self._normalize_globals(content, ctx)
        #3. collect and normalize kernel/function declarations
        content = self._normalize_entry_and_func_names(content, ctx)
        #4. normalize parameters
        content = self._normalize_parameters(content, ctx)
        #5. normalize registers (in function bodies only)
        #content = self._normalize_registers(content)
        #6. bucket immediates
        content = self._bucket_immediates(content)
        #7. normalize basic block labels
        content = self._normalize_bb_labels(content, ctx)
        return content
    
    def _remove_internal_globals(self, content: str) -> str:
        """Remove .global declarations containing _INTERNAL_."""
        lines = content.split('\n')
        filtered = [line for line in lines if '_INTERNAL_' not in line or not line.strip().startswith('.global')]
        return '\n'.join(filtered)
    
    def _normalize_globals(self, content: str, ctx: NormalizationContext) -> str:
        content = self.ARRAY_INIT_PATTERN.sub(r'\1;', content)
        
        lines = content.split('\n')
        result_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('.global') or stripped.startswith('.const'):
                vtable_match = self.VTABLE_PATTERN.search(line)
                if vtable_match:
                    vtable_name = vtable_match.group(1)
                    if vtable_name not in ctx.vtable_map:
                        ctx.vtable_map[vtable_name] = f'<VTABLE_{ctx.vtable_counter}>'
                        ctx.vtable_counter += 1
                    line = line.replace(vtable_name, ctx.vtable_map[vtable_name])
                str_match = self.STRING_PATTERN.search(line)
                if str_match:
                    str_name = str_match.group(1)  
                    str_num = str_match.group(2)  
                    if str_name not in ctx.str_map:
                        if str_num:
                            ctx.str_map[str_name] = f'<STR_{str_num}>'
                        else:
                            ctx.str_map[str_name] = f'<STR_{ctx.str_counter}>'
                            ctx.str_counter += 1
                    line = line.replace(str_name, ctx.str_map[str_name])
                table_match = self.TABLE_DECL_PATTERN.match(stripped)
                if table_match:
                    table_name = table_match.group(2)  
                    table_size = int(table_match.group(3))
                    if table_size > 1000 and table_name not in ctx.vtable_map and table_name not in ctx.str_map:
                        if table_name not in ctx.table_map:
                            ctx.table_map[table_name] = f'<TABLE_{ctx.table_counter}>'
                            ctx.table_counter += 1
                        line = line.replace(table_name, ctx.table_map[table_name])
            
            result_lines.append(line)
        
        content = '\n'.join(result_lines)
        for original, normalized in sorted(ctx.vtable_map.items(), key=lambda x: -len(x[0])):
            content = content.replace(original, normalized)
        for original, normalized in sorted(ctx.str_map.items(), key=lambda x: -len(x[0])):
            content = content.replace(original, normalized)
        for original, normalized in sorted(ctx.table_map.items(), key=lambda x: -len(x[0])):
            content = content.replace(original, normalized)
        
        return content
    
    def _normalize_entry_and_func_names(self, content: str, ctx: NormalizationContext) -> str:
        entry_pattern = re.compile(r'\.visible\s+\.entry\s+(_Z[A-Za-z0-9_]+)')
        for match in entry_pattern.finditer(content):
            mangled = match.group(1)
            if mangled not in ctx.kernel_map:
                ctx.kernel_map[mangled] = self.normalize_kernel_name(mangled)
        func_pattern = re.compile(r'\.func\s+(?:\([^)]*\)\s+)?(_Z[A-Za-z0-9_]+)')
        for match in func_pattern.finditer(content):
            mangled = match.group(1)
            if mangled not in ctx.func_map:
                ctx.func_map[mangled] = self.normalize_func_name(mangled)
        for mangled, normalized in ctx.kernel_map.items():
            content = content.replace(mangled, normalized)
        for mangled, normalized in ctx.func_map.items():
            content = content.replace(mangled, normalized)
        
        return content
    
    def _normalize_parameters(self, content: str, ctx: NormalizationContext) -> str:
        param_pattern = re.compile(r'([A-Za-z_<>][A-Za-z0-9_<>]*_param_\d+)')
        
        for match in param_pattern.finditer(content):
            full_param = match.group(1)
            if full_param not in ctx.param_map:
                ctx.param_map[full_param] = f'<PARAM_{ctx.param_counter}>'
                ctx.param_counter += 1
        norm_param_pattern = re.compile(r'(<[A-Z]+_[A-Za-z0-9_]+>_param_\d+)')
        for match in norm_param_pattern.finditer(content):
            full_param = match.group(1)
            if full_param not in ctx.param_map:
                ctx.param_map[full_param] = f'<PARAM_{ctx.param_counter}>'
                ctx.param_counter += 1
        for full_param, normalized in sorted(ctx.param_map.items(), key=lambda x: -len(x[0])):
            content = content.replace(full_param, normalized)
        
        return content
    
    def _normalize_registers(self, content: str) -> str:
        lines = content.split('\n')
        result_lines = []
        
        for line in lines:
            if self.REG_DECL_PATTERN.search(line):
                result_lines.append(line)
                continue
            
            normalized_line = self._normalize_line_registers(line)
            result_lines.append(normalized_line)
        
        return '\n'.join(result_lines)
    
    def _normalize_line_registers(self, line: str) -> str:
        protected = {}
        def protect_special(match):
            key = f'__SPECIAL_{len(protected)}__'
            protected[key] = match.group(0)
            return key
        line = self._special_reg_pattern.sub(protect_special, line)
        line = re.sub(r'@!%p\d+', '@!<PRED_REG>', line)
        line = re.sub(r'@%p\d+', '@<PRED_REG>', line)
        line = re.sub(r'%p\d+', '<PRED_REG>', line)
        #normalize each register type (order matters: longer patterns first)
        # %fd before %f, %rd before %r
        for reg_type in ['FP64_REG', 'INT64_REG', 'FP32_REG', 'INT32_REG', 'INT16_REG']:
            pattern = self.REGISTER_PATTERNS[reg_type]
            token = self.REGISTER_TOKENS[reg_type]
            line = pattern.sub(token, line)
        for placeholder, original in protected.items():
            line = line.replace(placeholder, original)
        return line
    
    def _bucket_immediates(self, content: str) -> str:
        lines = content.split('\n')
        result_lines = []
        
        for line in lines:
            if line.strip().startswith('//'):
                result_lines.append(line)
                continue
            
            result_line = self._bucket_line_immediates(line)
            result_lines.append(result_line)
        
        return '\n'.join(result_lines)
    
    def _bucket_line_immediates(self, line: str) -> str:
        hex_floats: List[str] = []
        
        def save_hex(match):
            hex_floats.append(match.group(0))
            return f'__HEX_{len(hex_floats)-1}__'
        
        line = self.HEX_FLOAT_PATTERN.sub(save_hex, line)
        hex_ints: List[str] = []
        
        def save_hex_int(match):
            hex_ints.append(match.group(0))
            return f'__HEXINT_{len(hex_ints)-1}__'
        
        line = re.sub(r'0[xX][0-9A-Fa-f]+', save_hex_int, line)
        def replace_int(match):
            try:
                val = int(match.group(0))
                return self.bucket_immediate(val)
            except ValueError:
                return match.group(0)
        
        # Match integers with proper boundaries:
        # - After: whitespace, comma, [, (, +, -, or start of line
        # - Before: whitespace, comma, ], ), ;, :, or end of line
        # Negative numbers: must be preceded by appropriate context
        #pattern = r'(?<=[\s,\[\(+])-?\d+(?=[\s,\]\);:])|(?<=[\s,\[\(+])-?\d+$|^-?\d+(?=[\s,\]\);:])'
        pattern = r'(?<=[\s,\[\(+-])-?\d+(?=[\s,\]\);:])|(?<=[\s,\[\(+-])-?\d+$|^-?\d+(?=[\s,\]\);:])|^-?\d+$'
        line = re.sub(pattern, replace_int, line)
        for i, hi in enumerate(hex_ints):
            line = line.replace(f'__HEXINT_{i}__', hi)
        for i, hf in enumerate(hex_floats):
            line = line.replace(f'__HEX_{i}__', hf)
        
        return line
    
    def _normalize_bb_labels(self, content: str, ctx: NormalizationContext) -> str:
        for match in self.BB_LABEL_PATTERN.finditer(content):
            full_label = match.group(0)  # e.g., "$L__BB0_9" or "$L__BB0_9:"
            inner_num = match.group(1)   # e.g., "9"
            has_colon = match.group(2)   # ":" or None
            
            if full_label not in ctx.bb_map:
                if has_colon:
                    ctx.bb_map[full_label] = f'<BB_{inner_num}>:'
                else:
                    ctx.bb_map[full_label] = f'<BB_{inner_num}>'
        for original, normalized in sorted(ctx.bb_map.items(), key=lambda x: -len(x[0])):
            content = content.replace(original, normalized)
        
        return content


def normalize_directory(input_dir: Path, output_dir: Path, verbose: bool = False) -> Tuple[int, int]:
    normalizer = PTXNormalizer(verbose=verbose)
    output_dir.mkdir(parents=True, exist_ok=True) 
    ptx_files = list(input_dir.glob('*.ptx'))
    success_count = 0
    error_count = 0
    for ptx_file in ptx_files:
        try:
            with open(ptx_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            normalized = normalizer.normalize_file(content)
            output_file = output_dir / ptx_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(normalized)
            success_count += 1
            if verbose:
                print(f"Normalized: {ptx_file.name}")
        except Exception as e:
            error_count += 1
            print(f"Error processing {ptx_file.name}: {e}")
    return success_count, error_count


def main():
    parser = argparse.ArgumentParser(description='Normalize PTX files for pre-training')
    parser.add_argument('--input', '-i', type=str, default='data/cleaned',help='Input directory containing PTX files')
    parser.add_argument('--output', '-o', type=str, default='data/processed',help='Output directory for normalized files')
    parser.add_argument('--verbose', '-v', action='store_true',help='Print progress for each file')
    parser.add_argument('--single', '-s', type=str, default=None,help='Process a single file (for testing)')
    args = parser.parse_args()
    project_root = Path(__file__).parent.parent.parent
    
    if args.single:
        normalizer = PTXNormalizer(verbose=True)
        input_path = Path(args.single)
        if not input_path.is_absolute():
            input_path = project_root / input_path
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        normalized = normalizer.normalize_file(content)
        print(normalized)
    else:
        input_dir = Path(args.input)
        output_dir = Path(args.output)
        if not input_dir.is_absolute():
            input_dir = project_root / input_dir
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        success, errors = normalize_directory(input_dir, output_dir, args.verbose)
        print(f"\nCompleted: {success} files normalized, {errors} errors")


if __name__ == '__main__':
    main()
