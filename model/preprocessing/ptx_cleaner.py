import re
from pathlib import Path
from typing import List, Tuple
import logging

#configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PTXCleaner: 
    def __init__(self, input_dir: str, output_dir: str = "cleaned"):
        self.input_dir = Path(input_dir)
        self.output_dir = self.input_dir.parent / output_dir
        self.multi_kernel_files = []
        
    def remove_comments(self, lines: List[str]) -> List[str]:
        cleaned_lines = []
        in_block_comment = False
        
        for line in lines:
            if '/*' in line:
                in_block_comment = True
                line = line[:line.index('/*')]
            if in_block_comment:
                if '*/' in line:
                    in_block_comment = False
                    line = line[line.index('*/') + 2:]
                else:
                    continue
            if '//' in line:
                line = line[:line.index('//')]
            if line.strip():
                cleaned_lines.append(line)
        
        return cleaned_lines
    
    def remove_empty_lines(self, lines: List[str]) -> List[str]:
        return [line for line in lines if line.strip()]
    
    def get_indentation_level(self, line: str) -> int:
        return len(line) - len(line.lstrip())
    
    def extract_declarations(self, lines: List[str]) -> List[str]:
        """Extract only forward declarations (ending with ;) and .extern/.global/.shared"""
        declarations = []
        extern_patterns = ['.extern', '.shared','.const','.param']
        
        i = 0
        while i < len(lines):
            line = lines[i]
            if ('.entry' in line) or ('.visible' in line):
                break
            if any(pattern in line for pattern in extern_patterns):
                declarations.append(line)
                i += 1
            elif '.func' in line and '.extern' not in line:
                '''temp_lines = [line]
                j = i + 1
                #read ahead until we find ; or {
                while j < len(lines):
                    temp_lines.append(lines[j])
                    if ';' in lines[j]:
                        declarations.extend(temp_lines)
                        i = j + 1
                        break
                    elif '{' in lines[j]:
                        i = j
                        break
                    j += 1
                if j >= len(lines):
                    i += 1'''
                break
            else:
                i += 1
        
        return declarations
    
    def extract_functions(self, lines: List[str]) -> List[Tuple[str, List[str]]]:
        """Extract function definitions (with bodies) from PTX file.
        
        Handles three cases:
        1. Single-line declarations ending with ;
        2. Multi-line declarations with params in () ending with ;
        3. Functions with { } bodies (single or multi-line)
        """
        functions = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if '.func' in line:
                #extract function name - pattern: .func (optional_return) function_name (params)
                match = re.search(r'\.func\s+(?:\([^)]+\)\s+)?([\w_]+)\s*\(', line)
                func_name = match.group(1) if match else f"func_{len(functions)}"
                start_indent = self.get_indentation_level(line)
                func_lines = [line]
                
                # Case 1: Single-line declaration with semicolon
                if ';' in line:
                    functions.append((func_name, func_lines))
                    i += 1
                    continue
                
                # Case 2: Function has opening brace on first line (has a body)
                if '{' in line:
                    i += 1
                    brace_count = line.count('{') - line.count('}')
                    
                    while i < len(lines):
                        current_line = lines[i]
                        func_lines.append(current_line)
                        brace_count += current_line.count('{') - current_line.count('}')
                        
                        if brace_count == 0 and '}' in current_line:
                            current_indent = self.get_indentation_level(current_line)
                            if current_indent == start_indent:
                                functions.append((func_name, func_lines))
                                break
                        i += 1
                    i += 1
                    continue
                
                # Case 3: Multi-line declaration with params in parentheses
                # Read lines until we find either { or ;
                i += 1
                while i < len(lines):
                    current_line = lines[i]
                    func_lines.append(current_line)
                    
                    # Found semicolon - declaration ends (no body)
                    if ';' in current_line:
                        functions.append((func_name, func_lines))
                        i += 1
                        break
                    
                    # Found opening brace - function has a body
                    if '{' in current_line:
                        brace_count = current_line.count('{') - current_line.count('}')
                        i += 1
                        
                        while i < len(lines):
                            body_line = lines[i]
                            func_lines.append(body_line)
                            brace_count += body_line.count('{') - body_line.count('}')
                            
                            if brace_count == 0 and '}' in body_line:
                                body_indent = self.get_indentation_level(body_line)
                                if body_indent == start_indent:
                                    functions.append((func_name, func_lines))
                                    break
                            i += 1
                        i += 1
                        break
                    
                    i += 1
            else:
                i += 1
        
        return functions
    
    def extract_kernels(self, lines: List[str]) -> List[Tuple[str, List[str]]]:
        kernels = []
        i = 0
        while i < len(lines):
            line = lines[i]
            #handling both .visible .entry and just .entry declarations
            if '.entry' in line:
                match = re.search(r'\.entry\s+(\w+)', line)
                kernel_name = match.group(1) if match else f"kernel_{len(kernels)}"
                start_indent = self.get_indentation_level(line)
                kernel_lines = [line]
                i += 1
                brace_count = line.count('{') - line.count('}')
                
                while i < len(lines):
                    current_line = lines[i]
                    kernel_lines.append(current_line)
                    brace_count += current_line.count('{') - current_line.count('}')
                    #check if we've found the matching closing brace
                    if brace_count == 0 and '}' in current_line:
                        #and then we verify it's at the same indentation level
                        current_indent = self.get_indentation_level(current_line)
                        if current_indent == start_indent:
                            kernels.append((kernel_name, kernel_lines))
                            break
                    i += 1
            i += 1
        
        return kernels
    def remove_global_declarations(self, lines: List[str]) -> List[str]:
        remove_patterns = ['.version', '.target', '.address_size']
        return [
            line for line in lines
            if not any(pattern in line for pattern in remove_patterns)
        ]
    
    def clean_kernel(self, kernel_lines: List[str]) -> List[str]:
        cleaned = self.remove_empty_lines(kernel_lines)
        return cleaned
    
    def clean_file(self, file_path: Path) -> bool:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.rstrip('\n\r') for line in f.readlines()]
            lines = self.remove_comments(lines)
            lines = self.remove_empty_lines(lines)
            lines = self.remove_global_declarations(lines)
            declarations = self.extract_declarations(lines)
            functions = self.extract_functions(lines)
            kernels = self.extract_kernels(lines)
            
            if not kernels:
                logger.warning(f"No .entry declarations found in {file_path.name}")
                return False
            if len(kernels) > 1:
                logger.info(f"Multiple kernels ({len(kernels)}) found in {file_path.name}")
                self.multi_kernel_files.append((file_path.name, len(kernels)))
            all_cleaned_kernels = []
            if declarations:
                all_cleaned_kernels.extend(declarations)
                all_cleaned_kernels.append('')
            for idx, (func_name, func_lines) in enumerate(functions):
                cleaned_func = self.clean_kernel(func_lines)
                all_cleaned_kernels.extend(cleaned_func)
                if idx < len(functions) - 1:
                    all_cleaned_kernels.append('')
            if functions and kernels:
                all_cleaned_kernels.append('')
            for idx, (kernel_name, kernel_lines) in enumerate(kernels):
                cleaned_kernel = self.clean_kernel(kernel_lines)
                all_cleaned_kernels.extend(cleaned_kernel)
                if len(kernels) > 1 and idx < len(kernels) - 1:
                    all_cleaned_kernels.append('')
            output_name = file_path.name
            output_path = self.output_dir / output_name
            logger.debug(f"Writing to: {output_path.absolute()}")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(all_cleaned_kernels))
                f.write('\n')
            logger.debug(f"Wrote cleaned kernel(s) to {output_name}")
            return True
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return False
    
    def clean_directory(self) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {self.output_dir}")
        # ptx_files = list(self.input_dir.glob("*.ptx"))
        ptx_files = [
            self.input_dir / "file_50142.ptx",
            self.input_dir / "file_50143.ptx",
            self.input_dir / "file_25464.ptx",
            self.input_dir / "00006_aligned_vectorized_matmul_base_base.ptx",
            self.input_dir / "00026_block_optimized_conv2d_relu_bias_base_base.ptx",
            self.input_dir / "file_30194.ptx",
            self.input_dir / "file_30199.ptx",
            self.input_dir /"file_10221.ptx"
            ]
        if not ptx_files:
            logger.warning(f"No .ptx files found in {self.input_dir}")
            return {
                'total_files': 0,
                'processed': 0,
                'skipped': 0,
                'multi_kernel_files': []
            }
        logger.info(f"Found {len(ptx_files)} PTX files to process")
        processed = 0
        skipped = 0
        for ptx_file in ptx_files:
            logger.info(f"Processing {ptx_file.name}...")
            if self.clean_file(ptx_file):
                processed += 1
            else:
                skipped += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"Cleaning Summary:")
        logger.info(f"  Total files: {len(ptx_files)}")
        logger.info(f"  Processed: {processed}")
        logger.info(f"  Skipped: {skipped}")
        if self.multi_kernel_files:
            logger.info(f"\n  Files with multiple kernels:")
            for filename, count in self.multi_kernel_files:
                logger.info(f"    - {filename}: {count} kernels")
        logger.info(f"{'='*60}\n")
        
        return {
            'total_files': len(ptx_files),
            'processed': processed,
            'skipped': skipped,
            'multi_kernel_files': self.multi_kernel_files
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Clean PTX files for transformer preprocessing')
    parser.add_argument('input_dir', type=str,help='Directory containing .ptx files')
    parser.add_argument('--output-dir',type=str,default='cleaned',help='Output directory name (default: cleaned)')
    parser.add_argument('--verbose',action='store_true',help='Enable verbose logging')
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    cleaner = PTXCleaner(args.input_dir, args.output_dir)
    stats = cleaner.clean_directory()
    return stats

if __name__ == "__main__":
    main()
