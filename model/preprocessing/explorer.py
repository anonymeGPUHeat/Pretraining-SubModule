import re
from pathlib import Path
from typing import List, Tuple
import json
import logging

#config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PTXExplorer:
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        self.declarations_before_kernel = []  # [(file, declaration)]
        self.cache_modifier_usage = []  # [(file, line_num, line)]
        self.kernels_with_atomics = []  # [(file, kernel_name)]
        self.kernels_with_tensor_cores = []  # [(file, kernel_name)]
        self.kernels_with_functions = []  # [(file, kernel_name)]
        self.kernels_with_texref = []  # [(file, kernel_name)]
        self.standard_declarations = {'.target', '.global', '.address_size', '.version'}
        self.cache_modifiers = {'.ca', '.cg', '.cs', '.lu', '.cv', '.wb', '.wt'}
        
    def is_comment_line(self, line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith('//') or stripped.startswith('/*')
    
    def extract_kernel_name(self, line: str) -> str:
        match = re.search(r'\.entry\s+(\w+)', line)
        return match.group(1) if match else "unknown"
    
    def scan_declarations_before_kernel(self, file_path: Path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            found_count = 0
            for line in lines:
                stripped = line.strip()
                if '.entry' in stripped:
                    break
                if not stripped or self.is_comment_line(stripped):
                    continue
                if stripped.startswith('.'):
                    decl_match = re.match(r'\.(\w+)', stripped)
                    if decl_match:
                        decl_type = '.' + decl_match.group(1)
                        if decl_type not in self.standard_declarations:
                            self.declarations_before_kernel.append({
                                'file': file_path.name,
                                'declaration': stripped
                            })
                            found_count += 1
            
            if found_count > 0:
                logger.info(f"  {file_path.name}: Found {found_count} declarations")
        
        except Exception as e:
            logger.error(f"Error scanning declarations in {file_path.name}: {e}")
    
    def scan_cache_modifiers(self, file_path: Path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            found_count = 0
            for line_num, line in enumerate(lines, 1):
                stripped = line.strip()
                if self.is_comment_line(stripped):
                    continue
                for modifier in self.cache_modifiers:
                    if modifier in stripped:
                        self.cache_modifier_usage.append({
                            'file': file_path.name,
                            'line_number': line_num,
                            'line': stripped,
                            'modifier': modifier
                        })
                        found_count += 1
                        break
            
            if found_count > 0:
                logger.info(f"  {file_path.name}: Found {found_count} cache modifier usages")
        
        except Exception as e:
            logger.error(f"Error scanning cache modifiers in {file_path.name}: {e}")
    
    def scan_kernel_features(self, file_path: Path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            kernels = self.extract_kernels_with_content(content)
            
            features_found = {
                'atomics': [],
                'tensor_cores': [],
                'functions': [],
                'texref': []
            }
            
            for kernel_name, kernel_content in kernels:
                if re.search(r'\batom\.[a-z.]+', kernel_content):
                    self.kernels_with_atomics.append({
                        'file': file_path.name,
                        'kernel': kernel_name
                    })
                    features_found['atomics'].append(kernel_name)
                #check for tensor core operations (mma.*, wmma.*, ldmatrix.*, etc.)
                tensor_patterns = [
                    r'\bmma\.[a-z.]+',
                    r'\bwmma\.[a-z.]+',
                    r'\bldmatrix\.[a-z.]+',
                    r'\bhmma\.[a-z.]+'
                ]
                if any(re.search(pattern, kernel_content) for pattern in tensor_patterns):
                    self.kernels_with_tensor_cores.append({
                        'file': file_path.name,
                        'kernel': kernel_name
                    })
                    features_found['tensor_cores'].append(kernel_name)
                #check for function definitions (.func)
                if re.search(r'\bfunc\b', kernel_content):
                    self.kernels_with_functions.append({
                        'file': file_path.name,
                        'kernel': kernel_name
                    })
                    features_found['functions'].append(kernel_name)
                # check for texture references (.texref)
                if re.search(r'\btexref\b', kernel_content):
                    self.kernels_with_texref.append({
                        'file': file_path.name,
                        'kernel': kernel_name
                    })
                    features_found['texref'].append(kernel_name)
            
            if any(features_found.values()):
                logger.info(f"  {file_path.name}: {len(kernels)} kernels found")
                if features_found['atomics']:
                    logger.info(f"    - Atomics: {', '.join(features_found['atomics'])}")
                if features_found['tensor_cores']:
                    logger.info(f"    - Tensor cores: {', '.join(features_found['tensor_cores'])}")
                if features_found['functions']:
                    logger.info(f"    - Functions: {', '.join(features_found['functions'])}")
                if features_found['texref']:
                    logger.info(f"    - Texref: {', '.join(features_found['texref'])}")
        except Exception as e:
            logger.error(f"Error scanning kernel features in {file_path.name}: {e}")
    
    def extract_kernels_with_content(self, content: str) -> List[Tuple[str, str]]:
        kernels = []
        lines = content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            if '.entry' in line:
                kernel_name = self.extract_kernel_name(line)
                kernel_lines = [line]
                i += 1
                brace_count = line.count('{') - line.count('}')
                
                while i < len(lines) and brace_count > 0:
                    current_line = lines[i]
                    kernel_lines.append(current_line)
                    brace_count += current_line.count('{') - current_line.count('}')
                    i += 1
                
                kernels.append((kernel_name, '\n'.join(kernel_lines)))
                continue
            
            i += 1
        return kernels
    
    def explore_directory(self):
        ptx_files = list(self.input_dir.glob("*.ptx"))
        
        if not ptx_files:
            logger.warning(f"No .ptx files found in {self.input_dir}")
            return
        
        logger.info(f"\nExploring {len(ptx_files)} PTX files from {self.input_dir}...")
        logger.info("\n[SCANNING DECLARATIONS]")
        #for ptx_file in ptx_files:
        #    self.scan_declarations_before_kernel(ptx_file)
        
        #logger.info("\n[SCANNING CACHE MODIFIERS]")
        #for ptx_file in ptx_files:
        #    self.scan_cache_modifiers(ptx_file)
        
        logger.info("\n[SCANNING KERNEL FEATURES]")
        for ptx_file in ptx_files:
            self.scan_kernel_features(ptx_file)
        
        logger.info(f"\nExploration complete!")
    
    def save_results(self, output_file: str = "ptx_exploration_results.json"):
        stats_dir = self.input_dir.parent / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n[SAVING RESULTS]")
        logger.info(f"Output directory: {stats_dir}")
        
        results = {
            'summary': {
                'unique_declarations': len(set(d['declaration'] for d in self.declarations_before_kernel)),
                'total_declaration_instances': len(self.declarations_before_kernel),
                'cache_modifier_usages': len(self.cache_modifier_usage),
                'kernels_with_atomics': len(self.kernels_with_atomics),
                'kernels_with_tensor_cores': len(self.kernels_with_tensor_cores),
                'kernels_with_functions': len(self.kernels_with_functions),
                'kernels_with_texref': len(self.kernels_with_texref)
            },
            'declarations_before_kernel': self.declarations_before_kernel,
            'cache_modifier_usage': self.cache_modifier_usage,
            'kernels_with_atomics': self.kernels_with_atomics,
            'kernels_with_tensor_cores': self.kernels_with_tensor_cores,
            'kernels_with_functions': self.kernels_with_functions,
            'kernels_with_texref': self.kernels_with_texref
        }
        
        output_path = stats_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"  Saved: {output_file}")
        
        decl_file = stats_dir / "declarations_found.txt"
        with open(decl_file, 'w', encoding='utf-8') as f:
            f.write("DECLARATIONS BEFORE .entry (excluding .target, .global, .address_size, .version)\n")
            f.write("="*80 + "\n\n")
            for item in self.declarations_before_kernel:
                f.write(f"File: {item['file']}\n")
                f.write(f"Declaration: {item['declaration']}\n")
                f.write("-"*80 + "\n")
        logger.info(f"  Saved: declarations_found.txt ({len(self.declarations_before_kernel)} items)")
        
        cache_file = stats_dir / "cache_modifiers_usage.txt"
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write("CACHE MODIFIER USAGE (.ca, .cg, .cs, .lu, .cv, .wb, .wt)\n")
            f.write("="*80 + "\n\n")
            for item in self.cache_modifier_usage:
                f.write(f"File: {item['file']} (Line {item['line_number']})\n")
                f.write(f"Modifier: {item['modifier']}\n")
                f.write(f"Line: {item['line']}\n")
                f.write("-"*80 + "\n")
        logger.info(f"  Saved: cache_modifiers_usage.txt ({len(self.cache_modifier_usage)} items)")
        
        features_file = stats_dir / "kernel_features_summary.txt"
        with open(features_file, 'w', encoding='utf-8') as f:
            f.write("KERNEL FEATURES SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"ATOMICS ({len(self.kernels_with_atomics)} kernels):\n")
            for item in self.kernels_with_atomics:
                f.write(f"  - {item['file']}: {item['kernel']}\n")
            
            f.write(f"\nTENSOR CORES ({len(self.kernels_with_tensor_cores)} kernels):\n")
            for item in self.kernels_with_tensor_cores:
                f.write(f"  - {item['file']}: {item['kernel']}\n")
            
            f.write(f"\nFUNCTIONS ({len(self.kernels_with_functions)} kernels):\n")
            for item in self.kernels_with_functions:
                f.write(f"  - {item['file']}: {item['kernel']}\n")
            
            f.write(f"\nTEXREF ({len(self.kernels_with_texref)} kernels):\n")
            for item in self.kernels_with_texref:
                f.write(f"  - {item['file']}: {item['kernel']}\n")
        logger.info(f"  Saved: kernel_features_summary.txt")
        logger.info(f"SUMMARY:\n")
        logger.info(f"  Declarations found: {len(self.declarations_before_kernel)}")
        logger.info(f"  Cache modifier usages: {len(self.cache_modifier_usage)}")
        logger.info(f"  Kernels with atomics: {len(self.kernels_with_atomics)}")
        logger.info(f"  Kernels with tensor cores: {len(self.kernels_with_tensor_cores)}")
        logger.info(f"  Kernels with functions: {len(self.kernels_with_functions)}")
        logger.info(f"  Kernels with texref: {len(self.kernels_with_texref)}")
        
        return results
    
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Explore PTX files for patterns and features')
    parser.add_argument('input_dir', type=str, help='Directory containing .ptx files')
    parser.add_argument('--output', type=str, default='ptx_exploration_results.json',
                        help='Output JSON file name (default: ptx_exploration_results.json)')
    args = parser.parse_args()
    explorer = PTXExplorer(args.input_dir)
    explorer.explore_directory()
    explorer.save_results(args.output)


if __name__ == "__main__":
    main()
