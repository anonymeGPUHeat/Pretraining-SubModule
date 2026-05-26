from pathlib import Path
from typing import List
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PTXCleaner:
    def __init__(self, input_dir: str, output_dir: str = "cleaned"):
        self.input_dir = Path(input_dir)
        self.output_dir = self.input_dir.parent / output_dir
        
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
    
    def remove_header_directives(self, lines: List[str]) -> List[str]:
        remove_patterns = ['.version', '.target', '.address_size']
        return [
            line for line in lines
            if not any(pattern in line for pattern in remove_patterns)
        ]
    
    def clean_file(self, file_path: Path) -> bool:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.rstrip('\n\r') for line in f.readlines()]
            lines = self.remove_comments(lines)
            lines = self.remove_header_directives(lines)
            output_path = self.output_dir / file_path.name
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
                f.write('\n')
            logger.info(f"Successfully cleaned {file_path.name}")
            return True 
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return False
    
    def clean_directory(self) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {self.output_dir}")
        ptx_files = list(self.input_dir.glob("*.ptx"))
        if not ptx_files:
            logger.warning(f"No .ptx files found in {self.input_dir}")
            return {
                'total_files': 0,
                'processed': 0,
                'failed': 0
            }
        logger.info(f"Found {len(ptx_files)} PTX files to process")
        processed = 0
        failed = 0
        for ptx_file in ptx_files:
            if self.clean_file(ptx_file):
                processed += 1
            else:
                failed += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"Cleaning Summary:")
        logger.info(f"  Total files: {len(ptx_files)}")
        logger.info(f"  Processed: {processed}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"{'='*60}\n")
        
        return {
            'total_files': len(ptx_files),
            'processed': processed,
            'failed': failed
        }
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Clean PTX files - remove comments and header directives')
    parser.add_argument('input_dir', type=str, help='Directory containing .ptx files')
    parser.add_argument('--output-dir', type=str, default='cleaned', help='Output directory name (default: cleaned)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args() 
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    cleaner = PTXCleaner(args.input_dir, args.output_dir)
    stats = cleaner.clean_directory()
    return stats


if __name__ == "__main__":
    main()
