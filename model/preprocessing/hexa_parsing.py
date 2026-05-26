import re
from collections import Counter
from pathlib import Path

def extract_float_constants(ptx_files):
    """
    From NVIDIA's doc : find all IEEE 754 hex float constants in PTX files
    - Double-precision: 0d/0D followed by 16 hex digits
    - Single-precision: 0f/0F followed by 8 hex digits
    """
    constant_counter = Counter()
    pattern_double = re.compile(r'0[dD][0-9A-Fa-f]{16}')
    pattern_single = re.compile(r'0[fF][0-9A-Fa-f]{8}')
    
    for file_path in ptx_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                constants_double = pattern_double.findall(content)
                constants_single = pattern_single.findall(content)
                constant_counter.update(constants_double)
                constant_counter.update(constants_single)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return constant_counter

def main():
    data_dir = Path(__file__).parent.parent.parent / "data/raw"
    ptx_files = list(data_dir.glob("**/*.ptx"))
    print(f"Found {len(ptx_files)} PTX files")
    if not ptx_files:
        print("No PTX files found!")
        return
    constants_counter = extract_float_constants(ptx_files)
    print(f"Found {len(constants_counter)} unique hex constants")
    print(f"Total occurrences: {sum(constants_counter.values())}")
    output_file = Path(__file__).parent.parent.parent / "ptx_hex_constants.txt"
    with open(output_file, 'w') as f:
        f.write("PTX Hexadecimal Floating-Point Constants\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total unique constants: {len(constants_counter)}\n")
        f.write(f"Total occurrences: {sum(constants_counter.values())}\n\n")
        f.write("DOUBLE-PRECISION CONSTANTS (0d/0D + 16 hex digits):\n")
        f.write("-" * 80 + "\n")
        double_consts = sorted([c for c in constants_counter if c[1] in 'dD'], 
                              key=lambda x: constants_counter[x], reverse=True)
        for const in double_consts:
            f.write(f"{const}: {constants_counter[const]} occurrences\n")
        f.write("\n")
        f.write("SINGLE-PRECISION CONSTANTS (0f/0F + 8 hex digits):\n")
        f.write("-" * 80 + "\n")
        single_consts = sorted([c for c in constants_counter if c[1] in 'fF'], 
                              key=lambda x: constants_counter[x], reverse=True)
        for const in single_consts:
            f.write(f"{const}: {constants_counter[const]} occurrences\n")
    
    print(f"\nResults written to {output_file}")
if __name__ == "__main__":
    main()