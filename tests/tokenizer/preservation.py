import sentencepiece as spm
import numpy as np
from pathlib import Path


def test_critical_token_preservation(tokenizer_model: str):
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_model)
    print("\n" + "="*80)
    print("CRITICAL TOKEN PRESERVATION TEST")
    print("="*80)
    critical_tokens = {
        'Memory Operations': [
            'ld.global.ca.f32',
            'ld.global.cg.f32',
            'ld.global.nc.u32',
            'ld.shared.f32',
            'st.global.f32',
            'st.shared.u32',
            'atom.global.add.u32',
        ],
        'Special Registers': [
            '%tid.x', '%tid.y', '%tid.z',
            '%ctaid.x', '%ntid.x',
            '%laneid', '%warpid',
        ],
        'Control Flow': [
            'bar.sync',
            'bra', 'ret',
        ],
        'Synchronization': [
            'bar.sync', 'membar.cta',
        ],
        'Data Types': [
            '.f32', '.f64', '.u32', '.s32',
            '.b32', '.pred',
        ],
    }
    
    results = {}
    
    for category, tokens in critical_tokens.items():
        print(f"\n{category}:")
        preserved = 0
        split = 0
        
        for token in tokens:
            pieces = sp.EncodeAsPieces(token)
            
            if len(pieces) == 1:
                print(f"'{token}' → {pieces}")
                preserved += 1
            else:
                print(f" '{token}' → {pieces} (SPLIT)")
                split += 1
        
        preservation_rate = (preserved / len(tokens)) * 100
        results[category] = preservation_rate
        
        print(f"  Preservation: {preserved}/{len(tokens)} ({preservation_rate:.1f}%)")
    overall_preservation = np.mean(list(results.values()))
    print(f"\n" + "="*80)
    print(f"OVERALL PRESERVATION RATE: {overall_preservation:.1f}%")
    
    if overall_preservation >= 80:
        print("EXCELLENT - Most critical tokens preserved")
    elif overall_preservation >= 60:
        print("GOOD - Some splitting, consider adding more user_defined_symbols")
    else:
        print("POOR - Too much splitting, increase vocab size or add symbols")

    
    return results


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python preservation.py <tokenizer_model>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_critical_token_preservation(model_path)