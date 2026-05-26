import sentencepiece as spm

def test_instruction_boundaries(tokenizer_model: str):
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_model)
    
    print("\n" + "="*80)
    print("INSTRUCTION BOUNDARY TEST")
    print("="*80)
    ptx_block = """
    mov.u32 %r1, %tid.x;
    mul.lo.s32 %r2, %r1, 4;
    ld.global.f32 %f1, [%r3+%r2];
    add.f32 %f2, %f1, %f1;
    st.global.f32 [%r3+%r2], %f2;
    """
    
    lines = [l.strip() for l in ptx_block.strip().split('\n')]
    
    print(f"Testing {len(lines)} instructions:\n")
    
    for line in lines:
        tokens = sp.EncodeAsPieces(line)
        has_separate_semicolon = ';' in tokens or '▁;' in tokens
        
        print(f"  {line}")
        print(f"    → Tokens ({len(tokens)}): {tokens}")
        
        if has_separate_semicolon:
            print(f"Semicolon separate (good instruction boundary)")
        else:
            print(f"Semicolon merged (may blur boundaries)")
        print()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inst_boundary_detection.py <tokenizer_model>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_instruction_boundaries(model_path)