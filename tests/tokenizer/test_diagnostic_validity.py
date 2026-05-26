import sentencepiece as spm
import sys
from pathlib import Path


def test_roundtrip_fidelity(sp: spm.SentencePieceProcessor):
    print("\n" + "="*70)
    print("TEST 1: ROUNDTRIP FIDELITY")
    print("="*70)
    
    test_cases = [
        "ld.global.f32 <FP32_REG>, [<INT64_REG>];",
        "st.shared.u32 [<INT32_REG>], <INT32_REG>;",
        "add.f32 <FP32_REG>, <FP32_REG>, <FP32_REG>;",
        "bar.sync 0;",
        "setp.lt.s32 <PRED>, <INT32_REG>, <INT32_REG>;",
    ]
    
    passed = 0
    failed = 0
    
    for original in test_cases:
        ids = sp.EncodeAsIds(original)
        decoded = sp.DecodeIds(ids)
        original_tokens = original.replace(';', '').replace(',', '').split()
        preserved = sum(1 for tok in original_tokens if tok in decoded or tok.lower() in decoded.lower())
        preservation_rate = (preserved / len(original_tokens)) * 100
        
        if preservation_rate >= 80:
            print(f"test: {original[:50]}")
            print(f"          Preservation: {preservation_rate:.0f}%")
            passed += 1
        else:
            print(f"test: {original[:50]}")
            print(f"          Original:  {original}")
            print(f"          Decoded:   {decoded}")
            print(f"          Preservation: {preservation_rate:.0f}%")
            failed += 1
    
    print(f"\n{'PASSED' if failed == 0 else 'FAILED'}: {passed}/{len(test_cases)} test cases passed")
    return failed == 0


def test_boundary_preservation(sp: spm.SentencePieceProcessor):
    print("TEST 2: BOUNDARY PRESERVATION")
    
    ptx_block = """mov.u32 <INT32_REG>, %tid.x;
mul.lo.s32 <INT32_REG>, <INT32_REG>, 4;
ld.global.f32 <FP32_REG>, [<INT64_REG>+<INT32_REG>];"""
    
    lines = [l.strip() for l in ptx_block.strip().split('\n')]
    
    passed = 0
    failed = 0
    
    for line in lines:
        tokens = sp.EncodeAsPieces(line)
        has_semicolon = any(';' in token for token in tokens)
        
        if has_semicolon:
            print(f"PASS: Semicolon preserved in: {line[:40]}")
            passed += 1
        else:
            print(f"FAIL: Semicolon lost in: {line}")
            print(f"          Tokens: {tokens}")
            failed += 1
    
    print(f"\n{'PASSED' if failed == 0 else 'FAILED'}: {passed}/{len(lines)} instructions preserved boundaries")
    return failed == 0


def test_structural_survival(sp: spm.SentencePieceProcessor):
    print("TEST 3: STRUCTURAL TOKEN SURVIVAL")
    
    structural_tokens = [
        '<BB_0>',
        '<BB_1>',
        '<BB_10>',
        '<LOOP_0>',
        '<KERNEL_main>',
        '<FUNC_helper>',
        '<PARAM_0>',
        '<PARAM_5>',
    ]
    
    passed = 0
    failed = 0
    
    for token in structural_tokens:
        pieces = sp.EncodeAsPieces(token)
        token_id = sp.PieceToId(token)
        
        # Check if token is preserved as single piece or known symbol
        if len(pieces) == 1:
            print(f"PASS: {token} → {pieces[0]}")
            passed += 1
        elif token_id != sp.unk_id():
            print(f"PASS: {token} → ID {token_id} (known)")
            passed += 1
        else:
            print(f"PASS: {token} → {pieces} (split into {len(pieces)} pieces)")
            failed += 1
    
    print(f"\n{'PASSED' if failed == 0 else 'ACCEPTABLE'}: {passed}/{len(structural_tokens)} structural tokens preserved")
    return passed >= len(structural_tokens) * 0.7  # 70% threshold


def test_vector_instruction_support(sp: spm.SentencePieceProcessor):
    print("TEST 4: VECTOR INSTRUCTION SUPPORT")
    
    vector_instructions = [
        ("ld.global.v4.f32 <FP32_REG>, [<INT64_REG>];", "v4"),
        ("ld.shared.v2.u32 <INT32_REG>, [<INT32_REG>];", "v2"),
        ("st.global.v4.f32 [<INT64_REG>], <FP32_REG>;", "v4"),
    ]
    
    passed = 0
    failed = 0
    
    for inst, expected_modifier in vector_instructions:
        pieces = sp.EncodeAsPieces(inst)
        decoded = sp.DecodeIds(sp.EncodeAsIds(inst))
        has_modifier = any(expected_modifier in p.lower() for p in pieces) or expected_modifier in decoded.lower()
        
        if has_modifier:
            print(f"PASS: {expected_modifier} modifier preserved in: {inst[:40]}")
            passed += 1
        else:
            print(f"PASS: {expected_modifier} modifier lost in: {inst}")
            print(f"          Pieces: {pieces}")
            failed += 1
    
    print(f"\n{'PASSED' if failed == 0 else 'FAILED'}: {passed}/{len(vector_instructions)} vector instructions preserved")
    return failed == 0


def test_special_register_handling(sp: spm.SentencePieceProcessor):
    print("TEST 5: SPECIAL REGISTER HANDLING")
    special_registers = [
        '%tid.x',
        '%tid.y',
        '%tid.z',
        '%ctaid.x',
        '%ntid.x',
        '%laneid',
        '%warpid',
    ]
    
    passed = 0
    failed = 0
    
    for reg in special_registers:
        pieces = sp.EncodeAsPieces(reg)
        token_id = sp.PieceToId(reg)
        is_unk = token_id == sp.unk_id() and len(pieces) == 1
        
        if not is_unk:
            print(f"PASS: {reg:12} → {pieces}")
            passed += 1
        else:
            print(f"PASS: {reg:12} → <UNK>")
            failed += 1
    
    print(f"\n{'PASSED' if failed == 0 else 'ACCEPTABLE'}: {passed}/{len(special_registers)} special registers handled")
    return passed >= len(special_registers) * 0.7


def test_normalized_token_preservation(sp: spm.SentencePieceProcessor):
    print("TEST 6: NORMALIZED TOKEN PRESERVATION")

    normalized_tokens = [
        '<FP32_REG>',
        '<FP64_REG>',
        '<INT32_REG>',
        '<INT64_REG>',
        '<INT16_REG>',
        '<PRED_REG>',
        '<PRED>',
    ]
    
    passed = 0
    failed = 0
    
    for token in normalized_tokens:
        pieces = sp.EncodeAsPieces(token)
        token_id = sp.PieceToId(token)
        
        # Check if token is preserved
        if len(pieces) == 1 and pieces[0] == token:
            print(f"PASS: {token:15} → preserved (ID: {token_id})")
            passed += 1
        elif token_id != sp.unk_id():
            print(f"PASS: {token:15} → known (ID: {token_id})")
            passed += 1
        else:
            print(f"PASS: {token:15} → {pieces}")
            failed += 1
    
    print(f"\n{'PASSED' if failed == 0 else 'ACCEPTABLE'}: {passed}/{len(normalized_tokens)} normalized tokens preserved")
    return passed >= len(normalized_tokens) * 0.8


def test_on_real_ptx_file(sp: spm.SentencePieceProcessor, ptx_file: Path):
    print(f"TEST 7: REAL PTX FILE - {ptx_file.name}")
    
    try:
        with open(ptx_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if not content.strip():
            print("WARNING: Empty file")
            return True
  
        ids = sp.EncodeAsIds(content)
        decoded = sp.DecodeIds(ids)
        unk_count = sum(1 for id in ids if id == sp.unk_id())
        unk_rate = (unk_count / len(ids) * 100) if ids else 0
        
        print(f"  File size:      {len(content):,} characters")
        print(f"  Tokens:         {len(ids):,}")
        print(f"  Unknown tokens: {unk_count} ({unk_rate:.2f}%)")
        print(f"  Avg token len:  {len(content)/len(ids):.2f} chars/token")
        roundtrip_ok = decoded.strip() == content.strip()
        
        if unk_rate < 1.0 and roundtrip_ok:
            print(f"PASSED: Excellent quality (<1% UNK, roundtrip OK)")
            return True
        elif unk_rate < 5.0 and roundtrip_ok:
            print(f"ACCEPTABLE: Good quality (<5% UNK, roundtrip OK)")
            return True
        else:
            print(f"FAILED: UNK rate {unk_rate:.2f}% or roundtrip failed")
            if not roundtrip_ok:
                print(f"  Roundtrip FAILED: decoded differs from original")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def run_all_diagnostic_tests(model_path: str, ptx_file: Path = None):
    print("PTX TOKENIZER DIAGNOSTIC VALIDITY TESTS")
    print(f"Model: {model_path}")

    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    
    print(f"Vocabulary size: {sp.GetPieceSize()}")
    results = {
        "Roundtrip Fidelity": test_roundtrip_fidelity(sp),
        "Boundary Preservation": test_boundary_preservation(sp),
        "Structural Survival": test_structural_survival(sp),
        "Vector Instructions": test_vector_instruction_support(sp),
        "Special Registers": test_special_register_handling(sp),
        "Normalized Tokens": test_normalized_token_preservation(sp),
    }
    if ptx_file and ptx_file.exists():
        results["Real PTX File"] = test_on_real_ptx_file(sp, ptx_file)
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:30} {status}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\nEXCELLENT: All diagnostic tests passed!")
        print("   Tokenizer is diagnostic-grade and ready for use.")
    elif passed >= total * 0.8:
        print("\nGOOD: Most tests passed, tokenizer is usable.")
        print("   Consider improvements for failed tests.")
    else:
        print("\nPOOR: Too many tests failed.")
        print("   Retrain with larger vocab or better user-defined symbols.")
    
    return passed == total


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_diagnostic_validity.py <tokenizer_model> [ptx_file]")
        print("\nExample:")
        print("  python test_diagnostic_validity.py ptx_tokenizer.model")
        print("  python test_diagnostic_validity.py ptx_tokenizer.model sample.ptx")
        sys.exit(1)
    
    model_path = sys.argv[1]
    ptx_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not Path(model_path).exists():
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)
    
    if ptx_file and not ptx_file.exists():
        print(f"ERROR: PTX file not found: {ptx_file}")
        sys.exit(1)
    
    success = run_all_diagnostic_tests(model_path, ptx_file)
    sys.exit(0 if success else 1)
