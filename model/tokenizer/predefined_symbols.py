from pathlib import Path


def create_user_symbols():
    """
    this is the list of user-defined symbols that SentencePiece must preserve
    These will be added to the vocabulary as single, unsplittable tokens
    """
    symbols = []
    symbols.extend([
        '<KERNEL_', '<FUNC_', '<PARAM_', '<BB_', '<VTABLE_',  '<STR_', '<TABLE_',
    #    '<FP32_REG>',
    #    '<FP64_REG>',
    #    '<INT32_REG>',
    #    '<INT64_REG>',
    #    '<INT16_REG>',
    #    '<PRED_REG>',
    #    '@!<PRED_REG>',
    #    '@<PRED_REG>',
        '<IMM_6K>', '<IMM_10K>', '<IMM_100K>','<IMM_1M>','<IMM_LARGE>','<IMM_NEG_6K>','<IMM_NEG_10K>','<IMM_NEG_100K>','<IMM_NEG_1M>','<IMM_NEG_LARGE>',])
    
    #memory modifiers
    symbols.extend([
        '.nc',    # non-cached
        '.ca',    # cache all levels
        '.cg',    # cache global
        '.cs',    # cache streaming
        '.lu',    # last use
        '.cv',    # cache volatile
        '.v2',    # vector 2
        '.v4',    # vector 4
        '.rn',    # round to nearest
        '.rz',    # round to zero
        '.rm',    # round to minus infinity
        '.rp',    # round to plus infinity
        '.ftz',   # flush to zero
        '.sat',   # saturate
        '.lo',    # low bits
        '.hi',    # high bits
        '.wide',  # wide multiply
        '.cta'
    ])


    symbols.extend([  '.entry', '.func', '.reg','.param','.shared', '.local', '.global', '.const','.visible','.maxnreg', '.reqntid', '.maxntid',])

    symbols.extend([
        '%tid.x','%tid.y', '%tid.z', '%ntid.x', '%ntid.y', '%ntid.z', '%ctaid.x', '%ctaid.y', '%ctaid.z',
        '%nctaid.x', '%nctaid.y', '%nctaid.z', '%laneid', '%warpid',
        '%clock', '%clock64', 
        '%pm0', '%pm1', '%pm2', '%pm3', '%pm4', '%pm5', '%pm6', '%pm7',
        '%pm0_64', '%pm1_64', '%pm2_64', '%pm3_64', '%pm4_64', '%pm5_64', '%pm6_64', '%pm7_64',
        *[f"%envreg{i}" for i in range(32)],
        '%globaltimer', '%globaltimer_lo', '%globaltimer_hi',
        '%reserved_smem_offset_begin', '%reserved_smem_offset_end', '%reserved_smem_offset_cap',
        '%reserved_smem_offset0', '%reserved_smem_offset1',
        '%total_smem_size', '%aggr_smem_size', '%dynamic_smem_size', '%current_graph_exec',
        '%r', '%rd', '%f', '%b', '%fd', '%p',
        '@%p', '@!%p',
    ])
    
    symbols.extend(['.visible', '.entry', '.func', '.param','.reg','.local','.global','.shared', '.const', '.align', '.extern',])
    
    #data types
    symbols.extend([
        '.b8',
        '.b16',
        '.b32',
        '.b64',
        '.u8',
        '.u16',
        '.u32',
        '.u64',
        '.s8',
        '.s16',
        '.s32',
        '.s64',
        '.f32',
        '.f64',
        '.pred',
        '.f16',
        '.f16x2',
        '.f32',
        '.f64',
        '.b128',
        '.bf16',
        '.tf32'])

    symbols.extend(['<mask>', '<cls>', '<sep>'])
    
    # PTX instruction opcodes (must be single tokens for ITC classifier)
    # Memory opcodes
    symbols.extend(['ld', 'st', 'ldu', 'ldmatrix', 'stmatrix'])
    
    # Arithmetic opcodes
    symbols.extend([
        'add', 'addc', 'sub', 'subc', 'mul', 'mul24', 'mad', 'mad24', 'madc',
        'fma', 'div', 'rem', 'abs', 'neg', 'min', 'max',
        'rcp', 'sqrt', 'rsqrt', 'sin', 'cos', 'lg2', 'ex2', 'tanh',
        'sad', 'dp2a', 'dp4a',
        'vadd', 'vadd2', 'vadd4', 'vsub', 'vsub2', 'vsub4',
        'vmad', 'vmax', 'vmax2', 'vmax4', 'vmin', 'vmin2', 'vmin4',
        'vavrg2', 'vavrg4', 'vabsdiff', 'vabsdiff2', 'vabsdiff4',
        'vset', 'vset2', 'vset4', 'vshl', 'vshr', 'multimem',
    ])
    
    # Logic and bitwise opcodes
    symbols.extend([
        'and', 'or', 'xor', 'not', 'cnot',
        'shl', 'shr', 'shf', 'bfe', 'bfi', 'bfind', 'brev', 'bmsk',
        'popc', 'clz', 'lop3', 'szext', 'copysign',
    ])
    
    # Comparison and predicate opcodes
    symbols.extend([
        'setp', 'set', 'selp', 'slct', 'vote', 'match',
        'testp', 'isspacep', 'istypep',
    ])
    
    # Control flow opcodes
    symbols.extend([
        'bra', 'brx', 'ret', 'exit', 'call',
        'brkpt', 'trap', 'alloca', 'applypriority', 'discard',
        'nanosleep', 'pmevent', 'setmaxnreg',
        'stackrestore', 'stacksave', 'griddepcontrol',
        'clusterlaunchcontrol', 'elect', 'getctarank',
    ])
    
    # Synchronization and barrier opcodes
    symbols.extend([
        'bar', 'barrier', 'membar', 'fence', 'mbarrier', 'activemask',
    ])
    
    # Conversion and data movement opcodes
    symbols.extend([
        'cvt', 'mov', 'movmatrix', 'shfl', 'prmt',
        'redux', 'fns', 'mapa', 'createpolicy',
    ])
    
    # Atomic and reduction opcodes
    symbols.extend([
        'atom', 'red', 'sured',
    ])
    
    # Special opcodes (texture, surface, tensor, async, etc.)
    symbols.extend([
        'tex', 'tld4', 'txq',
        'suld', 'sust', 'suq',
        'wmma', 'mma', 'wgmma',
        'cp', 'tcgen05', 'tensormap',
        'prefetch', 'prefetchu',
    ])
    
    #add hexadecimals
    hex_file = Path(__file__).parent.parent.parent / 'ptx_hex_constants.txt'
    if hex_file.exists():
        with open(hex_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith('0d') or line.startswith('0D') or line.startswith('0f') or line.startswith('0F'):
                    symbol = line.split(':')[0].strip()
                    if symbol:
                        symbols.append(symbol)
    else:
        print(f"Warning: Hex constants file not found at {hex_file}")
    
    return symbols

if __name__ == '__main__':
    symbols = create_user_symbols()
    with open('user_defined_symbols.txt', 'w') as f:
        for symbol in symbols:
            f.write(f'{symbol}\n')
    print(f"Created user_defined_symbols.txt with {len(symbols)} symbols")