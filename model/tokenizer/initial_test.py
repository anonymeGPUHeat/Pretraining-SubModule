import sentencepiece as spm

ptx_corpus = """add.s32 %r1, %r2, %r3;
ld.global.f32 %f1, [%r4+12];
mov.b32 %r5, %r1;
bar.sync 0;
ret;""" * 1000

with open('ptx_corpus.txt', 'w') as f:
    f.write(ptx_corpus)

spm.SentencePieceTrainer.train(
    input='ptx_corpus.txt',
    model_prefix='ptx',
    vocab_size=97,
    model_type='bpe'
)

sp = spm.SentencePieceProcessor(model_file='ptx.model')
ptx_test = 'add.s32 %r1, %r2, %r3; ld.global.f32 %f1, [%r4+12];'
print('Tokens:', sp.encode_as_pieces(ptx_test))
print('Decoded:', sp.decode(sp.encode(ptx_test)))
