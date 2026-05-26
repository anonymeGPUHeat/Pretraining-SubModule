#!/bin/bash
# =============================================================================
# PTX Transformer Pre-training
# =============================================================================
# Run from project root:  bash scripts/train.sh
#
# Prerequisites:
#   pip install torch numpy tqdm pyyaml sentencepiece transformers tensorboard
# =============================================================================

# ---------------------------------------------------------------------------
# Quick smoke test  (10 files, 2 epochs — validates entire pipeline)
# ---------------------------------------------------------------------------
# python model/training/train.py \
#     --data-dir ./data/cleaned \
#     --cache-dir ./data/cache \
#     --tokenizer-path ./ptx_tokenizer.model \
#     --max-files 10 \
#     --num-epochs 2 \
#     --batch-size 4 \
#     --grad-accum 1 \
#     --num-workers 0 \
#     --warmup-steps 10 \
#     --logging-steps 5 \
#     --save-steps 50 \
#     --checkpoint-dir ./checkpoints_test \
#     --log-dir ./logs_test

# ---------------------------------------------------------------------------
# Full training
# ---------------------------------------------------------------------------
python model.training.train \
    --data-dir ~/processed \
    --cache-dir ~/model/cache \
    --tokenizer-path ~/tokenizer_16k/ptx_tokenizer_16k.model \
    --vocab-size 16000 \
    --d-model 768 \
    --num-layers 12 \
    --num-heads 12 \
    --d-ff 3072 \
    --max-seq-length 2048 \
    --dropout 0.1 \
    --num-epochs 100 \
    --batch-size 16 \
    --grad-accum 4 \
    --lr 5e-5 \
    --weight-decay 0.01 \
    --warmup-steps 1000 \
    --mask-prob 0.15 \
    --label-smoothing 0.1 \
    --seed 42 \
    --logging-steps 100 \
    --save-steps 5000 \
    --checkpoint-dir ~/model/checkpoints \
    --log-dir ~/model/logs \
    --patience 5 \
    --num-workers 20 \
    --overlap 128

# ---------------------------------------------------------------------------
# Resume from checkpoint (uncomment to use)
# ---------------------------------------------------------------------------
# python model/training/train.py \
#     --data-dir ~/processed \
#     --cache-dir ~/model/cache \
#     --tokenizer-path ~/tokenizer_16k/ptx_tokenizer_16k.model \
#     --resume-from ./checkpoints/best_model.pt \
#     --num-epochs 100 \
#     --batch-size 16 \
#     --grad-accum 4 \
#     --lr 5e-5 \
#     --checkpoint-dir ./checkpoints \
#     --log-dir ./logs

# ---------------------------------------------------------------------------
# TensorBoard monitoring (run in separate terminal)
# ---------------------------------------------------------------------------
# tensorboard --logdir ./logs/tensorboard


python3 -m model.training.train     --data-dir ~/processed     --cache-dir ~/model/cache     --tokenizer-path ~/tokenizer_16k/ptx_tokenizer_16k.model     --vocab-size 16000     --d-model 768     --num-layers 6     --num-heads 8     --d-ff 3072     --max-seq
-length 1024     --dropout 0.1     --num-epochs 4     --batch-size 8     --grad-accum 4     --lr 5e-5     --weight-decay 0.01     --warmup-steps 1000     --
mask-prob 0.15     --label-smoothing 0.1     --seed 42     --logging-steps 100     --save-steps 5000     --checkpoint-dir ~/model/checkpoints     --log-dir
~/model/logs     --patience 5     --num-workers 20     --overlap 128