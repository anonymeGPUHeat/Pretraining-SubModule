import time, random, math, yaml, torch
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
from typing import Optional
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from ..architecture.complete_model import PTXTransformerForPretraining
from ..preprocessing.dataset_builder import PTXDataset
from .dataloader import create_pretraining_dataloader
from .objectives import PretrainingObjectives, compute_mlm_accuracy, compute_mlm_perplexity, compute_itc_accuracy
from .callbacks import ( TrainingState, LoggingCallback, TensorBoardCallback,
    CheckpointCallback, EarlyStoppingCallback, CallbackManager,)



def _cosine_warmup_scheduler(optimizer, warmup_steps, total_steps):
    """Manual cosine schedule with linear warmup (fallback if transformers missing)"""
    from torch.optim.lr_scheduler import LambdaLR
    print("Using manual cosine warmup scheduler (install transformers for built-in version)")
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate(model, val_dataloader, objectives, device, use_amp=True, loss_weights=None):
    model.eval()
    val_loss_sum = 0.0
    val_mlm_loss_sum = 0.0
    val_itc_loss_sum = 0.0
    val_steps = 0
    mlm_correct = 0
    mlm_total = 0
    itc_correct = 0
    itc_total = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation", leave=False, dynamic_ncols=True):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            mlm_labels = batch['mlm_labels'].to(device, non_blocking=True)
            itc_labels = batch.get('itc_labels')
            if itc_labels is not None:
                itc_labels = itc_labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(input_ids, attention_mask=attention_mask)
                loss_dict = objectives.compute_loss(
                    mlm_logits=outputs['logits'], mlm_labels=mlm_labels,
                    itc_logits=outputs.get('itc_logits'), itc_labels=itc_labels,
                    weights=loss_weights)
                loss = loss_dict['total_loss']
            val_loss_sum += loss.item()
            if 'mlm_loss' in loss_dict:
                val_mlm_loss_sum += loss_dict['mlm_loss'].item()
            if 'itc_loss' in loss_dict:
                val_itc_loss_sum += loss_dict['itc_loss'].item()
            val_steps += 1
            mlm_correct += (outputs['logits'].argmax(dim=-1) == mlm_labels).masked_select(mlm_labels != -100).sum().item()
            mlm_total += (mlm_labels != -100).sum().item()
            if itc_labels is not None and 'itc_logits' in outputs:
                itc_correct += (outputs['itc_logits'].argmax(dim=-1) == itc_labels).masked_select(itc_labels != -100).sum().item()
                itc_total += (itc_labels != -100).sum().item()
    val_loss = val_loss_sum / max(val_steps, 1)
    val_mlm_loss = val_mlm_loss_sum / max(val_steps, 1)
    val_accuracy = mlm_correct / max(mlm_total, 1)      
    val_perplexity = compute_mlm_perplexity(val_mlm_loss) 
    result = {
        'val_loss': val_loss,
        'val_mlm_loss': val_mlm_loss,
        'val_accuracy': val_accuracy,
        'val_perplexity': val_perplexity,
    }
    if itc_total > 0:
        val_itc_loss = val_itc_loss_sum / max(val_steps, 1)
        result['val_itc_loss'] = val_itc_loss
        result['val_itc_accuracy'] = itc_correct / max(itc_total, 1)
    return result



def train( data_dir: str = '~/processed',cache_dir: str = '~/model/cache',
    tokenizer_path: str = '~/tokenizer/ptx_tokenizer.model',
    vocab_size: int = 8000,d_model: int = 768,num_layers: int = 12,num_heads: int = 8,
    d_ff: int = 3072,max_seq_length: int = 2048,dropout: float = 0.1,
    num_epochs: int = 100,batch_size: int = 16,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-5,weight_decay: float = 0.01,
    adam_beta1: float = 0.9,adam_beta2: float = 0.999,adam_epsilon: float = 1e-8,
    max_grad_norm: float = 1.0,warmup_steps: int = 10000,mask_prob: float = 0.15,
    label_smoothing: float = 0.0,seed: int = 42,
    use_amp: bool = True,logging_steps: int = 100,histogram_steps: int = 1000,
    checkpoint_dir: str = '~/model/checkpoints',
    log_dir: str = '~/model/logs',
    save_steps: int = 5000,keep_last_n: int = 3,resume_from: Optional[str] = None,
    early_stopping_patience: int = 5,early_stopping_min_delta: float = 1e-4,
    num_workers: int = 20,pin_memory: bool = True,overlap: int = 128,
    max_files: Optional[int] = None,config_path: Optional[str] = None,
    training_mode: str = 'instruction_mlm',
    mlm_weight: float = 1.0, itc_weight: float = 1.0,
    warm_start_from: Optional[str] = None,):
    """
    PTX Transformer Pre-training.

    Args:
        training_mode: One of
            - ``'classic_mlm'``      – per-token random masking, MLM only
            - ``'instruction_mlm'``  – per-instruction masking, MLM only (default)
            - ``'mlm_itc'``          – per-instruction masking + ITC auxiliary head
        mlm_weight: Weight for the MLM loss (used in ``'mlm_itc'`` mode).
        itc_weight: Weight for the ITC loss (used in ``'mlm_itc'`` mode).
    """
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            yaml_cfg = yaml.safe_load(f) or {}
        data_dir                    = yaml_cfg.get('data_dir',                    data_dir)
        cache_dir                   = yaml_cfg.get('cache_dir',                   cache_dir)
        tokenizer_path              = yaml_cfg.get('tokenizer_path',              tokenizer_path)
        vocab_size                  = yaml_cfg.get('vocab_size',                  vocab_size)
        d_model                     = yaml_cfg.get('d_model',                     d_model)
        num_layers                  = yaml_cfg.get('num_layers',                  num_layers)
        num_heads                   = yaml_cfg.get('num_heads',                   num_heads)
        d_ff                        = yaml_cfg.get('d_ff',                        d_ff)
        max_seq_length              = yaml_cfg.get('max_seq_length',              max_seq_length)
        dropout                     = yaml_cfg.get('dropout',                     dropout)
        num_epochs                  = yaml_cfg.get('num_epochs',                  num_epochs)
        batch_size                  = yaml_cfg.get('batch_size',                  batch_size)
        gradient_accumulation_steps = yaml_cfg.get('gradient_accumulation_steps', gradient_accumulation_steps)
        learning_rate               = yaml_cfg.get('learning_rate',               learning_rate)
        weight_decay                = yaml_cfg.get('weight_decay',                weight_decay)
        adam_beta1                  = yaml_cfg.get('adam_beta1',                  adam_beta1)
        adam_beta2                  = yaml_cfg.get('adam_beta2',                  adam_beta2)
        adam_epsilon                = yaml_cfg.get('adam_epsilon',                adam_epsilon)
        max_grad_norm               = yaml_cfg.get('max_grad_norm',               max_grad_norm)
        warmup_steps                = yaml_cfg.get('warmup_steps',                warmup_steps)
        mask_prob                   = yaml_cfg.get('mask_prob',                   mask_prob)
        label_smoothing             = yaml_cfg.get('label_smoothing',             label_smoothing)
        use_amp                     = yaml_cfg.get('use_amp',                     use_amp)
        logging_steps               = yaml_cfg.get('logging_steps',               logging_steps)
        histogram_steps             = yaml_cfg.get('histogram_steps',             histogram_steps)
        checkpoint_dir              = yaml_cfg.get('checkpoint_dir',              checkpoint_dir)
        log_dir                     = yaml_cfg.get('log_dir',                     log_dir)
        save_steps                  = yaml_cfg.get('save_steps',                  save_steps)
        keep_last_n                 = yaml_cfg.get('keep_last_n',                 keep_last_n)
        resume_from                 = yaml_cfg.get('resume_from',                 resume_from)
        early_stopping_patience     = yaml_cfg.get('early_stopping_patience',     early_stopping_patience)
        early_stopping_min_delta    = yaml_cfg.get('early_stopping_min_delta',    early_stopping_min_delta)
        num_workers                 = yaml_cfg.get('num_workers',                 num_workers)
        pin_memory                  = yaml_cfg.get('pin_memory',                  pin_memory)
        overlap                     = yaml_cfg.get('overlap',                     overlap)
        max_files                   = yaml_cfg.get('max_files',                   max_files)
        seed                        = yaml_cfg.get('seed',                        seed)
        training_mode               = yaml_cfg.get('training_mode',               training_mode)
        mlm_weight                  = yaml_cfg.get('mlm_weight',                  mlm_weight)
        itc_weight                  = yaml_cfg.get('itc_weight',                  itc_weight)
        warm_start_from             = yaml_cfg.get('warm_start_from',             warm_start_from)

    # ── Validate training_mode ──
    VALID_MODES = ('classic_mlm', 'instruction_mlm', 'mlm_itc')
    if training_mode not in VALID_MODES:
        raise ValueError(f"training_mode must be one of {VALID_MODES}, got '{training_mode}'")

    # Derive flags from training_mode
    use_itc = (training_mode == 'mlm_itc')
    masking_mode = 'token' if training_mode == 'classic_mlm' else 'instruction'

    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    effective_batch_size = batch_size * gradient_accumulation_steps
    checkpoint_dir = Path(checkpoint_dir)
    log_dir = Path(log_dir)

    print(f"PTX Transformer Pre-training\n")
    print(f"Training mode: {training_mode}")
    print(f"  Masking: {'per-token (classic)' if masking_mode == 'token' else 'per-instruction'}")
    print(f"  ITC head: {'enabled' if use_itc else 'disabled'}")
    if use_itc:
        print(f"  Loss weights: MLM={mlm_weight}, ITC={itc_weight}")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    from ..tokenizer.tokenizer import PTXTokenizer
    tokenizer = PTXTokenizer(str(Path(tokenizer_path).expanduser()))
    vocab_size = tokenizer.vocab_size 
    print(f"Tokenizer loaded: vocab_size={vocab_size}")
    print(f"\nLoading normalized PTX datasets from {data_dir} ...")
    print("Creating train dataset...")
    train_dataset = PTXDataset(  tokenizer=tokenizer, 
        data_dir=Path(data_dir).expanduser(),  max_seq_length=max_seq_length,overlap=overlap,  
        cache_dir=Path(cache_dir).expanduser(),    max_files=max_files,
        verbose=True, split='train',  seed=seed, )
    print("\nCreating validation dataset...")
    val_dataset = PTXDataset(tokenizer=tokenizer, data_dir=Path(data_dir).expanduser(), 
        max_seq_length=max_seq_length, overlap=overlap,   cache_dir=Path(cache_dir).expanduser(),   max_files=max_files,
        verbose=True,  split='val',  seed=seed, ) 
    train_dataloader = create_pretraining_dataloader(
        dataset=train_dataset,tokenizer=tokenizer, 
        batch_size=batch_size, mask_prob=mask_prob,
        shuffle=True, num_workers=num_workers,pin_memory=pin_memory, seed=seed,
        use_itc=use_itc, masking_mode=masking_mode,)
    val_dataloader = create_pretraining_dataloader(dataset=val_dataset, tokenizer=tokenizer, 
        batch_size=batch_size,   mask_prob=mask_prob,  shuffle=False,  
        num_workers=num_workers,pin_memory=pin_memory, seed=seed,
        use_itc=use_itc, masking_mode=masking_mode,)
    print(f"\nTrain: {len(train_dataset):,} chunks → {len(train_dataloader):,} batches/epoch")
    print(f"Val:   {len(val_dataset):,} chunks → {len(val_dataloader):,} batches/epoch")
    model = PTXTransformerForPretraining(vocab_size=vocab_size,d_model=d_model,
        num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, dropout=dropout,
        max_seq_length=max_seq_length, padding_idx=tokenizer.pad_id,
        use_itc=use_itc,).to(device)
    param_counts = model.count_parameters()
    print(f"\nModel: {param_counts['total']:,} parameters")
    for k, v in param_counts.items():
        print(f"{k}: {v:,}")

    no_decay = {'bias', 'LayerNorm.weight', 'LayerNorm.bias', 'layer_norm.weight', 'layer_norm.bias'}
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    optimizer = AdamW(param_groups, lr=learning_rate, betas=(adam_beta1, adam_beta2), eps=adam_epsilon)

    steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    total_steps = steps_per_epoch * num_epochs
    try:
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    except ImportError:
        scheduler = _cosine_warmup_scheduler(optimizer, warmup_steps, total_steps)

    print(f"\nScheduler: cosine warmup ({warmup_steps:,} warmup / {total_steps:,} total steps)")

    use_amp = use_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    print(f"AMP: {'enabled' if use_amp else 'disabled'}")

    objectives = PretrainingObjectives(label_smoothing=label_smoothing)
    loss_weights = {'mlm': mlm_weight}
    if use_itc:
        loss_weights['itc'] = itc_weight

    callbacks = CallbackManager()
    callbacks.add(EarlyStoppingCallback(patience=early_stopping_patience, min_delta=early_stopping_min_delta))
    callbacks.add(LoggingCallback(log_dir, logging_steps))
    tb_dir = log_dir / 'tensorboard'
    callbacks.add(TensorBoardCallback(tb_dir, logging_steps, histogram_steps))
    print(f"TensorBoard: {tb_dir}")
    callbacks.add(CheckpointCallback(checkpoint_dir, save_steps, save_best=True, keep_last_n=keep_last_n))
    state = TrainingState()
    start_epoch = 0
    if resume_from and Path(resume_from).exists():
        state = CheckpointCallback.load_checkpoint(
            Path(resume_from), model, optimizer, scheduler, scaler, device)
        start_epoch = state.epoch
        print(f"Resumed from epoch {start_epoch}, step {state.global_step}")
        state.training_start_time = time.time()
    elif warm_start_from and Path(warm_start_from).exists():
        ckpt = torch.load(warm_start_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if missing:
            print(f"  warm-start: {len(missing)} missing keys (new params will be randomly initialised — expected for new heads)")
        if unexpected:
            print(f"  warm-start: {len(unexpected)} unexpected keys (ignored)")
        print(f"Warm-started encoder weights from {warm_start_from}")
        state.training_start_time = time.time()
    else:
        state.training_start_time = time.time()

    config_dict = {
        'vocab_size': vocab_size, 'd_model': d_model, 'num_layers': num_layers,
        'num_heads': num_heads, 'd_ff': d_ff, 'max_seq_length': max_seq_length,
        'dropout': dropout, 'batch_size': batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'learning_rate': learning_rate, 'weight_decay': weight_decay,
        'warmup_steps': warmup_steps, 'mask_prob': mask_prob,
        'seed': seed, 'tokenizer_path': str(tokenizer_path),
        'training_mode': training_mode, 'masking_mode': masking_mode,
        'use_itc': use_itc, 'mlm_weight': mlm_weight, 'itc_weight': itc_weight,
        'warm_start_from': str(warm_start_from) if warm_start_from else None,
    }

    callbacks.on_train_begin(state=state,
        model_info={
            **param_counts,
            'total_params': param_counts['total'],
            'device': str(device),
            'amp': use_amp,
            'grad_accum': gradient_accumulation_steps,
            'effective_batch_size': effective_batch_size,
            'total_steps': total_steps,
        },)


    print(f"\nStarting training: {num_epochs} epochs, {total_steps:,} optimizer steps\n")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        state.epoch = epoch + 1
        state.epoch_start_time = time.time()
        epoch_loss_sum = 0.0
        epoch_steps = 0
        optimizer.zero_grad()
        batch_idx = -1
        accum_start_time = time.time()
        accum_tokens = 0
        accum_loss_sum = 0.0
        accum_mlm_loss_sum = 0.0
        accum_itc_loss_sum = 0.0
        accum_masked_tokens = 0
        accum_itc_tokens = 0
        accum_mlm_correct = 0
        accum_mlm_total = 0
        accum_itc_correct = 0
        accum_itc_total = 0
        progress = tqdm(train_dataloader,desc=f"Epoch {state.epoch}/{num_epochs}",leave=True,dynamic_ncols=True,)
        for batch_idx, batch in enumerate(progress):
            if batch_idx % gradient_accumulation_steps == 0:
                state.step_start_time = time.time()
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            mlm_labels = batch['mlm_labels'].to(device, non_blocking=True)
            itc_labels = batch.get('itc_labels')
            if itc_labels is not None:
                itc_labels = itc_labels.to(device, non_blocking=True)
            accum_tokens += attention_mask.sum().item()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(input_ids, attention_mask=attention_mask)
                loss_dict = objectives.compute_loss(
                    mlm_logits=outputs['logits'], mlm_labels=mlm_labels,
                    itc_logits=outputs.get('itc_logits'), itc_labels=itc_labels,
                    weights=loss_weights)
                loss = loss_dict['total_loss'] / gradient_accumulation_steps
            scaler.scale(loss).backward()
            accum_loss_sum += loss_dict['total_loss'].item()
            if 'mlm_loss' in loss_dict:
                accum_mlm_loss_sum += loss_dict['mlm_loss'].item()
            if 'itc_loss' in loss_dict:
                accum_itc_loss_sum += loss_dict['itc_loss'].item()
            with torch.no_grad():
                _mlm_mask = (mlm_labels != -100)
                accum_masked_tokens += _mlm_mask.sum().item()
                accum_mlm_correct += (outputs['logits'].argmax(dim=-1) == mlm_labels).masked_select(_mlm_mask).sum().item()
                accum_mlm_total += _mlm_mask.sum().item()
                if itc_labels is not None and 'itc_logits' in outputs:
                    _itc_mask = (itc_labels != -100)
                    accum_itc_tokens += _itc_mask.sum().item()
                    accum_itc_correct += (outputs['itc_logits'].argmax(dim=-1) == itc_labels).masked_select(_itc_mask).sum().item()
                    accum_itc_total += _itc_mask.sum().item()
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                state.global_step += 1
                step_loss = accum_loss_sum / gradient_accumulation_steps
                mlm_loss_val = accum_mlm_loss_sum / gradient_accumulation_steps
                itc_loss_val = accum_itc_loss_sum / gradient_accumulation_steps
                accum_loss_sum = 0.0
                accum_mlm_loss_sum = 0.0
                accum_itc_loss_sum = 0.0
                state.recent_losses.append(step_loss)
                epoch_loss_sum += step_loss
                epoch_steps += 1
                metrics = {
                    'grad_norm': grad_norm,
                    'masked_tokens': accum_masked_tokens,
                    'mlm_loss': mlm_loss_val,
                    'step_loss': step_loss,
                }
                if use_itc:
                    metrics['itc_loss'] = itc_loss_val
                if accum_itc_tokens > 0:
                    metrics['itc_tokens'] = accum_itc_tokens
                if state.global_step % logging_steps == 0:
                    metrics['mlm_accuracy'] = accum_mlm_correct / max(accum_mlm_total, 1)
                    metrics['perplexity'] = compute_mlm_perplexity(mlm_loss_val)
                    if accum_itc_total > 0:
                        metrics['itc_accuracy'] = accum_itc_correct / max(accum_itc_total, 1)
                    accum_elapsed = time.time() - accum_start_time
                    metrics['tokens_per_sec'] = accum_tokens / max(accum_elapsed, 1e-6)
                    accum_start_time = time.time()
                    accum_tokens = 0
                accum_masked_tokens = 0
                accum_itc_tokens = 0
                accum_mlm_correct = 0
                accum_mlm_total = 0
                accum_itc_correct = 0
                accum_itc_total = 0
                lr = scheduler.get_last_lr()[0]
                callbacks.on_step(state=state,metrics=metrics,lr=lr,current_loss=step_loss,model=model,optimizer=optimizer,
                    scheduler=scheduler,scaler=scaler,config=config_dict,)
                window = min(logging_steps, len(state.recent_losses))
                avg = sum(state.recent_losses[-window:]) / window
                progress.set_postfix({
                    'loss': f'{avg:.4f}',
                    'lr': f'{lr:.2e}',
                    'step': state.global_step,
                })
        if batch_idx >= 0 and (batch_idx + 1) % gradient_accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            state.global_step += 1
            leftover_batches = (batch_idx + 1) % gradient_accumulation_steps
            step_loss = accum_loss_sum / leftover_batches
            accum_loss_sum = 0.0
            accum_mlm_loss_sum = 0.0
            accum_itc_loss_sum = 0.0
            accum_masked_tokens = 0
            accum_itc_tokens = 0
            accum_mlm_correct = 0
            accum_mlm_total = 0
            accum_itc_correct = 0
            accum_itc_total = 0
            state.recent_losses.append(step_loss)
            epoch_loss_sum += step_loss
            epoch_steps += 1

        avg_epoch_loss = epoch_loss_sum / max(epoch_steps, 1)
        val_rng_state = random.getstate()
        random.seed(seed * 10000 + epoch + 1)
        val_metrics = validate(model, val_dataloader, objectives, device, use_amp, loss_weights=loss_weights)
        random.setstate(val_rng_state)
        val_loss = val_metrics['val_loss']
        val_acc = val_metrics['val_accuracy']
        val_ppl = val_metrics['val_perplexity']
        state.val_loss_history.append(val_loss)
        state.val_accuracy_history.append(val_acc)
        state.val_perplexity_history.append(val_ppl)
        should_stop = callbacks.on_epoch_end(state=state, train_loss=avg_epoch_loss, val_metrics=val_metrics, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, config=config_dict,)
        is_best = state.is_new_best
        epoch_msg = (f"Epoch {state.epoch} - Train Loss: {avg_epoch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val MLM Acc: {val_acc:.4f}, ")
        if 'val_itc_accuracy' in val_metrics:
            epoch_msg += f"Val ITC Acc: {val_metrics['val_itc_accuracy']:.4f}, "
        epoch_msg += f"Val PPL: {val_ppl:.4f}"
        if is_best:
            epoch_msg += " [BEST]"
        print(epoch_msg)
        if should_stop:
            print(f"\nEarly stopping at epoch {state.epoch}.")
            break
        
    callbacks.on_training_end(state=state, model=model, config=config_dict)
    total_time = time.time() - state.training_start_time
    print(f"\nTraining complete!")
    print(f"  Total steps:     {state.global_step:,}")
    print(f"  Best loss:       {state.best_loss:.4f} (epoch {state.best_epoch})")
    print(f"  Best accuracy:   {state.best_val_accuracy:.4f}")
    print(f"  Best perplexity: {state.best_val_perplexity:.4f}")
    print(f"  Total time:      {timedelta(seconds=int(total_time))}")
    print(f"  Checkpoints:     {checkpoint_dir}")
    print(f"  TensorBoard:     tensorboard --logdir {log_dir / 'tensorboard'}")




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PTX Transformer Pre-training')
    parser.add_argument('--data-dir', type=str, default='data/sprocessed/sprocessed')
    parser.add_argument('--cache-dir', type=str, default='data/cache')
    parser.add_argument('--tokenizer-path', type=str, default='data/tokenizer/ptx_tokenizer.model')
    parser.add_argument('--max-files', type=int, default=None, help='Limit files (for testing)')
    parser.add_argument('--vocab-size', type=int, default=8000)
    parser.add_argument('--d-model', type=int, default=768)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--d-ff', type=int, default=3072)
    parser.add_argument('--max-seq-length', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--warmup-steps', type=int, default=10000)
    parser.add_argument('--mask-prob', type=float, default=0.15)
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--logging-steps', type=int, default=100)
    parser.add_argument('--save-steps', type=int, default=5000)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--resume-from', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=20)
    parser.add_argument('--overlap', type=int, default=128)
    parser.add_argument('--config', type=str, default=None, help='YAML config file (overrides CLI args)')
    parser.add_argument('--training-mode', type=str, default='instruction_mlm',
                        choices=['classic_mlm', 'instruction_mlm', 'mlm_itc'],
                        help='Training mode: classic_mlm (per-token), instruction_mlm (per-instruction), mlm_itc (instruction MLM + ITC)')
    parser.add_argument('--mlm-weight', type=float, default=1.0, help='MLM loss weight (for mlm_itc mode)')
    parser.add_argument('--itc-weight', type=float, default=1.0, help='ITC loss weight (for mlm_itc mode)')
    parser.add_argument('--warm-start-from', type=str, default=None,
                        help='Load encoder weights only from a checkpoint (no optimizer/scheduler restore). Use to switch training mode, e.g. instruction_mlm → mlm_itc.')
    args = parser.parse_args()
    train(data_dir=args.data_dir,cache_dir=args.cache_dir,tokenizer_path=args.tokenizer_path,vocab_size=args.vocab_size,
        d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads, d_ff=args.d_ff,
        max_seq_length=args.max_seq_length,dropout=args.dropout,num_epochs=args.num_epochs,
        batch_size=args.batch_size,gradient_accumulation_steps=args.grad_accum,learning_rate=args.lr,
        weight_decay=args.weight_decay,max_grad_norm=args.max_grad_norm,warmup_steps=args.warmup_steps,mask_prob=args.mask_prob,
        label_smoothing=args.label_smoothing,seed=args.seed,use_amp=not args.no_amp,logging_steps=args.logging_steps,
        save_steps=args.save_steps,checkpoint_dir=args.checkpoint_dir,log_dir=args.log_dir,resume_from=args.resume_from,
        early_stopping_patience=args.patience,num_workers=args.num_workers,overlap=args.overlap,max_files=args.max_files,
        config_path=args.config,training_mode=args.training_mode,
        mlm_weight=args.mlm_weight,itc_weight=args.itc_weight,
        warm_start_from=args.warm_start_from,)
