import torch
import time
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter


#stock the training state
@dataclass
class TrainingState:
    epoch: int = 0
    global_step: int = 0
    best_loss: float = float('inf')
    best_epoch: int = 0
    epochs_no_improve: int = 0
    recent_losses: List[float] = field(default_factory=list)
    training_start_time: float = 0.0
    epoch_start_time: float = 0.0
    step_start_time: float = 0.0
    val_loss_history: List[float] = field(default_factory=list)
    val_accuracy_history: List[float] = field(default_factory=list)
    val_perplexity_history: List[float] = field(default_factory=list)
    best_val_accuracy: float = 0.0
    best_val_perplexity: float = float('inf')
    is_new_best: bool = False
    
    def elapsed(self) -> str:
        secs = time.time() - self.training_start_time
        return str(timedelta(seconds=int(secs)))

    def epoch_elapsed(self) -> str:
        secs = time.time() - self.epoch_start_time
        return str(timedelta(seconds=int(secs)))

    def step_time_ms(self) -> float:
        return (time.time() - self.step_start_time) * 1000



#write in a text file + the console
class LoggingCallback:
    def __init__(self, log_dir: Path, logging_steps: int = 100):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logging_steps = logging_steps
        self.log_file = self.log_dir / 'training.log'

        with open(self.log_file, 'w') as f:
            f.write(f"{'='*90}\n")
            f.write(f"PTX Transformer Pre-training Log\n")
            f.write(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*90}\n\n")

    def _log(self, msg: str, console: bool = True):
        ts = datetime.now().strftime('%H:%M:%S')
        line = f"[{ts}] {msg}"
        if console:
            print(line)
        with open(self.log_file, 'a') as f:
            f.write(line + '\n')

    def on_train_begin(self, state: TrainingState, model_info: Dict):
        self._log(f"Model parameters: {model_info.get('total_params', '?'):,}")
        self._log(f"  Embedding:  {model_info.get('embedding', '?'):,}")
        self._log(f"  Encoder:    {model_info.get('encoder_layers', '?'):,}")
        self._log(f"  MLM head:   {model_info.get('mlm_head', '?'):,}")
        self._log(f"Device: {model_info.get('device', '?')}")
        self._log(f"AMP enabled: {model_info.get('amp', False)}")
        self._log(f"Grad accum steps: {model_info.get('grad_accum', 1)}")
        self._log(f"Effective batch size: {model_info.get('effective_batch_size', '?')}")
        self._log(f"Total training steps: {model_info.get('total_steps', '?'):,}")
        self._log("")

    def on_step(self, state: TrainingState, metrics: Dict[str, float], lr: float, **kwargs):
        if state.global_step % self.logging_steps != 0:
            return

        window = min(self.logging_steps, len(state.recent_losses))
        avg_loss = sum(state.recent_losses[-window:]) / window if window > 0 else 0.0
        step_ms = state.step_time_ms()

        msg = (
            f"step {state.global_step:>7,d} | "
            f"epoch {state.epoch} | "
            f"loss {avg_loss:.4f} | "
            f"lr {lr:.2e} | "
            f"step {step_ms:.0f}ms | "
            f"elapsed {state.elapsed()}"
        )

        if 'mlm_accuracy' in metrics:
            msg += f" | acc {metrics['mlm_accuracy']:.3f}"
        if 'itc_accuracy' in metrics:
            msg += f" | itc_acc {metrics['itc_accuracy']:.3f}"
        if 'perplexity' in metrics:
            msg += f" | ppl {metrics['perplexity']:.1f}"
        if 'grad_norm' in metrics:
            msg += f" | gnorm {metrics['grad_norm']:.2f}"
        if 'tokens_per_sec' in metrics:
            msg += f" | tok/s {metrics['tokens_per_sec']:.0f}"

        self._log(msg)

    def on_epoch_end(self, state: TrainingState, train_loss: float, val_metrics: Dict = None, **kwargs):
        self._log(f"\n{'─'*70}")
        msg = (
            f"Epoch {state.epoch} complete | "
            f"train avg loss {train_loss:.4f} | "
            f"epoch time {state.epoch_elapsed()} | "
            f"total {state.elapsed()}"
        )
        
        if val_metrics:
            msg += f" | val loss {val_metrics.get('val_loss', 0):.4f}"
            msg += f" | val acc {val_metrics.get('val_accuracy', 0):.4f}"
            msg += f" | val ppl {val_metrics.get('val_perplexity', 0):.2f}"
            if state.is_new_best:
                msg += " [BEST]"
        
        self._log(msg)
        self._log(f"{'─'*70}\n")


#write the metrics into tensorboard
class TensorBoardCallback:
    def __init__(self, log_dir: Path, logging_steps: int = 100, histogram_steps: int = 1000):
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.logging_steps = logging_steps
        self.histogram_steps = histogram_steps

    def on_step(self, state: TrainingState, metrics: Dict[str, float], lr: float,
                model=None, **kwargs):
        step = state.global_step

        if step % self.logging_steps != 0:
            return
        window = min(self.logging_steps, len(state.recent_losses))
        avg_loss = sum(state.recent_losses[-window:]) / window if window > 0 else 0.0
        self.writer.add_scalar('train/loss', avg_loss, step)
        self.writer.add_scalar('train/learning_rate', lr, step)
        self.writer.add_scalar('train/step_time_ms', state.step_time_ms(), step)
        if 'mlm_accuracy' in metrics:
            self.writer.add_scalar('train/mlm_accuracy', metrics['mlm_accuracy'], step)
        if 'perplexity' in metrics:
            self.writer.add_scalar('train/perplexity', metrics['perplexity'], step)
        if 'grad_norm' in metrics:
            self.writer.add_scalar('train/grad_norm', metrics['grad_norm'], step)
        if 'tokens_per_sec' in metrics:
            self.writer.add_scalar('train/tokens_per_sec', metrics['tokens_per_sec'], step)
        if 'masked_tokens' in metrics:
            self.writer.add_scalar('train/masked_tokens_per_step', metrics['masked_tokens'], step)
        if 'step_loss' in metrics:
            self.writer.add_scalar('train/step_loss', metrics['step_loss'], step)
        if 'mlm_loss' in metrics:
            self.writer.add_scalar('train/mlm_loss', metrics['mlm_loss'], step)
        if 'itc_loss' in metrics:
            self.writer.add_scalar('train/itc_loss', metrics['itc_loss'], step)
        if 'itc_accuracy' in metrics:
            self.writer.add_scalar('train/itc_accuracy', metrics['itc_accuracy'], step)
        if 'itc_tokens' in metrics:
            self.writer.add_scalar('train/itc_tokens_per_step', metrics['itc_tokens'], step)
        if model is not None and step % self.histogram_steps == 0:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.writer.add_histogram(f'params/{name}', param, step)
                    self.writer.add_histogram(f'grads/{name}', param.grad, step)

    def on_epoch_end(self, state: TrainingState, train_loss: float, val_metrics: Dict = None, **kwargs):
        self.writer.add_scalar('train/epoch_loss', train_loss, state.epoch)
        if val_metrics:
            self.writer.add_scalar('validation/loss', val_metrics.get('val_loss', 0), state.epoch)
            self.writer.add_scalar('validation/accuracy', val_metrics.get('val_accuracy', 0), state.epoch)
            self.writer.add_scalar('validation/perplexity', val_metrics.get('val_perplexity', 0), state.epoch)
            self.writer.add_scalar('validation/best_loss', state.best_loss, state.epoch)
            self.writer.add_scalar('validation/best_accuracy', state.best_val_accuracy, state.epoch)
            self.writer.add_scalar('validation/best_perplexity', state.best_val_perplexity, state.epoch)
            if train_loss > 0:
                self.writer.add_scalar('comparison/train_val_loss_ratio', val_metrics.get('val_loss', 0) / train_loss, state.epoch)
            if 'val_mlm_loss' in val_metrics:
                self.writer.add_scalar('validation/mlm_loss', val_metrics['val_mlm_loss'], state.epoch)
            if 'val_itc_loss' in val_metrics:
                self.writer.add_scalar('validation/itc_loss', val_metrics['val_itc_loss'], state.epoch)
            if 'val_itc_accuracy' in val_metrics:
                self.writer.add_scalar('validation/itc_accuracy', val_metrics['val_itc_accuracy'], state.epoch)

    def on_training_end(self, state: TrainingState = None, **kwargs):
        #log final summary statistics
        if state:
            self.writer.add_text('summary/training_complete', 
                                f"Total epochs: {state.epoch}\n"
                                f"Total steps: {state.global_step}\n"
                                f"Best loss: {state.best_loss:.4f}\n"
                                f"Best epoch: {state.best_epoch}\n"
                                f"Best accuracy: {state.best_val_accuracy:.4f}\n"
                                f"Best perplexity: {state.best_val_perplexity:.4f}")
        self.writer.flush()
        self.writer.close()


#save the best checkpoints (whole model + encoder only)
class CheckpointCallback:
    def __init__( self, checkpoint_dir: Path, save_steps: int = 5000,
        save_best: bool = True,keep_last_n: int = 3,):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_steps = save_steps
        self.save_best = save_best
        self.keep_last_n = keep_last_n
        self.saved_checkpoints: List[Path] = []

    def _save(self,state: TrainingState,model,optimizer,scheduler,scaler,config: Dict,name: str,) -> Path:
        path = self.checkpoint_dir / f'{name}.pt'
        checkpoint = {
            'epoch': state.epoch,
            'global_step': state.global_step,
            'best_loss': state.best_loss,
            'best_epoch': state.best_epoch,
            'best_val_accuracy': state.best_val_accuracy,
            'best_val_perplexity': state.best_val_perplexity,
            'val_loss_history': state.val_loss_history,
            'val_accuracy_history': state.val_accuracy_history,
            'val_perplexity_history': state.val_perplexity_history,
            'epochs_no_improve': state.epochs_no_improve,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'config': config,
            'timestamp': datetime.now().isoformat(),
        }
        torch.save(checkpoint, path)
        return path

    def _save_encoder_only(self, state: TrainingState, model, config: Dict, name: str) -> Path:
        """Save only the encoder (base transformer) without MLM head for fine-tuning."""
        path = self.checkpoint_dir / f'{name}_encoder.pt'
        encoder_state_dict = model.encoder.state_dict()
        encoder_checkpoint = {
            'encoder_state_dict': encoder_state_dict,
            'epoch': state.epoch,
            'global_step': state.global_step,
            'best_loss': state.best_loss,
            'best_epoch': state.best_epoch,
            'best_val_accuracy': state.best_val_accuracy,
            'best_val_perplexity': state.best_val_perplexity,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'note': 'This checkpoint contains only the encoder (base transformer) without the MLM head. '
                   'Load this for fine-tuning with custom heads.'
        }
        torch.save(encoder_checkpoint, path)
        return path

    def _cleanup(self):
        while len(self.saved_checkpoints) > self.keep_last_n:
            old = self.saved_checkpoints.pop(0)
            if old.exists() and 'best' not in old.stem:
                old.unlink()

    def on_step(self, state: TrainingState, current_loss: float,
                model=None, optimizer=None, scheduler=None, scaler=None,
                config: Dict = None, **kwargs):
        if state.global_step > 0 and state.global_step % self.save_steps == 0:
            path = self._save(state, model, optimizer, scheduler, scaler, config,
                              f'checkpoint-step{state.global_step}')
            self.saved_checkpoints.append(path)
            self._cleanup()
            ts = datetime.now().strftime('%H:%M:%S')
            print(f"[{ts}] Checkpoint saved: {path.name}")
            encoder_path = self._save_encoder_only(state, model, config, f'checkpoint-step{state.global_step}')
            print(f"[{ts}] Encoder-only checkpoint saved: {encoder_path.name}")
    
    def on_epoch_end(self, state: TrainingState, val_metrics: Dict = None,
        model=None, optimizer=None, scheduler=None, scaler=None,config: Dict = None, **kwargs):
        if not self.save_best or val_metrics is None:
            return
        if state.is_new_best:
            val_loss = val_metrics.get('val_loss', float('inf'))
            path = self._save(state, model, optimizer, scheduler, scaler, config, 'best_model')
            ts = datetime.now().strftime('%H:%M:%S')
            print(f"[{ts}] Best model saved (val_loss {val_loss:.4f}): {path.name}")
            encoder_path = self._save_encoder_only(state, model, config, 'best_model')
            print(f"[{ts}] Best encoder saved: {encoder_path.name}")

    def on_training_end(self, state: TrainingState, model=None, config: Dict = None, **kwargs):
        ts = datetime.now().strftime('%H:%M:%S')
        path = self.checkpoint_dir / 'final_model.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'total_steps': state.global_step,
            'total_epochs': state.epoch,
            'final_loss': state.best_loss,
            'best_epoch': state.best_epoch,
            'best_val_accuracy': state.best_val_accuracy,
            'best_val_perplexity': state.best_val_perplexity,
            'val_loss_history': state.val_loss_history,
            'val_accuracy_history': state.val_accuracy_history,
            'val_perplexity_history': state.val_perplexity_history,
            'timestamp': datetime.now().isoformat(),
        }, path)
        print(f"[{ts}] Final model saved: {path}")
        encoder_path = self._save_encoder_only(state, model, config, 'final_model')
        print(f"[{ts}] Final encoder (for fine-tuning) saved: {encoder_path}")
        metrics_path = self.checkpoint_dir / 'training_metrics.pt'
        torch.save({
            'val_loss_history': state.val_loss_history,
            'val_accuracy_history': state.val_accuracy_history,
            'val_perplexity_history': state.val_perplexity_history,
            'best_loss': state.best_loss,
            'best_epoch': state.best_epoch,
            'best_val_accuracy': state.best_val_accuracy,
            'best_val_perplexity': state.best_val_perplexity,
            'total_epochs': state.epoch,
            'total_steps': state.global_step,
            'timestamp': datetime.now().isoformat(),
        }, metrics_path)
        print(f"[{ts}] Training metrics saved: {metrics_path}")

    @staticmethod
    def load_checkpoint(path: Path, model, optimizer=None, scheduler=None, scaler=None, device='cuda'):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        # Use strict=False to allow resuming across different training modes
        # (e.g. a checkpoint with ITC head loaded into a model without one)
        missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if missing:
            print(f"  Checkpoint: {len(missing)} missing keys (expected if switching training mode)")
        if unexpected:
            print(f"  Checkpoint: {len(unexpected)} unexpected keys (expected if switching training mode)")
        if optimizer and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler and ckpt.get('scheduler_state_dict'):
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if scaler and ckpt.get('scaler_state_dict'):
            scaler.load_state_dict(ckpt['scaler_state_dict'])

        state = TrainingState(
            epoch=ckpt.get('epoch', 0),
            global_step=ckpt.get('global_step', 0),
            best_loss=ckpt.get('best_loss', float('inf')),
            best_epoch=ckpt.get('best_epoch', 0),
            best_val_accuracy=ckpt.get('best_val_accuracy', 0.0),
            best_val_perplexity=ckpt.get('best_val_perplexity', float('inf')),
            val_loss_history=ckpt.get('val_loss_history', []),
            val_accuracy_history=ckpt.get('val_accuracy_history', []),
            val_perplexity_history=ckpt.get('val_perplexity_history', []),
            epochs_no_improve=ckpt.get('epochs_no_improve', 0),
        )
        print(f"Resumed from checkpoint: step {state.global_step}, epoch {state.epoch}, "
              f"best_loss {state.best_loss:.4f}, best_acc {state.best_val_accuracy:.4f}")
        return state


class EarlyStoppingCallback:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta

    def on_epoch_end(self, state: TrainingState, train_loss: float, val_metrics: Dict = None, **kwargs) -> bool:
        current_loss = val_metrics.get('val_loss', train_loss) if val_metrics else train_loss
        if current_loss < state.best_loss - self.min_delta:
            state.best_loss = current_loss
            state.best_epoch = state.epoch
            state.epochs_no_improve = 0
            state.is_new_best = True
            if val_metrics:
                val_acc = val_metrics.get('val_accuracy', 0.0)
                val_ppl = val_metrics.get('val_perplexity', float('inf'))
                if val_acc > state.best_val_accuracy:
                    state.best_val_accuracy = val_acc
                if val_ppl < state.best_val_perplexity:
                    state.best_val_perplexity = val_ppl
            return False
        state.epochs_no_improve += 1
        state.is_new_best = False
        ts = datetime.now().strftime('%H:%M:%S')
        if state.epochs_no_improve >= self.patience:
            print(f"[{ts}] Early stopping: no improvement for "
                  f"{state.epochs_no_improve} epochs (patience={self.patience})")
            return True
        print(f"[{ts}] No improvement: {state.epochs_no_improve}/{self.patience}")
        return False



class CallbackManager:
    def __init__(self):
        self.callbacks = []

    def add(self, callback):
        self.callbacks.append(callback)

    def on_train_begin(self, **kwargs):
        for cb in self.callbacks:
            if hasattr(cb, 'on_train_begin'):
                cb.on_train_begin(**kwargs)

    def on_step(self, **kwargs):
        for cb in self.callbacks:
            if hasattr(cb, 'on_step'):
                cb.on_step(**kwargs)

    def on_epoch_end(self, **kwargs) -> bool:
        should_stop = False
        for cb in self.callbacks:
            if hasattr(cb, 'on_epoch_end'):
                result = cb.on_epoch_end(**kwargs)
                if result is True:
                    should_stop = True
        return should_stop

    def on_training_end(self, **kwargs):
        for cb in self.callbacks:
            if hasattr(cb, 'on_training_end'):
                cb.on_training_end(**kwargs)
