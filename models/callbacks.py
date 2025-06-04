"""
Clean training callbacks for VAE model training.
"""

import os
import torch
import numpy as np
from copy import deepcopy
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class Callback:
    """Base callback class for training events."""
    
    def __init__(self):
        self.model = None
        self.params = {}
    
    def set_model(self, model: torch.nn.Module) -> None:
        """Set the model being trained."""
        self.model = model
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set training parameters."""
        self.params = params
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of each batch."""
        pass

class EarlyStopping(Callback):
    """
    Stop training when a monitored metric has stopped improving.
    
    Args:
        monitor: Quantity to monitor (e.g., 'val_loss')
        min_delta: Minimum change to qualify as improvement
        patience: Number of epochs with no improvement to stop training
        verbose: Whether to print messages
        mode: 'min' or 'max' for minimizing or maximizing the monitored quantity
        restore_best_weights: Whether to restore model weights from best epoch
        baseline: Baseline value for the monitored quantity
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        min_delta: float = 0.0,
        patience: int = 10,
        verbose: bool = True,
        mode: str = 'min',
        restore_best_weights: bool = True,
        baseline: Optional[float] = None
    ):
        super().__init__()
        
        self.monitor = monitor
        self.min_delta = abs(min_delta)
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.baseline = baseline
        
        # Internal state
        self.best_weights = None
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.stop_training = False
        
        # Set comparison function and initial best value
        if mode == 'min':
            self.monitor_op = self._is_improvement_min
            self.best = float('inf') if baseline is None else baseline
        elif mode == 'max':
            self.monitor_op = self._is_improvement_max
            self.best = float('-inf') if baseline is None else baseline
        else:
            raise ValueError(f"Mode {mode} is not supported, use 'min' or 'max'")
        
        logger.info(f"EarlyStopping initialized: monitor={monitor}, patience={patience}, mode={mode}")
    
    def _is_improvement_min(self, current: float, best: float) -> bool:
        """Check if current value is improvement for minimization."""
        return current < best - self.min_delta
    
    def _is_improvement_max(self, current: float, best: float) -> bool:
        """Check if current value is improvement for maximization."""
        return current > best + self.min_delta
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Reset state at the beginning of training."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.stop_training = False
        
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = float('inf') if self.mode == 'min' else float('-inf')
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Check for improvement at the end of each epoch."""
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            if self.verbose:
                logger.warning(f"EarlyStopping: {self.monitor} not found in logs")
            return
        
        if self.monitor_op(current, self.best):
            # Improvement found
            self.best = current
            self.best_epoch = epoch
            self.wait = 0
            
            if self.restore_best_weights and self.model is not None:
                self.best_weights = deepcopy(self.model.state_dict())
            
            if self.verbose:
                logger.info(f"EarlyStopping: {self.monitor} improved to {current:.6f}")
        
        else:
            # No improvement
            self.wait += 1
            
            if self.verbose:
                logger.info(f"EarlyStopping: no improvement for {self.wait}/{self.patience} epochs, "
                          f"best {self.monitor}: {self.best:.6f} (epoch {self.best_epoch})")
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                
                if self.restore_best_weights and self.best_weights is not None and self.model is not None:
                    if self.verbose:
                        logger.info(f"EarlyStopping: restoring model weights from epoch {self.best_epoch}")
                    self.model.load_state_dict(self.best_weights)
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log final early stopping information."""
        if self.stopped_epoch > 0 and self.verbose:
            logger.info(f"EarlyStopping: training stopped at epoch {self.stopped_epoch}")
        elif self.verbose:
            logger.info(f"EarlyStopping: training completed without early stopping")

class ModelCheckpoint(Callback):
    """
    Save the model after every epoch or when a monitored metric improves.
    
    Args:
        filepath: Path to save the model file (supports formatting)
        monitor: Quantity to monitor
        verbose: Whether to print messages
        save_best_only: Only save when monitor quantity is best so far
        save_weights_only: If True, only save model weights
        mode: 'min' or 'max' for minimizing or maximizing the monitored quantity
        period: Interval between checkpoints
        save_optimizer: Whether to save optimizer state
    """
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        verbose: bool = True,
        save_best_only: bool = False,
        save_weights_only: bool = False,
        mode: str = 'min',
        period: int = 1,
        save_optimizer: bool = False
    ):
        super().__init__()
        
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.period = period
        self.save_optimizer = save_optimizer
        
        # Internal state
        self.epochs_since_last_save = 0
        self.best = None
        self.best_epoch = 0
        
        # Set comparison function and initial best value
        if mode == 'min':
            self.monitor_op = lambda x, y: x < y
            self.best = float('inf')
        elif mode == 'max':
            self.monitor_op = lambda x, y: x > y
            self.best = float('-inf')
        else:
            raise ValueError(f"Mode {mode} is not supported, use 'min' or 'max'")
        
        logger.info(f"ModelCheckpoint initialized: monitor={monitor}, save_best_only={save_best_only}")
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Reset state at the beginning of training."""
        self.epochs_since_last_save = 0
        self.best = float('inf') if self.mode == 'min' else float('-inf')
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Save model at the end of each epoch if conditions are met."""
        logs = logs or {}
        self.epochs_since_last_save += 1
        
        # Format filepath with epoch and logs
        save_path = self.filepath.format(epoch=epoch + 1, **logs)
        
        should_save = False
        
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                if self.verbose:
                    logger.warning(f"ModelCheckpoint: {self.monitor} not found in logs")
                return
            
            if self.monitor_op(current, self.best):
                self.best = current
                self.best_epoch = epoch
                should_save = True
                
                if self.verbose:
                    logger.info(f"ModelCheckpoint: {self.monitor} improved from {self.best:.6f} to {current:.6f}")
        
        else:
            # Save every period epochs
            if self.epochs_since_last_save >= self.period:
                should_save = True
        
        if should_save:
            self._save_model(save_path, epoch, logs)
            self.epochs_since_last_save = 0
    
    def _save_model(self, filepath: str, epoch: int, logs: Dict[str, Any]) -> None:
        """Save the model to filepath."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            if self.save_weights_only:
                # Save only model weights
                torch.save(self.model.state_dict(), filepath)
            else:
                # Save complete checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'logs': logs
                }
                
                # Add optimizer state if requested and available
                if self.save_optimizer and hasattr(self, 'optimizer'):
                    checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
                
                torch.save(checkpoint, filepath)
            
            if self.verbose:
                logger.info(f"ModelCheckpoint: saved model to {filepath}")
                
        except Exception as e:
            logger.error(f"ModelCheckpoint: failed to save model to {filepath}: {e}")

class ReduceLROnPlateau(Callback):
    """
    Reduce learning rate when a metric has stopped improving.
    
    Args:
        monitor: Quantity to monitor
        factor: Factor by which to reduce the learning rate
        patience: Number of epochs with no improvement to reduce LR
        verbose: Whether to print messages
        mode: 'min' or 'max' for minimizing or maximizing the monitored quantity
        min_delta: Threshold for measuring the new optimum
        cooldown: Number of epochs to wait before resuming normal operation
        min_lr: Lower bound on the learning rate
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        monitor: str = 'val_loss',
        factor: float = 0.1,
        patience: int = 10,
        verbose: bool = True,
        mode: str = 'min',
        min_delta: float = 1e-4,
        cooldown: int = 0,
        min_lr: float = 0
    ):
        super().__init__()
        
        self.optimizer = optimizer
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.min_delta = abs(min_delta)
        self.cooldown = cooldown
        self.min_lr = min_lr
        
        # Internal state
        self.wait = 0
        self.cooldown_counter = 0
        self.best = None
        
        # Set comparison function
        if mode == 'min':
            self.monitor_op = lambda x, y: x < y - self.min_delta
            self.best = float('inf')
        elif mode == 'max':
            self.monitor_op = lambda x, y: x > y + self.min_delta
            self.best = float('-inf')
        else:
            raise ValueError(f"Mode {mode} is not supported, use 'min' or 'max'")
        
        logger.info(f"ReduceLROnPlateau initialized: monitor={monitor}, factor={factor}, patience={patience}")
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Reset state at the beginning of training."""
        self.wait = 0
        self.cooldown_counter = 0
        self.best = float('inf') if self.mode == 'min' else float('-inf')
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Check for LR reduction at the end of each epoch."""
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            if self.verbose:
                logger.warning(f"ReduceLROnPlateau: {self.monitor} not found in logs")
            return
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait = 0
            return
        
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                self._reduce_lr()
                self.cooldown_counter = self.cooldown
                self.wait = 0
    
    def _reduce_lr(self) -> None:
        """Reduce learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            
            if self.verbose:
                logger.info(f"ReduceLROnPlateau: reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")

class CallbackList:
    """Container for managing a list of callbacks."""
    
    def __init__(self, callbacks: Optional[list] = None):
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback) -> None:
        """Add a callback to the list."""
        self.callbacks.append(callback)
    
    def set_model(self, model: torch.nn.Module) -> None:
        """Set the model for all callbacks."""
        for callback in self.callbacks:
            callback.set_model(model)
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set parameters for all callbacks."""
        for callback in self.callbacks:
            callback.set_params(params)
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_train_begin for all callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_train_end for all callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_epoch_begin for all callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_epoch_end for all callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_batch_begin for all callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_batch_end for all callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)
    
    def get_stop_training(self) -> bool:
        """Check if any callback requests to stop training."""
        return any(getattr(callback, 'stop_training', False) for callback in self.callbacks)