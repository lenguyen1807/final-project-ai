"""
Checkpointing utilities for saving and loading model states.
"""

import os
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    metrics: Optional[Dict[str, float]] = None,
    checkpoint_dir: str = "checkpoints",
    filename: Optional[str] = None,
    save_optimizer: bool = True
) -> str:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer state
        epoch: Current epoch
        step: Current step
        loss: Current loss value
        metrics: Optional metrics dictionary
        checkpoint_dir: Directory to save checkpoint
        filename: Custom filename (optional)
        save_optimizer: Whether to save optimizer state
        
    Returns:
        Path to saved checkpoint
    """
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint_data = {
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    if save_optimizer:
        checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
    
    if metrics:
        checkpoint_data['metrics'] = metrics
    
    # Generate filename
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_epoch_{epoch}_step_{step}_{timestamp}.pt"
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    # Save checkpoint
    torch.save(checkpoint_data, checkpoint_path)
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu"
) -> Tuple[int, int, float, Optional[Dict[str, float]]]:
    """Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load checkpoint on
        
    Returns:
        Tuple of (epoch, step, loss, metrics)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Extract metadata
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    loss = checkpoint.get('loss', float('inf'))
    metrics = checkpoint.get('metrics', None)
    
    return epoch, step, loss, metrics


def save_best_model(
    model: torch.nn.Module,
    metrics: Dict[str, float],
    checkpoint_dir: str = "checkpoints",
    metric_name: str = "val_loss",
    lower_is_better: bool = True
) -> str:
    """Save best model based on metric.
    
    Args:
        model: PyTorch model
        metrics: Current metrics
        checkpoint_dir: Directory to save model
        metric_name: Metric to use for comparison
        lower_is_better: Whether lower values are better
        
    Returns:
        Path to saved model
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Save model state dict
    model_path = os.path.join(checkpoint_dir, "best_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }, model_path)
    
    return model_path


def save_model_config(
    config: Dict[str, Any],
    checkpoint_dir: str = "checkpoints"
) -> str:
    """Save model configuration.
    
    Args:
        config: Configuration dictionary
        checkpoint_dir: Directory to save config
        
    Returns:
        Path to saved config
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Get the latest checkpoint file.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("checkpoint_") and f.endswith(".pt")
    ]
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time
    checkpoint_files.sort(
        key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)),
        reverse=True
    )
    
    return os.path.join(checkpoint_dir, checkpoint_files[0])
