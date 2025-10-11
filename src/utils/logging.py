"""
Logging utilities for the medical image captioning pipeline.
"""

import os
import logging
import torch
from pathlib import Path
from typing import Optional
from torch.utils.tensorboard import SummaryWriter


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    tensorboard: bool = True,
    experiment_name: Optional[str] = None
) -> tuple[logging.Logger, Optional[SummaryWriter]]:
    """Setup logging and tensorboard.
    
    Args:
        log_dir: Directory to save logs
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        tensorboard: Whether to enable tensorboard logging
        experiment_name: Name for the experiment
        
    Returns:
        Tuple of (logger, tensorboard_writer)
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = logging.getLogger("medical_captioning")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = os.path.join(log_dir, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Setup tensorboard
    tb_writer = None
    if tensorboard:
        tb_log_dir = os.path.join(log_dir, "tensorboard")
        if experiment_name:
            tb_log_dir = os.path.join(tb_log_dir, experiment_name)
        tb_writer = SummaryWriter(tb_log_dir)
        logger.info(f"TensorBoard logging enabled at {tb_log_dir}")
    
    logger.info(f"Logging setup complete. Logs saved to {log_dir}")
    return logger, tb_writer


def log_metrics(
    logger: logging.Logger,
    tb_writer: Optional[SummaryWriter],
    metrics: dict,
    step: int,
    prefix: str = ""
) -> None:
    """Log metrics to both logger and tensorboard.
    
    Args:
        logger: Logger instance
        tb_writer: TensorBoard writer
        metrics: Dictionary of metrics to log
        step: Current step number
        prefix: Prefix for metric names
    """
    # Log to console/file
    metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"{prefix} {metric_str}")
    
    # Log to tensorboard
    if tb_writer:
        for key, value in metrics.items():
            tb_writer.add_scalar(f"{prefix}/{key}", value, step)


def log_model_info(logger: logging.Logger, model: torch.nn.Module) -> None:
    """Log model information.
    
    Args:
        logger: Logger instance
        model: PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")


def close_logging(tb_writer: Optional[SummaryWriter]) -> None:
    """Close tensorboard writer.
    
    Args:
        tb_writer: TensorBoard writer to close
    """
    if tb_writer:
        tb_writer.close()
