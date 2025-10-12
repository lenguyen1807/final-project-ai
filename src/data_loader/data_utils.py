"""
Data loading utilities for medical image captioning.
"""

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, List
from torchvision import transforms
from transformers import PreTrainedTokenizer

from .dataset import MedicalImageDataset

def load_indiana_dataset(
    data_dir: str,
    image_dir: str,
    train_df: str,
    val_df: str,
    test_df: str
) -> pd.DataFrame:
    """Load the Indiana dataset."""
    train_df = pd.read_csv(train_df)
    val_df = pd.read_csv(val_df)
    test_df = pd.read_csv(test_df)
    return train_df, val_df, test_df

def preprocess_dataframe(
    df: pd.DataFrame,
    text_preprocessing: bool = True
) -> pd.DataFrame:
    """Preprocess the dataframe.
    
    Args:
        df: Input dataframe
        text_preprocessing: Whether to apply text preprocessing
        
    Returns:
        Preprocessed dataframe
    """
    if text_preprocessing:
        from ..utils.text_processing import clean_caption
        df['captions'] = df['captions'].apply(clean_caption)
    
    # Remove rows with empty captions
    df = df[df['captions'].str.len() > 0]
    
    return df


def create_data_loaders(
    tokenizer: PreTrainedTokenizer,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 16,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    max_caption_length: int = 128,
    num_workers: int = 4,
    text_preprocessing: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders.
    
    Args:
        df: DataFrame with image paths and captions
        tokenizer: Tokenizer for text processing
        image_size: Size to resize images to
        batch_size: Batch size for data loaders
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        max_caption_length: Maximum caption length
        num_workers: Number of worker processes
        text_preprocessing: Whether to apply text preprocessing
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Define image transforms
    image_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_df = "chest-xrays-indiana-university/train_df.csv"
    val_df = "chest-xrays-indiana-university/val_df.csv"
    test_df = "chest-xrays-indiana-university/test_df.csv"

   # Create datasets
    train_df = pd.read_csv(train_df)
    val_df = pd.read_csv(val_df)
    test_df = pd.read_csv(test_df)

    train_dataset = MedicalImageDataset(
        image_paths=train_df['imgs'].tolist(),
        captions=train_df['captions'].tolist(),
        tokenizer=tokenizer,
        image_transform=image_transforms,
        max_caption_length=max_caption_length,
        text_preprocessing=text_preprocessing
    )
    
    val_dataset = MedicalImageDataset(
        image_paths=val_df['imgs'].tolist(),
        captions=val_df['captions'].tolist(),
        tokenizer=tokenizer,
        image_transform=image_transforms,
        max_caption_length=max_caption_length,
        text_preprocessing=text_preprocessing
    )
    
    test_dataset = MedicalImageDataset(
        image_paths=test_df['imgs'].tolist(),
        captions=test_df['captions'].tolist(),
        tokenizer=tokenizer,
        image_transform=image_transforms,
        max_caption_length=max_caption_length,
        text_preprocessing=text_preprocessing
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader