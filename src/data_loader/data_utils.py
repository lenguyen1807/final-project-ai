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
    reports_file: str = "indiana_reports.csv",
    projections_file: str = "indiana_projections.csv",
    image_dir: str = "images/images_normalized"
) -> pd.DataFrame:
    """Load Indiana University chest X-ray dataset.
    
    Args:
        data_dir: Root directory containing the dataset
        reports_file: Name of the reports CSV file
        projections_file: Name of the projections CSV file
        image_dir: Name of the images directory
        
    Returns:
        DataFrame with image paths and captions
    """
    # Load CSV files
    reports_path = os.path.join(data_dir, reports_file)
    projections_path = os.path.join(data_dir, projections_file)
    
    df_reports = pd.read_csv(reports_path)
    df_projections = pd.read_csv(projections_path)
    
    # Create combined dataset
    df = pd.DataFrame({'imgs': [], 'captions': []})
    
    for i in range(len(df_projections)):
        uid = df_projections.iloc[i]['uid']
        image_filename = df_projections.iloc[i]['filename']
        
        # Find corresponding report
        matching_reports = df_reports.loc[df_reports['uid'] == uid]
        
        if not matching_reports.empty:
            report_idx = matching_reports.index[0]
            caption = df_reports.iloc[report_idx]['findings']
            
            # Skip if caption is NaN or empty
            if pd.isna(caption) or not isinstance(caption, str):
                continue
            
            # Add full image path
            image_path = os.path.join(data_dir, image_dir, image_filename)
            
            df = pd.concat([
                df, 
                pd.DataFrame([{'imgs': image_path, 'captions': caption}])
            ], ignore_index=True)
    
    return df


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
    df: pd.DataFrame,
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
    
    # Split data
    train_df, temp_df = train_test_split(
        df, 
        test_size=(val_split + test_split), 
        random_state=42
    )
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=test_split / (val_split + test_split), 
        random_state=42
    )
    
    print(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets
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