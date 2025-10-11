"""
Dataset classes for medical image captioning.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, Any, Optional, List
from transformers import PreTrainedTokenizer


class MedicalImageDataset(Dataset):
    """Dataset class for medical image captioning.
    
    This dataset handles loading and preprocessing of medical images and their
    corresponding text captions for training medical image captioning models.
    """
    
    def __init__(
        self,
        image_paths: List[str],
        captions: List[str],
        tokenizer: PreTrainedTokenizer,
        image_transform: Optional[Any] = None,
        max_caption_length: int = 128,
        text_preprocessing: bool = True
    ):
        """Initialize the dataset.
        
        Args:
            image_paths: List of paths to medical images
            captions: List of corresponding text captions
            tokenizer: Tokenizer for text processing
            image_transform: Transformations to apply to images
            max_caption_length: Maximum length for caption tokens
            text_preprocessing: Whether to apply text preprocessing
        """
        self.image_paths = image_paths
        self.captions = captions
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_caption_length = max_caption_length
        self.text_preprocessing = text_preprocessing
        
        # Validate inputs
        assert len(image_paths) == len(captions), \
            "Number of images and captions must match"
        
        # Filter out invalid samples
        self.valid_indices = []
        for i, (img_path, caption) in enumerate(zip(image_paths, captions)):
            if self._is_valid_sample(img_path, caption):
                self.valid_indices.append(i)
        
        print(f"Loaded {len(self.valid_indices)} valid samples out of {len(image_paths)} total")
    
    def _is_valid_sample(self, img_path: str, caption: str) -> bool:
        """Check if a sample is valid.
        
        Args:
            img_path: Path to image file
            caption: Text caption
            
        Returns:
            True if sample is valid, False otherwise
        """
        # Check if image path exists
        try:
            Image.open(img_path).convert('RGB')
        except (FileNotFoundError, OSError, Exception):
            return False
        
        # Check if caption is valid
        if not caption or not isinstance(caption, str):
            return False
        
        if len(caption.strip()) < 5:  # Minimum caption length
            return False
        
        return True
    
    def __len__(self) -> int:
        """Return the number of valid samples."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing image and text data
        """
        actual_idx = self.valid_indices[idx]
        
        # Load and process image
        image_path = self.image_paths[actual_idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.image_transform:
            image = self.image_transform(image)
        
        # Process caption
        caption = self.captions[actual_idx]
        if self.text_preprocessing:
            from ..utils.text_processing import clean_caption
            caption = clean_caption(caption)
        
        # Tokenize caption
        encoding = self.tokenizer(
            caption,
            max_length=self.max_caption_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image,  # shape [C, H, W]
            "input_ids": encoding["input_ids"].squeeze(0),  # shape [max_length]
            "attention_mask": encoding["attention_mask"].squeeze(0)  # shape [max_length]
        }
    
    def get_caption(self, idx: int) -> str:
        """Get the raw caption text for a sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Raw caption text
        """
        actual_idx = self.valid_indices[idx]
        return self.captions[actual_idx]
    
    def get_image_path(self, idx: int) -> str:
        """Get the image path for a sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Path to the image file
        """
        actual_idx = self.valid_indices[idx]
        return self.image_paths[actual_idx]
