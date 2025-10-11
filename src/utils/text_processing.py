"""
Text processing utilities for medical image captioning.
"""

import re
from typing import str


def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_special_chars: bool = True,
    normalize_whitespace: bool = True
) -> str:
    """Preprocess text for medical captions.
    
    Args:
        text: Input text to preprocess
        lowercase: Whether to convert to lowercase
        remove_special_chars: Whether to remove special characters
        normalize_whitespace: Whether to normalize whitespace
        
    Returns:
        Preprocessed text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    # Remove special characters (keep letters, numbers, spaces, and basic punctuation)
    if remove_special_chars:
        text = re.sub(r"[^a-z0-9.,!?;:\s]", " ", text)
    
    # Normalize whitespace
    if normalize_whitespace:
        # Replace multiple spaces with single space
        text = re.sub(r"\s+", " ", text)
        # Remove leading/trailing whitespace
        text = text.strip()
    
    return text


def normalize_medical_text(text: str) -> str:
    """Normalize medical text with specific medical terminology handling.
    
    Args:
        text: Input medical text
        
    Returns:
        Normalized medical text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Handle common medical abbreviations and terms
    medical_replacements = {
        r'\bno acute\b': 'no acute',
        r'\bacute\b': 'acute',
        r'\bchronic\b': 'chronic',
        r'\bmild\b': 'mild',
        r'\bmoderate\b': 'moderate',
        r'\bsevere\b': 'severe',
        r'\bnormal\b': 'normal',
        r'\babnormal\b': 'abnormal',
        r'\bopacity\b': 'opacity',
        r'\bconsolidation\b': 'consolidation',
        r'\beffusion\b': 'effusion',
        r'\batelectasis\b': 'atelectasis',
        r'\bcardiomegaly\b': 'cardiomegaly',
        r'\bpleural\b': 'pleural',
        r'\bpneumonia\b': 'pneumonia',
        r'\bedema\b': 'edema',
        r'\bemphysema\b': 'emphysema',
        r'\bfibrosis\b': 'fibrosis',
        r'\bcalcification\b': 'calcification',
        r'\bmass\b': 'mass',
        r'\bnodule\b': 'nodule',
        r'\blesion\b': 'lesion',
        r'\binfiltration\b': 'infiltration',
        r'\binflammation\b': 'inflammation'
    }
    
    # Apply medical term normalization
    for pattern, replacement in medical_replacements.items():
        text = re.sub(pattern, replacement, text)
    
    # Remove excessive punctuation
    text = re.sub(r'\.+', '.', text)
    text = re.sub(r',+', ',', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def clean_caption(caption: str) -> str:
    """Clean and normalize caption text.
    
    Args:
        caption: Input caption text
        
    Returns:
        Cleaned caption text
    """
    if not caption or not isinstance(caption, str):
        return ""
    
    # Handle NaN or None values
    if str(caption).lower() in ['nan', 'none', 'null']:
        return ""
    
    # Apply text preprocessing
    caption = preprocess_text(
        caption,
        lowercase=True,
        remove_special_chars=True,
        normalize_whitespace=True
    )
    
    # Apply medical text normalization
    caption = normalize_medical_text(caption)
    
    return caption


def validate_caption(caption: str, min_length: int = 5, max_length: int = 500) -> bool:
    """Validate caption text.
    
    Args:
        caption: Caption text to validate
        min_length: Minimum caption length
        max_length: Maximum caption length
        
    Returns:
        True if caption is valid, False otherwise
    """
    if not caption or not isinstance(caption, str):
        return False
    
    # Check length
    if len(caption) < min_length or len(caption) > max_length:
        return False
    
    # Check if caption is meaningful (not just punctuation)
    if not re.search(r'[a-zA-Z]', caption):
        return False
    
    return True
