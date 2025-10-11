"""
Factory for creating encoder instances.
"""

from typing import Dict, Any, Optional

from .base_encoder import ImageEncoder
from .vit_encoder import ViTEncoder
from .swin_encoder import SwinEncoder
from .resnet_encoder import ResNetEncoder


def create_encoder(
    encoder_type: str,
    model_name: str,
    output_dim: Optional[int] = None,
    freeze_encoder: bool = False,
    device: str = "cpu",
    projection_dim: Optional[int] = None,
    **kwargs
) -> ImageEncoder:
    """Create an encoder instance based on type.
    
    Args:
        encoder_type: Type of encoder ('vit', 'swin', 'resnet')
        model_name: Name of the model to load
        output_dim: Output dimension
        freeze_encoder: Whether to freeze encoder parameters
        device: Device to run on
        projection_dim: Dimension for projection layer
        **kwargs: Additional arguments for encoder initialization
        
    Returns:
        Encoder instance
        
    Raises:
        ValueError: If encoder_type is not supported
    """
    encoder_type = encoder_type.lower()
    
    if encoder_type == "vit":
        return ViTEncoder(
            model_name=model_name,
            output_dim=output_dim,
            freeze_encoder=freeze_encoder,
            device=device,
            projection_dim=projection_dim,
            **kwargs
        )
    elif encoder_type == "swin":
        return SwinEncoder(
            model_name=model_name,
            output_dim=output_dim,
            freeze_encoder=freeze_encoder,
            device=device,
            projection_dim=projection_dim,
            **kwargs
        )
    elif encoder_type == "resnet":
        return ResNetEncoder(
            model_name=model_name,
            output_dim=output_dim,
            freeze_encoder=freeze_encoder,
            device=device,
            projection_dim=projection_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}. "
                       f"Supported types: vit, swin, resnet")


def get_supported_encoders() -> Dict[str, Dict[str, Any]]:
    """Get information about supported encoders.
    
    Returns:
        Dictionary mapping encoder types to their information
    """
    return {
        "vit": {
            "name": "Vision Transformer",
            "description": "Vision Transformer with patch-based attention",
            "models": [
                "google/vit-base-patch16-224",
                "google/vit-large-patch16-224",
                "google/vit-small-patch16-224"
            ],
            "sequence_length": 197,
            "medical_specialized": True
        },
        "swin": {
            "name": "Swin Transformer",
            "description": "Hierarchical Vision Transformer with shifted windows",
            "models": [
                "microsoft/swin-tiny-patch4-window7-224",
                "microsoft/swin-small-patch4-window7-224",
                "microsoft/swin-base-patch4-window7-224"
            ],
            "sequence_length": 49,
            "medical_specialized": True
        },
        "resnet": {
            "name": "ResNet",
            "description": "Residual Neural Network for image classification",
            "models": ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
            "sequence_length": 1,
            "medical_specialized": False
        }
    }


def get_encoder_recommendations() -> Dict[str, str]:
    """Get encoder recommendations for medical image captioning.
    
    Returns:
        Dictionary with recommendations for different use cases
    """
    return {
        "best_overall": "vit",
        "best_quality": "vit",
        "best_speed": "resnet",
        "best_medical": "vit",
        "best_hierarchical": "swin",
        "best_resource_efficient": "resnet"
    }
