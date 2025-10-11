"""
Configuration management for the medical image captioning pipeline.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str
    data_dir: str
    image_dir: str
    reports_file: str
    projections_file: str
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    max_caption_length: int = 128
    image_size: list = field(default_factory=lambda: [224, 224])


@dataclass
class PreprocessingConfig:
    """Text preprocessing configuration."""
    text_normalization: bool = True
    lowercase: bool = True
    remove_special_chars: bool = True
    normalize_whitespace: bool = True


@dataclass
class EncoderConfig:
    """Image encoder configuration."""
    type: str = "vit"
    model_name: str = "google/vit-base-patch16-224"
    freeze_encoder: bool = False
    projection_dim: Optional[int] = None


@dataclass
class DecoderConfig:
    """Text decoder configuration."""
    type: str = "gpt2"
    model_name: str = "gpt2"
    add_cross_attention: bool = True
    freeze_decoder: bool = False


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "ViT-GPT2"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    num_epochs: int = 5
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_clip_norm: float = 1.0
    log_steps: int = 20
    save_steps: int = 500
    eval_steps: int = 100


@dataclass
class GenerationConfig:
    """Text generation configuration."""
    max_length: int = 100
    top_p: float = 0.9
    temperature: float = 1.0
    do_sample: bool = True
    num_beams: int = 1


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    metrics: list = field(default_factory=lambda: ["bleu1", "bleu4", "meteor", "rouge_l", "bert_score"])
    num_samples: int = 200
    bert_score_lang: str = "en"


@dataclass
class SystemConfig:
    """System configuration."""
    device: str = "auto"
    num_workers: int = 4
    seed: int = 42
    mixed_precision: bool = False


@dataclass
class LoggingConfig:
    """Logging and checkpointing configuration."""
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    save_best_model: bool = True
    save_last_model: bool = True
    tensorboard: bool = True


@dataclass
class PathsConfig:
    """Paths configuration."""
    output_dir: str = "outputs"
    model_save_path: str = "models"
    results_save_path: str = "results"


@dataclass
class Config:
    """Main configuration class."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Config object loaded from file
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls._from_dict(config_dict)
    
    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        # Create nested config objects
        dataset_config = DatasetConfig(**config_dict.get('dataset', {}))
        preprocessing_config = PreprocessingConfig(**config_dict.get('preprocessing', {}))
        
        encoder_config = EncoderConfig(**config_dict.get('model', {}).get('encoder', {}))
        decoder_config = DecoderConfig(**config_dict.get('model', {}).get('decoder', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        model_config.encoder = encoder_config
        model_config.decoder = decoder_config
        
        training_config = TrainingConfig(**config_dict.get('training', {}))
        generation_config = GenerationConfig(**config_dict.get('generation', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        system_config = SystemConfig(**config_dict.get('system', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        paths_config = PathsConfig(**config_dict.get('paths', {}))
        
        return cls(
            dataset=dataset_config,
            preprocessing=preprocessing_config,
            model=model_config,
            training=training_config,
            generation=generation_config,
            evaluation=evaluation_config,
            system=system_config,
            logging=logging_config,
            paths=paths_config
        )
    
    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            output_path: Path to save YAML file
        """
        config_dict = self._to_dict()
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        return {
            'dataset': self.dataset.__dict__,
            'preprocessing': self.preprocessing.__dict__,
            'model': {
                'name': self.model.name,
                'encoder': self.model.encoder.__dict__,
                'decoder': self.model.decoder.__dict__
            },
            'training': self.training.__dict__,
            'generation': self.generation.__dict__,
            'evaluation': self.evaluation.__dict__,
            'system': self.system.__dict__,
            'logging': self.logging.__dict__,
            'paths': self.paths.__dict__
        }
