import math
import os
import random
from typing import Dict, Type

import numpy as np
import torch
import transformers
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from src.config import TrainingConfig
from src.dataset import DatasetCollator, XRayDataset
from src.models.medblip_t5 import MedBLIPModel_t5
from src.models.medllm import MedBLIPModel_biomedlm
from src.models.vit_gpt2 import ViT_GPT2

KAGGLE_WORKING_DIR = os.environ.get("KAGGLE_DIR", "/kaggle/working")
TRAIN_CSV_PATH = "/kaggle/input/chest-imagecaptioning/train_df.csv"
VAL_CSV_PATH = "/kaggle/input/chest-imagecaptioning/val_df.csv"
IMG_SIZE = 224


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_model(config: TrainingConfig) -> torch.nn.Module:
    """Factory function to create a model based on the config."""
    if config.model_type == "t5":
        model = MedBLIPModel_t5(t5_model="google/flan-t5-xl")
    elif config.model_type == "biomedlm":
        model = MedBLIPModel_biomedlm(lm_model="stanford-crfm/BioMedLM")
    elif config.model_type == "vit_gpt2":
        model = ViT_GPT2()
    elif config.model_type == "vit_biogpt":
        model = ViT_GPT2(gpt2_model="microsoft/biogpt")
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    if config.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        target_modules = (
            ["c_attn", "c_proj"]
            if "gpt2" in config.model_type or "biomedlm" in config.model_type
            else ["q", "k", "v"]
        )

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
            if "gpt2" in config.model_type or "biomedlm" in config.model_type
            else None,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model


def create_dataloaders(config: TrainingConfig) -> (DataLoader, DataLoader):
    """Factory function to create train and validation dataloaders."""
    data_transforms = v2.Compose(
        [
            v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print("INFO: Setting up data loaders for Kaggle Chest X-ray dataset...")

    train_dataset = XRayDataset(csv_path=TRAIN_CSV_PATH, transform=data_transforms)
    train_collator = DatasetCollator()
    trainloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=train_collator,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
    )

    val_dataset = XRayDataset(csv_path=VAL_CSV_PATH, transform=data_transforms)
    val_collator = DatasetCollator()
    valloader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        collate_fn=val_collator,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )
    return trainloader, valloader


def create_optimizer(
    model: torch.nn.Module, config: TrainingConfig
) -> (Optimizer, Dict):
    """
    Factory function to create a default optimizer.
    Returns the optimizer and the parameter groups for customization.
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in param_optimizer
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in param_optimizer
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.lr)
    return optimizer, optimizer_grouped_parameters


def create_scheduler(optimizer: Optimizer, config: TrainingConfig, num_train_steps: int):
    """Factory function to create a learning rate scheduler."""
    warmup_steps = math.ceil(num_train_steps * config.warmup)
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps
    )
    return scheduler
