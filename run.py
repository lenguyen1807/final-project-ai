import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from src.dataset import DatasetCollator, XRayDataset
from src.models.medblip_t5 import MedBLIPModel_t5
from src.models.medllm import MedBLIPModel_biomedlm
from src.models.vit_gpt2 import ViT_GPT2
from src.trainer import Trainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


TRAIN_CSV_PATH = "/kaggle/input/chest-imagecaptioning/train_df.csv"
VAL_CSV_PATH = "/kaggle/input/chest-imagecaptioning/val_df.csv"
IMG_SIZE = 224


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    set_seed(args.seed)

    # --- Data Loading ---
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
        batch_size=args.batch_size,
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
        batch_size=args.eval_batch_size,
        collate_fn=val_collator,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    # --- Model Selection ---
    if args.model_type == "t5":
        model = MedBLIPModel_t5(t5_model="google/flan-t5-xl")
    elif args.model_type == "biomedlm":
        model = MedBLIPModel_biomedlm(lm_model="stanford-crfm/BioMedLM")
    elif args.model_type == "vit_gpt2":
        model = ViT_GPT2(gpt2_model="openai-community/gpt2")
    elif args.model_type == "vit_biogpt":
        model = ViT_GPT2(gpt2_model="microsoft/biogpt")
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # --- PEFT/LoRA Configuration ---
    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        # Define LoRA config. Target modules depend on the model architecture.
        # 'c_attn' is common for GPT-2. For T5, it might be 'q', 'k', 'v'.
        # Inspect model architecture to find correct module names.
        target_modules = (
            ["c_attn", "c_proj"]
            if "gpt2" in args.model_type or "biomedlm" in args.model_type
            else ["q", "k", "v"]
        )

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
            if "gpt2" in args.model_type or "biomedlm" in args.model_type
            else None,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.cuda()
    model_save_path = f"./checkpoints/{args.run_name}"

    # --- Training ---
    trainer = Trainer()
    trainer.train(
        model,
        trainloader,
        valloader,
        warmup_ratio=args.warmup,
        epochs=args.num_epochs,
        optimizer_params={"lr": args.lr},
        output_path=model_save_path,
        weight_decay=args.weight_decay,
        use_amp=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedBLIP Pre-training Script")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["t5", "biomedlm", "vit_gpt2"],
        help="Type of model to train.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="A name for the training run, used for the output directory.",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Enable LoRA for parameter-efficient training.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use.")
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Training batch size."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="Evaluation batch size."
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1, help="Warmup ratio.")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay."
    )

    args = parser.parse_args()
    main(args)
