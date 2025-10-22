"""
Standalone evaluation script for running models on the test set.

This script loads a trained model (either full or LoRA adapter) and runs a
full evaluation, saving both the computed metrics and a sample of
generated reports.

Example Usages:

ython evaluate-test.py \
    --model_type vit_gpt2 \
    --checkpoint_path ./kaggle/working/checkpoints/vit-gpt2-full-finetune/epoch_10 \
    --test_csv_path ./kaggle/input/chest-imagecaptioning/test_df.csv \
    --output_dir ./kaggle/working/checkpoints/vit-gpt2-full-finetune/
"""

import argparse
import dataclasses
import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    VisionEncoderDecoderModel,
)

from src.config import TrainingConfig
from src.dataset import DatasetCollator, XRayDataset
from src.factories import create_model, set_seed
from src.metrics import compute_all_metrics
from src.models.vit_gpt2 import ViT_GPT2

# PEFT might not be installed, so handle import error
try:
    from peft import PeftModel

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("WARNING: PEFT not found. LoRA model loading will not be available.")


# --- Constants ---
IMG_SIZE = 224
NUM_EXAMPLES_TO_SAVE = 100


def create_test_dataloader(test_csv_path: str, batch_size: int) -> DataLoader:
    """Creates a DataLoader for the test set."""
    data_transforms = v2.Compose(
        [
            v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    print(f"INFO: Loading test dataset from {test_csv_path}...")
    test_dataset = XRayDataset(csv_path=test_csv_path, transform=data_transforms)
    test_collator = DatasetCollator()
    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=test_collator,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )
    return testloader


def load_model_for_evaluation(
    config: TrainingConfig, checkpoint_path: str
) -> nn.Module:
    """
    Loads the model from a checkpoint, correctly handling both
    full-finetuned models (saved directory) and LoRA adapters (saved directory).
    """
    print(f"INFO: Loading model for evaluation with config: {config.model_type}")

    # 1. Create the base ViT_GPT2 wrapper.
    #    This wrapper class contains the .generate() and .forward() logic.
    #    The internal .model will be replaced by the finetuned weights.
    model_wrapper = create_model(config)

    if config.use_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT must be installed to load LoRA models.")

        print(f"INFO: Loading LoRA (PEFT) adapter from {checkpoint_path}...")
        # 1. Load the PEFT model from the saved adapter checkpoint.
        #    This loads the base model (model_wrapper.model) and attaches
        #    the adapter weights from checkpoint_path.
        try:
            peft_model = PeftModel.from_pretrained(model_wrapper.model, checkpoint_path)
        except Exception as e:
            print(f"ERROR: Failed to load PEFT model from {checkpoint_path}.")
            print(
                "Ensure the path points to a directory containing 'adapter_config.json'."
            )
            raise e

        # 2. Put the new PeftModel back into our custom wrapper
        model_wrapper.model = peft_model

    else:
        # Full model, checkpoint is a directory (as per your image)
        print(f"INFO: Loading Full-Finetuned model from {checkpoint_path}...")

        # 1. Replace the wrapper's internal .model with the full finetuned
        #    VisionEncoderDecoderModel saved at the checkpoint path.
        #    This also correctly loads the `vision_proj` layer if it exists.
        try:
            model_wrapper.model = VisionEncoderDecoderModel.from_pretrained(
                checkpoint_path
            )
        except Exception as e:
            print(f"ERROR: Failed to load full model from {checkpoint_path}.")
            print(
                "Ensure the path points to a directory containing 'pytorch_model.bin'."
            )
            raise e

    # 3. (For BOTH cases) Load the tokenizer and feature extractor that were
    #    saved alongside the model/adapter. This is crucial.
    print(f"INFO: Loading tokenizer and feature extractor from {checkpoint_path}...")
    model_wrapper.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model_wrapper.feature_extractor = AutoImageProcessor.from_pretrained(
        checkpoint_path
    )

    print("INFO: Model loaded successfully.")
    return model_wrapper


def run_evaluation(
    model: nn.Module,
    eval_dataloader: DataLoader,
    compute_clinical: bool = False,
) -> Tuple[Dict[str, float], List[str], List[str]]:
    """
    Runs the full evaluation loop and computes all metrics.
    Returns:
        - metrics (dict): Dictionary of computed metric scores.
        - all_predictions (list): List of all generated strings.
        - all_references (list): List of all ground-truth strings.
    """
    model.eval()
    all_predictions = []
    all_references = []

    is_captioning_model = (
        isinstance(model, ViT_GPT2) or "vit_gpt2" in model.__class__.__name__.lower()
    )

    def parse_qa_report_for_eval(doc: str):
        # This logic is copied directly from Trainer.evaluate
        if "The diagnosis is" in doc:
            text = doc.split("The diagnosis is ")[0]
            label = doc.split("The diagnosis is ")[1].split(".")[0]
            label_replacements = {
                "AD": "Dementia",
                "Demented": "Dementia",
                "NC": "Not demented",
                "CN": "Not demented",
                "Nondemented": "Not demented",
                "control": "Not demented",
                "MCI": "mild cognitive impairment (MCI)",
            }
            for k, v in label_replacements.items():
                label = label.replace(k, v)
            prompt = (
                text + "Question: What will this subject be diagnosed with? Answer: "
            )
            return prompt, label
        else:
            return doc, doc

    progress_bar = tqdm(eval_dataloader, desc="Running Evaluation")
    for eval_data in progress_bar:
        # Move images to GPU and use half precision for inference
        images = eval_data["images"].cuda()
        # Use bfloat16 if available, otherwise float16
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        images = images.to(dtype)

        reports = eval_data["reports"]
        generate_input = {"images": images}

        if is_captioning_model:
            references = reports
        else:
            prompts = []
            references = []
            for report in reports:
                prompt, reference = parse_qa_report_for_eval(report)
                prompts.append(prompt)
                references.append(reference)
            generate_input["prompt"] = prompts

        with torch.no_grad():
            # Use autocast for mixed-precision generation
            with torch.amp.autocast("cuda", dtype=dtype):
                predictions = model.generate(generate_input)

        all_predictions.extend(predictions)
        all_references.extend(references)

    print("INFO: Generation complete. Computing metrics...")
    metrics = compute_all_metrics(all_predictions, all_references, compute_clinical)
    return metrics, all_predictions, all_references


def main():
    parser = argparse.ArgumentParser(description="Run evaluation on a test set.")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Model type key (e.g., 'vit_gpt2', 'vit_biogpt', 'dino_gpt2').",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the saved model directory (e.g., 'epoch_10').",
    )
    parser.add_argument(
        "--test_csv_path", type=str, required=True, help="Path to the test_df.csv file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save metrics and generated examples.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Set this flag if the checkpoint is a LoRA adapter.",
    )
    parser.add_argument(
        "--compute_clinical",
        action="store_true",
        help="Set this flag to run CheXpert clinical factuality metrics.",
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    # --- 1. Setup Environment ---
    print(f"INFO: Setting up environment on GPU {args.gpu_id} with seed {args.seed}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 2. Create Config ---
    # We use TrainingConfig to pass parameters to our factory functions
    config = TrainingConfig(
        model_type=args.model_type,
        run_name=f"evaluation_{args.model_type}",
        use_lora=args.use_lora,
        compute_clinical=args.compute_clinical,
        eval_batch_size=args.batch_size,
        seed=args.seed,
        gpu_id=args.gpu_id,
        # Other fields are not needed for evaluation but required by dataclass
        num_epochs=0,
        batch_size=0,
        lr=0,
        warmup=0,
        weight_decay=0,
        accumulation_steps=0,
        max_grad_norm=0,
        use_amp=True,
    )

    # --- 3. Load Model ---
    model = load_model_for_evaluation(config, args.checkpoint_path)
    model = model.cuda()
    model.eval()

    # --- 4. Create DataLoader ---
    test_dataloader = create_test_dataloader(args.test_csv_path, args.batch_size)

    # --- 5. Run Evaluation ---
    metrics, predictions, references = run_evaluation(
        model, test_dataloader, args.compute_clinical
    )

    # --- 6. Save Results ---

    # Save Metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n--- Evaluation Results ---")
    print(json.dumps(metrics, indent=4))
    print(f"\nMetrics saved to {metrics_path}")

    # Save Generated Examples
    examples = []
    for i in range(min(len(predictions), NUM_EXAMPLES_TO_SAVE)):
        examples.append(
            {
                "id": i,
                "reference": references[i],
                "prediction": predictions[i],
            }
        )

    examples_path = os.path.join(args.output_dir, "generated_examples.json")
    with open(examples_path, "w") as f:
        json.dump(examples, f, indent=4)

    print(f"Saved {len(examples)} generated examples to {examples_path}")
    print("--- Evaluation Complete ---")


if __name__ == "__main__":
    main()
