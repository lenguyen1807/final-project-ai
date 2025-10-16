import json
import math
import os
from typing import Dict, Type

import torch
import transformers
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import compute_all_metrics
from src.models.vit_gpt2 import ViT_GPT2


class Trainer:
    """
    A refactored trainer for single-GPU training with proper evaluation hooks.
    """

    def __init__(self, args=None):
        self.accumulation_steps = 1

    def train(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        epochs: int = 1,
        scheduler: str = "WarmupCosine",
        warmup_ratio: float = 0.01,
        output_path: str = "./checkpoints/pretrain",
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params: Dict[str, object] = {"lr": 2e-5},
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        use_amp: bool = False,
        accumulation_steps: int = 1,
        compute_clinical: bool = False,
    ):
        """
        Main training loop.
        """
        self.accumulation_steps = accumulation_steps
        if use_amp and torch.cuda.is_available():
            from torch.cuda.amp import autocast

            scaler = torch.amp.GradScaler("cuda")
            print("INFO: Using Automatic Mixed Precision (AMP).")

        steps_per_epoch = len(train_dataloader)
        num_train_steps = int((steps_per_epoch / accumulation_steps) * epochs)
        warmup_steps = math.ceil(num_train_steps * warmup_ratio)

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
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

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        scheduler = self._get_scheduler(
            optimizer,
            scheduler=scheduler,
            warmup_steps=warmup_steps,
            t_total=num_train_steps,
        )

        model = model.cuda()

        for epoch in range(epochs):
            print(f"--- Starting Epoch {epoch + 1}/{epochs} ---")

            # --- Training Step ---
            model.train()
            total_loss = 0
            optimizer.zero_grad()

            progress_bar = tqdm(
                enumerate(train_dataloader),
                total=steps_per_epoch,
                desc=f"Epoch {epoch + 1} Training",
            )
            for train_iter, data in progress_bar:
                if use_amp and torch.cuda.is_available():
                    with autocast():
                        loss_dict = model(data)
                        loss = loss_dict["loss"] / self.accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    loss_dict = model(data)
                    loss = loss_dict["loss"] / self.accumulation_steps
                    loss.backward()

                total_loss += loss.item()

                if (train_iter + 1) % self.accumulation_steps == 0:
                    if use_amp and torch.cuda.is_available():
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_grad_norm
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_grad_norm
                        )
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()

                progress_bar.set_postfix(
                    {"loss": f"{loss.item() * self.accumulation_steps:.4f}"}
                )

            avg_train_loss = total_loss / steps_per_epoch
            print(
                f"Epoch {epoch + 1} finished. Average Training Loss: {avg_train_loss:.4f}"
            )

            # --- Evaluation Step ---
            print(f"--- Running Evaluation for Epoch {epoch + 1} ---")
            metrics = self.evaluate(model, eval_dataloader, compute_clinical)

            # Save metrics to a file
            metrics_path = os.path.join(output_path, f"epoch_{epoch + 1}_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"Evaluation metrics saved to {metrics_path}")

            # --- Save Checkpoint ---
            self._save_ckpt(model, epoch + 1, output_path)

    def evaluate(
        self,
        model: torch.nn.Module,
        eval_dataloader: DataLoader,
        compute_clinical: bool = False,
    ) -> Dict[str, float]:
        """
        Runs the full evaluation loop and computes all metrics.
        """
        model.eval()
        all_predictions = []
        all_references = []

        # Logic for parsing reports is specific to the dataset and model's forward pass.
        # This needs to be consistent with how the `forward` pass parses text.
        # This is a simplified version; you might need to adapt it if your eval data format differs.
        def parse_report_for_eval(doc: str):
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

                # The 'prompt' should match what the model expects for generation
                # For QA models, this includes the question. For captioning, it might be empty.
                prompt = (
                    text
                    + "Question: What will this subject be diagnosed with? Answer: "
                )
                return prompt, label
            else:
                return doc, ""

        progress_bar = tqdm(eval_dataloader, desc="Evaluation")
        for eval_data in progress_bar:
            images = eval_data["images"].cuda().half()

            prompts = []
            references = []
            for report in eval_data["reports"]:
                prompt, reference = parse_report_for_eval(report)
                prompts.append(prompt)
                references.append(reference)

            # The generate function signature might vary. Using a flexible dict.
            # The new ViT-GPT2 model does not need a 'prompt'.
            # The existing models do. A check on model type is needed for compatibility.
            generate_input = {"images": images}
            if not isinstance(
                model, ViT_GPT2
            ):  # A bit of a hack, better to standardize generate interface
                generate_input["prompt"] = prompts

            with torch.no_grad():
                predictions = model.generate(generate_input)

            all_predictions.extend(predictions)
            all_references.extend(references)

            # For debugging, print one batch
            if len(all_predictions) <= len(eval_data["reports"]):
                print("\n--- Example Generation Batch ---")
                for i in range(len(predictions)):
                    print(f"  Reference: {references[i]}")
                    print(f"  Prediction: {predictions[i]}")
                    print("  --------------------")

        print("--- Computing All Metrics ---")
        metrics = compute_all_metrics(all_predictions, all_references, compute_clinical)
        print("\n--- Evaluation Results ---")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        print("--------------------------\n")

        return metrics

    @staticmethod
    def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        scheduler = scheduler.lower()
        if scheduler == "warmupcosine":
            return transformers.get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
        else:
            raise ValueError(f"Unknown scheduler {scheduler}")

    def _save_ckpt(self, model, epoch, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # If using PEFT, save the adapter model
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(os.path.join(save_dir, f"epoch_{epoch}"))
            print(f"PEFT adapter saved to {os.path.join(save_dir, f'epoch_{epoch}')}")
        else:
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(save_dir, f"epoch{epoch}.pth"))
            print(
                f"Full model checkpoint saved to {os.path.join(save_dir, f'epoch{epoch}.pth')}"
            )
