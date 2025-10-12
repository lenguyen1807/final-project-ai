"""
Training utilities for medical image captioning.
"""

import os
import random
import torch
from tqdm import tqdm
from typing import Dict, Any, Optional
from transformers import PreTrainedTokenizer


def validate_model(model, val_loader, device):
    """Validate the model on validation dataset.
    
    Args:
        model: The model to validate
        val_loader: Validation data loader
        device: Device to run on
        
    Returns:
        Average validation loss
    """
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss


def generate_caption_sanity_check(model, val_dataset, tokenizer, device, idx=0, max_new_tokens=100):
    """Generate a caption for sanity check during training.
    
    Args:
        model: The model to generate with
        val_dataset: Validation dataset
        tokenizer: Tokenizer for decoding
        device: Device to run on
        idx: Index of sample to use (if 0, random sample is chosen)
        max_new_tokens: Maximum number of new tokens to generate
    """
    model.eval()
    if idx == 0:
        idx = random.randint(0, len(val_dataset) - 1)
    sample = val_dataset[idx]
    pixel_values = sample["pixel_values"].to(device)

    with torch.no_grad():
        generated_caption = model.generate(
            image=pixel_values,
            max_length=max_new_tokens,
            device=device
        )

    reference_caption = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)

    print(f"\n--- Sanity Check ---")
    print(f"Reference: {reference_caption}")
    print(f"Generated: {generated_caption}")
    print("-------------------\n")


def train_model(model, train_loader, val_loader, val_dataset, tokenizer, device, 
                epochs=5, lr=5e-5, log_steps=20, save_dir="checkpoints"):
    """Train the medical image captioning model.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        val_dataset: Validation dataset (for sanity checks)
        tokenizer: Tokenizer for text processing
        device: Device to run on
        epochs: Number of training epochs
        lr: Learning rate
        log_steps: Steps between logging
        save_dir: Directory to save checkpoints
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float("inf")  # initialize best loss
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (step + 1) % log_steps == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {avg_loss:.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # run validation
        avg_val_loss = validate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # check if this is the best model so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(save_dir, f"best_model_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved new best model at {save_path} with Val Loss: {avg_val_loss:.4f}")

        # sanity check
        generate_caption_sanity_check(model, val_dataset, tokenizer, device)
