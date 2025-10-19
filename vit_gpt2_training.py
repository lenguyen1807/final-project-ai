import os

import torch

from src.config import TrainingConfig
from src.factories import (
    KAGGLE_WORKING_DIR,
    create_dataloaders,
    create_model,
    create_scheduler,
    set_seed,
)
from src.trainer import Trainer


def run_vit_gpt2_custom_training():
    """
    An example script demonstrating how to use the refactored code as a library
    to run a custom training loop for the ViT-GPT2 model.

    This example shows how to:
    1. Programmatically define a training configuration.
    2. Create custom optimizer parameter groups for differential learning rates.
    3. Assemble the components and run the training.
    """
    print("--- Starting Custom ViT-GPT2 Training Example ---")

    # 1. Define the training configuration programmatically.
    # This overrides the defaults in the TrainingConfig class.
    # We use a lower learning rate as recommended for full fine-tuning.
    config = TrainingConfig(
        model_type="vit_gpt2",
        run_name="vit-gpt2-full-finetune-custom-lr",
        num_epochs=5,
        batch_size=4,  # Use a smaller batch size for full fine-tuning if memory is a concern
        eval_batch_size=8,
        lr=1e-6,  # A much lower learning rate for the sensitive vision encoder
        use_lora=False,
        compute_clinical=False,
        weight_decay=0.05,
        accumulation_steps=2,
    )

    # --- Boilerplate Setup ---
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)

    # --- Component Assembly ---
    print("INFO: Creating model and dataloaders...")
    model = create_model(config)
    train_dataloader, eval_dataloader = create_dataloaders(config)

    # 2. *** Define Custom Optimizer Parameter Groups ***
    # This is where you can specify different settings (like learning rates)
    # for different parts of the model, as you requested.
    print("INFO: Creating custom optimizer with differential learning rates...")
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # It's a good practice to apply weight decay to all parameters except biases and LayerNorm weights
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.model.encoder.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.weight_decay,
            "lr": config.lr,  # Low learning rate for the encoder
        },
        {
            "params": [
                p
                for n, p in model.model.encoder.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": config.lr,  # Low learning rate for the encoder
        },
        {
            "params": [
                p
                for n, p in model.model.decoder.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.weight_decay,
            "lr": 5e-5,  # Higher learning rate for the decoder
        },
        {
            "params": [
                p
                for n, p in model.model.decoder.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": 5e-5,  # Higher learning rate for the decoder
        },
    ]

    # 3. Create your custom optimizer and the scheduler
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    steps_per_epoch = len(train_dataloader)
    num_train_steps = int(
        (steps_per_epoch / config.accumulation_steps) * config.num_epochs
    )
    scheduler = create_scheduler(optimizer, config, num_train_steps)

    model.cuda()
    model_save_path = f"{KAGGLE_WORKING_DIR}/checkpoints/{config.run_name}"

    # 4. Instantiate the Trainer and run the training
    print("INFO: Starting training...")
    trainer = Trainer()
    trainer.train(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=config.num_epochs,
        output_path=model_save_path,
        max_grad_norm=config.max_grad_norm,
        use_amp=config.use_amp,
        accumulation_steps=config.accumulation_steps,
        compute_clinical=config.compute_clinical,
    )

    print("--- Custom ViT-GPT2 Training Example Finished ---")


if __name__ == "__main__":
    run_vit_gpt2_custom_training()
