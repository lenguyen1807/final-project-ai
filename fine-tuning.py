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

# =====================================================================================
# 1. HELPER FUNCTIONS (BOILERPLATE)
# =====================================================================================


def _setup_environment(config: TrainingConfig):
    """Sets up seeds, GPU, and environment variables."""
    os.environ["HF_HOME"] = "./cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)


def _create_optimizer(config: TrainingConfig, optimizer_grouped_parameters):
    """Creates an optimizer, using bitsandbytes if available."""
    try:
        import bitsandbytes.optim as bnb_optim

        print("INFO: Using 8-bit AdamW optimizer (bitsandbytes).")
        optimizer = bnb_optim.AdamW8bit(optimizer_grouped_parameters, lr=config.lr)
    except ImportError:
        print(
            "WARNING: bitsandbytes not found. Falling back to standard AdamW optimizer."
        )
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.lr)
    return optimizer


# =====================================================================================
# 2. PARAMETER SETUP STRATEGIES
# These functions define which parts of the model to train.
# =====================================================================================


def setup_full_finetuning(model: torch.nn.Module, config: TrainingConfig):
    """
    Strategy for ViT-GPT2: Fine-tune all parameters in both the encoder and decoder.
    Uses a lower learning rate for the pre-trained encoder.
    """
    print("INFO: Setting up FULL fine-tuning. All model parameters will be trained.")

    # Verify no parameters are frozen
    for param in model.parameters():
        param.requires_grad = True

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # Differential learning rates
    encoder_lr = 1e-6
    decoder_lr = config.lr

    optimizer_grouped_parameters = [
        # Encoder weights with decay
        {
            "params": [
                p
                for n, p in model.model.encoder.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.weight_decay,
            "lr": encoder_lr,
        },
        # Encoder biases/norms without decay
        {
            "params": [
                p
                for n, p in model.model.encoder.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": encoder_lr,
        },
        # Decoder weights with decay
        {
            "params": [
                p
                for n, p in model.model.decoder.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.weight_decay,
            "lr": decoder_lr,
        },
        # Decoder biases/norms without decay
        {
            "params": [
                p
                for n, p in model.model.decoder.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": decoder_lr,
        },
    ]
    return optimizer_grouped_parameters


def setup_decoder_only_finetuning(model: torch.nn.Module, config: TrainingConfig):
    """
    Strategy for DINO-GPT2: Freeze the encoder and only fine-tune the decoder.
    This applies to standard cross-attention based models.
    """
    print("INFO: Setting up DECODER-ONLY fine-tuning.")
    print("INFO: Freezing ENCODER parameters...")
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    # Ensure decoder is trainable
    for param in model.model.decoder.parameters():
        param.requires_grad = True

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.model.decoder.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.model.decoder.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def setup_prefix_decoder_finetuning(model: torch.nn.Module, config: TrainingConfig):
    """
    Strategy for BioGPT models: Freeze the encoder, fine-tune the new vision
    projection layer and the entire decoder.
    """
    print("INFO: Setting up PREFIX-DECODER fine-tuning for BioGPT.")
    print("INFO: Freezing ENCODER parameters...")
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    # Ensure vision_proj and decoder are trainable
    for param in model.vision_proj.parameters():
        param.requires_grad = True
    for param in model.model.decoder.parameters():
        param.requires_grad = True

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        # Decoder parameters
        {
            "params": [
                p
                for n, p in model.model.decoder.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.model.decoder.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        # Vision projection layer parameters
        {"params": model.vision_proj.parameters(), "weight_decay": config.weight_decay},
    ]
    return optimizer_grouped_parameters


# =====================================================================================
# 3. GENERIC TRAINING RUNNER
# =====================================================================================


def run_training_strategy(config: TrainingConfig, parameter_setup_fn: callable):
    """A generic function to run a training experiment."""
    print(f"\n{'=' * 30}\nStarting Training Run: {config.run_name}\n{'=' * 30}")

    _setup_environment(config)

    print("INFO: Creating model and dataloaders...")
    model = create_model(config)
    train_dataloader, eval_dataloader = create_dataloaders(config)

    # Apply the selected parameter freezing/setup strategy
    optimizer_grouped_parameters = parameter_setup_fn(model, config)

    # Sanity check trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable params: {trainable_params:,} || Total params: {total_params:,} || Trainable %: {100 * trainable_params / total_params:.2f}"
    )

    optimizer = _create_optimizer(config, optimizer_grouped_parameters)
    steps_per_epoch = len(train_dataloader)
    num_train_steps = int(
        (steps_per_epoch / config.accumulation_steps) * config.num_epochs
    )
    scheduler = create_scheduler(optimizer, config, num_train_steps)

    model.cuda()
    model_save_path = f"{KAGGLE_WORKING_DIR}/checkpoints/{config.run_name}"

    print("\nINFO: Initializing Trainer and starting training loop...")
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
    print(f"--- Finished Training Run: {config.run_name} ---")


# =====================================================================================
# 4. PUBLIC STRATEGY FUNCTIONS
# These are the functions you will call to run each experiment.
# =====================================================================================


def train_vit_gpt2_full():
    """Strategy 1: Full fine-tuning for ViT-GPT2."""
    config = TrainingConfig(
        model_type="vit_gpt2",
        run_name="vit-gpt2-full-finetune",
        num_epochs=10,
        batch_size=32,
        eval_batch_size=64,
        lr=5e-5,  # Decoder LR, encoder LR is hardcoded in setup function
        use_lora=False,
        compute_clinical=False,
        weight_decay=0.01,
        accumulation_steps=1,
    )
    run_training_strategy(config, setup_full_finetuning)


def train_dino_gpt2_decoder():
    """Strategy 2: Decoder-only fine-tuning for DINO-GPT2."""
    config = TrainingConfig(
        model_type="dino_gpt2",
        run_name="dino-gpt2-decoder-only",
        num_epochs=10,
        batch_size=32,
        eval_batch_size=64,
        lr=5e-5,
        use_lora=False,
        compute_clinical=False,
        weight_decay=0.01,
        accumulation_steps=1,
    )
    run_training_strategy(config, setup_decoder_only_finetuning)


def train_vit_biogpt_decoder():
    """Strategy 3: Freeze ViT encoder, fine-tune vision_proj and BioGPT decoder."""
    config = TrainingConfig(
        model_type="vit_biogpt",
        run_name="vit-biogpt-decoder-side",
        num_epochs=10,
        batch_size=32,
        eval_batch_size=64,
        lr=5e-5,
        use_lora=False,
        compute_clinical=False,
        weight_decay=0.01,
        accumulation_steps=1,
    )
    run_training_strategy(config, setup_prefix_decoder_finetuning)


def train_dino_biogpt_decoder():
    """Strategy 4: Freeze DINO encoder, fine-tune vision_proj and BioGPT decoder."""
    config = TrainingConfig(
        model_type="dino_biogpt",
        run_name="dino-biogpt-decoder-side",
        num_epochs=10,
        batch_size=32,
        eval_batch_size=64,
        lr=5e-5,
        use_lora=False,
        compute_clinical=False,
        weight_decay=0.01,
        accumulation_steps=1,
    )
    run_training_strategy(config, setup_prefix_decoder_finetuning)


# =====================================================================================
# 5. MAIN EXECUTION BLOCK
# =====================================================================================

if __name__ == "__main__":
    # --- CHOOSE WHICH STRATEGY TO RUN ---
    # Uncomment the function call for the experiment you want to perform.

    # train_vit_gpt2_full()
    train_dino_gpt2_decoder()
    # train_vit_biogpt_decoder()
    # train_dino_biogpt_decoder()
