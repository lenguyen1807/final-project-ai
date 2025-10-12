"""
Simple trainer for medical image captioning.
Focus on decoder training with minimal complexity.
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from simple_config import SimpleConfig


class SimpleTrainer:
    """Simple trainer for captioning models."""
    
    def __init__(self, config: SimpleConfig):
        """Initialize trainer with configuration."""
        self.config = config
        # Handle auto device detection
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        # Set random seeds
        self._set_seed(config.seed)
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        
        print("🚀 Simple Trainer Initialized")
        config.print_config()
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def setup_model(self):
        """Setup model and tokenizer based on config."""
        try:
            from src.decoders.decoder_factory_fixed import create_decoder
            from transformers import GPT2Tokenizer, T5Tokenizer, LlamaTokenizer
            
            print(f"🏗️  Setting up {self.config.decoder_type.upper()} model...")
            
            # Create tokenizer
            if self.config.decoder_type == "gpt2":
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.config.decoder_model)
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.config.decoder_type == "t5":
                self.tokenizer = T5Tokenizer.from_pretrained(self.config.decoder_model)
            elif self.config.decoder_type == "llama":
                self.tokenizer = LlamaTokenizer.from_pretrained(self.config.decoder_model)
            
            # Create decoder
            self.model = create_decoder(
                decoder_type=self.config.decoder_type,
                model_name=self.config.decoder_model,
                tokenizer=self.tokenizer,
                max_length=self.config.max_length,
                device=self.device,
                add_cross_attention=(self.config.decoder_type == "gpt2")
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Setup optimizer
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=0.01
            )
            
            print(f"✅ {self.config.decoder_type.upper()} model setup complete!")
            print(f"   - Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"   - Device: {self.device}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup model: {e}")
            return False
    
    def create_dummy_data(self, num_samples: int = 100):
        """Create dummy training data for testing."""
        try:
            from src.data_loader.dataset import MedicalImageDataset
            from torchvision import transforms
            from PIL import Image
            import torch
            
            print(f"📊 Creating dummy data ({num_samples} samples)...")
            
            # Create dummy data
            dummy_data = []
            for i in range(num_samples):
                dummy_data.append({
                    'imgs': f'dummy_images/dummy_{i}.jpg',
                    'captions': f'Medical image showing normal findings. Sample {i}.'
                })
            
            # Create dummy images directory
            os.makedirs('dummy_images', exist_ok=True)
            for i in range(num_samples):
                # Create dummy image
                img = Image.new('RGB', (224, 224), color='white')
                img.save(f'dummy_images/dummy_{i}.jpg')
            
            # Create transforms
            image_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            
            # Create dataset
            dataset = MedicalImageDataset(
                image_paths=[item['imgs'] for item in dummy_data],
                captions=[item['captions'] for item in dummy_data],
                tokenizer=self.tokenizer,
                image_transform=image_transforms,
                max_caption_length=self.config.max_length,
                text_preprocessing=True
            )
            
            # Split dataset
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                drop_last=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                drop_last=True
            )
            
            print(f"✅ Dummy data created - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
            return train_loader, val_loader, val_dataset
            
        except Exception as e:
            print(f"❌ Failed to create dummy data: {e}")
            return None, None, None
    
    def create_data_loaders_from_dataframes(self, train_df, val_df, test_df=None):
        """Create data loaders from pre-split dataframes.
        
        Args:
            train_df: Training dataframe with 'image_path' and 'caption' columns
            val_df: Validation dataframe with 'image_path' and 'caption' columns  
            test_df: Test dataframe (optional)
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        try:
            from src.data_loader.dataset import MedicalImageDataset
            from torchvision import transforms
            import torch
            
            print(f"📊 Creating data loaders from pre-split data...")
            print(f"   Train: {len(train_df)} samples")
            print(f"   Val: {len(val_df)} samples")
            if test_df is not None:
                print(f"   Test: {len(test_df)} samples")
            
            # Create transforms
            image_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            
            # Create datasets
            train_dataset = MedicalImageDataset(
                image_paths=train_df['image_path'].tolist(),
                captions=train_df['caption'].tolist(),
                tokenizer=self.tokenizer,
                image_transform=image_transforms,
                max_caption_length=self.config.max_length,
                text_preprocessing=True
            )
            
            val_dataset = MedicalImageDataset(
                image_paths=val_df['image_path'].tolist(),
                captions=val_df['caption'].tolist(),
                tokenizer=self.tokenizer,
                image_transform=image_transforms,
                max_caption_length=self.config.max_length,
                text_preprocessing=True
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                drop_last=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                drop_last=True
            )
            
            test_loader = None
            if test_df is not None:
                test_dataset = MedicalImageDataset(
                    image_paths=test_df['image_path'].tolist(),
                    captions=test_df['caption'].tolist(),
                    tokenizer=self.tokenizer,
                    image_transform=image_transforms,
                    max_caption_length=self.config.max_length,
                    text_preprocessing=True
                )
                
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=self.config.num_workers,
                    drop_last=True
                )
            
            print(f"✅ Data loaders created successfully!")
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            print(f"❌ Failed to create data loaders: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def train_epoch(self, train_loader: DataLoader, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Get batch data
                images = batch['images'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                # Create dummy encoder features for now
                batch_size = images.size(0)
                encoder_hidden_states = torch.randn(
                    batch_size, 197, 768, device=self.device
                )
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    labels=labels
                )
                
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/num_batches:.4f}'
                })
                
            except Exception as e:
                print(f"❌ Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def validate(self, val_loader: DataLoader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                try:
                    images = batch['images'].to(self.device)
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Create dummy encoder features
                    batch_size = images.size(0)
                    encoder_hidden_states = torch.randn(
                        batch_size, 197, 768, device=self.device
                    )
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        labels=labels
                    )
                    
                    total_loss += outputs.loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"❌ Validation error: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Main training loop."""
        print(f"\n🚀 Starting training for {self.config.num_epochs} epochs...")
        print("=" * 50)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            print(f"\n📚 Epoch {epoch+1}/{self.config.num_epochs}")
            print("-" * 30)
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            print(f"📈 Train Loss: {train_loss:.4f}")
            
            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                print(f"📊 Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, val_loss, is_best=True)
                    print(f"💾 New best model saved!")
        
        print(f"\n🎉 Training completed!")
        print(f"📁 Checkpoints saved in: {self.config.checkpoint_dir}")
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config.to_dict()
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, f"{self.config.checkpoint_dir}/checkpoint_epoch_{epoch}.pt")
        
        # Save best model
        if is_best:
            torch.save(checkpoint, f"{self.config.checkpoint_dir}/best_model.pt")
    
    def generate_sample(self, max_length: int = 50):
        """Generate a sample caption."""
        self.model.eval()
        
        with torch.no_grad():
            # Create dummy encoder features
            encoder_hidden_states = torch.randn(1, 197, 768, device=self.device)
            
            # Generate caption
            generated_text = self.model.generate(
                encoder_hidden_states=encoder_hidden_states,
                max_length=max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample
            )
            
            return generated_text


def main():
    """Main training function."""
    print("🏥 Simple Medical Image Captioning Trainer")
    print("=" * 50)
    
    # Create config
    config = SimpleConfig()
    
    # Initialize trainer
    trainer = SimpleTrainer(config)
    
    # Setup model
    if not trainer.setup_model():
        return False
    
    # Create dummy data
    train_loader, val_loader, val_dataset = trainer.create_dummy_data(50)
    if train_loader is None:
        return False
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    # Generate sample
    print(f"\n📝 Generating sample caption...")
    sample_caption = trainer.generate_sample()
    print(f"Generated: {sample_caption}")
    
    # Cleanup
    import shutil
    if os.path.exists('dummy_images'):
        shutil.rmtree('dummy_images')
        print("🧹 Cleaned up dummy images")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
