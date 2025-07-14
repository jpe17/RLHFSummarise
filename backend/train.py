#!/usr/bin/env python3
"""
LoRA Training for Summarization with Base QWEN (No Preprompting)
Trains the model to generate summaries directly from post text without instructions.
"""

import torch
from tqdm import tqdm
from model import setup_lora_model
from data_loader import load_data, setup_tokenizer, create_dataloaders
import math
import argparse

# Config - Using base QWEN model (no instruct training)
MODEL_ID = "Qwen/Qwen2-0.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Fast iteration config - CPU/GPU friendly
BATCH_SIZE = 2  # Single sample per batch to prevent instability
GRADIENT_ACCUMULATION_STEPS = 8  # Accumulate to effective batch size of 8
LEARNING_RATE = 1e-6  # Very conservative learning rate to prevent NaN
NUM_EPOCHS = 5  # Just 2 epochs for testing
MAX_GRAD_NORM = 0.5  # Aggressive gradient clipping
MAX_TRAIN_SAMPLES = 10000  # Small dataset for testing
MAX_VAL_SAMPLES = 6000   # Small validation set
USE_MIXED_PRECISION = False  # Disable mixed precision to avoid FP16 issues


# Add this method to the LoRAModel class (around line 196)
def load_lora_weights(self, path="lora_weights.pt"):
    """Load LoRA adapter weights for resuming training"""
    if self.model is None:
        raise ValueError("Model not loaded! Call load_and_setup() first.")
    
    print(f"🔄 Loading LoRA weights from: {path}")
    
    try:
        # Load the weights
        lora_weights = torch.load(path, map_location=self.device)
        
        # Get the target dtype from the model
        target_dtype = next(self.model.parameters()).dtype
        print(f"🔍 Target model dtype: {target_dtype}")
        
        # Convert weights to the correct dtype if needed
        converted_weights = {}
        dtype_conversions = 0
        
        for key, weight in lora_weights.items():
            if isinstance(weight, torch.Tensor) and weight.dtype != target_dtype:
                converted_weights[key] = weight.to(target_dtype)
                dtype_conversions += 1
            else:
                converted_weights[key] = weight
        
        if dtype_conversions > 0:
            print(f"🔄 Converted {dtype_conversions} weights to {target_dtype}")
        
        # Load weights into model
        missing_keys, unexpected_keys = self.model.load_state_dict(converted_weights, strict=False)
        
        print(f"✅ LoRA weights loaded successfully")
        if missing_keys:
            print(f"⚠️  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading LoRA weights: {e}")
        return False
    
    
def train_epoch(model, dataloader, optimizer, scaler):
    """Train one epoch with simplified training (no mixed precision)."""
    model.train()
    total_loss = 0
    valid_steps = 0
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        try:
            # Simple forward pass (no mixed precision)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Check for NaN or infinite loss
            if not torch.isfinite(loss):
                print(f"❌ NaN/Inf loss detected at step {i}: {loss}")
                print(f"   Input shape: {input_ids.shape}")
                print(f"   Learning rate: {optimizer.param_groups[0]['lr']}")
                print("🛑 Stopping training to prevent further instability")
                return float('inf')  # Return early to stop training
                
            if not loss.requires_grad:
                print(f"⚠️  Warning: Loss doesn't require grad at step {i}")
                continue
            
            # Scale loss for gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            # Simple backward pass
            loss.backward()
            
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            valid_steps += 1
            
            # Gradient accumulation and optimization step
            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                try:
                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    
                    # Check if gradients are finite before stepping
                    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                        print(f"⚠️  Warning: Non-finite gradients detected at step {i}, skipping optimizer step")
                    else:
                        optimizer.step()
                        
                    optimizer.zero_grad()
                    
                except Exception as e:
                    print(f"⚠️  Optimizer step error at step {i}: {e}")
                    optimizer.zero_grad()
                    continue
                
        except Exception as e:
            print(f"❌ Error at step {i}: {e}")
            print(f"🔄 Skipping batch {i} and continuing...")
            
            # Reset gradients and continue
            optimizer.zero_grad()
            continue
    
    # Return average loss over valid steps
    if valid_steps > 0:
        return total_loss / valid_steps
    else:
        print("⚠️  Warning: No valid training steps completed")
        return float('inf')

def validate(model, dataloader):
    """Validate model with simple forward pass."""
    model.eval()
    total_loss = 0
    valid_steps = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Validating")):
            try:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                
                # Simple forward pass (no mixed precision)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                # Check for finite loss
                if torch.isfinite(loss):
                    total_loss += loss.item()
                    valid_steps += 1
                else:
                    print(f"⚠️  Warning: Non-finite validation loss at step {i}")
                    
            except Exception as e:
                print(f"⚠️  Validation error at step {i}: {e}")
                continue
    
    if valid_steps > 0:
        return total_loss / valid_steps
    else:
        return float('inf')

def main():
    print("🚀 Starting FAST LoRA Training for Iteration...")
    print(f"Using device: {DEVICE}")
    print(f"Model: {MODEL_ID}")
    print(f"Fast iteration settings:")
    print(f"  - Batch size: {BATCH_SIZE}, Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  - Learning rate: {LEARNING_RATE}, Epochs: {NUM_EPOCHS}")
    print(f"  - Max train samples: {MAX_TRAIN_SAMPLES}, Max val samples: {MAX_VAL_SAMPLES}")
    print("📝 Training format: post_text → summary (no preprompting)")
    
    # Load data and setup tokenizer
    dataset = load_data()
    tokenizer = setup_tokenizer(MODEL_ID)
    
    # Setup model with LoRA
    model, lora_model = setup_lora_model(MODEL_ID, DEVICE)
    
    # Resume from checkpoint if provided
    if args.resume:
        print(f"🔄 Resuming training from: {args.resume}")
        if lora_model.load_lora_weights(args.resume):
            print("✅ Successfully resumed from checkpoint")
        else:
            print("❌ Failed to load checkpoint, starting from scratch")
    
    # Create dataloaders with limited data for fast iteration
    train_loader, val_loader = create_dataloaders(
        dataset, tokenizer, 
        batch_size=BATCH_SIZE, 
        max_train_samples=MAX_TRAIN_SAMPLES,
        max_val_samples=MAX_VAL_SAMPLES
    )
    
    # Optimizer - only train LoRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found!")
    
    print(f"Setting up optimizer with {len(trainable_params)} parameter groups")
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)
    
    # No mixed precision - keep it simple for debugging
    scaler = None
    print("✅ Using simple float32 training (no mixed precision)")
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scaler)
        
        # Validate
        val_loss = validate(model, val_loader)
        
        print(f"\n📊 Epoch {epoch + 1} Results:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        
        # Stop early if training went bad
        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            print("🛑 Training instability detected - stopping early")
            break
        
        # Save best model
        if val_loss < best_val_loss and math.isfinite(val_loss):
            best_val_loss = val_loss
            print("💾 Saving best model...")
            lora_model.save_lora_weights("best_lora_weights.pt")
    
    # Save final LoRA weights
    print("\n💾 Saving final LoRA weights...")
    lora_model.save_lora_weights()
    print("✅ Training complete!")
    print(f"📈 Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast LoRA training for summarization")
    parser.add_argument("--test", action="store_true", help="Ultra-fast test mode (10 samples, 1 epoch)")
    parser.add_argument("--samples", type=int, help="Max training samples")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--resume", type=str, help="Path to LoRA weights to resume from")
    
    args = parser.parse_args()
    
    # Ultra-fast test mode
    if args.test:
        print("🚀 ULTRA-FAST TEST MODE")
        MAX_TRAIN_SAMPLES = 1000
        MAX_VAL_SAMPLES = 200
        NUM_EPOCHS = 10
        BATCH_SIZE = 4
        GRADIENT_ACCUMULATION_STEPS = 1
    
    # Override with command line args
    if args.samples:
        MAX_TRAIN_SAMPLES = args.samples
        MAX_VAL_SAMPLES = max(5, args.samples // 5)
    
    if args.epochs:
        NUM_EPOCHS = args.epochs
    
    main() 