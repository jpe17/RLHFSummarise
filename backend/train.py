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

# Config - Using base QWEN model (no instruct training)
MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Training config - More conservative settings for stability
BATCH_SIZE = 4  # Reduced for stability
GRADIENT_ACCUMULATION_STEPS = 8  # Increased to maintain effective batch size
LEARNING_RATE = 1e-5  # Reduced learning rate for better stability
NUM_EPOCHS = 3
MAX_GRAD_NORM = 1.0  # Gradient clipping to prevent instability

def train_epoch(model, dataloader, optimizer, scaler):
    """Train one epoch with mixed precision and improved error handling."""
    model.train()
    total_loss = 0
    valid_steps = 0
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        try:
            # Mixed precision forward pass
            if DEVICE == "cuda" and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            
            # Check for NaN or infinite loss
            if not torch.isfinite(loss):
                print(f"⚠️  Warning: Non-finite loss at step {i}: {loss}")
                continue
                
            if not loss.requires_grad:
                print(f"⚠️  Warning: Loss doesn't require grad at step {i}")
                continue
            
            # Scale loss for gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass with proper error handling
            if DEVICE == "cuda" and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            valid_steps += 1
            
            # Gradient accumulation and optimization step
            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                try:
                    if DEVICE == "cuda" and scaler is not None:
                        # Unscale gradients for clipping
                        scaler.unscale_(optimizer)
                        
                        # Clip gradients to prevent explosion
                        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                        
                        # Check if gradients are finite before stepping
                        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                            print(f"⚠️  Warning: Non-finite gradients detected at step {i}, skipping optimizer step")
                            scaler.update()  # Update scaler but skip step
                        else:
                            scaler.step(optimizer)
                            scaler.update()
                    else:
                        # Clip gradients for non-CUDA devices
                        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                        optimizer.step()
                        
                    optimizer.zero_grad()
                    
                except RuntimeError as e:
                    if "Attempting to unscale FP16 gradients" in str(e):
                        print(f"⚠️  FP16 gradient scaling error at step {i}: {e}")
                        print("🔄 Resetting gradient scaler and continuing...")
                        
                        # Reset the scaler and skip this step
                        if scaler is not None:
                            scaler.update()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise e
                
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
    """Validate model with proper error handling."""
    model.eval()
    total_loss = 0
    valid_steps = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Validating")):
            try:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                
                if DEVICE == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                else:
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
    print("🚀 Starting LoRA Training with Base QWEN (No Preprompting)...")
    print(f"Using device: {DEVICE}")
    print(f"Model: {MODEL_ID}")
    print(f"Batch size: {BATCH_SIZE}, Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Learning rate: {LEARNING_RATE}, Max grad norm: {MAX_GRAD_NORM}")
    print("📝 Training format: post_text → summary (no instructions)")
    
    # Load data and setup tokenizer
    dataset = load_data()
    tokenizer = setup_tokenizer(MODEL_ID)
    
    # Setup model with LoRA
    model, lora_model = setup_lora_model(MODEL_ID, DEVICE)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(dataset, tokenizer, batch_size=BATCH_SIZE)
    
    # Optimizer - only train LoRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found!")
    
    print(f"Setting up optimizer with {len(trainable_params)} parameter groups")
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)
    
    # Mixed precision scaler with more conservative settings
    scaler = None
    if DEVICE == "cuda":
        scaler = torch.cuda.amp.GradScaler(
            init_scale=2**10,  # Lower initial scale for stability
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=1000  # Less aggressive growth
        )
        print("✅ Using mixed precision training with conservative scaler settings")
    
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
        
        # Save best model
        if val_loss < best_val_loss and math.isfinite(val_loss):
            best_val_loss = val_loss
            print("💾 Saving best model...")
            lora_model.save_lora_weights("best_lora_weights.pt")
        
        # Display scaler state if using mixed precision
        if scaler is not None:
            print(f"   Scaler scale: {scaler.get_scale()}")
    
    # Save final LoRA weights
    print("\n💾 Saving final LoRA weights...")
    lora_model.save_lora_weights()
    print("✅ Training complete!")
    print(f"📈 Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main() 