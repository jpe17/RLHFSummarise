#!/usr/bin/env python3
"""
LoRA Training for Summarization with Base QWEN (No Preprompting)
Trains the model to generate summaries directly from post text without instructions.
"""

import torch
from tqdm import tqdm
from model import setup_lora_model
from data_loader import load_data, setup_tokenizer, create_dataloaders

# Config - Using base QWEN model (no instruct training)
MODEL_ID = "Qwen/Qwen2-0.5B"  # Base QWEN model: 0.5B params (no instruct)
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Training config
BATCH_SIZE = 4  # Reduced for stability
GRADIENT_ACCUMULATION_STEPS = 8  # Increased to maintain effective batch size
LEARNING_RATE = 1e-4  # Reduced learning rate
NUM_EPOCHS = 3

def train_epoch(model, dataloader, optimizer, scaler):
    """Train one epoch with mixed precision."""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        # Debug: Check if inputs have gradients
        if i == 0:
            print(f"Input IDs shape: {input_ids.shape}")
            print(f"Attention mask shape: {attention_mask.shape}")
            print(f"Labels shape: {labels.shape}")
        
        try:
            # Mixed precision forward pass
            if DEVICE == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
            # Debug: Check loss properties
            if i == 0:
                print(f"Loss: {loss}")
                print(f"Loss requires grad: {loss.requires_grad}")
                print(f"Loss grad_fn: {loss.grad_fn}")
            
            if loss.requires_grad:
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            else:
                print(f"Warning: Loss doesn't require grad at step {i}")
                continue
            
            # Backward pass
            if DEVICE == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            # Gradient accumulation
            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                if DEVICE == "cuda":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                
        except Exception as e:
            print(f"Error at step {i}: {e}")
            print(f"Skipping batch {i}")
            continue
    
    return total_loss / len(dataloader)

def validate(model, dataloader):
    """Validate model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            with torch.cuda.amp.autocast() if DEVICE == "cuda" else torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()
    
    return total_loss / len(dataloader)

def main():
    print("🚀 Starting LoRA Training with Base QWEN (No Preprompting)...")
    print(f"Using device: {DEVICE}")
    print(f"Model: {MODEL_ID}")
    print(f"Batch size: {BATCH_SIZE}, Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
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
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scaler)
        
        # Validate
        val_loss = validate(model, val_loader)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Save LoRA weights
    lora_model.save_lora_weights()
    print("✅ Training complete!")

if __name__ == "__main__":
    main() 