#!/usr/bin/env python3
"""
Simple LoRA Training for Summarization
"""

import torch
from tqdm import tqdm
from model import setup_lora_model, save_lora_weights
from data_loader import load_data, setup_tokenizer, create_dataloaders

def main():
    # Config
    MODEL_ID = "Qwen/Qwen1.5-0.5B"
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 5
    MAX_SAMPLES = 10000
    
    # Load everything
    dataset = load_data()
    tokenizer = setup_tokenizer(MODEL_ID)
    model = setup_lora_model(MODEL_ID, DEVICE)
    train_loader, val_loader = create_dataloaders(dataset, tokenizer, BATCH_SIZE, max_train_samples=MAX_SAMPLES)
    
    # Optimizer
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}")
        
        # Train
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            optimizer.zero_grad()
            loss = model(input_ids=input_ids, labels=labels).loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                
                # Debug prints
                print(f"Val batch shape: {input_ids.shape}, Labels shape: {labels.shape}")
                print(f"Labels not -100: {(labels != -100).sum().item()}")
                print(f"Input IDs range: {input_ids.min().item()} to {input_ids.max().item()}")
                
                loss = model(input_ids=input_ids, labels=labels).loss
                print(f"Individual val loss: {loss.item()}")
                val_loss += loss.item()
        
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")
    
    # Save
    save_lora_weights(model, "lora_weights.pt")
    print("Done!")

if __name__ == "__main__":
    main() 