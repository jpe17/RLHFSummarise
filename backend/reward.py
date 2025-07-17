import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm
import json
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
from model import setup_lora_model, load_lora_weights
from data_loader import setup_tokenizer

# Configuration
MODEL_ID = "Qwen/Qwen1.5-0.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 16  # Increased for much faster training
MAX_LENGTH = 256  # Reduced for faster processing
LR = 5e-5  # Higher learning rate for faster convergence
EPOCHS = 3  # Fewer epochs, but more efficient
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
LORA_WEIGHTS_PATH = "rlhf_summarizer/lora_weights.pt"

# Improved reward model
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # Better reward head architecture
        hidden_size = base_model.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Better initialization - start with smaller weights
        for module in self.reward_head:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=0.1)  # Smaller gain
                torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask):
        # Get model outputs
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
        
        # Get the last hidden state
        last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # Vectorized way to get last non-padding token for each sequence
        batch_size = last_hidden_state.size(0)
        seq_len = last_hidden_state.size(1)
        
        # Find the last non-padding position for each sequence in the batch
        # attention_mask is 1 for real tokens, 0 for padding
        last_positions = attention_mask.sum(dim=1) - 1  # [batch_size]
        last_positions = last_positions.clamp(min=0)  # Ensure non-negative
        
        # Use advanced indexing to get the last token embeddings
        batch_indices = torch.arange(batch_size, device=last_hidden_state.device)
        last_token_embeddings = last_hidden_state[batch_indices, last_positions]  # [batch_size, hidden_size]
        
        # Get rewards for all sequences at once
        rewards = self.reward_head(last_token_embeddings).squeeze(-1)  # [batch_size]
        
        return rewards

# Load and process data
def load_data():
    with open("data/comparisons_train.jsonl", 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Better data filtering and processing
    filtered = []
    for item in data:
        pos_ex = item["pos_ex"].strip()
        neg_ex = item["neg_ex"].strip()
        post = item["post"].strip()
        
        # Skip if examples are identical or too short
        if pos_ex != neg_ex and len(pos_ex) > 10 and len(neg_ex) > 10 and len(post) > 10:
            filtered.append({
                "prompt": post,
                "chosen": pos_ex,
                "rejected": neg_ex
            })
    
    return filtered

# Better preprocessing - process individual examples, not batches
def preprocess_example(example):
    prompt = example["prompt"]
    chosen = example["chosen"]
    rejected = example["rejected"]
    
    # Create full text for chosen and rejected
    chosen_text = f"Post: {prompt}\nSummary: {chosen}"
    rejected_text = f"Post: {prompt}\nSummary: {rejected}"
    
    return {
        "chosen_text": chosen_text,
        "rejected_text": rejected_text
    }

# Custom collate function for DataLoader
def collate_fn(batch):
    chosen_texts = [item["chosen_text"] for item in batch]
    rejected_texts = [item["rejected_text"] for item in batch]
    
    # Tokenize the batch
    chosen_tokens = tokenizer(
        chosen_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    rejected_tokens = tokenizer(
        rejected_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    return {
        "chosen_input_ids": chosen_tokens["input_ids"],
        "chosen_attention_mask": chosen_tokens["attention_mask"],
        "rejected_input_ids": rejected_tokens["input_ids"],
        "rejected_attention_mask": rejected_tokens["attention_mask"]
    }

# Save model
def save_reward_model(model, path="rlhf_summarizer/qwen_reward_model.pt"):
    reward_state = {
        'reward_head': model.reward_head.state_dict(),
        'model_config': {
            'model_id': MODEL_ID,
            'hidden_size': model.base_model.config.hidden_size,
            'lora_weights_path': LORA_WEIGHTS_PATH
        }
    }
    torch.save(reward_state, path)
    print(f"✅ Reward model saved as {path}")

def load_reward_model(reward_model_path="rlhf_summarizer/qwen_reward_model.pt", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    reward_state = torch.load(reward_model_path, map_location=device)
    model_config = reward_state['model_config']
    
    # Use the same tokenizer setup function
    tokenizer = setup_tokenizer(model_config['model_id'])
    
    base_model = setup_lora_model(model_config['model_id'], device)
    
    # Handle relative path when called from backend directory
    lora_path = model_config['lora_weights_path']
    
    # Update old paths to new rlhf_summarizer location
    if lora_path == "lora_weights.pt":
        lora_path = "rlhf_summarizer/lora_weights.pt"
    elif lora_path == "simple_ppo_lora_final_20250716_130239.pt":
        lora_path = "rlhf_summarizer/simple_ppo_lora_final_20250716_130239.pt"
    
    if not os.path.isabs(lora_path) and not os.path.exists(lora_path):
        # Try parent directory
        parent_path = os.path.join('..', lora_path)
        if os.path.exists(parent_path):
            lora_path = parent_path
    
    base_model = load_lora_weights(base_model, lora_path)
    
    reward_model = RewardModel(base_model).to(device)
    reward_model.reward_head.load_state_dict(reward_state['reward_head'])
    
    print(f"✅ Loaded reward model from {reward_model_path}")
    return reward_model, tokenizer

if __name__ == '__main__':
    print(f"Starting training for {EPOCHS} epochs...")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Using device: {DEVICE}")
    if DEVICE == "cuda":
        print("Using mixed precision training")

    # Setup tokenizer using the same function as data_loader
    tokenizer = setup_tokenizer(MODEL_ID)

    base_model = setup_lora_model(MODEL_ID, DEVICE)
    base_model = load_lora_weights(base_model, LORA_WEIGHTS_PATH)

    model = RewardModel(base_model).to(DEVICE)

    data = load_data()
    print(f"Loaded {len(data)} comparison pairs")

    # Create dataset
    dataset = Dataset.from_list(data)
    split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split["train"]
    test_dataset = split["test"]

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Process datasets - map individual examples
    train_data = train_dataset.map(preprocess_example)
    test_data = test_dataset.map(preprocess_example)

    # Create dataloaders with optimization - set num_workers=0 for macOS compatibility
    train_loader = DataLoader(
        train_data, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 for macOS compatibility
        pin_memory=True if DEVICE == "cuda" else False  # Faster GPU transfer
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 for macOS compatibility
        pin_memory=True if DEVICE == "cuda" else False
    )

    # Training setup with optimizations
    optimizer = AdamW(model.reward_head.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Mixed precision training for faster performance (if using CUDA)
    scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None

    best_accuracy = 0.0
    EVAL_EVERY_N_EPOCHS = 1  # Evaluate every epoch, but you can increase this

    for epoch in range(EPOCHS):
        # Training
        model.train()
        total_loss = 0
        num_batches = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Training")
        
        for step, batch in enumerate(train_bar):
            chosen_ids = batch["chosen_input_ids"].to(DEVICE, non_blocking=True)
            chosen_mask = batch["chosen_attention_mask"].to(DEVICE, non_blocking=True)
            rejected_ids = batch["rejected_input_ids"].to(DEVICE, non_blocking=True)
            rejected_mask = batch["rejected_attention_mask"].to(DEVICE, non_blocking=True)
            
            # Mixed precision forward pass
            if DEVICE == "cuda":
                with torch.cuda.amp.autocast():
                    # Forward pass
                    chosen_rewards = model(chosen_ids, chosen_mask)
                    rejected_rewards = model(rejected_ids, rejected_mask)
                    
                    # Better loss function - Bradley-Terry model
                    logits = chosen_rewards - rejected_rewards
                    loss = -torch.nn.functional.logsigmoid(logits).mean()
                    
                    # Reduced regularization to prevent reward hacking
                    reward_penalty = 0.001 * (chosen_rewards.pow(2).mean() + rejected_rewards.pow(2).mean())
                    loss = loss + reward_penalty
                    
                    # Scale loss for gradient accumulation
                    loss = loss / GRADIENT_ACCUMULATION_STEPS
            else:
                # Forward pass without mixed precision
                chosen_rewards = model(chosen_ids, chosen_mask)
                rejected_rewards = model(rejected_ids, rejected_mask)
                
                # Better loss function - Bradley-Terry model
                logits = chosen_rewards - rejected_rewards
                loss = -torch.nn.functional.logsigmoid(logits).mean()
                
                # Reduced regularization to prevent reward hacking
                reward_penalty = 0.001 * (chosen_rewards.pow(2).mean() + rejected_rewards.pow(2).mean())
                loss = loss + reward_penalty
                
                # Scale loss for gradient accumulation
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass with mixed precision
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.reward_head.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.reward_head.parameters(), max_norm=1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
            
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS  # Unscale for display
            num_batches += 1
            
            # Update progress bar with more detailed info (less frequently for speed)
            if step % 10 == 0:  # Update every 10 steps instead of every step
                train_bar.set_postfix({
                    'loss': f'{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}',
                    'main_loss': f'{(-torch.nn.functional.logsigmoid(logits).mean()).item():.4f}',
                    'penalty': f'{reward_penalty.item():.4f}',
                    'chosen_reward': f'{chosen_rewards.mean().item():.2f}',
                    'rejected_reward': f'{rejected_rewards.mean().item():.2f}',
                    'avg_loss': f'{total_loss/num_batches:.4f}'
                })
        
        scheduler.step()
        
        # Evaluation
        if (epoch + 1) % EVAL_EVERY_N_EPOCHS == 0:
            model.eval()
            correct = 0
            total = 0
            eval_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Evaluating"):
                    chosen_ids = batch["chosen_input_ids"].to(DEVICE, non_blocking=True)
                    chosen_mask = batch["chosen_attention_mask"].to(DEVICE, non_blocking=True)
                    rejected_ids = batch["rejected_input_ids"].to(DEVICE, non_blocking=True)
                    rejected_mask = batch["rejected_attention_mask"].to(DEVICE, non_blocking=True)
                    
                    chosen_rewards = model(chosen_ids, chosen_mask)
                    rejected_rewards = model(rejected_ids, rejected_mask)
                    
                    # Calculate accuracy
                    predictions = (chosen_rewards > rejected_rewards).float()
                    correct += predictions.sum().item()
                    total += predictions.size(0)
                    
                    # Calculate evaluation loss
                    logits = chosen_rewards - rejected_rewards
                    batch_loss = -torch.nn.functional.logsigmoid(logits).mean()
                    eval_loss += batch_loss.item()
            
            accuracy = correct / total if total > 0 else 0.0
            avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0
            avg_eval_loss = eval_loss / len(test_loader) if len(test_loader) > 0 else 0.0
            
            print(f"Epoch {epoch + 1}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Eval Loss: {avg_eval_loss:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                print(f"  🎉 New best accuracy: {best_accuracy:.4f}")
        else:
            # Just print training loss when not evaluating
            avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0
            print(f"Epoch {epoch + 1}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")

    save_reward_model(model)
    print("✅ Training completed successfully!")
    print(f"Best accuracy achieved: {best_accuracy:.4f}")