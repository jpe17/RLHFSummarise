from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

class SummarizationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Simple format: post text directly followed by summary
        # This teaches the model to generate summaries without preprompting
        text = f"{example['prompt']}\n\n{example['ideal_summary']}"
        
        # Tokenize
        tokens = self.tokenizer(
            text, 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "labels": tokens["input_ids"].squeeze()
        }

def load_data():
    """Load dataset."""
    dataset = load_dataset("json", data_files={
        "train": "data/train.jsonl",
        "validation": "data/valid.jsonl"
    })
    return dataset

def setup_tokenizer(model_id):
    """Setup tokenizer for base QWEN models."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Set pad token - base models typically use eos_token as pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    return tokenizer

def create_dataloaders(dataset, tokenizer, batch_size=4, max_length=512):
    """Create train and validation dataloaders."""
    train_dataset = SummarizationDataset(dataset["train"], tokenizer, max_length)
    val_dataset = SummarizationDataset(dataset["validation"], tokenizer, max_length)
    
    # Reduce num_workers to avoid tokenizer parallelism issues
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    
    print(f"Created dataloaders - Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    return train_loader, val_loader 