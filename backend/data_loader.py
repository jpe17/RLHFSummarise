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
        
        input_text = example['prompt'].strip()
        target_text = example['ideal_summary'].strip()
        
        # Tokenize the separator WITHOUT truncation to get true length
        separator = f"Please summarize:\n\n{input_text}\n\nSummary:"
        sep_tokens = self.tokenizer(separator, return_tensors="pt")["input_ids"].squeeze()
        
        # Now tokenize the full text WITH truncation
        full_text = f"Please summarize:\n\n{input_text}\n\nSummary: {target_text}"
        encoding = self.tokenizer(full_text, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze()
        
        # Calculate mask length properly
        mask_length = min(len(sep_tokens), len(input_ids))
        
        labels = input_ids.clone()
        labels[:mask_length] = -100
        
        return {"input_ids": input_ids, "labels": labels}

def load_data():
    return load_dataset("json", data_files={"train": "data/train.jsonl", "validation": "data/valid.jsonl"})

def setup_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def create_dataloaders(dataset, tokenizer, batch_size=4, max_length=512, max_train_samples=None):
    train_data = dataset["train"]
    val_data = dataset["validation"]
    
    if max_train_samples:
        train_data = train_data.select(range(min(max_train_samples, len(train_data))))
        val_data = val_data.select(range(min(max_train_samples//5, len(val_data))))
    
    train_dataset = SummarizationDataset(train_data, tokenizer, max_length)
    val_dataset = SummarizationDataset(val_data, tokenizer, max_length)
    
    # Simple padding function
    def pad_batch(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        padded = []
        for item in batch:
            input_ids = item["input_ids"]
            labels = item["labels"]
            pad_len = max_len - len(input_ids)
            if pad_len > 0:
                input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=input_ids.dtype)])
                labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=labels.dtype)])
            padded.append({"input_ids": input_ids, "labels": labels})
        
        return {
            "input_ids": torch.stack([item["input_ids"] for item in padded]),
            "labels": torch.stack([item["labels"] for item in padded])
        }
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=pad_batch)
    
    return train_loader, val_loader