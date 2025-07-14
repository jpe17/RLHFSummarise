# Simple LoRA Training for Summarization

Clean, modular implementation of LoRA training for the Qwen model on summarization tasks.

## Structure

```
backend/
├── train.py      # Main training script
├── model.py      # LoRA model implementation
├── data.py       # Dataset handling
└── requirements.txt
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run training:**
   ```bash
   python train.py
   ```

## What it does

- ✅ **LoRA only**: Trains only LoRA parameters in attention layers (q_proj, v_proj)
- ✅ **Base model frozen**: Qwen1.5-7B stays frozen, only ~0.1% parameters trained
- ✅ **Simple structure**: Clean separation of concerns
- ✅ **Efficient**: Uses PyTorch with custom LoRA implementation

## Files

### `train.py`
- Main training loop
- Handles training and validation
- Saves LoRA weights

### `model.py`
- LoRA layer implementation
- Model setup with LoRA application
- Weight saving utilities

### `data.py`
- Dataset loading and processing
- Tokenizer setup
- DataLoader creation

## Configuration

- **Model**: Qwen/Qwen1.5-7B
- **LoRA rank**: 8
- **Target modules**: q_proj, v_proj (attention only)
- **Training**: 3 epochs, batch size 2
- **Learning rate**: 2e-4

## Output

Training saves `lora_weights.pt` containing only the trained LoRA parameters. 