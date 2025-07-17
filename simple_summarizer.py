#!/usr/bin/env python3
"""
Simple Text Summarizer using LoRA Model
A standalone script that takes a text file and generates a summary using the fine-tuned LoRA model.
Now supports longer texts through chunking and longer summaries.
"""

import torch
import torch.nn as nn
import argparse
import sys
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

class LoRALinear(nn.Module):
    def __init__(self, original_layer, r=16, alpha=32):
        super().__init__()
        self.original_layer = original_layer
        self.scaling = alpha / r
        
        # Create LoRA matrices
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        
        # Match original layer dtype
        if hasattr(original_layer, 'weight'):
            dtype = original_layer.weight.dtype
            self.lora_A = self.lora_A.to(dtype)
            self.lora_B = self.lora_B.to(dtype)
        
        # Freeze original layer
        for param in original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        original = self.original_layer(x)
        lora = self.lora_B(self.lora_A(x)) * self.scaling
        
        if original.dtype != lora.dtype:
            lora = lora.to(original.dtype)
            
        return original + lora

def setup_lora_model(model_id, device="cuda", r=16, alpha=32):
    """Set up the model with LoRA layers."""
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Apply LoRA to attention layers
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(target in name for target in targets):
            # Replace module with LoRA version
            *parent_path, child_name = name.split('.')
            parent = model.get_submodule('.'.join(parent_path)) if parent_path else model
            
            setattr(parent, child_name, LoRALinear(module, r, alpha))
            count += 1
    
    model = model.to(device)
    
    # Print stats
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"Applied LoRA to {count} layers")
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    return model

def load_lora_weights(model, path):
    """Load LoRA weights from a saved file."""
    if not torch.cuda.is_available():
        weights = torch.load(path, map_location='cpu')
    else:
        weights = torch.load(path)
    
    loaded_count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A'):
            lora_A_key = f"{name}.lora_A.weight"
            lora_B_key = f"{name}.lora_B.weight"
            
            if lora_A_key in weights and lora_B_key in weights:
                module.lora_A.weight.data = weights[lora_A_key].to(module.lora_A.weight.device)
                module.lora_B.weight.data = weights[lora_B_key].to(module.lora_B.weight.device)
                loaded_count += 1
    
    print(f"Loaded LoRA weights from {path}")
    print(f"Loaded weights for {loaded_count} LoRA modules")
    return model

def setup_tokenizer(model_id):
    """Set up the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Ensure proper tokenizer configuration for Qwen
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    return tokenizer

def setup_model_and_tokenizer(model_id="Qwen/Qwen1.5-0.5B", 
                             lora_weights_path="rlhf_summarizer/simple_ppo_lora_final_20250716_130239.pt",
                             device=None):
    """
    Set up the model and tokenizer with LoRA weights.
    
    Args:
        model_id: Hugging Face model ID
        lora_weights_path: Path to LoRA weights file
        device: Device to run on (auto-detected if None)
    
    Returns:
        tuple: (model, tokenizer, device)
    """
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"🔧 Setting up model on device: {device}")
    print(f"📝 Loading tokenizer for {model_id}...")
    
    # Setup tokenizer
    tokenizer = setup_tokenizer(model_id)
    
    # Setup model with LoRA
    print(f"🤖 Loading model with LoRA...")
    model = setup_lora_model(model_id, device=device, r=16, alpha=32)
    
    # Load LoRA weights
    if os.path.exists(lora_weights_path):
        model = load_lora_weights(model, lora_weights_path)
        print(f"✅ Loaded LoRA weights from {lora_weights_path}")
    else:
        print(f"⚠️ LoRA weights not found at {lora_weights_path}")
        print(f"⚠️ Using base model without fine-tuning")
    
    model.eval()
    return model, tokenizer, device

def chunk_text(text, tokenizer, max_chunk_tokens=1000, overlap_tokens=100):
    """
    Split long text into chunks that fit within the model's context window.
    
    Args:
        text: Text to chunk
        tokenizer: Tokenizer to use for token counting
        max_chunk_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks
    
    Returns:
        list: List of text chunks
    """
    # Tokenize the entire text
    tokens = tokenizer.encode(text)
    
    if len(tokens) <= max_chunk_tokens:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        # Calculate end position for this chunk
        end = min(start + max_chunk_tokens, len(tokens))
        
        # Extract tokens for this chunk
        chunk_tokens = tokens[start:end]
        
        # Decode back to text
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        
        # Move start position, accounting for overlap
        start = end - overlap_tokens
        if start >= len(tokens):
            break
    
    return chunks

def generate_summary_for_chunk(model, tokenizer, text, max_length=300, temperature=0.7, device="cpu"):
    """
    Generate a summary for a single chunk of text.
    
    Args:
        model: The LoRA model
        tokenizer: The tokenizer
        text: Text to summarize
        max_length: Maximum length of summary
        temperature: Generation temperature
        device: Device to run on
    
    Returns:
        str: Generated summary
    """
    if not text.strip():
        return ""
    
    # Clean and prepare the text
    text = text.strip()
    
    # Create prompt for summarization
    prompt = f"Please summarize the following text:\n\n{text}\n\nSummary:"
    
    # Tokenize input with higher max_length for longer texts
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=2048,  # Increased from 512 to handle longer texts
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate summary
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decode and extract summary
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = full_output[len(prompt):].strip()
    
    return summary

def generate_summary(model, tokenizer, text, max_length=300, temperature=0.7, device="cpu", chunk_size=1000):
    """
    Generate a summary of the given text, handling long texts through chunking.
    
    Args:
        model: The LoRA model
        tokenizer: The tokenizer
        text: Text to summarize
        max_length: Maximum length of summary
        temperature: Generation temperature
        device: Device to run on
        chunk_size: Maximum tokens per chunk
    
    Returns:
        str: Generated summary
    """
    if not text.strip():
        return ""
    
    # Clean and prepare the text
    text = text.strip()
    
    print(f"📏 Input text length: {len(text)} characters")
    
    # Check if text needs chunking
    tokens = tokenizer.encode(text)
    print(f"📏 Input text tokens: {len(tokens)}")
    
    if len(tokens) <= chunk_size:
        # Text fits in one chunk
        print(f"🔄 Generating summary (max {max_length} tokens, temp {temperature})...")
        summary = generate_summary_for_chunk(model, tokenizer, text, max_length, temperature, device)
        print(f"✅ Generated summary ({len(summary)} characters)")
        return summary
    else:
        # Text needs chunking
        print(f"📦 Text is too long, splitting into chunks...")
        chunks = chunk_text(text, tokenizer, chunk_size, overlap_tokens=100)
        print(f"📦 Split into {len(chunks)} chunks")
        
        # Generate summaries for each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"🔄 Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
            chunk_summary = generate_summary_for_chunk(
                model, tokenizer, chunk, 
                max_length=max_length//len(chunks),  # Distribute tokens across chunks
                temperature=temperature, 
                device=device
            )
            chunk_summaries.append(chunk_summary)
        
        # Combine chunk summaries
        combined_summary = " ".join(chunk_summaries)
        
        # If the combined summary is still too long, generate a final summary
        if len(combined_summary) > max_length * 2:
            print(f"🔄 Generating final summary from chunk summaries...")
            final_summary = generate_summary_for_chunk(
                model, tokenizer, combined_summary,
                max_length=max_length,
                temperature=temperature,
                device=device
            )
            print(f"✅ Generated final summary ({len(final_summary)} characters)")
            return final_summary
        else:
            print(f"✅ Generated combined summary ({len(combined_summary)} characters)")
            return combined_summary

def read_text_file(file_path):
    """
    Read text from a file.
    
    Args:
        file_path: Path to the text file
    
    Returns:
        str: File contents
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading file {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate summaries using LoRA model")
    parser.add_argument("input_file", help="Path to the input text file")
    parser.add_argument("--output", "-o", help="Output file path (default: print to stdout)")
    parser.add_argument("--max-length", "-l", type=int, default=300, 
                       help="Maximum length of summary in tokens (default: 300)")
    parser.add_argument("--temperature", "-t", type=float, default=0.7,
                       help="Generation temperature (default: 0.7)")
    parser.add_argument("--model", default="Qwen/Qwen1.5-0.5B",
                       help="Hugging Face model ID (default: Qwen/Qwen1.5-0.5B)")
    parser.add_argument("--weights", default="rlhf_summarizer/simple_ppo_lora_final_20250716_130239.pt",
                       help="Path to LoRA weights file")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], 
                       help="Device to run on (auto-detected if not specified)")
    parser.add_argument("--chunk-size", type=int, default=1000,
                       help="Maximum tokens per chunk for long texts (default: 1000)")
    parser.add_argument("--no-chunking", action="store_true",
                       help="Disable chunking (may truncate long texts)")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"❌ Input file not found: {args.input_file}")
        sys.exit(1)
    
    print("🚀 Enhanced LoRA Text Summarizer")
    print("=" * 50)
    
    try:
        # Setup model and tokenizer
        model, tokenizer, device = setup_model_and_tokenizer(
            model_id=args.model,
            lora_weights_path=args.weights,
            device=args.device
        )
        
        # Read input file
        print(f"📖 Reading input file: {args.input_file}")
        text = read_text_file(args.input_file)
        if text is None:
            sys.exit(1)
        
        # Generate summary
        if args.no_chunking:
            # Use original method with truncation
            summary = generate_summary_for_chunk(
                model, tokenizer, text, 
                max_length=args.max_length, 
                temperature=args.temperature,
                device=device
            )
        else:
            # Use enhanced method with chunking
            summary = generate_summary(
                model, tokenizer, text, 
                max_length=args.max_length, 
                temperature=args.temperature,
                device=device,
                chunk_size=args.chunk_size
            )
        
        # Output results
        print("\n" + "=" * 50)
        print("📋 SUMMARY")
        print("=" * 50)
        print(summary)
        print("=" * 50)
        
        # Save to file if requested
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(summary)
                print(f"💾 Summary saved to: {args.output}")
            except Exception as e:
                print(f"❌ Error saving to {args.output}: {e}")
        
        # Print statistics
        compression_ratio = len(summary) / len(text) if len(text) > 0 else 0
        print(f"📊 Compression ratio: {compression_ratio:.1%}")
        print(f"📏 Original: {len(text)} characters")
        print(f"📏 Summary: {len(summary)} characters")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 