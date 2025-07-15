import torch
from transformers import AutoTokenizer
from model import setup_lora_model
import os
import glob
import re
from datetime import datetime

# Config
model_id = "Qwen/Qwen1.5-0.5B"
device = "cuda" if torch.cuda.is_available() else "cpu"
max_length = 100

def find_latest_weights():
    """Find the latest weights file based on timestamp"""
    # Look for weight files in current directory and parent directory
    search_paths = [".", ".."]
    weight_files = []
    
    for search_path in search_paths:
        # Pattern to match both epoch and final weights
        patterns = [
            os.path.join(search_path, "lora_weights_epoch*_*.pt"),
            os.path.join(search_path, "lora_weights_final_*.pt")
        ]
        
        for pattern in patterns:
            weight_files.extend(glob.glob(pattern))
    
    if not weight_files:
        return None
    
    # Parse timestamps and find the latest
    latest_file = None
    latest_timestamp = None
    
    for file_path in weight_files:
        # Extract timestamp from filename
        # Pattern: lora_weights_epoch{epoch}_{timestamp}.pt or lora_weights_final_{timestamp}.pt
        match = re.search(r'_(\d{8}_\d{6})\.pt$', file_path)
        if match:
            timestamp_str = match.group(1)
            try:
                # Parse timestamp (format: YYYYMMDD_HHMMSS)
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                if latest_timestamp is None or timestamp > latest_timestamp:
                    latest_timestamp = timestamp
                    latest_file = file_path
            except ValueError:
                continue
    
    return latest_file

# Find latest weights or use provided path
weights_path = find_latest_weights()
if weights_path:
    print(f"Using latest weights: {weights_path}")
else:
    print("No weights file found")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = setup_lora_model(model_id, device=device, r=16, alpha=32)
model.eval()

# Load LoRA weights if available
if weights_path:
    lora_weights = torch.load(weights_path, map_location=device)
    target_dtype = next(model.parameters()).dtype
    
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A'):
            lora_a_key = f"{name}.lora_A.weight"
            lora_b_key = f"{name}.lora_B.weight"
            
            if lora_a_key in lora_weights and lora_b_key in lora_weights:
                module.lora_A.weight.data = lora_weights[lora_a_key].to(target_dtype).to(device)
                module.lora_B.weight.data = lora_weights[lora_b_key].to(target_dtype).to(device)

# Sample post
post = """
TIFU by trying to impress my crush with my "cooking skills"

So this happened last weekend and I'm still cringing about it. I (22M) have been talking to this girl (21F) from my chemistry class for a few weeks now, and she mentioned she loves homemade pasta. Being the genius I am, I decided to invite her over for dinner and make "authentic Italian carbonara" despite never having cooked anything more complex than ramen.

I watched exactly one YouTube video and thought I was Gordon Ramsay. Bought all the ingredients, spent way too much money on "good" parmesan, and hyped it up all week. She comes over, I'm trying to be all smooth and confident, talking about how my "nonna taught me this recipe" (I'm not even Italian lol).

Everything that could go wrong did. First, I didn't realize you're supposed to take the pan off heat before adding the eggs. Scrambled eggs in pasta. Then I somehow managed to burn the pancetta while it was literally sitting in water. The kitchen filled with smoke, fire alarm goes off, and I'm frantically trying to fan it with a dish towel while she's just standing there watching this disaster unfold.

The final product looked like cheesy scrambled eggs with burnt meat chunks. She was super sweet about it and we ended up ordering pizza, but I could tell she was trying not to laugh. We're still talking but now she keeps sending me cooking memes and asking if I need help with "basic life skills."

"""

# Create prompt and generate
prompt = f"{post.strip()}\n\n"
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1
    )

# Decode and extract summary
full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
summary = full_output[len(prompt):].strip()

# Print results
print(f"Original: {post.strip()[:100]}...")
print(f"Summary: {summary}")
print(f"Compression: {len(summary)/len(post.strip()):.1%}")