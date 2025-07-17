import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

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

def save_lora_weights(model, path="rlhf_summarizer/lora_weights.pt"):
    weights = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A'):
            weights[f"{name}.lora_A.weight"] = module.lora_A.weight.data
            weights[f"{name}.lora_B.weight"] = module.lora_B.weight.data
    
    torch.save(weights, path)
    print(f"Saved LoRA weights to {path}")

def load_lora_weights(model, path="rlhf_summarizer/lora_weights.pt"):
    """Load LoRA weights from a saved file"""
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