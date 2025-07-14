import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r=16, alpha=32):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        
        # Ensure parameters are trainable
        self.lora_A.weight.requires_grad = True
        self.lora_B.weight.requires_grad = True
    
    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scaling

class LoRALinear(nn.Module):
    def __init__(self, original_layer, r=16, alpha=32):
        super().__init__()
        self.original_layer = original_layer
        self.lora = LoRALayer(original_layer.in_features, original_layer.out_features, r, alpha)
        
        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.original_layer(x) + self.lora(x)

def freeze_base_model(model):
    """Freeze all parameters in the base model."""
    for param in model.parameters():
        param.requires_grad = False
    print("✅ All base model parameters frozen")

def apply_lora(model, target_modules=None, r=16, alpha=32):
    """Apply LoRA to specified modules."""
    if target_modules is None:
        # QWEN model attention layer names
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # First, freeze all parameters
    freeze_base_model(model)
    
    applied_count = 0
    
    # Find and replace target modules
    def find_and_replace_modules(model, target_modules, r, alpha):
        nonlocal applied_count
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                # Check if this is a target module
                if any(target in name for target in target_modules):
                    # Replace with LoRA layer
                    lora_layer = LoRALinear(module, r, alpha)
                    setattr(model, name, lora_layer)
                    applied_count += 1
                    print(f"Applied LoRA to: {name}")
            else:
                # Recursively apply to children
                find_and_replace_modules(module, target_modules, r, alpha)
    
    find_and_replace_modules(model, target_modules, r, alpha)
    
    # If no matches found, try broader search
    if applied_count == 0:
        print("Warning: No LoRA layers applied with standard names! Trying broader search...")
        
        def find_attention_layers(model, prefix=""):
            layers_found = []
            for name, module in model.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(module, nn.Linear):
                    layers_found.append(full_name)
                    print(f"Found linear layer: {full_name}")
                else:
                    layers_found.extend(find_attention_layers(module, full_name))
            return layers_found
        
        all_linear_layers = find_attention_layers(model)
        
        # Apply LoRA to attention-related layers
        for layer_name in all_linear_layers:
            if any(keyword in layer_name.lower() for keyword in ["attn", "attention", "self_attn"]):
                # Navigate to the layer
                parts = layer_name.split('.')
                current = model
                for part in parts[:-1]:
                    current = getattr(current, part)
                
                layer = getattr(current, parts[-1])
                if isinstance(layer, nn.Linear):
                    lora_layer = LoRALinear(layer, r, alpha)
                    setattr(current, parts[-1], lora_layer)
                    applied_count += 1
                    print(f"Applied LoRA to (broad search): {layer_name}")
    
    print(f"Applied LoRA to {applied_count} layers")
    return model

def verify_trainable_parameters(model):
    """Verify that the model has trainable parameters."""
    trainable_params = []
    frozen_params = []
    total_trainable = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            total_trainable += param.numel()
            if "lora" in name:
                print(f"✅ Trainable LoRA parameter: {name} ({param.numel():,} params)")
            else:
                print(f"⚠️  Trainable non-LoRA parameter: {name} ({param.numel():,} params)")
        else:
            frozen_params.append(param)
    
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found! Check LoRA application.")
    
    # Check if we have too many trainable parameters (should be only LoRA)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_ratio = total_trainable / total_params
    
    print(f"\nFound {len(trainable_params)} trainable parameter groups")
    print(f"Total trainable parameters: {total_trainable:,}")
    print(f"Trainable ratio: {trainable_ratio:.2%}")
    
    if trainable_ratio > 0.1:  # If more than 10% is trainable, something's wrong
        print("⚠️  Warning: Too many parameters are trainable! Expected only LoRA parameters.")
    
    return trainable_params

def setup_model(model_id, device):
    """Setup model with LoRA."""
    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    
    print("Applying LoRA...")
    model = apply_lora(model, r=16, alpha=32)
    
    # Verify trainable parameters
    verify_trainable_parameters(model)
    
    model = model.to(device)
    
    # Print parameter summary
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nParameter Summary:")
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    return model

def save_lora_weights(model, path="qwen_lora_weights.pt"):
    """Save only LoRA weights."""
    lora_weights = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_weights[f"{name}.lora.lora_A.weight"] = module.lora.lora_A.weight.data
            lora_weights[f"{name}.lora.lora_B.weight"] = module.lora.lora_B.weight.data
    
    torch.save(lora_weights, path)
    print(f"✅ LoRA weights saved to {path}")
    print(f"Saved {len(lora_weights)} LoRA weight tensors") 