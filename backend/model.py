import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer that learns a low-rank decomposition:
    ΔW = B @ A where A is (in_features, r) and B is (r, out_features)
    """
    def __init__(self, in_features, out_features, r=16, alpha=32):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r  # Scaling factor for stability
        
        # Low-rank matrices: W_down (A) and W_up (B)
        self.lora_A = nn.Linear(in_features, r, bias=False)  # Down-projection
        self.lora_B = nn.Linear(r, out_features, bias=False)  # Up-projection
        
        # Initialize: A with small random values, B with zeros
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        # Forward pass: x -> A -> B -> scale
        return self.lora_B(self.lora_A(x)) * self.scaling

class LoRALinear(nn.Module):
    """
    Wraps an existing linear layer with LoRA adaptation:
    output = original_layer(x) + lora_layer(x)
    """
    def __init__(self, original_layer, r=16, alpha=32):
        super().__init__()
        self.original_layer = original_layer
        self.lora = LoRALayer(original_layer.in_features, original_layer.out_features, r, alpha)
        
        # Ensure LoRA layers have the same dtype as the original layer
        if hasattr(original_layer, 'weight') and original_layer.weight is not None:
            self.lora = self.lora.to(dtype=original_layer.weight.dtype)
        
        # Freeze original weights - we only train LoRA parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Combine original output with LoRA adaptation
        original_output = self.original_layer(x)
        lora_output = self.lora(x)
        
        # Ensure dtype consistency
        if original_output.dtype != lora_output.dtype:
            lora_output = lora_output.to(original_output.dtype)
            
        return original_output + lora_output

class LoRAModel:
    """
    Main class that handles LoRA application and management
    """
    def __init__(self, model_id, device="cuda", r=16, alpha=32):
        self.model_id = model_id
        self.device = device
        self.r = r
        self.alpha = alpha
        self.model = None
        
    def load_and_setup(self):
        """Load model and apply LoRA in one step"""
        print(f"🔄 Loading model: {self.model_id}")
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            trust_remote_code=True,
            torch_dtype=torch.float32  # Force float32 for stability
        )
        
        # Apply LoRA
        print("🔄 Applying LoRA...")
        self._apply_lora()
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Show summary
        self._print_summary()
        
        return self.model
    
    def _apply_lora(self):
        """Apply LoRA to attention layers"""
        # Standard attention layer names for most transformer models
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        # First freeze everything
        self._freeze_base_model()
        
        # Apply LoRA to target modules
        applied_count = self._replace_modules_with_lora(target_modules)
        
        # If no standard modules found, try broader search
        if applied_count == 0:
            print("⚠️  Standard attention layers not found. Searching broadly...")
            applied_count = self._apply_lora_broadly()
        
        if applied_count == 0:
            raise ValueError("❌ No suitable layers found for LoRA application!")
        
        print(f"✅ Applied LoRA to {applied_count} layers")
    
    def _freeze_base_model(self):
        """Freeze all original model parameters"""
        for param in self.model.parameters():
            param.requires_grad = False
        print("❄️  Base model parameters frozen")
    
    def _replace_modules_with_lora(self, target_modules):
        """Replace target modules with LoRA versions"""
        applied_count = 0
        
        def replace_recursive(module, prefix=""):
            nonlocal applied_count
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                if isinstance(child, nn.Linear):
                    # Check if this is a target module
                    if any(target in name for target in target_modules):
                        # Replace with LoRA version
                        lora_layer = LoRALinear(child, self.r, self.alpha)
                        setattr(module, name, lora_layer)
                        applied_count += 1
                        print(f"  ✅ {full_name}")
                else:
                    # Recursively check children
                    replace_recursive(child, full_name)
        
        replace_recursive(self.model)
        return applied_count
    
    def _apply_lora_broadly(self):
        """Fallback: apply LoRA to any attention-related linear layers"""
        applied_count = 0
        attention_keywords = ["attn", "attention", "self_attn"]
        
        # Find all linear layers
        linear_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append(name)
        
        # Apply LoRA to attention-related layers
        for layer_name in linear_layers:
            if any(keyword in layer_name.lower() for keyword in attention_keywords):
                # Navigate to the layer and replace it
                parts = layer_name.split('.')
                parent = self.model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                
                original_layer = getattr(parent, parts[-1])
                lora_layer = LoRALinear(original_layer, self.r, self.alpha)
                setattr(parent, parts[-1], lora_layer)
                applied_count += 1
                print(f"  ✅ {layer_name}")
        
        return applied_count
    
    def _print_summary(self):
        """Print training parameter summary"""
        trainable_params = []
        total_trainable = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                total_trainable += param.numel()
                if "lora" in name:
                    print(f"  📚 {name}: {param.numel():,} params")
                else:
                    print(f"  ⚠️  Non-LoRA trainable: {name}: {param.numel():,} params")
        
        if len(trainable_params) == 0:
            raise ValueError("❌ No trainable parameters found!")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_ratio = total_trainable / total_params
        
        print(f"\n📊 Parameter Summary:")
        print(f"   Trainable: {total_trainable:,} / {total_params:,} ({100*trainable_ratio:.2f}%)")
        print(f"   LoRA rank (r): {self.r}, Alpha: {self.alpha}")
        
        if trainable_ratio > 0.1:
            print("⚠️  Warning: >10% parameters trainable - expected only LoRA!")
    
    def save_lora_weights(self, path="lora_weights.pt"):
        """Save only the LoRA adapter weights"""
        if self.model is None:
            raise ValueError("Model not loaded! Call load_and_setup() first.")
        
        lora_weights = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora'):
                # Save both A and B matrices
                lora_weights[f"{name}.lora.lora_A.weight"] = module.lora.lora_A.weight.data
                lora_weights[f"{name}.lora.lora_B.weight"] = module.lora.lora_B.weight.data
        
        torch.save(lora_weights, path)
        print(f"💾 LoRA weights saved to {path}")
        print(f"   Saved {len(lora_weights)} weight tensors")
        
        return path

# Convenience function for quick setup
def setup_lora_model(model_id, device="cuda", r=16, alpha=32):
    """
    Quick setup function - creates and configures a LoRA model in one call
    
    Args:
        model_id: HuggingFace model identifier
        device: Device to load model on
        r: LoRA rank (lower = fewer parameters)
        alpha: LoRA scaling factor (higher = stronger adaptation)
    """
    lora_model = LoRAModel(model_id, device, r, alpha)
    model = lora_model.load_and_setup()
    return model, lora_model 