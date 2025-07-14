import torch
from transformers import AutoTokenizer
from model import setup_lora_model
import argparse
import json
import os
from typing import Optional, Dict, Any

class SummarizationInference:
    """
    Simple inference class for post summarization using LoRA fine-tuned models
    """
    
    def __init__(self, model_id: str = "Qwen/Qwen2-0.5B", lora_weights_path: Optional[str] = None, 
                 device: str = "auto", max_length: int = 150, r: int = 16, alpha: int = 32):
        """
        Initialize the summarization inference system
        
        Args:
            model_id: HuggingFace model identifier (default: base Qwen2-0.5B)
            lora_weights_path: Path to saved LoRA weights (optional)
            device: Device to run inference on
            max_length: Maximum length of generated summary
            r: LoRA rank parameter
            alpha: LoRA alpha parameter
        """
        self.model_id = model_id
        self.lora_weights_path = lora_weights_path
        self.max_length = max_length
        self.r = r
        self.alpha = alpha
        
        # Auto-detect device if not specified
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"🚀 Initializing inference on {self.device}")
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()
        
        # Load LoRA weights if provided
        if lora_weights_path:
            self._load_lora_weights()
    
    def _load_model_and_tokenizer(self):
        """Load the base model with LoRA and tokenizer"""
        print(f"📦 Loading model: {self.model_id}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with LoRA setup
        self.model, self.lora_model = setup_lora_model(
            self.model_id, 
            device=self.device, 
            r=self.r, 
            alpha=self.alpha
        )
        
        # Set to evaluation mode
        self.model.eval()
        print("✅ Model loaded and ready for inference")
    
    def _load_lora_weights(self):
        """Load pre-trained LoRA weights with dtype compatibility handling"""
        print(f"🔄 Loading LoRA weights from: {self.lora_weights_path}")
        
        try:
            # Load the weights
            lora_weights = torch.load(self.lora_weights_path, map_location=self.device)
            
            # Get the target dtype from the model
            target_dtype = next(self.model.parameters()).dtype
            print(f"🔍 Target model dtype: {target_dtype}")
            
            # Convert weights to the correct dtype if needed
            converted_weights = {}
            dtype_conversions = 0
            
            for key, weight in lora_weights.items():
                if isinstance(weight, torch.Tensor) and weight.dtype != target_dtype:
                    converted_weights[key] = weight.to(target_dtype)
                    dtype_conversions += 1
                else:
                    converted_weights[key] = weight
            
            if dtype_conversions > 0:
                print(f"🔄 Converted {dtype_conversions} weights to {target_dtype}")
            
            # Load weights into model
            missing_keys, unexpected_keys = self.model.load_state_dict(converted_weights, strict=False)
            
            print(f"✅ LoRA weights loaded successfully")
            if missing_keys:
                print(f"⚠️  Missing keys: {len(missing_keys)}")
                # Only show first few missing keys to avoid spam
                if len(missing_keys) > 5:
                    print(f"   First few: {missing_keys[:5]}")
                else:
                    print(f"   Missing: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
                # Only show first few unexpected keys to avoid spam
                if len(unexpected_keys) > 5:
                    print(f"   First few: {unexpected_keys[:5]}")
                else:
                    print(f"   Unexpected: {unexpected_keys}")
                
        except Exception as e:
            print(f"❌ Error loading LoRA weights: {e}")
            print("🔄 Continuing with base LoRA initialization...")
            
            # Try to provide helpful debugging info
            try:
                checkpoint = torch.load(self.lora_weights_path, map_location='cpu')
                print(f"🔍 Checkpoint contains {len(checkpoint)} keys")
                print(f"🔍 Sample keys: {list(checkpoint.keys())[:3]}...")
                
                # Check if this looks like a LoRA checkpoint
                lora_keys = [k for k in checkpoint.keys() if 'lora' in k.lower()]
                print(f"🔍 Found {len(lora_keys)} LoRA-related keys")
                
            except Exception as debug_e:
                print(f"🔍 Could not inspect checkpoint: {debug_e}")
    
    def create_prompt(self, post: str) -> str:
        """
        Create input for summarization - just the raw post text
        
        Args:
            post: The post text to summarize
            
        Returns:
            The post text with newlines (model should continue with summary)
        """
        # No preprompting - just the post followed by newlines, model should generate summary
        return f"{post.strip()}\n\n"
    
    def summarize(self, post: str, temperature: float = 0.7, top_p: float = 0.9, 
                  do_sample: bool = True) -> Dict[str, Any]:
        """
        Summarize a given post
        
        Args:
            post: The post text to summarize
            temperature: Sampling temperature (higher = more creative)
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            Dictionary containing summary and metadata
        """
        if not post.strip():
            return {"error": "Empty post provided"}
        
        try:
            # Create prompt
            prompt = self.create_prompt(post)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,  # Adjust based on your model's context length
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate summary
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode output
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the generated part (everything after the original prompt)
            # The prompt ends with "\n\n", so everything after that is the generated summary
            summary = full_output[len(prompt):].strip()
            
            return {
                "summary": summary,
                "original_post": post,
                "prompt_used": prompt,
                "model_id": self.model_id,
                "parameters": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_length": self.max_length
                }
            }
            
        except Exception as e:
            return {"error": f"Inference failed: {str(e)}"}
    
    def batch_summarize(self, posts: list, **kwargs) -> list:
        """
        Summarize multiple posts
        
        Args:
            posts: List of post strings
            **kwargs: Additional arguments for summarize()
            
        Returns:
            List of summary results
        """
        results = []
        for i, post in enumerate(posts):
            print(f"Processing post {i+1}/{len(posts)}")
            result = self.summarize(post, **kwargs)
            results.append(result)
        
        return results

# Default usage example - runs when no specific arguments provided
def demo_inference():
    """Demonstration of base model inference without preprompting"""
    print("🤖 LoRA Summarization Inference")
    print("=" * 50)
    print()
    print("💡 Using base Qwen2-0.5B model")
    print("   - No preprompting - pure LoRA-trained summarization")
    print("   - Direct post text → summary generation")
    print()
    
    # Sample post to summarize
    sample_post = """
    Just had the most incredible weekend trip to the mountains! The weather was perfect 
    for hiking - sunny but not too hot. We started early Saturday morning and hiked 
    about 8 miles to this amazing viewpoint overlooking the valley. The trail was 
    challenging in some parts, especially the steep rocky section near the summit, 
    but totally worth it for the panoramic views. We packed a picnic lunch and ate 
    it while watching eagles soar below us. Sunday we did a more relaxed nature walk 
    and spotted several deer and even a fox. Already planning our next adventure!
    """
    
    try:
        # Check for weights file in parent directory
        weights_path = "../lora_weights.pt"
        if not os.path.exists(weights_path):
            weights_path = "lora_weights.pt"  # Try current directory
            if not os.path.exists(weights_path):
                weights_path = None  # No weights file found
                print("⚠️  No LoRA weights found, using base model")
        
        # Initialize inference with base model
        print("🔄 Loading base model...")
        
        inferencer = SummarizationInference(
            model_id="Qwen/Qwen2-0.5B",  # Base model matching training
            lora_weights_path=weights_path,  # Will use if exists
            device="auto",
            max_length=100
        )
        
        print("📝 Generating summary...")
        result = inferencer.summarize(sample_post, temperature=0.7)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"\n📄 Original Post ({len(sample_post.strip())} chars):")
            print(f"   {sample_post.strip()[:150]}...")
            
            print(f"\n📋 Generated Summary ({len(result['summary'])} chars):")
            print(f"   {result['summary']}")
            
            compression = len(result['summary']) / len(sample_post.strip())
            print(f"\n📊 Compression Ratio: {compression:.1%}")
            
            if weights_path:
                print(f"\n✅ Using LoRA weights: {weights_path}")
            else:
                print(f"\n⚠️  No LoRA weights - using base model only")
        
        print("\n✅ Demo complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Tips:")
        print("   - Make sure you have internet connection (for model download)")
        print("   - Try running with --device cpu if GPU memory is limited")
        print("   - Check that all dependencies are installed")

def main():
    """Command line interface for the summarization tool"""
    parser = argparse.ArgumentParser(description="Summarize posts using LoRA fine-tuned model")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-0.6B-Base", help="HuggingFace model ID (base model)")
    parser.add_argument("--lora_weights", type=str, help="Path to LoRA weights file")
    parser.add_argument("--post", type=str, help="Post text to summarize")
    parser.add_argument("--input_file", type=str, help="JSON file with posts to summarize")
    parser.add_argument("--output_file", type=str, help="Output file for results")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--max_length", type=int, default=150, help="Max summary length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    
    args = parser.parse_args()
    
    # If no arguments provided, run demo
    if not args.post and not args.input_file:
        demo_inference()
        return
    
    # Handle weights path
    weights_path = args.lora_weights
    if weights_path is None:
        # Check for weights file in parent directory then current directory
        if os.path.exists("../lora_weights.pt"):
            weights_path = "../lora_weights.pt"
        elif os.path.exists("lora_weights.pt"):
            weights_path = "lora_weights.pt"
    
    # Initialize inference system
    inferencer = SummarizationInference(
        model_id=args.model_id,
        lora_weights_path=weights_path,
        device=args.device,
        max_length=args.max_length
    )
    
    # Handle single post
    if args.post:
        result = inferencer.summarize(
            args.post, 
            temperature=args.temperature, 
            top_p=args.top_p
        )
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"\n📝 Original Post:\n{result['original_post']}")
            print(f"\n📋 Summary:\n{result['summary']}")
    
    # Handle batch processing
    elif args.input_file:
        print(f"📂 Loading posts from: {args.input_file}")
        
        with open(args.input_file, 'r') as f:
            data = json.load(f)
        
        # Support different JSON formats
        if isinstance(data, list):
            posts = data
        elif isinstance(data, dict) and "posts" in data:
            posts = data["posts"]
        else:
            print("❌ Error: Expected list of posts or {'posts': [...]} format")
            return
        
        # Process posts
        results = inferencer.batch_summarize(
            posts, 
            temperature=args.temperature, 
            top_p=args.top_p
        )
        
        # Save results
        output_file = args.output_file or "summaries.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to: {output_file}")
    
    else:
        print("❌ Please provide either --post or --input_file")

if __name__ == "__main__":
    main() 