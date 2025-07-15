#!/usr/bin/env python3
"""
Tweet Summarizer Pipeline - Command Line Interface

This script provides a simple command-line interface to run the complete
tweet summarization and scoring pipeline.

Usage:
    python run_tweet_summarizer.py username [options]

Examples:
    python run_tweet_summarizer.py elonmusk
    python run_tweet_summarizer.py dril --count 15 --save
    python run_tweet_summarizer.py horse_ebooks --count 5 --max-length 150
"""

import argparse
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.tweet_summarizer_pipeline import TweetSummarizerPipeline

def main():
    parser = argparse.ArgumentParser(
        description="Scrape tweets, generate summaries, and score them using AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s elonmusk
  %(prog)s dril --count 15 --save
  %(prog)s horse_ebooks --count 5 --max-length 150
  %(prog)s username --device cuda --lora-weights my_weights.pt
        """
    )
    
    # Required arguments
    parser.add_argument(
        "username",
        help="Twitter username (without @) to analyze"
    )
    
    # Optional arguments
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=10,
        help="Number of tweets to fetch (default: 10)"
    )
    
    parser.add_argument(
        "--max-length", "-l",
        type=int,
        default=200,
        help="Maximum length of generated summary (default: 200)"
    )
    
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Save results to a JSON file"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output filename (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "--device", "-d",
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="Device to run models on (default: auto)"
    )
    
    parser.add_argument(
        "--lora-weights",
        default="lora_weights.pt",
        help="Path to LoRA weights file (default: lora_weights.pt)"
    )
    
    parser.add_argument(
        "--reward-model",
        default="qwen_reward_model.pt",
        help="Path to reward model file (default: qwen_reward_model.pt)"
    )
    
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen1.5-0.5B",
        help="Hugging Face model ID (default: Qwen/Qwen1.5-0.5B)"
    )
    
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress detailed output, only show final results"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.count <= 0:
        parser.error("Tweet count must be positive")
    
    if args.max_length <= 0:
        parser.error("Max length must be positive")
    
    if not (0.0 <= args.temperature <= 2.0):
        parser.error("Temperature must be between 0.0 and 2.0")
    
    # Set device
    device = None if args.device == "auto" else args.device
    
    try:
        # Initialize pipeline
        if not args.quiet:
            print(f"🚀 Initializing Tweet Summarizer Pipeline...")
            print(f"   • Username: @{args.username}")
            print(f"   • Tweet count: {args.count}")
            print(f"   • Max summary length: {args.max_length}")
            print(f"   • Temperature: {args.temperature}")
            print(f"   • Device: {args.device}")
            print(f"   • LoRA weights: {args.lora_weights}")
            print(f"   • Reward model: {args.reward_model}")
            print()
        
        pipeline = TweetSummarizerPipeline(
            model_id=args.model_id,
            lora_weights_path=args.lora_weights,
            reward_model_path=args.reward_model,
            device=device
        )
        
        # Process user
        if not args.quiet:
            print(f"🔄 Processing @{args.username}...")
        
        results = pipeline.process_user(
            username=args.username,
            tweet_count=args.count,
            summary_max_length=args.max_length
        )
        
        # Print results
        if args.quiet:
            # Quiet mode - just essential info
            if "error" in results:
                print(f"❌ Error: {results['error']}")
                sys.exit(1)
            
            print(f"@{results['username']} Summary (Score: {results['score']:.4f}):")
            print(f"{results['summary']}")
        else:
            # Full output
            pipeline.print_results(results)
        
        # Save results if requested
        if args.save:
            pipeline.save_results(results, args.output)
        
        # Clean up
        pipeline.close()
        
        if not args.quiet:
            print(f"\n✅ Pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 