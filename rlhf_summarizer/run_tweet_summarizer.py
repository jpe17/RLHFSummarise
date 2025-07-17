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
    python run_tweet_summarizer.py username --since 2024-01-01 --until 2024-01-31
"""

import argparse
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from backend.tweet_summarizer_pipeline import TweetSummarizerPipeline

def parse_date(date_str):
    """Parse date string in various formats."""
    formats = [
        "%Y-%m-%d",           # 2024-01-01
        "%Y/%m/%d",           # 2024/01/01
        "%m/%d/%Y",           # 01/01/2024
        "%d/%m/%Y",           # 01/01/2024 (European)
        "%Y-%m-%d %H:%M:%S",  # 2024-01-01 12:00:00
        "%Y-%m-%dT%H:%M:%S",  # 2024-01-01T12:00:00
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # Try relative dates
    date_str_lower = date_str.lower()
    now = datetime.now()
    
    if date_str_lower in ['today', 'now']:
        return now
    elif date_str_lower == 'yesterday':
        return now - timedelta(days=1)
    elif date_str_lower.endswith('d'):
        try:
            days = int(date_str_lower[:-1])
            return now - timedelta(days=days)
        except ValueError:
            pass
    elif date_str_lower.endswith('h'):
        try:
            hours = int(date_str_lower[:-1])
            return now - timedelta(hours=hours)
        except ValueError:
            pass
    elif date_str_lower.endswith('w'):
        try:
            weeks = int(date_str_lower[:-1])
            return now - timedelta(weeks=weeks)
        except ValueError:
            pass
    
    raise ValueError(f"Unable to parse date: {date_str}")

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
  %(prog)s username --use-ppo --ppo-weights simple_ppo_lora_ep15_20250715_160654.pt
  %(prog)s username --since 2024-01-01 --until 2024-01-31
  %(prog)s username --since 7d --count 20
  %(prog)s username --since yesterday --until today
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
        default="rlhf_summarizer/lora_weights.pt",
        help="Path to LoRA weights file (default: rlhf_summarizer/lora_weights.pt)"
    )
    
    parser.add_argument(
        "--use-ppo",
        action="store_true",
        help="Use PPO-trained model instead of baseline LoRA model"
    )
    
    parser.add_argument(
        "--ppo-weights",
        default="simple_ppo_lora_ep15_20250715_160654.pt",
        help="Path to PPO-trained LoRA weights file (default: simple_ppo_lora_ep15_20250715_160654.pt)"
    )
    
    parser.add_argument(
        "--reward-model",
        default="rlhf_summarizer/qwen_reward_model.pt",
        help="Path to reward model file (default: rlhf_summarizer/qwen_reward_model.pt)"
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
        "--since",
        help="Filter tweets since this date (e.g., 2024-01-01, 7d, yesterday)"
    )
    
    parser.add_argument(
        "--until",
        help="Filter tweets until this date (e.g., 2024-01-31, today)"
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
    
    # Parse date filters
    since_date = None
    until_date = None
    
    if args.since:
        try:
            since_date = parse_date(args.since)
        except ValueError as e:
            parser.error(f"Invalid --since date: {e}")
    
    if args.until:
        try:
            until_date = parse_date(args.until)
        except ValueError as e:
            parser.error(f"Invalid --until date: {e}")
    
    # Validate date range
    if since_date and until_date and since_date > until_date:
        parser.error("--since date must be before --until date")
    
    # Set device
    device = None if args.device == "auto" else args.device
    
    try:
        # Choose pipeline based on --use-ppo flag
        if args.use_ppo:
            from backend.tweet_summarizer_pipeline_ppo import TweetSummarizerPipelinePPO
            
            if not args.quiet:
                print(f"🚀 Initializing PPO-Trained Tweet Summarizer Pipeline...")
                print(f"   • Username: @{args.username}")
                print(f"   • Tweet count: {args.count}")
                print(f"   • Max summary length: {args.max_length}")
                print(f"   • Temperature: {args.temperature}")
                print(f"   • Device: {args.device}")
                print(f"   • PPO weights: {args.ppo_weights}")
                print(f"   • Reward model: {args.reward_model}")
                if since_date:
                    print(f"   • Since: {since_date.strftime('%Y-%m-%d %H:%M:%S')}")
                if until_date:
                    print(f"   • Until: {until_date.strftime('%Y-%m-%d %H:%M:%S')}")
                print()
            
            pipeline = TweetSummarizerPipelinePPO(
                model_id=args.model_id,
                ppo_weights_path=args.ppo_weights,
                reward_model_path=args.reward_model,
                device=device
            )
        else:
            # Initialize baseline pipeline
            if not args.quiet:
                print(f"🚀 Initializing Tweet Summarizer Pipeline...")
                print(f"   • Username: @{args.username}")
                print(f"   • Tweet count: {args.count}")
                print(f"   • Max summary length: {args.max_length}")
                print(f"   • Temperature: {args.temperature}")
                print(f"   • Device: {args.device}")
                print(f"   • LoRA weights: {args.lora_weights}")
                print(f"   • Reward model: {args.reward_model}")
                if since_date:
                    print(f"   • Since: {since_date.strftime('%Y-%m-%d %H:%M:%S')}")
                if until_date:
                    print(f"   • Until: {until_date.strftime('%Y-%m-%d %H:%M:%S')}")
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
            summary_max_length=args.max_length,
            since_date=since_date,
            until_date=until_date
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