#!/usr/bin/env python3
"""
Integrated Database Pipeline

This pipeline uses the stored JSON files instead of scraping Twitter.
It allows selecting top 5 liked posts, latest 5 posts, or random 5 posts
before running the summarization and voice synthesis pipeline.
"""

import json
import os
import random
from datetime import datetime
import argparse
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from backend.tweet_summarizer_pipeline_ppo import TweetSummarizerPipelinePPO
from pipeline_twitter.integrated_tweet_voice_pipeline import IntegratedTweetVoicePipeline

class JSONTweetSelector:
    """Select tweets from JSON files based on different criteria."""
    
    def __init__(self, json_dir="data/json_tweets"):
        self.json_dir = json_dir
    
    def get_available_users(self):
        """Get list of users available in JSON files."""
        try:
            print(f"🔍 Looking for JSON files in: {self.json_dir}")
            if not os.path.exists(self.json_dir):
                print(f"❌ JSON directory does not exist: {self.json_dir}")
                return []
            
            files = os.listdir(self.json_dir)
            print(f"📁 Found {len(files)} files in directory")
            
            users = []
            for filename in files:
                if filename.endswith('_tweets.json'):
                    username = filename.replace('_tweets.json', '')
                    users.append(username)
                    print(f"   ✅ Found user: {username}")
            
            print(f"👥 Total users found: {len(users)}")
            return sorted(users)
            
        except Exception as e:
            print(f"❌ Error getting users from JSON: {e}")
            return []
    
    def load_user_tweets(self, username):
        """Load all tweets for a user from JSON file."""
        try:
            json_file = os.path.join(self.json_dir, f"{username}_tweets.json")
            
            if not os.path.exists(json_file):
                print(f"❌ JSON file not found: {json_file}")
                return []
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return data.get('tweets', [])
            
        except Exception as e:
            print(f"❌ Error loading tweets for @{username}: {e}")
            return []
    
    def get_top_engagement_tweets(self, username, count=5):
        """Get top N most engaged tweets for a user (likes + retweets + replies)."""
        tweets = self.load_user_tweets(username)
        
        if not tweets:
            return []
        
        # Sort by engagement_total (descending), then by date (most recent first)
        sorted_tweets = sorted(tweets, key=lambda x: (x.get('engagement_total', 0), x.get('parsed_date', '')), reverse=True)
        
        # Convert to standard format
        result = []
        for tweet in sorted_tweets[:count]:
            result.append({
                'content': tweet['content'],
                'timestamp': tweet['timestamp'],
                'parsed_date': tweet.get('parsed_date', ''),
                'likes': tweet['likes'],
                'retweets': tweet.get('retweets', 0),
                'replies': tweet.get('replies', 0),
                'engagement_total': tweet.get('engagement_total', 0),
                'scraped_at': tweet.get('scraped_at', 'Unknown')
            })
        
        return result
    
    def get_latest_tweets(self, username, count=5):
        """Get latest N tweets for a user (by parsed_date)."""
        tweets = self.load_user_tweets(username)
        
        if not tweets:
            return []
        
        # Sort by parsed_date (descending) - already most recent first
        sorted_tweets = sorted(tweets, key=lambda x: x.get('parsed_date', ''), reverse=True)
        
        # Convert to standard format
        result = []
        for tweet in sorted_tweets[:count]:
            result.append({
                'content': tweet['content'],
                'timestamp': tweet['timestamp'],
                'parsed_date': tweet.get('parsed_date', ''),
                'likes': tweet['likes'],
                'retweets': tweet.get('retweets', 0),
                'replies': tweet.get('replies', 0),
                'scraped_at': tweet.get('scraped_at', 'Unknown')
            })
        
        return result
    
    def get_random_tweets(self, username, count=5):
        """Get random N tweets for a user."""
        tweets = self.load_user_tweets(username)
        
        if not tweets:
            return []
        
        # Randomly sample tweets, then sort by date (most recent first)
        if len(tweets) <= count:
            selected_tweets = tweets
        else:
            selected_tweets = random.sample(tweets, count)
        
        # Sort selected tweets by date (most recent first)
        selected_tweets.sort(key=lambda x: x.get('parsed_date', ''), reverse=True)
        
        # Convert to standard format
        result = []
        for tweet in selected_tweets:
            result.append({
                'content': tweet['content'],
                'timestamp': tweet['timestamp'],
                'parsed_date': tweet.get('parsed_date', ''),
                'likes': tweet['likes'],
                'retweets': tweet.get('retweets', 0),
                'replies': tweet.get('replies', 0),
                'scraped_at': tweet.get('scraped_at', 'Unknown')
            })
        
        return result
    
    def get_tweets_by_date(self, username, count=5):
        """Get tweets grouped by date, showing the most recent tweets from each day."""
        tweets = self.load_user_tweets(username)
        
        if not tweets:
            return []
        
        # Group tweets by date (YYYY-MM-DD)
        from datetime import datetime
        tweets_by_date = {}
        
        for tweet in tweets:
            try:
                # Parse the parsed_date to get just the date part
                date_str = tweet.get('parsed_date', '')
                if date_str:
                    # Extract just the date part (YYYY-MM-DD)
                    date_only = date_str.split('T')[0]
                    if date_only not in tweets_by_date:
                        tweets_by_date[date_only] = []
                    tweets_by_date[date_only].append(tweet)
            except Exception as e:
                print(f"Warning: Could not parse date for tweet: {e}")
                continue
        
        # Sort dates in descending order (most recent first)
        sorted_dates = sorted(tweets_by_date.keys(), reverse=True)
        
        # Select tweets from each date, prioritizing by engagement
        result = []
        tweets_per_date = max(1, count // len(sorted_dates)) if sorted_dates else count
        
        for date in sorted_dates:
            if len(result) >= count:
                break
                
            date_tweets = tweets_by_date[date]
            # Sort tweets from this date by engagement (descending)
            date_tweets.sort(key=lambda x: x.get('engagement_total', 0), reverse=True)
            
            # Take the top tweets from this date
            for tweet in date_tweets[:tweets_per_date]:
                if len(result) >= count:
                    break
                    
                result.append({
                    'content': tweet['content'],
                    'timestamp': tweet['timestamp'],
                    'parsed_date': tweet.get('parsed_date', ''),
                    'likes': tweet['likes'],
                    'retweets': tweet.get('retweets', 0),
                    'replies': tweet.get('replies', 0),
                    'engagement_total': tweet.get('engagement_total', 0),
                    'scraped_at': tweet.get('scraped_at', 'Unknown'),
                    'date_group': date  # Add date group for UI display
                })
        
        return result

class IntegratedJSONPipeline:
    """Integrated pipeline using JSON files instead of database."""
    
    def __init__(self, json_dir="data/json_tweets"):
        self.json_dir = json_dir
        self.selector = JSONTweetSelector(json_dir)
        self.voice_pipeline = None
        self.summarizer_pipeline = None
    
    def initialize_pipelines(self, ppo_weights="rlhf_summarizer/lora_weights.pt", reward_model="rlhf_summarizer/qwen_reward_model.pt"):
        """Initialize the summarization and voice pipelines."""
        try:
            # Initialize PPO summarizer
            self.summarizer_pipeline = TweetSummarizerPipelinePPO(
                model_id="Qwen/Qwen1.5-0.5B",
                ppo_weights_path=ppo_weights,
                reward_model_path=reward_model,
                device=None  # Let the pipeline auto-detect
            )
            
            # Initialize voice pipeline with the same parameters as the working version
            self.voice_pipeline = IntegratedTweetVoicePipeline(
                model_id="Qwen/Qwen1.5-0.5B",
                ppo_weights_path="rlhf_summarizer/simple_ppo_lora_final_20250716_130239.pt",
                reward_model_path="rlhf_summarizer/qwen_reward_model.pt"
            )
            
            print("✅ Pipelines initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing pipelines: {e}")
            return False
    
    def select_tweets(self, username, selection_type="top", count=5):
        """Select tweets based on the specified criteria."""
        print(f"📊 Selecting {count} {selection_type} tweets for @{username}...")
        
        if selection_type == "top":
            tweets = self.selector.get_top_engagement_tweets(username, count)
            print(f"   🏆 Selected top {len(tweets)} most engaged tweets (sorted by date)")
        elif selection_type == "latest":
            tweets = self.selector.get_latest_tweets(username, count)
            print(f"   ⏰ Selected latest {len(tweets)} tweets")
        elif selection_type == "random":
            tweets = self.selector.get_random_tweets(username, count)
            print(f"   🎲 Selected random {len(tweets)} tweets (sorted by date)")
        else:
            print(f"❌ Unknown selection type: {selection_type}")
            return []
        
        if not tweets:
            print(f"⚠️ No tweets found for @{username}")
            return []
        
        # Display selected tweets with engagement info
        for i, tweet in enumerate(tweets, 1):
            engagement = tweet.get('engagement_total', tweet['likes'] + tweet.get('retweets', 0) + tweet.get('replies', 0))
            date_str = tweet.get('parsed_date', '').split('T')[0] if tweet.get('parsed_date') else 'Unknown'
            print(f"   {i}. 🔥 {engagement} total engagement ({date_str}) - {tweet['content'][:60]}...")
        
        return tweets
    
    def process_user(self, username, voice_name, selection_type="top", count=5, max_length=200):
        """Process a single user with selected tweets."""
        print(f"\n{'='*60}")
        print(f"🎯 PROCESSING @{username} with {voice_name} voice")
        print(f"{'='*60}")
        
        # Select tweets
        tweets = self.select_tweets(username, selection_type, count)
        
        if not tweets:
            return {"error": f"No tweets found for @{username}"}
        
        # Combine tweets into a single text for summarization
        combined_text = "\n\n".join([tweet['content'] for tweet in tweets])
        
        # Generate summary using PPO model
        print(f"🤖 Generating summary with PPO model...")
        try:
            summary = self.summarizer_pipeline.generate_summary(
                combined_text, 
                max_length=max_length
            )
            
            if not summary:
                return {"error": "Summarization failed: No summary generated"}
            
            # Score the summary
            score = self.summarizer_pipeline.score_summary(combined_text, summary)
            
            print(f"✅ Summary generated (Score: {score:.4f})")
            print(f"📝 {summary}")
            
        except Exception as e:
            return {"error": f"Summarization error: {e}"}
        
        # Generate voice
        print(f"🎤 Generating voice with {voice_name}...")
        try:
            # Use the same approach as the working integrated_tweet_voice_pipeline
            audio_path = self.voice_pipeline.synthesize_voice(
                text=summary,
                voice_name=voice_name
            )
            
            if not audio_path:
                return {"error": "Voice generation failed: No audio path returned"}
            
            print(f"✅ Voice generated: {audio_path}")
            
        except Exception as e:
            return {"error": f"Voice generation error: {e}"}
        
        # Return results
        return {
            "username": username,
            "voice_name": voice_name,
            "selection_type": selection_type,
            "tweet_count": len(tweets),
            "summary": summary,
            "score": score,
            "audio_path": audio_path,
            "tweets": tweets
        }
    
    def run_batch_processing(self, user_voice_pairs, selection_type="top", count=5):
        """Process multiple users with their assigned voices."""
        if not self.initialize_pipelines():
            return []
        
        results = []
        
        for username, voice_name in user_voice_pairs:
            try:
                result = self.process_user(username, voice_name, selection_type, count)
                results.append(result)
                
                # Add delay between users
                if len(user_voice_pairs) > 1:
                    print("⏳ Waiting 3 seconds before next user...")
                    import time
                    time.sleep(3)
                    
            except Exception as e:
                print(f"❌ Error processing @{username}: {e}")
                results.append({"username": username, "error": f"Processing failed: {e}"})
        
        return results
    
    def close(self):
        """Clean up resources."""
        if self.summarizer_pipeline:
            self.summarizer_pipeline.close()
        if self.voice_pipeline:
            self.voice_pipeline.close()

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Integrated JSON pipeline for tweet summarization and voice synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --users elonmusk AOC --voices elonmusk christina
  %(prog)s --users dril horse_ebooks --voices freeman barackobama --selection random
  %(prog)s --users NASA --voices freeman --selection latest --count 10
  %(prog)s --interactive
        """
    )
    
    # User and voice arguments
    parser.add_argument(
        "--users", "-u",
        nargs="+",
        help="List of usernames to process"
    )
    
    parser.add_argument(
        "--voices", "-v",
        nargs="+",
        help="List of voice names to use (must match number of users)"
    )
    
    parser.add_argument(
        "--selection", "-s",
        choices=["top", "latest", "random"],
        default="top",
        help="Tweet selection method (default: top)"
    )
    
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=5,
        help="Number of tweets to select per user (default: 5)"
    )
    
    parser.add_argument(
        "--max-length", "-l",
        type=int,
        default=200,
        help="Maximum summary length (default: 200)"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode to select users and voices"
    )
    
    parser.add_argument(
        "--json-dir",
        default="json_tweets",
        help="Directory containing JSON tweet files (default: json_tweets)"
    )
    
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = IntegratedJSONPipeline(args.json_dir)
    
    # Get available users
    available_users = pipeline.selector.get_available_users()
    
    if not available_users:
        print("❌ No users found in JSON files. Run the JSON converter first!")
        print(f"💡 Expected JSON files in: {args.json_dir}/")
        sys.exit(1)
    
    print(f"📊 Available users in JSON files: {', '.join(available_users)}")
    
    # Interactive mode
    if args.interactive:
        print("\n🎯 INTERACTIVE MODE")
        print("=" * 40)
        
        # Select users
        print("\nSelect users to process:")
        for i, user in enumerate(available_users, 1):
            print(f"  {i}. @{user}")
        
        user_indices = input("\nEnter user numbers (comma-separated): ").strip()
        try:
            selected_indices = [int(x.strip()) - 1 for x in user_indices.split(",")]
            selected_users = [available_users[i] for i in selected_indices if 0 <= i < len(available_users)]
        except:
            print("❌ Invalid user selection")
            sys.exit(1)
        
        # Select voices
        available_voices = ["christina", "elonmusk", "barackobama", "freeman", "angie", "daniel", "emma"]
        print(f"\nAvailable voices: {', '.join(available_voices)}")
        
        voice_input = input("Enter voice names (comma-separated, or 'auto' for automatic): ").strip()
        if voice_input.lower() == "auto":
            selected_voices = available_voices[:len(selected_users)]
        else:
            selected_voices = [v.strip() for v in voice_input.split(",")]
        
        # Select tweet selection method
        print(f"\nTweet selection methods: top, latest, random")
        selection = input("Enter selection method (default: top): ").strip() or "top"
        
        # Select count
        count_input = input("Enter number of tweets per user (default: 5): ").strip()
        count = int(count_input) if count_input.isdigit() else 5
        
        user_voice_pairs = list(zip(selected_users, selected_voices))
        
    else:
        # Command line mode
        if not args.users:
            parser.error("--users is required (or use --interactive)")
        
        if not args.voices:
            parser.error("--voices is required (or use --interactive)")
        
        if len(args.users) != len(args.voices):
            parser.error("Number of users must match number of voices")
        
        user_voice_pairs = list(zip(args.users, args.voices))
    
    # Validate users exist in JSON files
    for username, voice in user_voice_pairs:
        if username not in available_users:
            print(f"❌ User @{username} not found in JSON files")
            sys.exit(1)
    
    print(f"\n🚀 Starting integrated JSON pipeline...")
    print(f"📋 Processing: {user_voice_pairs}")
    print(f"🎯 Selection: {args.selection} tweets")
    print(f"📊 Count: {args.count} tweets per user")
    
    try:
        # Process users
        results = pipeline.run_batch_processing(
            user_voice_pairs, 
            selection_type=args.selection,
            count=args.count
        )
        
        # Print results
        print(f"\n{'='*80}")
        print("🎉 PROCESSING COMPLETE")
        print(f"{'='*80}")
        
        successful = 0
        for result in results:
            if "error" not in result:
                successful += 1
                print(f"✅ @{result['username']} → {result['voice_name']} → {result['audio_path']}")
                print(f"   📝 Summary: {result['summary'][:100]}...")
                print(f"   🏆 Score: {result['score']:.4f}")
            else:
                print(f"❌ @{result['username']}: {result['error']}")
        
        print(f"\n📊 Success Rate: {successful}/{len(user_voice_pairs)} users processed successfully")
        
        # Save results if requested
        if args.save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"json_pipeline_results_{timestamp}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Results saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
    finally:
        pipeline.close()

if __name__ == "__main__":
    main() 