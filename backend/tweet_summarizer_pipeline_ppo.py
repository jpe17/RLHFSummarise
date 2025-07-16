import torch
import sys
import os
from datetime import datetime
import json
import re

try:
    from dateutil import parser as date_parser
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False
    print("⚠️ Warning: python-dateutil not installed. Date filtering may be limited.")

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twitter_scraper_selenium import TwitterSeleniumScraper
from model import setup_lora_model, load_lora_weights
from reward import load_reward_model
from data_loader import setup_tokenizer

class TweetSummarizerPipelinePPO:
    def __init__(self, 
                 model_id="Qwen/Qwen1.5-0.5B",
                 ppo_weights_path="lora_weights.pt",
                 reward_model_path="qwen_reward_model.pt",
                 device=None):
        """
        Initialize the complete pipeline for tweet summarization and scoring using PPO-trained model.
        
        Args:
            model_id (str): Hugging Face model ID
            ppo_weights_path (str): Path to PPO-trained LoRA weights
            reward_model_path (str): Path to reward model weights
            device (str): Device to run models on
        """
        self.model_id = model_id
        self.ppo_weights_path = ppo_weights_path
        self.reward_model_path = reward_model_path
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"🔧 Initializing PPO-trained pipeline on device: {self.device}")
        print(f"🎯 Using PPO weights: {self.ppo_weights_path}")
        
        # Initialize components
        self.scraper = None
        self.summarizer_model = None
        self.reward_model = None
        self.tokenizer = None
        
        self._setup_models()
    
    def _setup_models(self):
        """Set up the summarizer and reward models."""
        try:
            # Setup tokenizer
            print("📝 Setting up tokenizer...")
            self.tokenizer = setup_tokenizer(self.model_id)
            
            # Setup summarizer model with PPO-trained LoRA weights
            print("🤖 Loading PPO-trained summarizer model...")
            self.summarizer_model = setup_lora_model(self.model_id, device=self.device, r=16, alpha=32)
            
            # Load PPO-trained LoRA weights
            if os.path.exists(self.ppo_weights_path):
                # Check if this is actually a PPO policy weights file
                try:
                    weights = torch.load(self.ppo_weights_path, map_location='cpu')
                    if len(weights) < 10:  # Value head has only 2 keys, LoRA should have many more
                        print(f"⚠️ Warning: {self.ppo_weights_path} appears to be value head weights, not policy weights")
                        print(f"⚠️ Falling back to regular LoRA weights: lora_weights.pt")
                        if os.path.exists("lora_weights.pt"):
                            self.summarizer_model = load_lora_weights(self.summarizer_model, "lora_weights.pt")
                            print(f"✅ Loaded fallback LoRA weights from lora_weights.pt")
                        else:
                            print(f"⚠️ No fallback weights found, using base model")
                    else:
                        self.summarizer_model = load_lora_weights(self.summarizer_model, self.ppo_weights_path)
                        print(f"✅ Loaded PPO-trained LoRA weights from {self.ppo_weights_path}")
                except Exception as e:
                    print(f"⚠️ Error loading PPO weights: {e}")
                    print(f"⚠️ Falling back to regular LoRA weights: lora_weights.pt")
                    if os.path.exists("lora_weights.pt"):
                        self.summarizer_model = load_lora_weights(self.summarizer_model, "lora_weights.pt")
                        print(f"✅ Loaded fallback LoRA weights from lora_weights.pt")
            else:
                print(f"⚠️ PPO weights not found at {self.ppo_weights_path}")
                print(f"⚠️ Falling back to regular LoRA weights: lora_weights.pt")
                if os.path.exists("lora_weights.pt"):
                    self.summarizer_model = load_lora_weights(self.summarizer_model, "lora_weights.pt")
                    print(f"✅ Loaded fallback LoRA weights from lora_weights.pt")
                else:
                    print(f"⚠️ No fallback weights found, using base model")
            
            self.summarizer_model.eval()
            
            # Setup reward model
            print("🏆 Loading reward model...")
            if os.path.exists(self.reward_model_path):
                self.reward_model, _ = load_reward_model(self.reward_model_path, self.device)
                self.reward_model.eval()
                print(f"✅ Loaded reward model from {self.reward_model_path}")
            else:
                print(f"⚠️ Reward model not found at {self.reward_model_path}")
                self.reward_model = None
            
        except Exception as e:
            print(f"❌ Error setting up models: {e}")
            raise
    
    def _setup_scraper(self):
        """Initialize the Twitter scraper when needed."""
        if self.scraper is None:
            print("🐦 Initializing Twitter scraper...")
            self.scraper = TwitterSeleniumScraper(headless=True)
    
    def get_tweets(self, username, count=10, since_date=None, until_date=None):
        """
        Get tweets from a Twitter user.
        
        Args:
            username (str): Twitter username (without @)
            count (int): Number of tweets to fetch
            since_date (datetime): Only include tweets after this date
            until_date (datetime): Only include tweets before this date
            
        Returns:
            list: List of tweet dictionaries
        """
        self._setup_scraper()
        
        print(f"📱 Fetching {count} tweets from @{username}...")
        if since_date or until_date:
            print(f"   Date range: {since_date.strftime('%Y-%m-%d') if since_date else 'any'} to {until_date.strftime('%Y-%m-%d') if until_date else 'any'}")
        
        tweets = self.scraper.get_user_tweets(username, count, since_date, until_date)
        
        if not tweets:
            print(f"⚠️ No tweets found for @{username}")
            return []
        
        print(f"✅ Successfully fetched {len(tweets)} tweets")
        return tweets
    
    def combine_tweets(self, tweets):
        """
        Combine tweets into a single text for summarization.
        
        Args:
            tweets (list): List of tweet dictionaries
            
        Returns:
            str: Combined tweet text
        """
        if not tweets:
            return ""
        
        # Sort tweets by engagement (likes + retweets + replies)
        def get_engagement_score(tweet):
            try:
                likes = int(tweet.get('likes', '0').replace(',', ''))
                retweets = int(tweet.get('retweets', '0').replace(',', ''))
                replies = int(tweet.get('replies', '0').replace(',', ''))
                return likes + retweets + replies
            except (ValueError, TypeError):
                return 0
        
        # Sort by engagement and take top tweets
        sorted_tweets = sorted(tweets, key=get_engagement_score, reverse=True)
        
        # Combine tweets into a single text (only content, no formatting)
        combined_text = ""
        for i, tweet in enumerate(sorted_tweets, 1):
            combined_text += f"{tweet['content']}\n\n"
        
        return combined_text.strip()
    
    def generate_summary(self, text, max_length=200, temperature=0.7):
        """
        Generate a summary of the given text using the PPO-trained model.
        
        Args:
            text (str): Text to summarize
            max_length (int): Maximum length of summary
            temperature (float): Generation temperature
            
        Returns:
            str: Generated summary
        """
        if not text.strip():
            return ""
        
        # Create prompt for summarization (same format as PPO training)
        prompt = f"Please summarize the following tweets:\n\n{text}\n\nSummary:"
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512, 
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        print("🔄 Generating summary with PPO-trained model...")
        
        with torch.no_grad():
            outputs = self.summarizer_model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode and extract summary
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = full_output[len(prompt):].strip()
        
        print(f"✅ Generated PPO summary ({len(summary)} characters)")
        return summary
    
    def score_summary(self, original_text, summary):
        """
        Score the quality of a summary using the reward model.
        
        Args:
            original_text (str): Original text that was summarized
            summary (str): Generated summary
            
        Returns:
            float: Quality score (higher is better)
        """
        if self.reward_model is None:
            print("⚠️ Reward model not available, returning default score")
            return 0.0
        
        if not summary.strip():
            return 0.0
        
        # Format text for reward model (same format as training)
        formatted_text = f"Post: {original_text}\nSummary: {summary}"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        print("🏆 Scoring summary...")
        
        with torch.no_grad():
            score = self.reward_model(
                inputs["input_ids"], 
                inputs["attention_mask"]
            )
            
            # Convert to float and handle batch dimension
            if score.dim() > 0:
                score = score.item()
            else:
                score = float(score)
        
        print(f"✅ Summary score: {score:.4f}")
        return score
    
    def filter_tweets_by_date(self, tweets, since_date=None, until_date=None):
        """
        Filters a list of tweets to include only those within a specified date range.
        
        Args:
            tweets (list): List of tweet dictionaries
            since_date (datetime): Only include tweets after this date
            until_date (datetime): Only include tweets before this date
            
        Returns:
            list: Filtered list of tweets
        """
        if not since_date and not until_date:
            return tweets
        
        filtered_tweets = []
        for tweet in tweets:
            tweet_timestamp = tweet.get('timestamp')
            if not tweet_timestamp or tweet_timestamp == "Unknown":
                continue  # Skip tweets without a valid timestamp
            
            try:
                # Parse the timestamp - Twitter uses ISO format
                if 'T' in tweet_timestamp and tweet_timestamp.endswith('Z'):
                    # ISO format: 2024-01-01T12:00:00.000Z
                    tweet_date = datetime.fromisoformat(tweet_timestamp.replace('Z', '+00:00'))
                elif 'T' in tweet_timestamp:
                    # ISO format without Z: 2024-01-01T12:00:00
                    tweet_date = datetime.fromisoformat(tweet_timestamp)
                elif HAS_DATEUTIL:
                    # Try to parse with dateutil as fallback
                    tweet_date = date_parser.parse(tweet_timestamp)
                else:
                    # Basic parsing for common formats
                    formats = [
                        "%Y-%m-%d %H:%M:%S",
                        "%Y-%m-%d",
                        "%m/%d/%Y",
                        "%d/%m/%Y"
                    ]
                    tweet_date = None
                    for fmt in formats:
                        try:
                            tweet_date = datetime.strptime(tweet_timestamp, fmt)
                            break
                        except ValueError:
                            continue
                    
                    if tweet_date is None:
                        raise ValueError(f"Could not parse timestamp format: {tweet_timestamp}")
                
                # Convert to naive datetime if needed (remove timezone info for comparison)
                if hasattr(tweet_date, 'tzinfo') and tweet_date.tzinfo is not None:
                    tweet_date = tweet_date.replace(tzinfo=None)
                
            except (ValueError, TypeError) as e:
                print(f"⚠️ Could not parse timestamp '{tweet_timestamp}': {e}")
                continue  # Skip tweets with invalid timestamp format
            
            # Apply date filters
            if since_date and tweet_date < since_date:
                continue
            if until_date and tweet_date > until_date:
                continue
            
            filtered_tweets.append(tweet)
        
        return filtered_tweets

    def process_user(self, username, tweet_count=10, summary_max_length=200, since_date=None, until_date=None):
        """
        Complete pipeline: scrape tweets, summarize, and score using PPO-trained model.
        
        Args:
            username (str): Twitter username (without @)
            tweet_count (int): Number of tweets to fetch
            summary_max_length (int): Maximum length of summary
            since_date (datetime): Only include tweets after this date
            until_date (datetime): Only include tweets before this date
            
        Returns:
            dict: Results containing tweets, summary, and score
        """
        try:
            # Step 1: Get latest tweets (just tweet_count)
            tweets = self.get_tweets(username, tweet_count, since_date, until_date)
            if not tweets:
                return {
                    "username": username,
                    "error": "No tweets found",
                    "tweets": [],
                    "combined_text": "",
                    "summary": "",
                    "score": 0.0,
                    "model_type": "PPO-trained"
                }
            
            # Step 2: Combine tweets
            combined_text = self.combine_tweets(tweets)
            if not combined_text:
                return {
                    "username": username,
                    "error": "No valid tweet content",
                    "tweets": tweets,
                    "combined_text": "",
                    "summary": "",
                    "score": 0.0,
                    "model_type": "PPO-trained"
                }
            
            # Step 3: Generate summary
            summary = self.generate_summary(combined_text, summary_max_length)
            if not summary:
                return {
                    "username": username,
                    "error": "Failed to generate summary",
                    "tweets": tweets,
                    "combined_text": combined_text,
                    "summary": "",
                    "score": 0.0,
                    "model_type": "PPO-trained"
                }
            
            # Step 4: Score summary
            score = self.score_summary(combined_text, summary)
            
            return {
                "username": username,
                "tweets": tweets,
                "combined_text": combined_text,
                "summary": summary,
                "score": score,
                "timestamp": datetime.now().isoformat(),
                "model_type": "PPO-trained",
                "ppo_weights_path": self.ppo_weights_path,
                "stats": {
                    "tweet_count": len(tweets),
                    "original_length": len(combined_text),
                    "summary_length": len(summary),
                    "compression_ratio": len(summary) / len(combined_text) if combined_text else 0
                }
            }
            
        except Exception as e:
            print(f"❌ Error processing user @{username}: {e}")
            return {
                "username": username,
                "error": str(e),
                "tweets": [],
                "combined_text": "",
                "summary": "",
                "score": 0.0,
                "model_type": "PPO-trained"
            }

    def print_results(self, results):
        """
        Print results in a formatted way.
        
        Args:
            results (dict): Results from process_user
        """
        print(f"\n{'='*80}")
        print(f"📊 PPO-TRAINED RESULTS FOR @{results['username']}")
        print(f"{'='*80}")
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
        
        # Print model info
        print(f"🎯 Model: {results.get('model_type', 'PPO-trained')}")
        print(f"📂 Weights: {results.get('ppo_weights_path', 'Unknown')}")
        
        # Print stats
        stats = results.get("stats", {})
        print(f"📈 Statistics:")
        print(f"   • Tweets processed: {stats.get('tweet_count', 0)}")
        print(f"   • Original length: {stats.get('original_length', 0):,} characters")
        print(f"   • Summary length: {stats.get('summary_length', 0):,} characters")
        print(f"   • Compression ratio: {stats.get('compression_ratio', 0):.1%}")
        print(f"   • Quality score: {results.get('score', 0):.4f}")
        
        # Print summary
        print(f"\n📝 PPO Summary:")
        print(f"   {results.get('summary', 'No summary available')}")
        
        # Print top tweets
        print(f"\n🐦 Top Tweets:")
        for i, tweet in enumerate(results.get('tweets', [])[:5], 1):
            print(f"   {i}. {tweet['content'][:100]}...")
            print(f"      ❤️ {tweet.get('likes', '0')} | 🔄 {tweet.get('retweets', '0')} | 💬 {tweet.get('replies', '0')}")
    
    def save_results(self, results, filename=None):
        """
        Save results to a JSON file.
        
        Args:
            results (dict): Results from process_user
            filename (str): Optional filename, auto-generated if None
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ppo_tweet_summary_{results['username']}_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 PPO results saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def close(self):
        """Clean up resources."""
        if self.scraper:
            self.scraper.close()
        print("🧹 PPO pipeline closed")

def main():
    """Demo the complete PPO pipeline."""
    # Initialize PPO pipeline
    pipeline = TweetSummarizerPipelinePPO()
    
    try:
        # Test usernames
        test_usernames = ["elonmusk", "@BarackObama", "@NASA"]
        
        for username in test_usernames:
            print(f"\n🚀 Processing @{username} with PPO-trained model...")
            
            # Process user
            results = pipeline.process_user(username, tweet_count=10)
            
            # Print results
            pipeline.print_results(results)
            
            # Save results
            pipeline.save_results(results)
            
            print(f"\n⏳ Waiting before next user...")
            import time
            time.sleep(5)  # Rate limiting
    
    finally:
        pipeline.close()

if __name__ == "__main__":
    main() 