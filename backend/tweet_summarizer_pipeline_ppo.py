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

# TwitterSeleniumScraper import removed - not needed for modular system
from model import setup_lora_model, load_lora_weights
from reward import load_reward_model
from data_loader import setup_tokenizer
from text_utils import format_tweet_for_summarization, clean_text_for_summarization
import re
import emoji

class TweetSummarizerPipelinePPO:
    def __init__(self, 
                 model_id="Qwen/Qwen1.5-0.5B",
                 ppo_weights_path="rlhf_summarizer/lora_weights.pt",
                 reward_model_path="rlhf_summarizer/qwen_reward_model.pt",
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
                        print(f"⚠️ Falling back to regular LoRA weights: rlhf_summarizer/lora_weights.pt")
                        if os.path.exists("rlhf_summarizer/lora_weights.pt"):
                            self.summarizer_model = load_lora_weights(self.summarizer_model, "rlhf_summarizer/lora_weights.pt")
                            print(f"✅ Loaded fallback LoRA weights from rlhf_summarizer/lora_weights.pt")
                        else:
                            print(f"⚠️ No fallback weights found, using base model")
                    else:
                        self.summarizer_model = load_lora_weights(self.summarizer_model, self.ppo_weights_path)
                        print(f"✅ Loaded PPO-trained LoRA weights from {self.ppo_weights_path}")
                except Exception as e:
                    print(f"⚠️ Error loading PPO weights: {e}")
                    print(f"⚠️ Falling back to regular LoRA weights: rlhf_summarizer/lora_weights.pt")
                    if os.path.exists("rlhf_summarizer/lora_weights.pt"):
                        self.summarizer_model = load_lora_weights(self.summarizer_model, "rlhf_summarizer/lora_weights.pt")
                        print(f"✅ Loaded fallback LoRA weights from rlhf_summarizer/lora_weights.pt")
            else:
                print(f"⚠️ PPO weights not found at {self.ppo_weights_path}")
                print(f"⚠️ Falling back to regular LoRA weights: rlhf_summarizer/lora_weights.pt")
                if os.path.exists("rlhf_summarizer/lora_weights.pt"):
                    self.summarizer_model = load_lora_weights(self.summarizer_model, "rlhf_summarizer/lora_weights.pt")
                    print(f"✅ Loaded fallback LoRA weights from rlhf_summarizer/lora_weights.pt")
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
            print("🐦 Twitter scraper not available - using modular system")
            # TwitterSeleniumScraper removed - modular system handles scraping
            self.scraper = None
    
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
        
        if self.scraper is None:
            print("⚠️ No scraper available - use modular system for tweet fetching")
            return []
        
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
        
        # Group tweets by similar themes or users if possible
        organized_tweets = self._organize_tweets_by_theme(sorted_tweets)
        
        # Use the new text utils to format and clean tweets
        combined_text = format_tweet_for_summarization(organized_tweets)
        
        print(f"🧹 Cleaned {len(organized_tweets)} tweets for summarization")
        
        # Debug: Show before/after cleaning
        print(f"\n🔍 DEBUG: TEXT CLEANING CHECK")
        print(f"📏 Combined text length: {len(combined_text)} characters")
        hashtags = re.findall(r'#\w+', combined_text)
        print(f"🔍 Hashtags found: {len(hashtags)}")
        print(f"🔍 Emojis found: {emoji.emoji_count(combined_text)}")
        if len(hashtags) > 0:
            print(f"⚠️ WARNING: Hashtags still present: {hashtags[:5]}")
        
        return combined_text
    
    def _organize_tweets_by_theme(self, tweets):
        """
        Organize tweets by themes or group similar content together.
        
        Args:
            tweets (list): List of tweet dictionaries
            
        Returns:
            list: Organized tweets
        """
        if not tweets:
            return []
        
        # Simple organization: group by user if multiple users, otherwise keep order
        user_groups = {}
        for tweet in tweets:
            # Extract user from tweet content or use 'unknown'
            user = tweet.get('user', 'unknown')
            if user not in user_groups:
                user_groups[user] = []
            user_groups[user].append(tweet)
        
        # If multiple users, organize by user
        if len(user_groups) > 1:
            organized = []
            for user, user_tweets in user_groups.items():
                # Take top 3 tweets per user to avoid overwhelming
                organized.extend(user_tweets[:3])
            return organized
        else:
            # Single user, return top tweets
            return tweets[:10]  # Limit to top 10 tweets
    
    def generate_summary(self, text, strategy="adaptive", **kwargs):
        """
        Generate a summary of the given text using the PPO-trained model.
        
        Args:
            text (str): Text to summarize
            strategy (str): Generation strategy - "adaptive", "conservative", "creative"
            **kwargs: Generation parameters (e.g., max_length, temperature)
            
        Returns:
            str: Generated summary
        """
        if not text.strip():
            return ""
        
        # Get generation parameters based on strategy
        gen_params = self._get_generation_params(strategy)
        
        # Update with any provided kwargs
        gen_params.update(kwargs)

        # Clean the text for better summarization
        cleaned_text = clean_text_for_summarization(text)
        
        # Create context-aware prompt based on content analysis
        prompt = self._create_context_aware_prompt(cleaned_text)
        
        # Debug: Print the input prompt
        print("\n" + "="*80)
        print("🔍 DEBUG: INPUT PROMPT TO MODEL")
        print("="*80)
        print(prompt)
        print("="*80)
        print(f"📏 Prompt length: {len(prompt)} characters")
        print(f"📏 Original text length: {len(text)} characters")
        print(f"📏 Cleaned text length: {len(cleaned_text)} characters")
        print(f"🎯 Strategy: {strategy}")
        print("="*80 + "\n")
        
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
        
        # Use mixed precision for faster inference if available
        with torch.no_grad():
            if hasattr(torch, 'autocast') and self.device != 'cpu':
                with torch.autocast(device_type='cuda' if 'cuda' in self.device else 'cpu'):
                    outputs = self.summarizer_model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        use_cache=True,
                        **gen_params
                    )
            else:
                outputs = self.summarizer_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=True,
                    **gen_params
                )
        
        # Decode and extract summary
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = full_output[len(prompt):].strip()
        
        # Post-process the summary to remove any remaining artifacts
        summary = self._post_process_summary(summary)
        
        print(f"✅ Generated PPO summary ({len(summary)} characters)")
        return summary
    
    def _get_generation_params(self, strategy):
        """
        Get generation parameters based on the chosen strategy.
        
        Args:
            strategy (str): Generation strategy
            
        Returns:
            dict: Generation parameters
        """
        if strategy == "conservative":
            # More deterministic, focused on accuracy
            return {
                "max_new_tokens": 200,
                "min_length": 40,
                "no_repeat_ngram_size": 3,
                "temperature": 0.6,
                "do_sample": True,
                "top_p": 0.8,
                "top_k": 30,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
        elif strategy == "creative":
            # More creative, diverse outputs
            return {
                "max_new_tokens": 350,
                "min_length": 60,
                "no_repeat_ngram_size": 2,
                "temperature": 1.0,
                "do_sample": True,
                "top_p": 0.95,
                "top_k": 100,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
        else:  # adaptive (default)
            # Balanced approach
            return {
                "max_new_tokens": 300,
                "min_length": 50,
                "no_repeat_ngram_size": 3,
                "temperature": 0.8,
                "do_sample": True,
                "top_p": 0.9,
                "top_k": 50,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
    
    def _create_context_aware_prompt(self, cleaned_text):
        """
        Create a context-aware prompt based on the content analysis.
        
        Args:
            cleaned_text (str): Cleaned text to analyze
            
        Returns:
            str: Context-aware prompt
        """
        # Analyze content to determine best prompt strategy
        lines = cleaned_text.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        # Check if content seems to be from multiple sources/topics
        if len(non_empty_lines) > 5:
            # Multiple diverse posts
            prompt = f"""Please create a concise summary of the following social media posts. Since these posts cover different topics, organize your summary by identifying the main themes and key points from each:

{cleaned_text}

Summary:"""
        elif len(non_empty_lines) <= 2:
            # Very short content
            prompt = f"""Please provide a brief summary of the following social media content, focusing on the main message:

{cleaned_text}

Summary:"""
        else:
            # Standard content
            prompt = f"""Please create a concise and coherent summary of the following social media posts. Focus on the main themes, key messages, and important information:

{cleaned_text}

Summary:"""
        
        return prompt
    
    def _post_process_summary(self, summary):
        """
        Post-process the generated summary to remove artifacts and improve quality.
        
        Args:
            summary (str): Raw generated summary
            
        Returns:
            str: Cleaned summary
        """
        if not summary:
            return ""
        
        # Remove any remaining hashtags that might have slipped through
        summary = re.sub(r'#\w+', '', summary)
        summary = re.sub(r'#[^\s]+', '', summary)
        
        # Remove any remaining @ mentions
        summary = re.sub(r'@\w+', '', summary)
        summary = re.sub(r'@[^\s]+', '', summary)
        
        # Remove URLs that might appear in summary
        summary = re.sub(r'https?://[^\s]+', '', summary)
        summary = re.sub(r'www\.[^\s]+', '', summary)
        
        # Remove emojis from summary
        summary = emoji.replace_emoji(summary, replace='')
        
        # Fix common generation artifacts
        summary = re.sub(r'\s+', ' ', summary)  # Multiple spaces
        summary = re.sub(r'\.{2,}', '.', summary)  # Multiple periods
        summary = re.sub(r',{2,}', ',', summary)  # Multiple commas
        
        # Remove repetitive phrases (common in generation)
        lines = summary.split('.')
        unique_lines = []
        for line in lines:
            line = line.strip()
            if line and line not in unique_lines:
                unique_lines.append(line)
        
        summary = '. '.join(unique_lines)
        
        # Ensure proper capitalization
        if summary and not summary[0].isupper():
            summary = summary[0].upper() + summary[1:]
        
        # Ensure proper ending
        if summary and not summary.endswith(('.', '!', '?')):
            summary += '.'
        
        # Final cleanup
        summary = summary.strip()
        
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

    def process_user(self, username, tweet_count=10, summary_max_length=200, since_date=None, until_date=None, strategy="adaptive"):
        """
        Complete pipeline: scrape tweets, summarize, and score using PPO-trained model.
        
        Args:
            username (str): Twitter username (without @)
            tweet_count (int): Number of tweets to fetch
            summary_max_length (int): Maximum length of summary
            since_date (datetime): Only include tweets after this date
            until_date (datetime): Only include tweets before this date
            strategy (str): Generation strategy - "adaptive", "conservative", "creative"
            
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
                    "model_type": "PPO-trained",
                    "strategy": strategy
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
                    "model_type": "PPO-trained",
                    "strategy": strategy
                }
            
            # Step 3: Generate summary with specified strategy
            summary = self.generate_summary(combined_text, strategy=strategy, max_length=summary_max_length)
            if not summary:
                return {
                    "username": username,
                    "error": "Failed to generate summary",
                    "tweets": tweets,
                    "combined_text": combined_text,
                    "summary": "",
                    "score": 0.0,
                    "model_type": "PPO-trained",
                    "strategy": strategy
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
                "strategy": strategy,
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
                "model_type": "PPO-trained",
                "strategy": strategy
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