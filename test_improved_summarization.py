#!/usr/bin/env python3
"""
Test script to demonstrate improved summarization with different strategies.
"""

import sys
import os
sys.path.append('backend')

from tweet_summarizer_pipeline_ppo import TweetSummarizerPipelinePPO

def test_summarization_strategies():
    """Test different summarization strategies."""
    
    # Sample tweet data (simulating mixed content from different users)
    sample_tweets = [
        {
            "content": "Just finished an amazing workout at the gym! 💪 Feeling stronger every day. #fitness #motivation #gymlife",
            "likes": "150",
            "retweets": "25",
            "replies": "10",
            "user": "user1"
        },
        {
            "content": "Breaking: New research shows that climate change is accelerating faster than previously thought. We need immediate action! 🌍 #ClimateChange #Environment",
            "likes": "500",
            "retweets": "200",
            "replies": "85",
            "user": "user2"
        },
        {
            "content": "Happy birthday to my amazing daughter! 🎂 So proud of how much you've grown. Love you! #family #birthday #proud",
            "likes": "75",
            "retweets": "5",
            "replies": "20",
            "user": "user1"
        },
        {
            "content": "The stock market is showing interesting patterns today. Tech stocks are up 3% while energy is down 2%. 📈📉 #stocks #investing #market",
            "likes": "300",
            "retweets": "50",
            "replies": "40",
            "user": "user3"
        },
        {
            "content": "Just tried the new restaurant downtown. The pasta was incredible! 🍝 Highly recommend for date night. #food #restaurant #datenight",
            "likes": "80",
            "retweets": "15",
            "replies": "12",
            "user": "user1"
        }
    ]
    
    print("🚀 Testing Improved Summarization Strategies")
    print("=" * 60)
    
    # Initialize pipeline
    try:
        pipeline = TweetSummarizerPipelinePPO()
        
        # Combine tweets
        combined_text = pipeline.combine_tweets(sample_tweets)
        
        print(f"\n📝 Original Combined Text ({len(combined_text)} chars):")
        print("-" * 40)
        print(combined_text[:500] + "..." if len(combined_text) > 500 else combined_text)
        
        # Test different strategies
        strategies = ["conservative", "adaptive", "creative"]
        
        for strategy in strategies:
            print(f"\n🎯 Testing {strategy.upper()} Strategy:")
            print("-" * 40)
            
            try:
                summary = pipeline.generate_summary(combined_text, strategy=strategy)
                score = pipeline.score_summary(combined_text, summary)
                
                print(f"📊 Summary ({len(summary)} chars, score: {score:.4f}):")
                print(f"   {summary}")
                
            except Exception as e:
                print(f"❌ Error with {strategy} strategy: {e}")
        
        # Test with custom parameters
        print(f"\n🔧 Testing Custom Parameters:")
        print("-" * 40)
        
        try:
            custom_summary = pipeline.generate_summary(
                combined_text, 
                strategy="adaptive",
                temperature=0.5,  # Lower temperature for more focused output
                max_new_tokens=150  # Shorter summary
            )
            custom_score = pipeline.score_summary(combined_text, custom_summary)
            
            print(f"📊 Custom Summary ({len(custom_summary)} chars, score: {custom_score:.4f}):")
            print(f"   {custom_summary}")
            
        except Exception as e:
            print(f"❌ Error with custom parameters: {e}")
        
        pipeline.close()
        
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        print("💡 Make sure the model weights are available")

if __name__ == "__main__":
    test_summarization_strategies() 