#!/usr/bin/env python3
"""
Simple JSON Pipeline Runner

This script runs the integrated JSON pipeline with predefined configurations.
"""

from integrated_db_pipeline import IntegratedJSONPipeline

def main():
    """Run the JSON pipeline with predefined settings."""
    
    # 🎯 CUSTOMIZE THESE SETTINGS
    # User-voice pairs (username -> voice_name)
    user_voice_pairs = [
        ("elonmusk", "elonmusk"),
        ("AOC", "christina"),
        ("sama", "freeman"),
        ("dril", "daniel"),
        ("NASA", "barackobama")
    ]
    
    # Tweet selection options
    SELECTION_TYPE = "top"  # Options: "top", "latest", "random"
    TWEET_COUNT = 5         # Number of tweets per user
    MAX_LENGTH = 256        # Maximum summary length
    JSON_DIR = "data/json_tweets"  # Directory with JSON files
    
    print("🎬 Starting Integrated JSON Pipeline")
    print("=" * 60)
    print(f"📋 Processing: {user_voice_pairs}")
    print(f"🎯 Selection: {SELECTION_TYPE} tweets")
    print(f"📊 Count: {TWEET_COUNT} tweets per user")
    print(f"📝 Max length: {MAX_LENGTH} characters")
    print(f"📁 JSON directory: {JSON_DIR}")
    print("=" * 60)
    
    # Initialize and run pipeline
    pipeline = IntegratedJSONPipeline(JSON_DIR)
    
    try:
        # Process all users
        results = pipeline.run_batch_processing(
            user_voice_pairs,
            selection_type=SELECTION_TYPE,
            count=TWEET_COUNT
        )
        
        # Print final summary
        print(f"\n{'='*80}")
        print("🎉 PROCESSING COMPLETE")
        print(f"{'='*80}")
        
        successful = 0
        for result in results:
            if "error" not in result:
                successful += 1
                print(f"✅ @{result['username']} → {result['voice_name']} → {result['audio_path']}")
                print(f"   📝 Summary: {result['summary'][:80]}...")
                print(f"   🏆 Score: {result['score']:.4f}")
                print(f"   📊 Tweets: {result['tweet_count']} {SELECTION_TYPE} tweets")
            else:
                # For error results, we need to handle the case where username might not be present
                username = result.get('username', 'Unknown')
                print(f"❌ @{username}: {result['error']}")
        
        print(f"\n📊 Success Rate: {successful}/{len(user_voice_pairs)} users processed successfully")
        
        # Show available audio files
        if successful > 0:
            print(f"\n🎵 Generated Audio Files:")
            for result in results:
                if "error" not in result:
                    print(f"   🔊 {result['audio_path']}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pipeline.close()

if __name__ == "__main__":
    main() 