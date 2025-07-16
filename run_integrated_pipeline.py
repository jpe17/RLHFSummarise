#!/usr/bin/env python3
"""
Simple usage example for the Integrated Tweet-to-Voice Pipeline.
Customize the usernames and voices below to fit your needs.
"""

from integrated_tweet_voice_pipeline import IntegratedTweetVoicePipeline

def main():
    # 🎯 CUSTOMIZE THESE SETTINGS
    test_usernames = ["celinedion", "elonmusk", "BarackObama", "NASA"]
    voices = ["christina", "elonmusk", "barackobama", "freeman"]
    
    # Create user-voice pairs
    user_voice_pairs = list(zip(test_usernames, voices))
    
    print("🎬 Starting Integrated Tweet-to-Voice Pipeline")
    print(f"📋 Processing: {user_voice_pairs}")
    
    # Initialize and run pipeline
    pipeline = IntegratedTweetVoicePipeline()
    
    try:
        # Process all users
        results = pipeline.run_batch_processing(user_voice_pairs)
        
        # Print final summary
        print(f"\n{'='*80}")
        print("🎉 PROCESSING COMPLETE")
        print(f"{'='*80}")
        
        successful = 0
        for result in results:
            if "error" not in result and result.get("voice_generated", False):
                successful += 1
                print(f"✅ @{result['username']} → {result['audio_path']}")
            else:
                print(f"❌ @{result['username']}: Failed")
        
        print(f"\n📊 Success Rate: {successful}/{len(user_voice_pairs)} users processed successfully")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
    finally:
        pipeline.close()

if __name__ == "__main__":
    main() 