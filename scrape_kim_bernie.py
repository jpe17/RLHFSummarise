#!/usr/bin/env python3
"""
Simple script to scrape tweets from Kim Kardashian and Bernie Sanders.
"""

from twitter_scraper_database import TwitterDatabaseScraper
import time

def main():
    """Scrape tweets from Kim Kardashian and Bernie Sanders."""
    
    # Initialize the scraper
    scraper = TwitterDatabaseScraper(headless=True)
    
    # Users to scrape
    users = [
        "KimKardashian",
        "BernieSanders"
    ]
    
    try:
        print("🚀 Starting Twitter Scraper for Kim Kardashian and Bernie Sanders")
        print("=" * 70)
        
        total_stored = 0
        
        for i, username in enumerate(users, 1):
            print(f"\n📱 Scraping @{username} ({i}/{len(users)})")
            print("-" * 50)
            
            # Scrape 100 tweets from each user
            stored_count = scraper.scrape_and_store_tweets(username, target_count=100)
            total_stored += stored_count
            
            # Wait between users to be respectful
            if i < len(users):
                print("⏳ Waiting 5 seconds...")
                time.sleep(5)
        
        print(f"\n🎉 Scraping Complete!")
        print(f"✅ Total new tweets stored: {total_stored}")
        
        # Show database stats
        scraper.get_database_stats()
        
        # Show top tweets from each user
        print(f"\n🏆 Top Tweets from Each User:")
        for username in users:
            top_tweets = scraper.get_top_liked_tweets_from_db(username, count=5)
            if top_tweets:
                print(f"\n📱 @{username} - Top 5 most liked tweets:")
                for i, tweet in enumerate(top_tweets, 1):
                    print(f"   {i}. ❤️ {tweet['likes']} - {tweet['content'][:100]}...")
    
    except KeyboardInterrupt:
        print("\n⚠️ Scraping interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during scraping: {e}")
    finally:
        scraper.close()
        print("\n🔒 Scraper closed")

if __name__ == "__main__":
    main() 