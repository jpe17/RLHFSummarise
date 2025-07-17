#!/usr/bin/env python3
"""
Twitter Database Scraper Runner

This script runs the Twitter database scraper to collect tweets from famous people
and store them in a SQLite database.
"""

import sys
import os
from twitter_scraper_database import TwitterDatabaseScraper

def main():
    """Run the Twitter database scraper."""
    
    # Configuration
    HEADLESS = True  # Set to False to see the browser
    DB_PATH = "tweets_database.db"
    TWEETS_PER_USER = 100
    
    # List of famous people to scrape
    FAMOUS_PEOPLE = [
        "AOC",              # Alexandria Ocasio-Cortez
        "sama",             # Sam Altman
        "elonmusk",         # Elon Musk
        "realDonaldTrump",  # Donald Trump
        "dril",             # Dril (famous Twitter personality)
        "horse_ebooks",     # Horse ebooks (famous Twitter bot)
        "BarackObama",      # Barack Obama
        "NASA",             # NASA
        "elonmusk",         # Elon Musk (duplicate, but keeping for variety)
    ]
    
    # Remove duplicates while preserving order
    unique_people = list(dict.fromkeys(FAMOUS_PEOPLE))
    
    print("🚀 Twitter Database Scraper")
    print("=" * 50)
    print(f"📋 Will scrape {TWEETS_PER_USER} tweets each from {len(unique_people)} users")
    print(f"💾 Database: {DB_PATH}")
    print(f"🌐 Headless mode: {HEADLESS}")
    print("=" * 50)
    
    # Initialize scraper
    scraper = TwitterDatabaseScraper(headless=HEADLESS, db_path=DB_PATH)
    
    if not scraper.driver:
        print("❌ Failed to initialize WebDriver. Exiting.")
        sys.exit(1)
    
    try:
        total_stored = 0
        
        for i, username in enumerate(unique_people, 1):
            print(f"\n📱 Processing @{username} ({i}/{len(unique_people)})")
            
            # Scrape and store tweets
            stored_count = scraper.scrape_and_store_tweets(username, target_count=TWEETS_PER_USER)
            total_stored += stored_count
            
            print(f"✅ Stored {stored_count} new tweets for @{username}")
            
            # Add delay between requests (be respectful to Twitter)
            if i < len(unique_people):
                print("⏳ Waiting 5 seconds before next user...")
                import time
                time.sleep(5)
        
        print(f"\n🎉 SCRAPING COMPLETE!")
        print(f"✅ Total new tweets stored: {total_stored}")
        
        # Show final statistics
        scraper.get_database_stats()
        
    except KeyboardInterrupt:
        print("\n⚠️ Scraping interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during scraping: {e}")
    finally:
        scraper.close()
        print("\n👋 Scraper closed")

if __name__ == "__main__":
    main() 