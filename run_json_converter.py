#!/usr/bin/env python3
"""
Simple JSON Converter Runner

This script runs the database to JSON converter with default settings.
"""

from convert_db_to_json import convert_all_users_to_json, show_json_sample

def main():
    """Run the JSON converter."""
    
    print("🔄 Converting Database to JSON Files")
    print("=" * 50)
    
    # Convert all users to JSON
    summary = convert_all_users_to_json(
        db_path="tweets_database.db",
        output_dir="json_tweets"
    )
    
    if summary and summary.get('total_tweets', 0) > 0:
        print(f"\n📋 CONVERSION SUMMARY:")
        print(f"   Total users: {summary['total_users']}")
        print(f"   Total tweets: {summary['total_tweets']}")
        print(f"   Files saved in: json_tweets/")
        
        # Show sample for first user
        users = list(summary.get('users', {}).keys())
        if users:
            print(f"\n📄 Sample JSON structure for @{users[0]}:")
            show_json_sample(users[0], "json_tweets")
    else:
        print("❌ No data found to convert. Make sure you have run the scraper first.")

if __name__ == "__main__":
    main() 