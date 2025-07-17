#!/usr/bin/env python3
"""
Database to JSON Converter

This script converts the SQLite tweets database to JSON files,
one file per Twitter user, ordered by date descending.
"""

import sqlite3
import json
import os
from datetime import datetime
import re

def parse_timestamp(timestamp_str):
    """
    Parse Twitter timestamp to a sortable datetime.
    Handles various formats like '2h', '1d', 'Jan 15', ISO datetime, etc.
    """
    if not timestamp_str or timestamp_str == "Recent":
        return datetime.now()
    
    # Try to parse ISO datetime format
    try:
        # Remove timezone info if present
        clean_timestamp = timestamp_str.split('+')[0].split('Z')[0]
        return datetime.fromisoformat(clean_timestamp.replace('T', ' '))
    except:
        pass
    
    # Try to parse relative time formats
    relative_patterns = [
        (r'(\d+)h', lambda m: datetime.now().replace(hour=datetime.now().hour - int(m.group(1)))),
        (r'(\d+)d', lambda m: datetime.now().replace(day=datetime.now().day - int(m.group(1)))),
        (r'(\d+)w', lambda m: datetime.now().replace(day=datetime.now().day - int(m.group(1)) * 7)),
    ]
    
    for pattern, func in relative_patterns:
        match = re.match(pattern, timestamp_str)
        if match:
            try:
                return func(match)
            except:
                pass
    
    # Try to parse month day format (e.g., "Jan 15")
    month_pattern = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d+)'
    match = re.match(month_pattern, timestamp_str)
    if match:
        try:
            month_map = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            }
            month = month_map[match.group(1)]
            day = int(match.group(2))
            current_year = datetime.now().year
            return datetime(current_year, month, day)
        except:
            pass
    
    # If all parsing fails, return current time
    return datetime.now()

def get_all_users(db_path="tweets_database.db"):
    """Get all unique usernames from the database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT DISTINCT username FROM tweets ORDER BY username')
        users = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return users
        
    except Exception as e:
        print(f"❌ Error getting users: {e}")
        return []

def convert_user_to_json(username, db_path="tweets_database.db", output_dir="data/json_tweets"):
    """
    Convert all tweets for a specific user to JSON format.
    
    Args:
        username (str): Twitter username
        db_path (str): Path to SQLite database
        output_dir (str): Directory to save JSON files
    
    Returns:
        int: Number of tweets converted
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tweets for this user
        cursor.execute('''
            SELECT content, timestamp, likes, retweets, replies, scraped_at
            FROM tweets 
            WHERE username = ?
            ORDER BY timestamp DESC
        ''', (username,))
        
        tweets = cursor.fetchall()
        conn.close()
        
        if not tweets:
            print(f"⚠️ No tweets found for @{username}")
            return 0
        
        # Convert to JSON format with parsed dates
        json_tweets = []
        for content, timestamp, likes, retweets, replies, scraped_at in tweets:
            # Parse the timestamp for sorting
            parsed_date = parse_timestamp(timestamp)
            
            tweet_data = {
                "content": content,
                "timestamp": timestamp,
                "parsed_date": parsed_date.isoformat(),
                "likes": likes,
                "retweets": retweets,
                "replies": replies,
                "scraped_at": scraped_at,
                "engagement_total": likes + retweets + replies
            }
            json_tweets.append(tweet_data)
        
        # Sort by parsed date (newest first)
        json_tweets.sort(key=lambda x: x['parsed_date'], reverse=True)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to JSON file
        output_file = os.path.join(output_dir, f"{username}_tweets.json")
        
        # Create the final JSON structure
        json_data = {
            "username": username,
            "total_tweets": len(json_tweets),
            "export_date": datetime.now().isoformat(),
            "tweets": json_tweets
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Converted {len(json_tweets)} tweets for @{username} -> {output_file}")
        return len(json_tweets)
        
    except Exception as e:
        print(f"❌ Error converting tweets for @{username}: {e}")
        return 0

def convert_all_users_to_json(db_path="tweets_database.db", output_dir="data/json_tweets"):
    """
    Convert all users in the database to JSON files.
    
    Args:
        db_path (str): Path to SQLite database
        output_dir (str): Directory to save JSON files
    
    Returns:
        dict: Summary of conversion results
    """
    print("🔄 Converting database to JSON files...")
    print(f"📁 Output directory: {output_dir}")
    print(f"💾 Database: {db_path}")
    
    # Get all users
    users = get_all_users(db_path)
    
    if not users:
        print("❌ No users found in database")
        return {}
    
    print(f"👥 Found {len(users)} users: {', '.join(users)}")
    
    # Convert each user
    results = {}
    total_tweets = 0
    
    for i, username in enumerate(users, 1):
        print(f"\n📱 Processing @{username} ({i}/{len(users)})")
        tweet_count = convert_user_to_json(username, db_path, output_dir)
        results[username] = tweet_count
        total_tweets += tweet_count
    
    # Create summary file
    summary_file = os.path.join(output_dir, "conversion_summary.json")
    summary = {
        "conversion_date": datetime.now().isoformat(),
        "database_path": db_path,
        "total_users": len(users),
        "total_tweets": total_tweets,
        "users": results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 CONVERSION COMPLETE!")
    print(f"📊 Total tweets converted: {total_tweets}")
    print(f"👥 Total users: {len(users)}")
    print(f"📁 Files saved in: {output_dir}")
    print(f"📋 Summary: {summary_file}")
    
    return summary

def show_json_sample(username, json_dir="data/json_tweets"):
    """Show a sample of the JSON structure for a user."""
    json_file = os.path.join(json_dir, f"{username}_tweets.json")
    
    if not os.path.exists(json_file):
        print(f"❌ JSON file not found: {json_file}")
        return
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n📄 JSON Structure for @{username}:")
        print("=" * 60)
        print(f"Username: {data['username']}")
        print(f"Total tweets: {data['total_tweets']}")
        print(f"Export date: {data['export_date']}")
        
        if data['tweets']:
            print(f"\n📝 Sample tweet (first one):")
            tweet = data['tweets'][0]
            print(f"   Content: {tweet['content'][:100]}...")
            print(f"   Timestamp: {tweet['timestamp']}")
            print(f"   Parsed date: {tweet['parsed_date']}")
            print(f"   Likes: {tweet['likes']}")
            print(f"   Retweets: {tweet['retweets']}")
            print(f"   Replies: {tweet['replies']}")
            print(f"   Total engagement: {tweet['engagement_total']}")
            print(f"   Scraped at: {tweet['scraped_at']}")
        
    except Exception as e:
        print(f"❌ Error reading JSON file: {e}")

def main():
    """Main function with options."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert SQLite database to JSON files')
    parser.add_argument('--db', default='tweets_database.db', help='Database file path')
    parser.add_argument('--output', default='data/json_tweets', help='Output directory')
    parser.add_argument('--user', help='Convert only specific user')
    parser.add_argument('--sample', help='Show sample JSON for specific user')
    
    args = parser.parse_args()
    
    if args.sample:
        show_json_sample(args.sample, args.output)
        return
    
    if args.user:
        # Convert only specific user
        tweet_count = convert_user_to_json(args.user, args.db, args.output)
        if tweet_count > 0:
            show_json_sample(args.user, args.output)
    else:
        # Convert all users
        convert_all_users_to_json(args.db, args.output)

if __name__ == "__main__":
    main() 