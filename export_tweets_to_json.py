#!/usr/bin/env python3
"""
Export scraped tweets from database to JSON format matching existing structure.
"""

import json
import sqlite3
from datetime import datetime
import os

def export_user_tweets_to_json(username, db_path="tweets_database.db", output_dir="data/posts/twitter"):
    """
    Export tweets for a specific user from database to JSON format.
    
    Args:
        username (str): Twitter username
        db_path (str): Path to SQLite database
        output_dir (str): Directory to save JSON files
    """
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tweets for the user
        cursor.execute('''
            SELECT content, timestamp, likes, retweets, replies, scraped_at
            FROM tweets 
            WHERE username = ?
            ORDER BY likes DESC
        ''', (username,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            print(f"⚠️ No tweets found for @{username}")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to the format matching existing JSON files
        posts = []
        for i, (content, timestamp, likes, retweets, replies, scraped_at) in enumerate(results):
            # Create a simple ID
            post_id = f"{username}_tweet_{i+1}"
            
            post = {
                "id": post_id,
                "platform": "twitter",
                "username": username,
                "user_display_name": None,
                "content": content,
                "content_type": "text",
                "media_items": [],
                "timestamp": timestamp,
                "url": None,
                "likes": likes,
                "shares": retweets,  # retweets are called shares in the format
                "comments": replies,
                "views": None,
                "platform_data": {
                    "engagement_total": likes + retweets + replies,
                    "scraped_at": scraped_at
                },
                "scraped_at": scraped_at,
                "processed_at": None,
                "engagement_total": likes + retweets + replies
            }
            posts.append(post)
        
        # Create the final JSON structure
        json_data = {
            "platform": "twitter",
            "username": username,
            "total_posts": len(posts),
            "last_updated": datetime.now().isoformat(),
            "posts": posts
        }
        
        # Save to JSON file
        output_file = os.path.join(output_dir, f"{username}_posts.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Exported {len(posts)} tweets for @{username} to {output_file}")
        
        # Show some stats
        total_likes = sum(post['likes'] for post in posts)
        avg_likes = total_likes / len(posts) if posts else 0
        max_likes = max(post['likes'] for post in posts) if posts else 0
        
        print(f"📊 Stats: Total likes: {total_likes}, Avg: {avg_likes:.0f}, Max: {max_likes}")
        
    except Exception as e:
        print(f"❌ Error exporting tweets for @{username}: {e}")

def main():
    """Export tweets for Kim Kardashian and Bernie Sanders."""
    
    users = ["KimKardashian", "BernieSanders"]
    
    print("📤 Exporting scraped tweets to JSON format...")
    print("=" * 60)
    
    for username in users:
        print(f"\n📱 Exporting tweets for @{username}")
        print("-" * 40)
        export_user_tweets_to_json(username)
    
    print(f"\n🎉 Export complete!")
    print("📁 Check the 'data/posts/twitter/' directory for the JSON files")

if __name__ == "__main__":
    main() 