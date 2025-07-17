"""
Storage implementations for the modular pipeline.
"""

import json
import os
from typing import List, Optional, Dict, Any
from datetime import datetime

from core.interfaces import BaseStorage
from core.data_models import SocialPost, Platform


class JSONStorage(BaseStorage):
    """
    JSON-based storage implementation.
    
    Stores posts in JSON files organized by platform and username.
    """
    
    def __init__(self, data_dir: str = "data/posts"):
        """
        Initialize JSON storage.
        
        Args:
            data_dir: Directory to store JSON files
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Create platform subdirectories
        for platform in Platform:
            platform_dir = os.path.join(data_dir, platform.value)
            os.makedirs(platform_dir, exist_ok=True)
            
    def store_posts(self, posts: List[SocialPost]) -> bool:
        """
        Store posts organized by platform and username.
        
        Args:
            posts: List of SocialPost objects to store
            
        Returns:
            True if storage was successful
        """
        try:
            # Group posts by platform and username
            grouped_posts: Dict[str, List[SocialPost]] = {}
            
            for post in posts:
                key = f"{post.platform.value}_{post.username}"
                if key not in grouped_posts:
                    grouped_posts[key] = []
                grouped_posts[key].append(post)
                
            # Store each group
            for key, group_posts in grouped_posts.items():
                platform_str, username = key.split('_', 1)
                platform = Platform(platform_str)
                
                # Load existing posts
                existing_posts = self.load_posts(username, platform)
                existing_ids = {post.id for post in existing_posts}
                
                # Add new posts (avoid duplicates)
                new_posts = [post for post in group_posts if post.id not in existing_ids]
                all_posts = existing_posts + new_posts
                
                # Sort by timestamp (newest first)
                all_posts.sort(key=lambda p: p.timestamp, reverse=True)
                
                # Prepare data structure
                data = {
                    "platform": platform.value,
                    "username": username,
                    "total_posts": len(all_posts),
                    "last_updated": datetime.now().isoformat(),
                    "posts": [post.to_dict() for post in all_posts]
                }
                
                # Write to file
                file_path = self._get_file_path(username, platform)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    
            return True
            
        except Exception as e:
            print(f"Error storing posts: {e}")
            return False
            
    def load_posts(self, username: str, platform: Platform, 
                   limit: Optional[int] = None) -> List[SocialPost]:
        """
        Load posts for a specific user and platform, handling different storage structures
        and case-insensitive usernames.
        """
        platform_dir = os.path.join(self.data_dir, platform.value)
        if not os.path.exists(platform_dir):
            return []

        file_path = None

        if platform == Platform.INSTAGRAM:
            # Strategy for Instagram: data/posts/instagram/{Username}/{Username}_instagram_posts.json
            try:
                for item_name in os.listdir(platform_dir):
                    if item_name.lower() == username.lower():
                        user_dir_path = os.path.join(platform_dir, item_name)
                        if os.path.isdir(user_dir_path):
                            filename_to_find = f"{item_name}_instagram_posts.json"
                            potential_path = os.path.join(user_dir_path, filename_to_find)
                            if os.path.exists(potential_path):
                                file_path = potential_path
                                break
            except FileNotFoundError:
                pass
        
        else: # Default strategy for Twitter and other platforms
            # Strategy: data/posts/twitter/{Username}_posts.json
            try:
                for filename in os.listdir(platform_dir):
                    if filename.lower().endswith('_posts.json'):
                        base_name = filename.replace('_posts.json', '')
                        if base_name.lower() == username.lower():
                            file_path = os.path.join(platform_dir, filename)
                            break
            except FileNotFoundError:
                pass
                
        if not file_path:
            print(f"No posts file found for {username} on {platform}")
            return []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            posts = [SocialPost.from_dict(post_data) for post_data in data.get("posts", [])]
                
            if limit:
                posts = posts[:limit]
                
            return posts
            
        except Exception as e:
            print(f"Error loading posts for {username} on {platform}: {e}")
            return []
            
    def get_available_users(self, platform: Platform) -> List[str]:
        """
        Get list of available users for a platform.
        
        Args:
            platform: The platform to get users for
            
        Returns:
            List of usernames
        """
        platform_dir = os.path.join(self.data_dir, platform.value)
        
        if not os.path.exists(platform_dir):
            return []
            
        try:
            users = []
            for filename in os.listdir(platform_dir):
                if filename.endswith('_posts.json'):
                    username = filename.replace('_posts.json', '')
                    users.append(username)
                    
            return sorted(users)
            
        except Exception as e:
            print(f"Error getting users for platform {platform}: {e}")
            return []
            
    def get_user_stats(self, username: str, platform: Platform) -> Dict[str, Any]:
        """
        Get statistics for a user on a platform.
        
        Args:
            username: The username to get stats for
            platform: The platform to check
            
        Returns:
            Dictionary with user statistics
        """
        file_path = self._get_file_path(username, platform)
        
        if not os.path.exists(file_path):
            return {}
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            posts = [SocialPost.from_dict(post_data) for post_data in data.get("posts", [])]
            
            if not posts:
                return {}
                
            total_engagement = sum(post.engagement_total for post in posts)
            avg_engagement = total_engagement / len(posts)
            
            return {
                "username": username,
                "platform": platform.value,
                "total_posts": len(posts),
                "total_engagement": total_engagement,
                "average_engagement": avg_engagement,
                "latest_post": posts[0].timestamp.isoformat() if posts else None,
                "oldest_post": posts[-1].timestamp.isoformat() if posts else None,
                "last_updated": data.get("last_updated")
            }
            
        except Exception as e:
            print(f"Error getting stats for {username} on {platform}: {e}")
            return {}


class MigrationHelper:
    """
    Helper class to migrate existing Twitter JSON data to the new format.
    """
    
    def __init__(self, old_data_dir: str = "pipeline_twitter/data/json_tweets"):
        self.old_data_dir = old_data_dir
        
    def migrate_twitter_data(self, storage: JSONStorage) -> bool:
        """
        Migrate existing Twitter JSON files to the new format.
        
        Args:
            storage: JSONStorage instance to migrate data to
            
        Returns:
            True if migration was successful
        """
        if not os.path.exists(self.old_data_dir):
            print(f"Old data directory not found: {self.old_data_dir}")
            return False
            
        try:
            migrated_count = 0
            
            for filename in os.listdir(self.old_data_dir):
                if filename.endswith('_tweets.json'):
                    username = filename.replace('_tweets.json', '')
                    
                    # Load old format
                    old_file_path = os.path.join(self.old_data_dir, filename)
                    with open(old_file_path, 'r', encoding='utf-8') as f:
                        old_data = json.load(f)
                        
                    # Convert to new format
                    posts = []
                    for tweet_data in old_data.get("tweets", []):
                        post = SocialPost(
                            id=str(hash(tweet_data.get("content", "") + str(tweet_data.get("timestamp", "")))),
                            platform=Platform.TWITTER,
                            username=username,
                            content=tweet_data.get("content", ""),
                            timestamp=datetime.fromisoformat(tweet_data.get("parsed_date", tweet_data.get("timestamp", ""))),
                            likes=tweet_data.get("likes", 0),
                            shares=tweet_data.get("retweets", 0),
                            comments=tweet_data.get("replies", 0),
                            platform_data={
                                "engagement_total": tweet_data.get("engagement_total", 0),
                                "scraped_at": tweet_data.get("scraped_at", "")
                            }
                        )
                        posts.append(post)
                        
                    # Store in new format
                    if posts:
                        storage.store_posts(posts)
                        migrated_count += 1
                        print(f"Migrated {len(posts)} posts for {username}")
                        
            print(f"Successfully migrated {migrated_count} users")
            return True
            
        except Exception as e:
            print(f"Error during migration: {e}")
            return False 