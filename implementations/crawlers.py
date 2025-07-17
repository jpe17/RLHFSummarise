"""
Platform-specific crawler implementations.
"""

import os
import sys
import hashlib
from typing import List, Optional
from datetime import datetime

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from core.interfaces import BaseCrawler
from core.data_models import SocialPost, Platform, ContentType, MediaItem


class TwitterCrawler(BaseCrawler):
    """
    Twitter crawler implementation using existing Twitter scraper.
    """
    
    def __init__(self, headless: bool = True, wait_time: int = 2):
        """
        Initialize Twitter crawler.
        
        Args:
            headless: Whether to run browser in headless mode
            wait_time: Wait time between requests
        """
        self.headless = headless
        self.wait_time = wait_time
        self._scraper = None
        
    @property
    def platform(self) -> Platform:
        """Return the platform this crawler handles."""
        return Platform.TWITTER
        
    def _get_scraper(self):
        """Lazily initialize the scraper."""
        if self._scraper is None:
            # TwitterSeleniumScraper not available - use placeholder
            print("⚠️ TwitterSeleniumScraper not available - use existing data")
            self._scraper = None
        return self._scraper
        
    def crawl_user(self, username: str, count: int = 10) -> List[SocialPost]:
        """
        Crawl tweets from a Twitter user.
        
        Args:
            username: Twitter username (without @)
            count: Number of tweets to fetch
            
        Returns:
            List of SocialPost objects
        """
        try:
            scraper = self._get_scraper()
            
            if scraper is None:
                print(f"⚠️ No scraper available - use existing data for {username}")
                return []
            
            # Use existing scraper method
            tweet_data = scraper.scrape_user_tweets(username, count)
            
            if not tweet_data:
                return []
                
            posts = []
            for tweet in tweet_data:
                # Create deterministic ID using content + timestamp
                content = tweet.get("content", "")
                timestamp = tweet.get("timestamp", "")
                id_string = f"{content}{timestamp}"
                post_id = hashlib.sha256(id_string.encode('utf-8')).hexdigest()[:16]
                
                post = SocialPost(
                    id=post_id,
                    platform=Platform.TWITTER,
                    username=username,
                    content=content,
                    timestamp=datetime.fromisoformat(tweet.get("parsed_date", tweet.get("timestamp", ""))),
                    likes=tweet.get("likes", 0),
                    shares=tweet.get("retweets", 0),
                    comments=tweet.get("replies", 0),
                    platform_data={
                        "engagement_total": tweet.get("engagement_total", 0),
                        "scraped_at": tweet.get("scraped_at", "")
                    }
                )
                posts.append(post)
                
            return posts
            
        except Exception as e:
            print(f"Error crawling Twitter user {username}: {e}")
            return []
            
    def validate_username(self, username: str) -> bool:
        """
        Validate if a Twitter username exists and is accessible.
        
        Args:
            username: Twitter username to validate
            
        Returns:
            True if username is valid and accessible
        """
        try:
            # Try to crawl just 1 tweet to validate
            posts = self.crawl_user(username, count=1)
            return len(posts) > 0
        except Exception:
            return False


class InstagramCrawler(BaseCrawler):
    """
    Instagram crawler implementation.
    
    Note: This is a placeholder implementation. Full Instagram crawling
    would require proper API access or web scraping implementation.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Instagram crawler.
        
        Args:
            api_key: Instagram API key (if using official API)
        """
        self.api_key = api_key
        
    @property
    def platform(self) -> Platform:
        """Return the platform this crawler handles."""
        return Platform.INSTAGRAM
        
    def crawl_user(self, username: str, count: int = 10) -> List[SocialPost]:
        """
        Crawl posts from an Instagram user.
        
        Args:
            username: Instagram username
            count: Number of posts to fetch
            
        Returns:
            List of SocialPost objects
        """
        # TODO: Implement actual Instagram crawling
        # This would require:
        # 1. Instagram API integration OR
        # 2. Web scraping implementation OR
        # 3. Third-party service integration
        
        print(f"Instagram crawling not yet implemented for {username}")
        
        # Return placeholder data for now
        posts = []
        for i in range(min(count, 3)):  # Limit to 3 placeholder posts
            post = SocialPost(
                id=f"instagram_{username}_{i}",
                platform=Platform.INSTAGRAM,
                username=username,
                content=f"Placeholder Instagram post {i+1} from @{username}",
                content_type=ContentType.MIXED,
                media_items=[
                    MediaItem(
                        url=f"https://placeholder.com/image_{i}.jpg",
                        type="image",
                        caption=f"Image caption {i+1}",
                        dimensions={"width": 1080, "height": 1080}
                    )
                ],
                timestamp=datetime.now(),
                likes=100 + i * 10,
                shares=5 + i,
                comments=20 + i * 2,
                platform_data={
                    "hashtags": ["#placeholder", "#instagram"],
                    "location": "Sample Location"
                }
            )
            posts.append(post)
            
        return posts
        
    def validate_username(self, username: str) -> bool:
        """
        Validate if an Instagram username exists and is accessible.
        
        Args:
            username: Instagram username to validate
            
        Returns:
            True if username is valid and accessible
        """
        # TODO: Implement actual validation
        # For now, return True for non-empty usernames
        return bool(username and username.strip())


class TikTokCrawler(BaseCrawler):
    """
    TikTok crawler implementation (placeholder for future implementation).
    """
    
    def __init__(self):
        """Initialize TikTok crawler."""
        pass
        
    @property
    def platform(self) -> Platform:
        """Return the platform this crawler handles."""
        return Platform.TIKTOK
        
    def crawl_user(self, username: str, count: int = 10) -> List[SocialPost]:
        """
        Crawl posts from a TikTok user.
        
        Args:
            username: TikTok username
            count: Number of posts to fetch
            
        Returns:
            List of SocialPost objects
        """
        # TODO: Implement TikTok crawling
        print(f"TikTok crawling not yet implemented for {username}")
        return []
        
    def validate_username(self, username: str) -> bool:
        """
        Validate if a TikTok username exists and is accessible.
        
        Args:
            username: TikTok username to validate
            
        Returns:
            True if username is valid and accessible
        """
        # TODO: Implement actual validation
        return bool(username and username.strip())


class CrawlerFactory:
    """Factory class for creating platform-specific crawlers."""
    
    @staticmethod
    def create_crawler(platform: Platform, **kwargs) -> BaseCrawler:
        """
        Create a crawler for the specified platform.
        
        Args:
            platform: Platform to create crawler for
            **kwargs: Additional arguments for crawler initialization
            
        Returns:
            BaseCrawler instance
            
        Raises:
            ValueError: If platform is not supported
        """
        if platform == Platform.TWITTER:
            return TwitterCrawler(**kwargs)
        elif platform == Platform.INSTAGRAM:
            return InstagramCrawler(**kwargs)
        elif platform == Platform.TIKTOK:
            return TikTokCrawler(**kwargs)
        else:
            raise ValueError(f"Unsupported platform: {platform}")
            
    @staticmethod
    def get_supported_platforms() -> List[Platform]:
        """
        Get list of supported platforms.
        
        Returns:
            List of supported Platform enums
        """
        return [Platform.TWITTER, Platform.INSTAGRAM, Platform.TIKTOK] 