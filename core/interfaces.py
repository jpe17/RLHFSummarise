"""
Abstract base classes and interfaces for pipeline components.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from .data_models import SocialPost, ProcessedContent, Summary, VoiceOutput, Platform


class BaseCrawler(ABC):
    """Abstract base class for platform-specific crawlers."""
    
    @property
    @abstractmethod
    def platform(self) -> Platform:
        """Return the platform this crawler handles."""
        pass
    
    @abstractmethod
    def crawl_user(self, username: str, count: int = 10) -> List[SocialPost]:
        """
        Crawl posts from a specific user.
        
        Args:
            username: The username to crawl
            count: Number of posts to fetch
            
        Returns:
            List of SocialPost objects
        """
        pass
    
    @abstractmethod
    def validate_username(self, username: str) -> bool:
        """
        Validate if a username exists and is accessible.
        
        Args:
            username: The username to validate
            
        Returns:
            True if username is valid and accessible
        """
        pass


class BaseStorage(ABC):
    """Abstract base class for data storage implementations."""
    
    @abstractmethod
    def store_posts(self, posts: List[SocialPost]) -> bool:
        """
        Store a list of social posts.
        
        Args:
            posts: List of SocialPost objects to store
            
        Returns:
            True if storage was successful
        """
        pass
    
    @abstractmethod
    def load_posts(self, username: str, platform: Platform, 
                   limit: Optional[int] = None) -> List[SocialPost]:
        """
        Load posts for a specific user and platform.
        
        Args:
            username: The username to load posts for
            platform: The platform to load from
            limit: Maximum number of posts to load
            
        Returns:
            List of SocialPost objects
        """
        pass
    
    @abstractmethod
    def get_available_users(self, platform: Platform) -> List[str]:
        """
        Get list of available users for a platform.
        
        Args:
            platform: The platform to get users for
            
        Returns:
            List of usernames
        """
        pass


class BaseContentProcessor(ABC):
    """Abstract base class for content processing components."""
    
    @abstractmethod
    def process_posts(self, posts: List[SocialPost]) -> ProcessedContent:
        """
        Process a list of social posts into content ready for summarization.
        
        Args:
            posts: List of SocialPost objects to process
            
        Returns:
            ProcessedContent object
        """
        pass


class BaseImageToText(ABC):
    """Abstract base class for image-to-text conversion."""
    
    @abstractmethod
    def extract_text(self, image_url: str) -> str:
        """
        Extract text description from an image.
        
        Args:
            image_url: URL or path to the image
            
        Returns:
            Text description of the image
        """
        pass


class BaseSummarizer(ABC):
    """Abstract base class for summarization components."""
    
    @abstractmethod
    def generate_summary(self, content: ProcessedContent) -> Summary:
        """
        Generate a summary from processed content.
        
        Args:
            content: ProcessedContent object to summarize
            
        Returns:
            Summary object
        """
        pass
    
    @abstractmethod
    def score_summary(self, content: ProcessedContent, summary: Summary) -> float:
        """
        Score a summary based on the original content.
        
        Args:
            content: Original processed content
            summary: Generated summary
            
        Returns:
            Score between 0.0 and 1.0
        """
        pass


class BaseVoiceSynthesizer(ABC):
    """Abstract base class for voice synthesis components."""
    
    @abstractmethod
    def synthesize(self, text: str, voice_name: str) -> VoiceOutput:
        """
        Synthesize voice from text.
        
        Args:
            text: Text to synthesize
            voice_name: Name of the voice to use
            
        Returns:
            VoiceOutput object
        """
        pass
    
    @abstractmethod
    def get_available_voices(self) -> List[str]:
        """
        Get list of available voices.
        
        Returns:
            List of voice names
        """
        pass


class BasePostSelector(ABC):
    """Abstract base class for post selection strategies."""
    
    @abstractmethod
    def select_posts(self, posts: List[SocialPost], 
                    selection_type: str = "top",
                    count: int = 5) -> List[SocialPost]:
        """
        Select posts based on criteria.
        
        Args:
            posts: List of posts to select from
            selection_type: Type of selection ("top", "latest", "random")
            count: Number of posts to select
            
        Returns:
            Selected posts
        """
        pass 