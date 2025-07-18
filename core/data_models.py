"""
Common data models and interfaces for the modular pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import json


class Platform(Enum):
    """Supported social media platforms."""
    TWITTER = "twitter"
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"


class ContentType(Enum):
    """Types of content that can be processed."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    MIXED = "mixed"


@dataclass
class MediaItem:
    """Represents a media item (image, video, etc.)."""
    url: str
    type: str  # "image", "video", "gif", etc.
    caption: Optional[str] = None
    alt_text: Optional[str] = None # Added to support Instagram alt text
    dimensions: Optional[Dict[str, int]] = None  # {"width": 1080, "height": 1080}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "type": self.type,
            "caption": self.caption,
            "alt_text": self.alt_text,
            "dimensions": self.dimensions
        }


@dataclass
class SocialPost:
    """Universal data model for social media posts across all platforms."""
    
    # Core identification (required fields first)
    id: str
    platform: Platform
    username: str
    content: str  # Main text content
    timestamp: datetime
    
    # Optional fields (with defaults)
    user_display_name: Optional[str] = None
    content_type: ContentType = ContentType.TEXT
    media_items: List[MediaItem] = field(default_factory=list)
    url: Optional[str] = None
    
    # Engagement metrics
    likes: int = 0
    shares: int = 0  # retweets, reposts, etc.
    comments: int = 0
    views: Optional[int] = None
    
    # Platform-specific data
    platform_data: Dict[str, Any] = field(default_factory=dict)
    
    # Processing metadata
    scraped_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    
    @property
    def engagement_total(self) -> int:
        """Calculate total engagement."""
        return self.likes + self.shares + self.comments
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "platform": self.platform.value,
            "username": self.username,
            "user_display_name": self.user_display_name,
            "content": self.content,
            "content_type": self.content_type.value,
            "media_items": [item.to_dict() for item in self.media_items],
            "timestamp": self.timestamp.isoformat(),
            "url": self.url,
            "likes": self.likes,
            "shares": self.shares,
            "comments": self.comments,
            "views": self.views,
            "platform_data": self.platform_data,
            "scraped_at": self.scraped_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "engagement_total": self.engagement_total
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SocialPost':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            platform=Platform(data["platform"]),
            username=data["username"],
            user_display_name=data.get("user_display_name"),
            content=data["content"],
            content_type=ContentType(data.get("content_type", "text")),
            media_items=[MediaItem(**item) for item in data.get("media_items", [])],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            url=data.get("url"),
            likes=data.get("likes", 0),
            shares=data.get("shares", 0),
            comments=data.get("comments", 0),
            views=data.get("views"),
            platform_data=data.get("platform_data", {}),
            scraped_at=datetime.fromisoformat(data["scraped_at"]),
            processed_at=datetime.fromisoformat(data["processed_at"]) if data.get("processed_at") else None
        )


@dataclass
class ProcessedContent:
    """Represents processed content ready for summarization."""
    
    original_posts: List[SocialPost]
    combined_text: str
    processed_text: str  # After cleaning, img2txt, etc.
    
    # Processing metadata
    processing_steps: List[str] = field(default_factory=list)
    processed_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_posts": [post.to_dict() for post in self.original_posts],
            "combined_text": self.combined_text,
            "processed_text": self.processed_text,
            "processing_steps": self.processing_steps,
            "processed_at": self.processed_at.isoformat()
        }


@dataclass
class Summary:
    """Represents a generated summary."""
    
    content: str
    score: float
    original_content: ProcessedContent
    
    # Generation metadata
    model_name: str
    generation_params: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "score": self.score,
            "original_content": self.original_content.to_dict(),
            "model_name": self.model_name,
            "generation_params": self.generation_params,
            "generated_at": self.generated_at.isoformat()
        }


@dataclass
class VoiceOutput:
    """Represents synthesized voice output."""
    
    audio_path: str
    voice_name: str
    text: str
    
    # Audio metadata
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    
    # Generation metadata
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_path": self.audio_path,
            "voice_name": self.voice_name,
            "text": self.text,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "generated_at": self.generated_at.isoformat()
        }


@dataclass
class PipelineResult:
    """Complete pipeline result."""
    
    platform: Union[Platform, str]
    username: str
    posts: List[SocialPost]
    selection_type: str
    processed_content: ProcessedContent
    summary: Summary
    voice_output: Optional[VoiceOutput]
    
    # Pipeline metadata
    pipeline_version: str = "1.0"
    total_duration: Optional[float] = None
    completed_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform.value if isinstance(self.platform, Platform) else self.platform,
            "username": self.username,
            "posts": [post.to_dict() for post in self.posts],
            "selection_type": self.selection_type,
            "processed_content": self.processed_content.to_dict(),
            "summary": self.summary.to_dict(),
            "voice_output": self.voice_output.to_dict() if self.voice_output else None,
            "pipeline_version": self.pipeline_version,
            "total_duration": self.total_duration,
            "completed_at": self.completed_at.isoformat()
        } 