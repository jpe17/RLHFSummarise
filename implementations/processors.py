"""
Content processing implementations.
"""

import os
import sys
import random
from typing import List, Optional
from datetime import datetime

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from core.interfaces import BaseContentProcessor, BaseImageToText, BasePostSelector
from core.data_models import SocialPost, ProcessedContent, ContentType, MediaItem


class MockImageToText(BaseImageToText):
    """
    Mock implementation of image-to-text for testing.
    
    In production, this would use a proper image captioning model
    like BLIP, CLIP, or a similar vision-language model.
    """
    
    def extract_text(self, image_url: str) -> str:
        """
        Extract text description from an image (mock implementation).
        
        Args:
            image_url: URL or path to the image
            
        Returns:
            Mock text description of the image
        """
        # TODO: Implement actual image-to-text using a vision model
        # This could use models like:
        # - BLIP (Bootstrapping Language-Image Pre-training)
        # - CLIP (Contrastive Language-Image Pre-training)
        # - LLaVA (Large Language and Vision Assistant)
        
        # For now, return a mock description based on the URL
        if "placeholder" in image_url:
            return "A placeholder image showing sample content"
        elif "profile" in image_url:
            return "A profile picture or avatar image"
        elif "post" in image_url:
            return "A social media post image with various content"
        else:
            return "An image shared on social media"


class VisionImageToText(BaseImageToText):
    """
    Vision model implementation for image-to-text.
    
    This would use a proper vision-language model in production.
    """
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        """
        Initialize vision model.
        
        Args:
            model_name: Name of the vision model to use
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        
    def _load_model(self):
        """Lazily load the vision model."""
        if self.model is None:
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                import torch
                
                self.processor = BlipProcessor.from_pretrained(self.model_name)
                self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
                
                # Move to appropriate device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(device)
                
            except ImportError:
                print("Vision model dependencies not available, falling back to mock")
                return MockImageToText()
                
    def extract_text(self, image_url: str) -> str:
        """
        Extract text description from an image using vision model.
        
        Args:
            image_url: URL or path to the image
            
        Returns:
            Text description of the image
        """
        try:
            self._load_model()
            
            if isinstance(self.model, MockImageToText):
                return self.model.extract_text(image_url)
            
            # TODO: Implement actual image processing
            # This would involve:
            # 1. Downloading/loading the image
            # 2. Processing with the vision model
            # 3. Generating caption
            
            # For now, return mock description
            return f"Generated caption for image: {image_url}"
            
        except Exception as e:
            print(f"Error processing image {image_url}: {e}")
            return "Image content could not be processed"


class ContentProcessor(BaseContentProcessor):
    """
    Main content processor that handles text cleaning, image-to-text, and content preparation.
    """
    
    def __init__(self, image_to_text: Optional[BaseImageToText] = None):
        """
        Initialize content processor.
        
        Args:
            image_to_text: Image-to-text processor (optional)
        """
        self.image_to_text = image_to_text or MockImageToText()
        
    def process_posts(self, posts: List[SocialPost]) -> ProcessedContent:
        """
        Process a list of social posts into content ready for summarization.
        
        Args:
            posts: List of SocialPost objects to process
            
        Returns:
            ProcessedContent object
        """
        processing_steps = []
        combined_text = ""
        processed_parts = []
        
        for post in posts:
            # Start with the original text content
            text_content = post.content
            processing_steps.append(f"Processing post from @{post.username}")
            
            # Clean the text content
            cleaned_text = self._clean_text(text_content)
            processing_steps.append("Cleaned text content")
            
            # Process media items if present
            media_descriptions = []
            if post.media_items:
                for media_item in post.media_items:
                    if media_item.type in ["image", "photo"]:
                        # For Instagram, extract text from images but don't include the "Image content:" prefix
                        description = self.image_to_text.extract_text(media_item.url)
                        # Only add meaningful descriptions (not empty or very short)
                        if description and len(description.strip()) > 10:
                            # Limit each image description to 200 characters
                            if len(description) > 200:
                                description = description[:200].rsplit(' ', 1)[0] + '...'
                            media_descriptions.append(description)
                        processing_steps.append(f"Processed image: {media_item.url}")
                    elif media_item.type == "video":
                        # Skip video content annotation to reduce noise
                        processing_steps.append(f"Noted video: {media_item.url}")
            
            # Combine text and media descriptions
            # The caption (cleaned_text) is the primary content, supplemented by media analysis
            post_full_content = cleaned_text
            if media_descriptions:
                post_full_content += "\n" + "\n".join(media_descriptions)
            
            # Limit each individual post to 200 characters total
            original_length = len(post_full_content)
            if len(post_full_content) > 200:
                post_full_content = post_full_content[:200].rsplit(' ', 1)[0] + '...'
                print(f"📏 Truncated post from @{post.username}: {original_length} → {len(post_full_content)} chars")
            else:
                print(f"📏 Post from @{post.username}: {len(post_full_content)} chars")
                
            processed_parts.append(post_full_content)
            
        # Combine all processed content
        processed_text = "\n\n".join(processed_parts)
        combined_text = processed_text  # Use processed_text as the main content
        
        print(f"📊 Final processed content: {len(processed_text)} chars from {len(processed_parts)} posts")
        
        return ProcessedContent(
            original_posts=posts,
            combined_text=combined_text,
            processed_text=processed_text,
            processing_steps=processing_steps
        )
        
    def _clean_text(self, text: str) -> str:
        """
        Clean text content for processing.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Import the existing text cleaning utility
        try:
            from backend.text_utils import clean_text_for_summarization
            return clean_text_for_summarization(text)
        except ImportError:
            # Fallback cleaning if import fails
            return self._basic_clean_text(text)
            
    def _basic_clean_text(self, text: str) -> str:
        """
        Basic text cleaning fallback.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        import re
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove empty lines
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text


class PostSelector(BasePostSelector):
    """
    Post selection implementation with various strategies.
    """
    
    def select_posts(self, posts: List[SocialPost], 
                    selection_type: str = "top",
                    count: int = 5) -> List[SocialPost]:
        """
        Select posts based on criteria.
        
        Args:
            posts: List of posts to select from
            selection_type: Type of selection ("top", "latest", "random", "diverse")
            count: Number of posts to select
            
        Returns:
            Selected posts
        """
        if not posts:
            return []
            
        # Ensure we don't select more posts than available
        count = min(count, len(posts))
        
        if selection_type == "top":
            return self._select_top_posts(posts, count)
        elif selection_type == "latest":
            return self._select_latest_posts(posts, count)
        elif selection_type == "random":
            return self._select_random_posts(posts, count)
        elif selection_type == "diverse":
            return self._select_diverse_posts(posts, count)
        elif selection_type == "balanced":
            return self._select_balanced_posts(posts, count)
        else:
            raise ValueError(f"Unknown selection type: {selection_type}")
            
    def _select_top_posts(self, posts: List[SocialPost], count: int) -> List[SocialPost]:
        """Select posts with highest engagement."""
        sorted_posts = sorted(posts, key=lambda p: p.engagement_total, reverse=True)
        return sorted_posts[:count]
        
    def _select_latest_posts(self, posts: List[SocialPost], count: int) -> List[SocialPost]:
        """Select most recent posts."""
        sorted_posts = sorted(posts, key=lambda p: p.timestamp, reverse=True)
        return sorted_posts[:count]
        
    def _select_random_posts(self, posts: List[SocialPost], count: int) -> List[SocialPost]:
        """Select random posts."""
        return random.sample(posts, count)
        
    def _select_diverse_posts(self, posts: List[SocialPost], count: int) -> List[SocialPost]:
        """
        Select diverse posts using a combination of engagement and recency.
        
        This tries to balance high engagement with temporal diversity.
        """
        # Sort by engagement
        top_posts = sorted(posts, key=lambda p: p.engagement_total, reverse=True)
        
        # Sort by recency
        recent_posts = sorted(posts, key=lambda p: p.timestamp, reverse=True)
        
        # Take a mix: 60% from top engagement, 40% from recent
        selected = []
        
        # Add top engagement posts
        top_count = int(count * 0.6)
        selected.extend(top_posts[:top_count])
        
        # Add recent posts (avoiding duplicates)
        remaining_count = count - len(selected)
        selected_ids = {post.id for post in selected}
        
        for post in recent_posts:
            if post.id not in selected_ids:
                selected.append(post)
                remaining_count -= 1
                if remaining_count <= 0:
                    break
                    
        return selected[:count]
    
    def _select_balanced_posts(self, posts: List[SocialPost], count: int) -> List[SocialPost]:
        """
        Select posts with balanced representation from each user.
        
        This ensures that posts from multiple users are included fairly.
        """
        # Group posts by username
        user_posts = {}
        for post in posts:
            username = post.username
            if username not in user_posts:
                user_posts[username] = []
            user_posts[username].append(post)
        
        print(f"🔍 Balanced selection: {len(user_posts)} users, {count} posts needed")
        for username, user_post_list in user_posts.items():
            print(f"  - @{username}: {len(user_post_list)} posts available")
        
        # Sort posts within each user by engagement (or timestamp for latest)
        for username in user_posts:
            user_posts[username].sort(key=lambda p: p.engagement_total, reverse=True)
        
        # Use round-robin selection to ensure fair distribution
        selected = []
        user_names = list(user_posts.keys())
        user_indices = {username: 0 for username in user_names}
        
        # Round-robin selection
        while len(selected) < count:
            added_any = False
            
            for username in user_names:
                if len(selected) >= count:
                    break
                    
                # Check if this user has more posts available
                user_index = user_indices[username]
                available_posts = user_posts[username]
                
                if user_index < len(available_posts):
                    selected.append(available_posts[user_index])
                    user_indices[username] += 1
                    added_any = True
                    print(f"  ✅ Selected post {user_index + 1} from @{username}")
            
            # If no user had any posts left, break
            if not added_any:
                break
        
        print(f"🔍 Final selection: {len(selected)} posts from {len(set(p.username for p in selected))} users")
        for username in user_names:
            user_count = sum(1 for p in selected if p.username == username)
            print(f"  - @{username}: {user_count} posts selected")
                
        return selected[:count] 