"""
Main pipeline manager that orchestrates all components.
"""

import time
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime

from .interfaces import (
    BaseCrawler, BaseStorage, BaseContentProcessor, 
    BaseSummarizer, BaseVoiceSynthesizer, BasePostSelector
)
from .data_models import (
    Platform, SocialPost, ProcessedContent, Summary, 
    VoiceOutput, PipelineResult
)


class PipelineManager:
    """
    Main pipeline manager that orchestrates the entire process.
    
    This class manages the flow from crawling to voice synthesis,
    allowing for easy swapping of components and debugging.
    """
    
    def __init__(self):
        self.crawlers: Dict[Platform, BaseCrawler] = {}
        self.storage: Optional[BaseStorage] = None
        self.content_processor: Optional[BaseContentProcessor] = None
        self.summarizer: Optional[BaseSummarizer] = None
        self.voice_synthesizer: Optional[BaseVoiceSynthesizer] = None
        self.post_selector: Optional[BasePostSelector] = None
        
        # Progress tracking
        self.progress_callback: Optional[Callable[[int, str], None]] = None
        
    def register_crawler(self, crawler: BaseCrawler):
        """Register a platform-specific crawler."""
        self.crawlers[crawler.platform] = crawler
        
    def register_storage(self, storage: BaseStorage):
        """Register a storage implementation."""
        self.storage = storage
        
    def register_content_processor(self, processor: BaseContentProcessor):
        """Register a content processor."""
        self.content_processor = processor
        
    def register_summarizer(self, summarizer: BaseSummarizer):
        """Register a summarizer."""
        self.summarizer = summarizer
        
    def register_voice_synthesizer(self, synthesizer: BaseVoiceSynthesizer):
        """Register a voice synthesizer."""
        self.voice_synthesizer = synthesizer
        
    def register_post_selector(self, selector: BasePostSelector):
        """Register a post selector."""
        self.post_selector = selector
        
    def set_progress_callback(self, callback: Callable[[int, str], None]):
        """Set a callback for progress updates."""
        self.progress_callback = callback
        
    def _update_progress(self, progress: int, message: str):
        """Update progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(progress, message)
            
    def get_available_users(self, platform: Platform) -> List[str]:
        """Get available users for a platform."""
        if not self.storage:
            raise ValueError("Storage not registered")
        return self.storage.get_available_users(platform)
        
    def get_available_voices(self) -> List[str]:
        """Get available voices."""
        if not self.voice_synthesizer:
            raise ValueError("Voice synthesizer not registered")
        return self.voice_synthesizer.get_available_voices()
        
    def crawl_user(self, platform: Platform, username: str, count: int = 10) -> List[SocialPost]:
        """
        Crawl posts from a user on a specific platform.
        
        Args:
            platform: Platform to crawl from
            username: Username to crawl
            count: Number of posts to fetch
            
        Returns:
            List of SocialPost objects
        """
        if platform not in self.crawlers:
            raise ValueError(f"No crawler registered for platform {platform}")
            
        crawler = self.crawlers[platform]
        return crawler.crawl_user(username, count)
        
    def load_posts(self, platform: Platform, username: str, limit: Optional[int] = None) -> List[SocialPost]:
        """
        Load posts from storage.
        
        Args:
            platform: Platform to load from
            username: Username to load
            limit: Maximum number of posts to load
            
        Returns:
            List of SocialPost objects
        """
        if not self.storage:
            raise ValueError("Storage not registered")
            
        return self.storage.load_posts(username, platform, limit)
        
    def select_posts(self, posts: List[SocialPost], selection_type: str = "top", 
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
        if not self.post_selector:
            raise ValueError("Post selector not registered")
            
        return self.post_selector.select_posts(posts, selection_type, count)
        
    def process_content(self, posts: List[SocialPost]) -> ProcessedContent:
        """
        Process posts into content ready for summarization.
        
        Args:
            posts: List of SocialPost objects to process
            
        Returns:
            ProcessedContent object
        """
        if not self.content_processor:
            raise ValueError("Content processor not registered")
            
        return self.content_processor.process_posts(posts)
        
    def generate_summary(self, content: ProcessedContent) -> Summary:
        """
        Generate a summary from processed content.
        
        Args:
            content: ProcessedContent object to summarize
            
        Returns:
            Summary object
        """
        if not self.summarizer:
            raise ValueError("Summarizer not registered")
            
        return self.summarizer.generate_summary(content)
        
    def synthesize_voice(self, text: str, voice_name: str) -> VoiceOutput:
        """
        Synthesize voice from text.
        
        Args:
            text: Text to synthesize
            voice_name: Name of the voice to use
            
        Returns:
            VoiceOutput object
        """
        if not self.voice_synthesizer:
            raise ValueError("Voice synthesizer not registered")
            
        return self.voice_synthesizer.synthesize(text, voice_name)
        
    def run_pipeline_without_tts(self, users: List[Dict[str, str]], 
                                selection_type: str = "top", count: int = 5) -> PipelineResult:
        """
        Run the pipeline from posts to summary, without voice synthesis.
        """
        start_time = time.time()
        
        try:
            self._update_progress(10, "Loading posts...")
            all_posts = []
            for user_info in users:
                platform = Platform(user_info['platform'])
                username = user_info['username']
                if username:
                    posts = self.load_posts(platform, username)
                    all_posts.extend(posts)

            if not all_posts:
                raise ValueError("No posts found for the specified users.")
                
            self._update_progress(20, f"Selecting {count} {selection_type} posts...")
            selected_posts = self.select_posts(all_posts, selection_type, count)
            
            self._update_progress(40, "🔎 Processing content...")
            processed_content = self.process_content(selected_posts)
            
            self._update_progress(60, "✍️ Generating summary...")
            summary = self.generate_summary(processed_content)
            
            self._update_progress(85, "📦 Finalizing results...")
            total_duration = time.time() - start_time
            
            return PipelineResult(
                platform="multiple",
                username=", ".join([u['username'] for u in users if u['username']]),
                posts=selected_posts,
                selection_type=selection_type,
                processed_content=processed_content,
                summary=summary,
                voice_output=None,
                total_duration=total_duration
            )
            
        except Exception as e:
            self._update_progress(0, f"Error: {str(e)}")
            raise

    def run_full_pipeline(self, users: List[Dict[str, str]], 
                         voice_name: str, selection_type: str = "top", 
                         count: int = 5, max_length: int = 200) -> PipelineResult:
        """
        Run the complete pipeline for multiple users.
        """
        start_time = time.time()
        
        try:
            # Run the pipeline up to the summary generation
            result = self.run_pipeline_without_tts(
                users=users,
                selection_type=selection_type,
                count=count,
                max_length=max_length
            )
            
            # Step 4: Synthesize voice
            self._update_progress(80, f"🎤 Synthesizing voice using '{voice_name}'...")
            voice_output = self.synthesize_voice(result.summary.content, voice_name)
            
            # Update result with voice output
            result.voice_output = voice_output
            result.total_duration = time.time() - start_time
            
            self._update_progress(100, "✅ Complete!")
            
            return result
            
        except Exception as e:
            self._update_progress(0, f"Error: {str(e)}")
            raise
            
    def run_pipeline_from_crawl(self, platform: Platform, username: str, 
                               voice_name: str, selection_type: str = "top", 
                               count: int = 5, max_length: int = 200) -> PipelineResult:
        """
        Run pipeline starting from crawling (not using stored data).
        
        Args:
            platform: Platform to crawl from
            username: Username to crawl
            voice_name: Voice to use for synthesis
            selection_type: Type of post selection
            count: Number of posts to process
            max_length: Maximum summary length
            
        Returns:
            PipelineResult object
        """
        start_time = time.time()
        
        try:
            # Step 1: Crawl posts
            self._update_progress(5, "Crawling posts...")
            posts = self.crawl_user(platform, username, count * 2)  # Get more to select from
            
            if not posts:
                raise ValueError(f"No posts found for {username} on {platform}")
                
            # Step 2: Store posts
            self._update_progress(10, "Storing posts...")
            if self.storage:
                self.storage.store_posts(posts)
                
            # Step 3: Select posts
            self._update_progress(20, "Selecting posts...")
            selected_posts = self.select_posts(posts, selection_type, count)
            
            # Step 4: Process content
            self._update_progress(40, "Processing content...")
            processed_content = self.process_content(selected_posts)
            
            # Step 5: Generate summary
            self._update_progress(60, "Generating summary...")
            summary = self.generate_summary(processed_content, max_length)
            
            # Step 6: Synthesize voice
            self._update_progress(80, "Synthesizing voice...")
            voice_output = self.synthesize_voice(summary.content, voice_name)
            
            # Step 7: Create result
            self._update_progress(100, "Complete!")
            total_duration = time.time() - start_time
            
            return PipelineResult(
                posts=selected_posts,
                processed_content=processed_content,
                summary=summary,
                voice_output=voice_output,
                total_duration=total_duration
            )
            
        except Exception as e:
            self._update_progress(0, f"Error: {str(e)}")
            raise
            
    def validate_configuration(self) -> Dict[str, bool]:
        """
        Validate that all required components are registered.
        
        Returns:
            Dictionary with validation results
        """
        return {
            "storage": self.storage is not None,
            "content_processor": self.content_processor is not None,
            "summarizer": self.summarizer is not None,
            "voice_synthesizer": self.voice_synthesizer is not None,
            "post_selector": self.post_selector is not None,
            "crawlers": len(self.crawlers) > 0
        } 