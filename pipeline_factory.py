"""
Pipeline factory for creating and configuring the complete modular pipeline.
"""

import os
from typing import Dict, Any, Optional
import time

from core.pipeline_manager import PipelineManager
from core.data_models import Platform

from implementations.storage import JSONStorage, MigrationHelper
from implementations.crawlers import CrawlerFactory
from implementations.processors import ContentProcessor, PostSelector, MockImageToText, VisionImageToText
from implementations.summarizer import SummarizerFactory
from implementations.voice_synthesizer import VoiceSynthesizerFactory


class PipelineFactory:
    """
    Factory class for creating and configuring complete pipeline instances.
    """
    
    @staticmethod
    def create_default_pipeline(
        data_dir: str = "data/posts",
        summarizer_type: str = "rlhf",
        voice_synthesizer_type: str = "tts",
        use_vision_model: bool = False,
        migrate_twitter_data: bool = False
    ) -> PipelineManager:
        """
        Create a pipeline with default configuration.
        
        Args:
            data_dir: Directory for storing posts
            summarizer_type: Type of summarizer to use ("rlhf", "simple")
            voice_synthesizer_type: Type of voice synthesizer ("tts" only)
            use_vision_model: Whether to use real vision model for image-to-text
            migrate_twitter_data: Whether to migrate existing Twitter data
            
        Returns:
            Configured PipelineManager instance
        """
        # Create pipeline manager
        manager = PipelineManager()
        
        # Configure storage
        storage = JSONStorage(data_dir=data_dir)
        manager.register_storage(storage)
        
        # Migrate existing Twitter data if requested
        if migrate_twitter_data:
            migration_helper = MigrationHelper()
            migration_helper.migrate_twitter_data(storage)
        
        # Configure crawlers
        twitter_crawler = CrawlerFactory.create_crawler(Platform.TWITTER)
        instagram_crawler = CrawlerFactory.create_crawler(Platform.INSTAGRAM)
        
        manager.register_crawler(twitter_crawler)
        manager.register_crawler(instagram_crawler)
        
        # Configure content processor
        image_to_text = VisionImageToText() if use_vision_model else MockImageToText()
        content_processor = ContentProcessor(image_to_text=image_to_text)
        manager.register_content_processor(content_processor)
        
        # Configure post selector
        post_selector = PostSelector()
        manager.register_post_selector(post_selector)
        
        # Configure summarizer
        summarizer = SummarizerFactory.create_summarizer(summarizer_type)
        manager.register_summarizer(summarizer)
        
        # Configure voice synthesizer
        voice_synthesizer = VoiceSynthesizerFactory.create_synthesizer(voice_synthesizer_type)
        manager.register_voice_synthesizer(voice_synthesizer)
        
        return manager
        
    @staticmethod
    def create_testing_pipeline() -> PipelineManager:
        """
        Create a pipeline configured for testing with mock components.
        
        Returns:
            PipelineManager configured for testing
        """
        return PipelineFactory.create_default_pipeline(
            data_dir="test_data/posts",
            summarizer_type="simple",
            voice_synthesizer_type="tts",
            use_vision_model=False,
            migrate_twitter_data=False
        )
        
    @staticmethod
    def create_production_pipeline() -> PipelineManager:
        """
        Create a pipeline configured for production with full features.
        
        Returns:
            PipelineManager configured for production
        """
        return PipelineFactory.create_default_pipeline(
            data_dir="data/posts",
            summarizer_type="rlhf",
            voice_synthesizer_type="tts",
            use_vision_model=True,
            migrate_twitter_data=True
        )
        
    @staticmethod
    def create_custom_pipeline(config: Dict[str, Any]) -> PipelineManager:
        """
        Create a pipeline with custom configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            PipelineManager configured according to config
        """
        # Extract configuration values
        data_dir = config.get("data_dir", "data/posts")
        summarizer_type = config.get("summarizer_type", "rlhf")
        voice_synthesizer_type = config.get("voice_synthesizer_type", "tts")
        use_vision_model = config.get("use_vision_model", False)
        migrate_twitter_data = config.get("migrate_twitter_data", False)
        
        # Create pipeline manager
        manager = PipelineManager()
        
        # Configure storage
        storage_config = config.get("storage", {})
        storage = JSONStorage(data_dir=data_dir, **storage_config)
        manager.register_storage(storage)
        
        # Migrate data if requested
        if migrate_twitter_data:
            migration_helper = MigrationHelper()
            migration_helper.migrate_twitter_data(storage)
        
        # Configure crawlers
        crawler_configs = config.get("crawlers", {})
        
        # Twitter crawler
        twitter_config = crawler_configs.get("twitter", {})
        twitter_crawler = CrawlerFactory.create_crawler(Platform.TWITTER, **twitter_config)
        manager.register_crawler(twitter_crawler)
        
        # Instagram crawler
        instagram_config = crawler_configs.get("instagram", {})
        instagram_crawler = CrawlerFactory.create_crawler(Platform.INSTAGRAM, **instagram_config)
        manager.register_crawler(instagram_crawler)
        
        # Configure content processor
        processor_config = config.get("content_processor", {})
        image_to_text = VisionImageToText() if use_vision_model else MockImageToText()
        content_processor = ContentProcessor(image_to_text=image_to_text, **processor_config)
        manager.register_content_processor(content_processor)
        
        # Configure YouTube processor (keep separate, don't register as content processor)
        from implementations.youtube_processor import YouTubeProcessor
        youtube_processor = YouTubeProcessor(**config.get("youtube_processor", {}))
        # Note: YouTubeProcessor is used directly for YouTube processing, not as the main content processor
        
        # Configure post selector
        selector_config = config.get("post_selector", {})
        post_selector = PostSelector(**selector_config)
        manager.register_post_selector(post_selector)
        
        # Configure summarizer
        summarizer_config = config.get("summarizer", {})
        summarizer = SummarizerFactory.create_summarizer(summarizer_type, **summarizer_config)
        manager.register_summarizer(summarizer)
        
        # Configure voice synthesizer (load on-demand)
        voice_config = config.get("voice_synthesizer", {})
        voice_synthesizer = VoiceSynthesizerFactory.create_synthesizer(voice_synthesizer_type, **voice_config)
        manager.register_voice_synthesizer(voice_synthesizer)
        
        print("✅ Pipeline initialization complete")
        return manager

    def run_pipeline_without_tts(self, platform: Platform, username: str, voice_name: str, selection_type: str = 'top', count: int = 5, max_length: int = 200):
        """
        Run the complete pipeline without voice synthesis.
        """
        start_time = time.time()
        
        # ... (rest of the pipeline logic from run_full_pipeline)
        
        # For now, returning a mock result, will implement fully later
        from core.data_models import Summary, VoiceOutput, PipelineResult, ProcessedContent
        
        mock_summary = Summary(content="This is a summary.", score=0.9)
        mock_voice = VoiceOutput(audio_path=None)
        mock_processed = ProcessedContent(cleaned_content="Cleaned text", processing_steps=["Step 1"])
        
        return PipelineResult(
            platform=platform,
            username=username,
            posts=[],
            summary=mock_summary,
            voice_output=mock_voice,
            processed_content=mock_processed,
            total_duration=time.time() - start_time
        )

    def synthesize_voice(self, text: str, voice_name: str) -> str:
        """
        Synthesize voice from text.
        """
        if self.voice_synthesizer is None:
            raise Exception("Voice synthesizer not registered")
        
        return self.voice_synthesizer.synthesize(text, voice_name)
        
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Get default configuration dictionary.
        
        Returns:
            Default configuration
        """
        return {
            "data_dir": "data/posts",
            "summarizer_type": "rlhf",
            "voice_synthesizer_type": "tts",
            "use_vision_model": False,
            "migrate_twitter_data": False,
            "storage": {},
            "crawlers": {
                "twitter": {
                    "headless": True,
                    "wait_time": 2
                },
                "instagram": {
                    "api_key": None
                }
            },
            "content_processor": {},
            "post_selector": {},
            "summarizer": {
                "model_id": "Qwen/Qwen1.5-0.5B",
                "ppo_weights_path": "rlhf_summarizer/simple_ppo_lora_final_20250716_130239.pt",
                "reward_model_path": "rlhf_summarizer/qwen_reward_model.pt"
            },
            "voice_synthesizer": {
                "device": None,
                "voices_dir": "voices"
            }
        }
        
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration dictionary.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required directories
        data_dir = config.get("data_dir", "data/posts")
        if not os.path.exists(data_dir):
            results["warnings"].append(f"Data directory will be created: {data_dir}")
            
        # Check summarizer type
        summarizer_type = config.get("summarizer_type", "rlhf")
        if summarizer_type not in ["rlhf", "simple"]:
            results["errors"].append(f"Invalid summarizer type: {summarizer_type}")
            results["valid"] = False
            
        # Check voice synthesizer type
        voice_type = config.get("voice_synthesizer_type", "tts")
        if voice_type != "tts":
            results["errors"].append(f"Invalid voice synthesizer type: {voice_type}")
            results["valid"] = False
            
        # Check RLHF files if using RLHF summarizer
        if summarizer_type == "rlhf":
            summarizer_config = config.get("summarizer", {})
            ppo_weights = summarizer_config.get("ppo_weights_path", "rlhf_summarizer/simple_ppo_lora_final_20250716_130239.pt")
            reward_model = summarizer_config.get("reward_model_path", "rlhf_summarizer/qwen_reward_model.pt")
            
            if not os.path.exists(ppo_weights):
                results["errors"].append(f"PPO weights file not found: {ppo_weights}")
                results["valid"] = False
                
            if not os.path.exists(reward_model):
                results["errors"].append(f"Reward model file not found: {reward_model}")
                results["valid"] = False
                
        # Check voices directory if using TTS
        if voice_type == "tts":
            voice_config = config.get("voice_synthesizer", {})
            voices_dir = voice_config.get("voices_dir", "voices")
            
            if not os.path.exists(voices_dir):
                results["errors"].append(f"Voices directory not found: {voices_dir}")
                results["valid"] = False
                
        return results


def create_pipeline_from_config_file(config_file: str) -> PipelineManager:
    """
    Create a pipeline from a configuration file.
    
    Args:
        config_file: Path to configuration file (JSON)
        
    Returns:
        Configured PipelineManager instance
    """
    import json
    
    with open(config_file, 'r') as f:
        config = json.load(f)
        
    return PipelineFactory.create_custom_pipeline(config)


def save_config_to_file(config: Dict[str, Any], config_file: str):
    """
    Save configuration to a file.
    
    Args:
        config: Configuration dictionary
        config_file: Path to save configuration file
    """
    import json
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2) 


def create_fast_pipeline(**config_overrides) -> PipelineManager:
    """
    Create a pipeline optimized for fast development.
    
    Args:
        **config_overrides: Additional configuration overrides
        
    Returns:
        PipelineManager instance
    """
    config = {
        **config_overrides
    }
    return PipelineFactory.create_custom_pipeline(config)


def create_preloaded_pipeline(**config_overrides) -> PipelineManager:
    """
    Create a pipeline with all models preloaded for fastest runtime performance.
    Models are initialized at startup to eliminate first-time loading delays.
    
    Args:
        **config_overrides: Additional configuration overrides
        
    Returns:
        PipelineManager instance with preloaded models
    """
    print("🔧 Creating preloaded pipeline...")
    config = {
        "migrate_twitter_data": False,  # Explicitly disable migration
        "voice_synthesizer_type": "tts",  # Use real TTS
        "summarizer": {
            "model_id": "Qwen/Qwen1.5-0.5B",
            "ppo_weights_path": "rlhf_summarizer/simple_ppo_lora_final_20250716_130239.pt",
            "reward_model_path": "rlhf_summarizer/qwen_reward_model.pt",
            "preload": True  # Preload RLHF model (this is the slow one)
        },
        "voice_synthesizer": {
            "device": None,
            "voices_dir": "voices",
            "preload": False  # Load on-demand for faster initialization
        },
        **config_overrides
    }
    print("📋 Configuration prepared, creating pipeline...")
    return PipelineFactory.create_custom_pipeline(config) 