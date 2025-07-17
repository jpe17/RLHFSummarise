"""
Summarization implementations.
"""

import os
import sys
import time
from typing import Dict, Any, Optional

# Fix tokenizer parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from core.interfaces import BaseSummarizer
from core.data_models import ProcessedContent, Summary


# Global model cache to avoid reloading
_model_cache = {}


class RLHFSummarizer(BaseSummarizer):
    """
    RLHF/PPO summarizer implementation that wraps the existing pipeline.
    """
    
    def __init__(self, 
                 model_id: str = "Qwen/Qwen1.5-0.5B",
                 ppo_weights_path: str = "rlhf_summarizer/simple_ppo_lora_final_20250716_130239.pt",
                 reward_model_path: str = "rlhf_summarizer/qwen_reward_model.pt",
                 preload: bool = False):
        """
        Initialize RLHF summarizer.
        
        Args:
            model_id: Hugging Face model ID
            ppo_weights_path: Path to PPO-trained LoRA weights
            reward_model_path: Path to reward model weights
            preload: Whether to preload the model immediately
        """
        self.model_id = model_id
        self.ppo_weights_path = ppo_weights_path
        self.reward_model_path = reward_model_path
        self.cache_key = f"{model_id}:{ppo_weights_path}:{reward_model_path}"
        
        if preload:
            print("🚀 Preloading RLHF model...")
            self._get_pipeline()
        
    def _get_pipeline(self):
        """Get cached pipeline or create new one."""
        if self.cache_key in _model_cache:
            return _model_cache[self.cache_key]
            
        print("🔄 Loading RLHF model...")
        try:
            from backend.tweet_summarizer_pipeline_ppo import TweetSummarizerPipelinePPO
            pipeline = TweetSummarizerPipelinePPO(
                model_id=self.model_id,
                ppo_weights_path=self.ppo_weights_path,
                reward_model_path=self.reward_model_path
            )
            
            # Cache the pipeline
            _model_cache[self.cache_key] = pipeline
            print("✅ RLHF model loaded and cached")
            return pipeline
            
        except Exception as e:
            print(f"Error initializing RLHF pipeline: {e}")
            return None
        
    def generate_summary(self, content: ProcessedContent, max_length: int = 200) -> Summary:
        """
        Generate a summary using the RLHF/PPO trained model.
        
        Args:
            content: ProcessedContent object containing posts to summarize
            max_length: Maximum length of the summary
            
        Returns:
            Summary object
        """
        try:
            pipeline = self._get_pipeline()
            
            if pipeline is None:
                return Summary(
                    content="Error: Could not load RLHF summarizer",
                    score=0.0,
                    original_content=content,
                    model_name=self.model_id
                )
            
            # Use the correct method name
            summary_text = pipeline.generate_summary(
                content.processed_text,
                max_length=max_length
            )
            
            return Summary(
                content=summary_text,
                score=0.8,  # Default score, can be improved later
                original_content=content,
                model_name=self.model_id,
                generation_params={"max_length": max_length}
            )
            
        except Exception as e:
            print(f"Error generating RLHF summary: {e}")
            return Summary(
                content=f"Error generating summary: {str(e)}",
                score=0.0,
                original_content=content,
                model_name=self.model_id
            )
            
    def score_summary(self, content: ProcessedContent, summary: Summary) -> float:
        """
        Score a summary based on the original content.
        
        Args:
            content: Original processed content
            summary: Generated summary
            
        Returns:
            Score between 0.0 and 1.0
        """
        try:
            pipeline = self._get_pipeline()
            
            if pipeline is None:
                return 0.5  # Default score if pipeline not available
            
            # Use the existing scoring method
            score = pipeline.score_summary(content.processed_text, summary.content)
            
            # Ensure score is between 0.0 and 1.0
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"Error scoring summary: {e}")
            return 0.5  # Default score if scoring fails


class SummarizerFactory:
    """Factory class for creating summarizers."""
    
    @staticmethod
    def create_summarizer(summarizer_type: str, **kwargs) -> BaseSummarizer:
        """
        Create a summarizer of the specified type.
        
        Args:
            summarizer_type: Type of summarizer ("rlhf")
            **kwargs: Additional arguments for summarizer initialization
            
        Returns:
            BaseSummarizer instance
            
        Raises:
            ValueError: If summarizer type is not supported
        """
        if summarizer_type == "rlhf":
            return RLHFSummarizer(**kwargs)
        else:
            raise ValueError(f"Unsupported summarizer type: {summarizer_type}")
            
    @staticmethod
    def get_supported_types() -> Dict[str, str]:
        """
        Get supported summarizer types and their descriptions.
        
        Returns:
            Dictionary mapping type names to descriptions
        """
        return {
            "rlhf": "RLHF/PPO trained model (high quality)"
        } 