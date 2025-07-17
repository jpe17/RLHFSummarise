"""
Voice synthesis implementations.
"""

import os
import sys
import hashlib
import torch
import signal
from typing import List, Optional, Dict
from datetime import datetime
import time

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from core.interfaces import BaseVoiceSynthesizer
from core.data_models import VoiceOutput


class TTSVoiceSynthesizer(BaseVoiceSynthesizer):
    """
    TTS voice synthesizer implementation that wraps the existing voice cloning pipeline.
    """
    
    def __init__(self, device: Optional[str] = None, voices_dir: str = "voices", preload: bool = False):
        """
        Initialize TTS voice synthesizer.
        
        Args:
            device: Device to run TTS on (auto-detected if None)
            voices_dir: Directory containing voice models
            preload: Whether to preload the TTS model immediately
        """
        self.device = device
        self.voices_dir = voices_dir
        self.tts = None
        self.model_loaded = False  # Flag to track if model is loaded
        self._available_voices = None
        self._audio_cache: Dict[str, VoiceOutput] = {}  # Cache for generated audio
        
        # Set MPS fallback for compatibility with TTS
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Reduce memory issues
        
        # Force CPU for problematic TTS operations
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Override MPS for TTS due to channel limitations
            os.environ['PYTORCH_FORCE_CPU_FALLBACK'] = '1'
        
        if preload:
            print("🚀 Preloading TTS model...")
            try:
                self._get_tts()
                print("✅ TTS model loaded and cached")
            except Exception as e:
                print(f"❌ Failed to preload TTS model: {e}")
                print("🔄 TTS will load on-demand instead")
                # Don't fail initialization, just load on demand
                pass
        
    def _get_cache_key(self, text: str, voice_name: str) -> str:
        """Generate a cache key for the given text and voice."""
        content = f"{text}:{voice_name}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
        
    def _get_tts(self):
        """Lazily initialize the TTS engine with speed optimizations."""
        if self.model_loaded and self.tts:
            return self.tts

        if self.tts is None:
            try:
                from TTS.api import TTS
                
                # Auto-detect device if not specified
                if self.device is None:
                    # SPEED OPTIMIZATION: Prefer CPU for TTS to avoid GPU memory transfers
                    if torch.cuda.is_available():
                        self.device = "cuda"
                    else:
                        self.device = "cpu"
                        print("🔧 Using CPU for TTS (optimized for speed)")
                
                print(f"🎤 Initializing TTS on device: {self.device}")
                
                # Initialize TTS with optimizations
                print("📥 Loading TTS model with speed optimizations...")
                
                # SPEED OPTIMIZATION: Load with minimal memory footprint
                self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
                
                # SPEED OPTIMIZATION: Set model to eval mode and optimize for inference
                self.tts.synthesizer.tts_model.eval()
                
                # SPEED OPTIMIZATION: Disable gradient computation permanently
                for param in self.tts.synthesizer.tts_model.parameters():
                    param.requires_grad = False
                
                self.model_loaded = True
                print("✅ TTS model initialized with speed optimizations")
                
            except Exception as e:
                print(f"❌ Error initializing TTS: {e}")
                self.model_loaded = False
                raise
                
        return self.tts
        
    def synthesize(self, text: str, voice_name: str) -> str:
        """
        Synthesize voice from text and return the audio file path.
        OPTIMIZED FOR SPEED while maintaining quality.
        
        Args:
            text: Text to synthesize
            voice_name: Name of the voice to use
            
        Returns:
            Path to the generated audio file
        """
        try:
            # SPEED OPTIMIZATION: More aggressive text truncation
            MAX_TTS_LENGTH = 1200  # Reduced from 2000 for faster processing
            if len(text) > MAX_TTS_LENGTH:
                # Find the last complete sentence within the limit
                truncated = text[:MAX_TTS_LENGTH]
                last_sentence_end = max(
                    truncated.rfind('.'),
                    truncated.rfind('!'),
                    truncated.rfind('?')
                )
                if last_sentence_end > MAX_TTS_LENGTH * 0.6:  # Reduced from 0.7
                    text = truncated[:last_sentence_end + 1]
                else:
                    text = truncated + "..."
                print(f"📝 Truncated text to {len(text)} characters for faster TTS")
            
            # Check cache first
            cache_key = self._get_cache_key(text, voice_name)
            if cache_key in self._audio_cache:
                print(f"🚀 Using cached audio for {voice_name}")
                return self._audio_cache[cache_key].audio_path
            
            print(f"🔊 Generating new audio for {voice_name}...")
            
            # Clean text (optimized)
            cleaned_text = self._clean_text_for_tts(text)
            
            # Get voice reference
            voice_ref_path = self._get_voice_reference(voice_name)
            
            if not voice_ref_path:
                raise ValueError(f"Voice reference not found for {voice_name}")
                
            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"synthesis_{voice_name}_{timestamp}.wav"
            output_path = os.path.join("outputs", output_filename)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Synthesize speech with SPEED OPTIMIZATIONS
            tts = self._get_tts()
            
            # SPEED OPTIMIZATION: Use smaller chunk size and process only first chunk
            text_chunks = self._chunk_text_intelligently(cleaned_text, max_chunk_size=300)  # Reduced from 400
            
            # Always process only the first chunk for maximum speed
            tts.tts_to_file(
                text=text_chunks[0],
                speaker_wav=voice_ref_path,
                language="en",
                file_path=output_path,
                split_sentences=False,  # Disable sentence splitting for speed
                speed=1.2,  # Increased from 1.1 for faster speech and quicker generation
            )
            
            if len(text_chunks) > 1:
                print(f"📄 Processed 1 of {len(text_chunks)} chunks for speed optimization")
            
            # Get audio duration (approximate)
            duration = len(cleaned_text) * 0.08  # Reduced estimate for faster speech
            
            result = VoiceOutput(
                audio_path=output_path,
                voice_name=voice_name,
                text=cleaned_text,
                duration=duration,
                sample_rate=22050  # Default sample rate for XTTS
            )
            
            # Cache the result
            self._audio_cache[cache_key] = result
            
            # Verify audio file was created
            if not os.path.exists(output_path):
                raise Exception(f"Audio file was not created: {output_path}")
                
            return output_path
            
        except Exception as e:
            print(f"Error synthesizing voice: {e}")
            return "" # Return empty string on error
            
    def get_available_voices(self) -> List[str]:
        """
        Get list of available voices.
        
        Returns:
            List of voice names
        """
        if self._available_voices is None:
            self._available_voices = self._scan_available_voices()
        return self._available_voices
        
    def _scan_available_voices(self) -> List[str]:
        """Scan the voices directory for available voices."""
        voices = []
        
        if not os.path.exists(self.voices_dir):
            return voices
            
        try:
            for item in os.listdir(self.voices_dir):
                voice_dir = os.path.join(self.voices_dir, item)
                if os.path.isdir(voice_dir):
                    # Check if the voice directory contains audio files
                    has_audio = any(
                        f.endswith(('.wav', '.mp3', '.flac'))
                        for f in os.listdir(voice_dir)
                    )
                    if has_audio:
                        voices.append(item)
                        
        except Exception as e:
            print(f"Error scanning voices directory: {e}")
            
        return sorted(voices)
        
    def _get_voice_reference(self, voice_name: str) -> Optional[str]:
        """Get the reference audio file for a voice."""
        voice_dir = os.path.join(self.voices_dir, voice_name)
        
        if not os.path.exists(voice_dir):
            return None
            
        # Look for audio files in the voice directory
        for filename in os.listdir(voice_dir):
            if filename.endswith(('.wav', '.mp3', '.flac')):
                return os.path.join(voice_dir, filename)
                
        return None
        
    def _clean_text_for_tts(self, text: str) -> str:
        """
        Clean text for TTS synthesis.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text suitable for TTS
        """
        try:
            # Import the existing text cleaning utility
            from backend.text_utils import clean_text_for_tts
            return clean_text_for_tts(text)
        except ImportError:
            # Fallback cleaning
            return self._basic_clean_text_for_tts(text)
            
    def _basic_clean_text_for_tts(self, text: str) -> str:
        """
        Basic text cleaning for TTS.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        import re
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags for cleaner speech
        text = re.sub(r'[@#]\w+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{2,}', '...', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Ensure text ends with punctuation for better speech rhythm
        if text and not text[-1] in '.!?':
            text += '.'
            
        return text

    def _chunk_text_intelligently(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """
        Split text into chunks at natural sentence boundaries for better TTS processing.
        
        Args:
            text: Text to chunk
            max_chunk_size: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        # Split by sentences first
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence) + 2  # +2 for '. '
            
            if current_length + sentence_length > max_chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = '. '.join(current_chunk)
                if not chunk_text.endswith('.'):
                    chunk_text += '.'
                chunks.append(chunk_text)
                
                # Start new chunk
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk)
            if not chunk_text.endswith('.'):
                chunk_text += '.'
            chunks.append(chunk_text)
        
        return chunks


class MockVoiceSynthesizer(BaseVoiceSynthesizer):
    """
    Mock voice synthesizer for testing.
    """
    
    def __init__(self):
        """Initialize mock voice synthesizer."""
        self.mock_voices = [
            "christina", "elonmusk", "barackobama", "freeman", 
            "angie", "daniel", "emma", "halle", "jlaw", "weaver"
        ]
        
    def synthesize(self, text: str, voice_name: str) -> VoiceOutput:
        """
        Mock synthesize voice from text.
        OPTIMIZED FOR SPEED with same text limits as real TTS.
        
        Args:
            text: Text to synthesize
            voice_name: Name of the voice to use
            
        Returns:
            VoiceOutput object with mock data
        """
        # SPEED OPTIMIZATION: Use same text limits as real TTS
        MAX_TTS_LENGTH = 1200  # Same as real TTS
        original_length = len(text)
        if len(text) > MAX_TTS_LENGTH:
            # Find the last complete sentence within the limit
            truncated = text[:MAX_TTS_LENGTH]
            last_sentence_end = max(
                truncated.rfind('.'),
                truncated.rfind('!'),
                truncated.rfind('?')
            )
            if last_sentence_end > MAX_TTS_LENGTH * 0.6:  # Same as real TTS
                text = truncated[:last_sentence_end + 1]
            else:
                text = truncated + "..."
            print(f"📝 Mock TTS: Truncated text from {original_length} to {len(text)} characters")
        
        # Simulate minimal processing time
        time.sleep(0.05)  # Reduced from 0.1 for speed
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mock_path = f"mock_synthesis_{voice_name}_{timestamp}.wav"
        
        return VoiceOutput(
            audio_path=mock_path,
            voice_name=voice_name,
            text=text,
            duration=len(text) * 0.06,  # Reduced for faster speech simulation
            sample_rate=22050
        )
        
    def get_available_voices(self) -> List[str]:
        """
        Get list of available voices.
        
        Returns:
            List of mock voice names
        """
        return self.mock_voices


class VoiceSynthesizerFactory:
    """Factory class for creating voice synthesizers."""
    
    @staticmethod
    def create_synthesizer(synthesizer_type: str, **kwargs) -> BaseVoiceSynthesizer:
        """
        Create a voice synthesizer of the specified type.
        
        Args:
            synthesizer_type: Type of synthesizer ("tts", "mock")
            **kwargs: Additional arguments for synthesizer initialization
            
        Returns:
            BaseVoiceSynthesizer instance
            
        Raises:
            ValueError: If synthesizer type is not supported
        """
        if synthesizer_type == "tts":
            return TTSVoiceSynthesizer(**kwargs)
        elif synthesizer_type == "mock":
            return MockVoiceSynthesizer(**kwargs)
        else:
            raise ValueError(f"Unsupported synthesizer type: {synthesizer_type}")
            
    @staticmethod
    def get_available_synthesizers() -> dict:
        """
        Get available synthesizer types.
        
        Returns:
            Dictionary mapping synthesizer types to descriptions
        """
        return {
            "tts": "TTS voice cloning synthesizer (recommended)",
            "mock": "Mock synthesizer for testing"
        } 