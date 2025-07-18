"""
YouTube processor implementation for the modular pipeline.
"""

import yt_dlp
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from faster_whisper import WhisperModel
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from collections import Counter
from tqdm import tqdm

from core.interfaces import Processor, SocialPost, ProcessedContent
from core.data_models import Platform
from implementations.summarizer import RLHFSummarizer

# Add these functions after the imports and before the LoRALinear class (around line 25)

class YouTubeCache:
    """Cache system for YouTube video processing to avoid reprocessing the same URLs."""
    
    def __init__(self, cache_file: str = "youtube_cache.json"):
        """
        Initialize the YouTube cache.
        
        Args:
            cache_file: Path to the cache JSON file
        """
        self.cache_file = cache_file
        self.cache_data = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache data from JSON file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                print(f"📋 Loaded YouTube cache with {len(cache_data.get('videos', {}))} videos")
                return cache_data
            except Exception as e:
                print(f"⚠️ Error loading cache: {e}")
                return {"videos": {}, "metadata": {"created": datetime.now().isoformat()}}
        else:
            print("📋 Creating new YouTube cache")
            return {"videos": {}, "metadata": {"created": datetime.now().isoformat()}}
    
    def _save_cache(self):
        """Save cache data to JSON file."""
        try:
            # Update metadata
            self.cache_data["metadata"]["last_updated"] = datetime.now().isoformat()
            self.cache_data["metadata"]["total_videos"] = len(self.cache_data.get("videos", {}))
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved YouTube cache with {len(self.cache_data.get('videos', {}))} videos")
        except Exception as e:
            print(f"❌ Error saving cache: {e}")
    
    def get_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        # Handle different YouTube URL formats
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
            r'youtube\.com/watch\?.*v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # If no pattern matches, use the URL as ID
        return url
    
    def is_cached(self, url: str) -> bool:
        """Check if a YouTube URL is already cached."""
        video_id = self.get_video_id(url)
        return video_id in self.cache_data.get("videos", {})
    
    def get_cached_result(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached result for a YouTube URL."""
        video_id = self.get_video_id(url)
        cached_video = self.cache_data.get("videos", {}).get(video_id)
        
        if cached_video:
            # Check if all required files exist
            required_files = ["transcription_file", "summary_file"]
            for file_key in required_files:
                file_path = cached_video.get(file_key)
                if file_path and not os.path.exists(file_path):
                    print(f"⚠️ Cached file not found: {file_path}")
                    return None
            
            print(f"🎯 Found cached result for video: {cached_video.get('title', 'Unknown')}")
            return cached_video
        
        return None
    
    def cache_result(self, url: str, result: Dict[str, Any]):
        """Cache the result for a YouTube URL."""
        video_id = self.get_video_id(url)
        
        # Prepare cache entry
        cache_entry = {
            "url": url,
            "video_id": video_id,
            "title": result.get("analysis_info", {}).get("title", "Unknown"),
            "uploader": result.get("analysis_info", {}).get("uploader", "Unknown"),
            "duration": result.get("analysis_info", {}).get("duration", "Unknown"),
            "processed_at": datetime.now().isoformat(),
            "transcription_file": result.get("analysis_info", {}).get("transcription_file"),
            "summary_file": result.get("analysis_info", {}).get("summary_file"),
            "audio_file": result.get("analysis_info", {}).get("audio_file"),
            "content": {
                "transcription": result.get("content", {}).get("transcription", ""),
                "summary": result.get("content", {}).get("summary", ""),
                "topics": result.get("content", {}).get("topics", [])
            }
        }
        
        # Add to cache
        if "videos" not in self.cache_data:
            self.cache_data["videos"] = {}
        
        self.cache_data["videos"][video_id] = cache_entry
        self._save_cache()
        
        print(f"💾 Cached result for video: {cache_entry['title']}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        videos = self.cache_data.get("videos", {})
        return {
            "total_videos": len(videos),
            "cache_created": self.cache_data.get("metadata", {}).get("created"),
            "last_updated": self.cache_data.get("metadata", {}).get("last_updated"),
            "recent_videos": list(videos.keys())[-5:] if videos else []
        }

# Global cache instance
youtube_cache = YouTubeCache()

# Simple directory utility function
def get_subdir(dir_name: str) -> str:
    """
    Create and return path to a subdirectory.
    
    Args:
        dir_name: Name of the subdirectory
        
    Returns:
        str: Path to the created directory
    """
    dir_path = os.path.join(os.getcwd(), dir_name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

# LoRA Model Integration
class LoRALinear(nn.Module):
    def __init__(self, original_layer, r=16, alpha=32):
        super().__init__()
        self.original_layer = original_layer
        self.scaling = alpha / r
        
        # Create LoRA matrices
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        
        # Match original layer dtype
        if hasattr(original_layer, 'weight'):
            dtype = original_layer.weight.dtype
            self.lora_A = self.lora_A.to(dtype)
            self.lora_B = self.lora_B.to(dtype)
        
        # Freeze original layer
        for param in original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        original = self.original_layer(x)
        lora = self.lora_B(self.lora_A(x)) * self.scaling
        
        if original.dtype != lora.dtype:
            lora = lora.to(original.dtype)
            
        return original + lora

class LoRASummarizer:
    """LoRA-based summarizer using the trained model from simple_summarizer.py"""
    
    def __init__(self, model_id="Qwen/Qwen1.5-0.5B", 
                 lora_weights_path="rlhf_summarizer/simple_ppo_lora_final_20250716_130239.pt",
                 device=None):
        """
        Initialize the LoRA summarizer.
        
        Args:
            model_id: Hugging Face model ID
            lora_weights_path: Path to LoRA weights file
            device: Device to run on (auto-detected if None)
        """
        self.model_id = model_id
        self.lora_weights_path = lora_weights_path
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"🔧 Setting up LoRA model on device: {self.device}")
        
        # Setup model and tokenizer
        self.tokenizer = self._setup_tokenizer()
        self.model = self._setup_model()
        
    def _setup_tokenizer(self):
        """Set up the tokenizer."""
        print(f"📝 Loading tokenizer for {self.model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        return tokenizer
    
    def _setup_model(self):
        """Set up the model with LoRA layers."""
        print(f" Loading model with LoRA...")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Apply LoRA to attention layers
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        count = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(target in name for target in targets):
                # Replace module with LoRA version
                *parent_path, child_name = name.split('.')
                parent = model.get_submodule('.'.join(parent_path)) if parent_path else model
                
                setattr(parent, child_name, LoRALinear(module, r=16, alpha=32))
                count += 1
        
        model = model.to(self.device)
        
        # Load LoRA weights
        if os.path.exists(self.lora_weights_path):
            self._load_lora_weights(model)
            print(f"✅ Loaded LoRA weights from {self.lora_weights_path}")
        else:
            print(f"⚠️ LoRA weights not found at {self.lora_weights_path}")
            print(f"⚠️ Using base model without fine-tuning")
        
        model.eval()
        
        # Print stats
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Applied LoRA to {count} layers")
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        
        return model
    
    def _load_lora_weights(self, model):
        """Load LoRA weights from a saved file."""
        if not torch.cuda.is_available():
            weights = torch.load(self.lora_weights_path, map_location='cpu')
        else:
            weights = torch.load(self.lora_weights_path)
        
        loaded_count = 0
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A'):
                lora_A_key = f"{name}.lora_A.weight"
                lora_B_key = f"{name}.lora_B.weight"
                
                if lora_A_key in weights and lora_B_key in weights:
                    module.lora_A.weight.data = weights[lora_A_key].to(module.lora_A.weight.device)
                    module.lora_B.weight.data = weights[lora_B_key].to(module.lora_B.weight.device)
                    loaded_count += 1
        
        print(f"Loaded weights for {loaded_count} LoRA modules")
    
    def chunk_text(self, text, max_chunk_tokens=1000, overlap_tokens=100):
        """Split long text into chunks that fit within the model's context window."""
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= max_chunk_tokens:
            return [text]
        
        chunks = []
        start = 0
        
        # Calculate total number of chunks for progress bar
        total_chunks = 0
        temp_start = 0
        while temp_start < len(tokens):
            temp_end = min(temp_start + max_chunk_tokens, len(tokens))
            total_chunks += 1
            temp_start = temp_end - overlap_tokens
            if temp_start >= len(tokens):
                break
        
        print(f"📦 Splitting text into {total_chunks} chunks...")
        with tqdm(total=total_chunks, desc="Creating chunks", unit="chunk") as pbar:
            while start < len(tokens):
                end = min(start + max_chunk_tokens, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text)
                start = end - overlap_tokens
                if start >= len(tokens):
                    break
                pbar.update(1)
        
        return chunks

    def generate_summary(self, text, max_length=300, temperature=0.7, chunk_size=2048, stream_callback=None):
        """
        Generate a summary of the given text using the LoRA model.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary in tokens
            temperature: Generation temperature
            chunk_size: Maximum tokens per chunk for long texts
            stream_callback: Optional callback function for streaming updates
            
        Returns:
            str: Generated summary
        """
        if not text.strip():
            return ""
        
        text = text.strip()
        print(f" Input text length: {len(text)} characters")
        
        # Check if text needs chunking
        tokens = self.tokenizer.encode(text)
        print(f" Input text tokens: {len(tokens)}")
        
        if len(tokens) <= chunk_size:
            # Text fits in one chunk
            print(f" Generating summary (max {max_length} tokens, temp {temperature})...")
            summary = self._generate_summary_for_chunk_streaming(text, max_length, temperature, stream_callback)
            print(f"✅ Generated summary ({len(summary)} characters)")
            return summary
        else:
            # Text needs chunking
            chunks = self.chunk_text(text, chunk_size, overlap_tokens=100)
            print(f" Split into {len(chunks)} chunks")
            
            # Generate summaries for each chunk with progress bar
            chunk_summaries = []
            print(f" Processing {len(chunks)} chunks...")
            with tqdm(total=len(chunks), desc="Generating summaries", unit="chunk") as pbar:
                for i, chunk in enumerate(chunks):
                    chunk_summary = self._generate_summary_for_chunk_streaming(
                        chunk, 
                        max_length=max_length//len(chunks),
                        temperature=temperature,
                        stream_callback=stream_callback
                    )
                    chunk_summaries.append(chunk_summary)
                    pbar.set_postfix({"chunk": f"{i+1}/{len(chunks)}", "chars": len(chunk_summary)})
                    pbar.update(1)
            
            # Combine chunk summaries
            combined_summary = " ".join(chunk_summaries)
            
            # If the combined summary is still too long, generate a final summary
            if len(combined_summary) > max_length * 2:
                print(f"🔄 Generating final summary from chunk summaries...")
                final_summary = self._generate_summary_for_chunk_streaming(
                    combined_summary,
                    max_length=max_length,
                    temperature=temperature,
                    stream_callback=stream_callback
                )
                print(f"✅ Generated final summary ({len(final_summary)} characters)")
                return final_summary
            else:
                print(f"✅ Generated combined summary ({len(combined_summary)} characters)")
                return combined_summary
    
    def _generate_summary_for_chunk(self, text, max_length=300, temperature=0.7):
        """Generate a summary for a single chunk of text."""
        if not text.strip():
            return ""
        
        text = text.strip()
        prompt = f"Please summarize the following text:\n\n{text}\n\nSummary:"
        
        # Tokenize input with higher max_length for longer texts
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048,  # Increased from 512 to handle longer texts
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode and extract summary
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = full_output[len(prompt):].strip()
        
        return summary
    
    def _generate_summary_for_chunk_streaming(self, text, max_length=300, temperature=0.7, stream_callback=None):
        """Generate a summary for a single chunk of text with streaming support."""
        if not text.strip():
            return ""
        
        text = text.strip()
        prompt = f"Please summarize the following text:\n\n{text}\n\nSummary:"
        
        # Tokenize input with higher max_length for longer texts
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048,  # Increased from 512 to handle longer texts
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate summary with streaming
        generated_tokens = []
        current_summary = ""
        
        with torch.no_grad():
            # Use generate with streaming
            for outputs in self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                streamer=None,  # We'll handle streaming manually
                return_dict_in_generate=True,
                output_scores=False
            ):
                # Get the new tokens
                if hasattr(outputs, 'sequences'):
                    new_tokens = outputs.sequences[0][len(inputs['input_ids'][0]):]
                else:
                    new_tokens = outputs[0][len(inputs['input_ids'][0]):]
                
                # Decode new tokens
                new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                # Update current summary
                current_summary = new_text
                
                # Call stream callback if provided
                if stream_callback:
                    stream_callback(current_summary)
        
        return current_summary.strip()
    
    def detect_transcription_errors(self, transcribed_text: str) -> List[Dict]:
        """
        Detect potential errors in the transcribed text using the LoRA model.
        
        Args:
            transcribed_text (str): The transcribed text to analyze
            
        Returns:
            List[Dict]: List of detected errors with their locations and descriptions
        """
        prompt = f"""Analyze this transcribed text and identify potential errors such as:
1. Misheard words or phrases
2. Incorrect homophones
3. Missing punctuation
4. Grammatical inconsistencies
5. Context-inappropriate words

For each error found, provide:
1. The error text
2. The likely correct text
3. A brief explanation of why it's probably an error
4. Confidence level (high/medium/low)

Format your response as a JSON array of objects, each containing:
{{
    "error_text": "the erroneous text",
    "likely_correction": "the probable correct text",
    "explanation": "why this is likely an error",
    "confidence": "high/medium/low"
}}

If no errors are found, return an empty array.

Text to analyze:
{transcribed_text}

JSON response:"""

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.3,  # Lower temperature for more focused analysis
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode and extract response
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_output[len(prompt):].strip()
            
            # Try to extract JSON from response
            try:
                # Find JSON array in the response
                json_start = response.find('[')
                json_end = response.rfind(']') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    return json.loads(json_str)
                else:
                    return []
            except json.JSONDecodeError:
                print("⚠️ Could not parse JSON response from error detection")
                return []
                
        except Exception as e:
            print(f"Error in error detection: {e}")
            return []
    
    def correct_transcription(self, transcribed_text: str, errors: List[Dict]) -> str:
        """
        Correct the detected errors in the transcribed text using the LoRA model.
        
        Args:
            transcribed_text (str): The original transcribed text
            errors (List[Dict]): List of detected errors from detect_transcription_errors
            
        Returns:
            str: The corrected transcription
        """
        if not errors:
            return transcribed_text
        
        prompt = f"""You are an expert at correcting transcription errors in text.
Your task is to take the original transcribed text and a list of detected errors,
and produce a corrected version that:
1. Fixes all identified errors
2. Maintains the original meaning and context
3. Sounds natural and fluent
4. Preserves any correct parts of the original text

Additionally:
- Only make corrections that have high or medium confidence
- Preserve the original text's style and tone
- Maintain proper formatting and punctuation

Original text:
{transcribed_text}

Detected errors:
{json.dumps(errors, indent=2)}

Corrected text:"""

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=len(transcribed_text) + 200,  # Allow for corrections
                    temperature=0.3,  # Lower temperature for more accurate corrections
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode and extract response
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            corrected_text = full_output[len(prompt):].strip()
            
            return corrected_text
            
        except Exception as e:
            print(f"Error in correction: {e}")
            return transcribed_text
    
    def extract_topics(self, text: str, max_topics: int = 5) -> List[Dict[str, Any]]:
        """
        Extract topics from the text using the LoRA model.
        
        Args:
            text (str): The text to analyze
            max_topics (int): Maximum number of topics to extract
            
        Returns:
            List[Dict]: List of topics with their relevance scores
        """
        prompt = f"""You are an expert at analyzing text and extracting key topics.
Your task is to identify the most relevant topics from the given text, considering:
1. Frequency of mention
2. Centrality to the main discussion
3. Specificity and uniqueness
4. Contextual importance

For each topic, provide:
1. The topic name
2. A relevance score (0-100)
3. A brief description of why it's relevant
4. Key terms or phrases associated with it

Format your response as a JSON array of objects, each containing:
{{
    "topic": "the topic name",
    "relevance_score": number between 0-100,
    "description": "why this topic is relevant",
    "key_terms": ["term1", "term2", ...]
}}

Sort topics by relevance score in descending order.
Limit the number of topics to {max_topics}.

Text to analyze:
{text}

JSON response:"""

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=400,
                    temperature=0.5,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode and extract response
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_output[len(prompt):].strip()
            
            # Try to extract JSON from response
            try:
                # Find JSON array in the response
                json_start = response.find('[')
                json_end = response.rfind(']') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    return json.loads(json_str)
                else:
                    # Fallback to keyword-based extraction
                    return self._extract_topics_keyword_based(text, max_topics)
            except json.JSONDecodeError:
                print("⚠️ Could not parse JSON response from topic extraction, using keyword-based method")
                return self._extract_topics_keyword_based(text, max_topics)
                
        except Exception as e:
            print(f"Error in topic extraction: {e}")
            return self._extract_topics_keyword_based(text, max_topics)
    
    def _extract_topics_keyword_based(self, text: str, max_topics: int = 5) -> List[Dict[str, Any]]:
        """
        Extract topics using keyword frequency analysis as fallback.
        
        Args:
            text (str): The text to analyze
            max_topics (int): Maximum number of topics to extract
            
        Returns:
            List[Dict]: List of topics with their relevance scores
        """
        # Common stop words to filter out
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'his', 'hers', 'ours', 'theirs'
        }
        
        # Extract words and filter
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Get the most common words as topics
        topics = []
        for word, count in word_counts.most_common(max_topics):
            # Calculate a simple relevance score based on frequency
            relevance_score = min(100, int((count / len(words)) * 1000))
            
            topics.append({
                "topic": word.title(),
                "relevance_score": relevance_score,
                "description": f"Appears {count} times in the text",
                "key_terms": [word.title()]
            })
        
        return topics

def download_youtube_audio(youtube_url: str, output_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Download audio from a YouTube video.
    
    Args:
        youtube_url (str): The YouTube video URL
        output_path (str): Path to save the audio file
        
    Returns:
        Tuple[str, Dict[str, Any]]: Tuple containing (path to audio file, video metadata)
    """
    try:
        # Ensure output_path doesn't have .mp3 extension (yt-dlp will add it)
        if output_path.endswith('.mp3'):
            output_path = output_path[:-4]
            
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
            'keepvideo': False,  # Don't keep the original video file
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            print(f"Downloading: {info['title']}")
            
            # Extract metadata
            metadata = {
                'title': info.get('title'),
                'upload_date': info.get('upload_date'),  # YYYYMMDD format
                'timestamp': info.get('timestamp'),      # Unix timestamp
                'uploader': info.get('uploader')
            }
            
            # Return both the file path and metadata
            return f"{output_path}.mp3", metadata
    except Exception as e:
        print(f"Error downloading YouTube audio: {str(e)}")
        raise Exception(f"Failed to download YouTube audio: {str(e)}")

def transcribe_audio(audio_path: str, model_size: str = "base", use_gpu: bool = False) -> Dict[str, Any]:
    """
    Transcribe audio from a file using faster-whisper.
    
    Args:
        audio_path (str): Path to the audio file
        model_size (str): Size of the Whisper model to use ("tiny", "base", "small", "medium", "large")
        use_gpu (bool): Whether to use GPU for transcription
        
    Returns:
        Dict[str, Any]: Dictionary containing transcription and segments
    """
    try:
        # Ensure the audio file exists
        if not os.path.exists(audio_path):
            raise Exception(f"Audio file not found: {audio_path}")
            
        # Load the Whisper model with specified size
        print(f"Loading Whisper {model_size} model...")
        # Force CPU usage for faster-whisper to avoid MPS issues
        device = "cpu"  # Always use CPU for faster-whisper
        model = WhisperModel(model_size, device=device)
        
        # Transcribe the audio
        print("Transcribing audio...")
        segments, info = model.transcribe(audio_path)
        
        # Process segments to match the expected format
        transcription_segments = []
        full_text = ""
        for segment in segments:
            transcription_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "avg_logprob": segment.avg_logprob
            })
            full_text += segment.text + " "
        
        return {
            "transcription": full_text.strip(),
            "segments": transcription_segments
        }
    except Exception as e:
        print(f"Error transcribing audio: {str(e)}")
        raise Exception(f"Failed to transcribe audio: {str(e)}")

def save_to_json(transcription: str, summary: str, topics: List[Dict], video_url: str, metadata: Optional[Dict[str, Any]] = None, output_dir: Optional[str] = None) -> Optional[str]:
    """
    Save the transcription, summary, and topics to a JSON file.
    
    Args:
        transcription (str): The transcribed text
        summary (str): The generated summary
        topics (List[Dict]): List of topics with their relevance scores
        video_url (str): The YouTube video URL
        metadata (Dict[str, Any]): Video metadata from YouTube
        output_dir (str): Directory to save the JSON file (default: "transcriptions")
        
    Returns:
        str: Path to the saved JSON file
    """
    if output_dir is None:
        output_dir = get_subdir("transcriptions")
    else:
        output_dir = get_subdir(output_dir)
    
    # Generate filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"transcription_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Prepare the data structure
    metadata_dict = {
        "timestamp": timestamp,
        "video_url": video_url,
        "source": "YouTube",
        "processing_date": datetime.now().isoformat()
    }
    
    # Add video metadata if available
    if metadata:
        metadata_dict.update(metadata)
    
    data = {
        "metadata": metadata_dict,
        "content": {
            "transcription": transcription,
            "summary": summary,
            "topics": topics
        }
    }
    
    # Write to JSON file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nTranscription saved to: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        return None

def save_summary_to_txt(summary: str, video_url: str, metadata: Optional[Dict[str, Any]] = None, output_dir: Optional[str] = None) -> Optional[str]:
    """
    Save the summary to a text file for easy reading by the UI.
    
    Args:
        summary (str): The generated summary
        video_url (str): The YouTube video URL
        metadata (Dict[str, Any]): Video metadata from YouTube
        output_dir (str): Directory to save the text file (default: "summaries")
        
    Returns:
        str: Path to the saved text file
    """
    if output_dir is None:
        output_dir = get_subdir("summaries")
    else:
        output_dir = get_subdir(output_dir)
    
    # Generate filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"summary_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    
    # Prepare the content to write
    content_lines = []
    
    # Add header with metadata
    content_lines.append("=" * 80)
    content_lines.append("YOUTUBE VIDEO SUMMARY")
    content_lines.append("=" * 80)
    content_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    content_lines.append(f"Video URL: {video_url}")
    
    if metadata:
        if 'title' in metadata:
            content_lines.append(f"Video Title: {metadata['title']}")
        if 'uploader' in metadata:
            content_lines.append(f"Uploader: {metadata['uploader']}")
        if 'duration' in metadata:
            content_lines.append(f"Duration: {metadata['duration']}")
    
    content_lines.append("=" * 80)
    content_lines.append("")
    
    # Add the summary
    content_lines.append("SUMMARY:")
    content_lines.append("-" * 40)
    content_lines.append(summary)
    content_lines.append("")
    
    # Write to text file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content_lines))
        print(f"✅ Summary saved to: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving summary file: {e}")
        return None

def process_video_audio(input_source: str, metadata: Optional[Dict[str, Any]] = None, model_size: str = "base", use_gpu: bool = False, progress_callback=None) -> Dict[str, Any]:
    """
    Process video or audio content and return the processed data.
    Checks cache first to avoid reprocessing the same URLs.
    
    Args:
        input_source (str): Path to the video/audio file or YouTube URL
        metadata (Dict[str, Any], optional): Additional metadata to include
        model_size (str): Size of the Whisper model to use ("tiny", "base", "small", "medium", "large")
        use_gpu (bool): Whether to use GPU for transcription
        
    Returns:
        Dict[str, Any]: Dictionary containing processed content and analysis info
    """
    try:
        # Check if this is a YouTube URL and if it's cached
        if input_source.startswith(('http://', 'https://')):
            print("🔍 Checking YouTube cache...")
            cached_result = youtube_cache.get_cached_result(input_source)
            
            if cached_result:
                print("✅ Found cached result - simulating processing for demo...")
                
                # Simulate processing time for demo purposes with progress updates
                import time
                print("⏳ Simulating 5 seconds of processing time...")
                
                if progress_callback:
                    # Simulate realistic processing steps
                    progress_callback(20, "🔍 Checking cache...")
                    time.sleep(1)
                    progress_callback(40, "📥 Loading cached data...")
                    time.sleep(1)
                    progress_callback(60, "🔄 Processing cached content...")
                    time.sleep(1)
                    progress_callback(80, "📝 Preparing cached summary...")
                    time.sleep(1)
                    progress_callback(90, "✅ Finalizing cached result...")
                    time.sleep(1)
                else:
                    time.sleep(5)
                
                print("✅ Cached result ready!")
                
                # Return cached result in the expected format
                return {
                    "content": {
                        "transcription": cached_result["content"]["transcription"],
                        "summary": cached_result["content"]["summary"],
                        "topics": cached_result["content"]["topics"],
                        "errors": []
                    },
                    "analysis_info": {
                        "source_file": input_source,
                        "content_type": "video",
                        "processed_at": cached_result["processed_at"],
                        "transcription_file": cached_result["transcription_file"],
                        "summary_file": cached_result["summary_file"],
                        "audio_file": cached_result.get("audio_file"),
                        "summarization_method": "LoRA (cached)",
                        "title": cached_result["title"],
                        "uploader": cached_result["uploader"],
                        "duration": cached_result["duration"],
                        "cached": True
                    }
                }
            else:
                print(" Video not in cache - processing required")
        
        # If not cached or not a YouTube URL, proceed with normal processing
        print("🚀 Starting video processing...")
        
        # Initialize LoRA summarizer
        print("🚀 Initializing LoRA summarizer...")
        lora_summarizer = LoRASummarizer()
        print("✅ LoRA summarizer initialized successfully")
        
        # Create temp directory for audio files
        temp_dir = get_subdir("temp")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_audio = os.path.join(temp_dir, f"temp_audio_{timestamp}")
        
        # Download audio if it's a YouTube URL
        if input_source.startswith(('http://', 'https://')):
            print("Downloading audio from YouTube...")
            audio_file, video_metadata = download_youtube_audio(input_source, temp_audio)
            if metadata is None:
                metadata = video_metadata
            else:
                metadata.update(video_metadata)
        else:
            audio_file = input_source
            if metadata is None:
                metadata = {}
        
        # Step 1: Transcribe audio
        print("\nTranscribing audio...")
        transcription_data = transcribe_audio(audio_file, model_size=model_size, use_gpu=use_gpu)
        print("\nInitial transcription:")
        print(transcription_data["transcription"])
        
        # Step 2: Detect and correct errors using LoRA model
        corrected_text = transcription_data["transcription"]
        errors = []  # Initialize errors list
        
        # Step 3: Generate summary using LoRA model with streaming
        print("\nGenerating summary...")
        
        # Create streaming callback if progress_callback is provided
        def stream_callback(partial_summary):
            # Emit streaming update
            if progress_callback:
                progress_callback(85, f"📝 Generating summary... ({len(partial_summary)} chars)")
        
        summary = lora_summarizer.generate_summary(
            corrected_text, 
            max_length=400, 
            temperature=0.7,
            stream_callback=stream_callback
        )
        print("\nLoRA-generated summary:")
        print(summary)
        
        # Step 4: Extract topics using LoRA model
        print("\nExtracting main topics...")
        topics = lora_summarizer.extract_topics(corrected_text, max_topics=5)
        
        if topics:
            print("\nMain topics identified:")
            for topic in topics:
                print(f"\nTopic: {topic['topic']}")
                print(f"Relevance Score: {topic['relevance_score']}")
                print(f"Description: {topic['description']}")
                if 'key_terms' in topic:
                    print(f"Key Terms: {', '.join(topic['key_terms'])}")
        
        # Step 5: Save transcription
        transcription_file = save_to_json(
            corrected_text,
            summary,
            topics,
            input_source,
            metadata
        )
        print(f"\n✅ Transcription saved to: {transcription_file}")
        
        # Step 5.5: Save summary to text file
        summary_file = save_summary_to_txt(
            summary,
            input_source,
            metadata
        )
        print(f"✅ Summary saved to: {summary_file}")
        
        # Step 6: Save audio file permanently if it's a YouTube URL
        if input_source.startswith(('http://', 'https://')):
            try:
                audio_dir = get_subdir("audio_content")
                permanent_audio = os.path.join(audio_dir, f"audio_{timestamp}.mp3")
                import shutil
                shutil.copy2(audio_file, permanent_audio)
                print(f"✅ Audio file saved to: {permanent_audio}")
                metadata["audio_file"] = permanent_audio
            except Exception as e:
                print(f"Warning: Could not save audio file permanently: {str(e)}")
        
        # Step 7: Prepare result
        result = {
            "content": {
                "transcription": corrected_text,
                "summary": summary,
                "topics": topics,
                "errors": errors
            },
            "analysis_info": {
                "source_file": input_source,
                "content_type": "video" if input_source.startswith(('http://', 'https://')) else "audio",
                "processed_at": datetime.now().isoformat(),
                "transcription_file": transcription_file,
                "summary_file": summary_file,
                "summarization_method": "LoRA",
                **metadata
            }
        }
        
        # Step 8: Cache the result if it's a YouTube URL
        if input_source.startswith(('http://', 'https://')):
            youtube_cache.cache_result(input_source, result)
        
        return result
        
    except Exception as e:
        raise Exception(f"Error processing video/audio content: {str(e)}")

class YouTubeProcessor(Processor):
    """Processor for YouTube videos that downloads, transcribes, and summarizes content."""
    
    @property
    def name(self) -> str:
        """Return the name of this processor."""
        return "YouTube Processor"
    
    @property
    def description(self) -> str:
        """Return a description of what this processor does."""
        return "Downloads YouTube videos, transcribes audio, and generates summaries"
    
    def can_process(self, post: SocialPost) -> bool:
        """Check if this processor can handle the given post."""
        return post.platform == Platform.YOUTUBE
    
    def process(self, post: SocialPost, progress_callback=None) -> ProcessedContent:
        """Process a YouTube video post."""
        try:
            print(f"🎥 Processing YouTube video: {post.content}")
            
            # Process the video using the existing function
            result = process_video_audio(
                post.content,  # This should be the YouTube URL
                model_size="base",
                use_gpu=False,  # Force CPU to avoid MPS issues
                progress_callback=progress_callback
            )
            
            # Create ProcessedContent object
            processed = ProcessedContent(
                original_posts=[post],
                combined_text=result["content"]["transcription"],
                processed_text=result["content"]["summary"],
                processing_steps=["youtube_download", "audio_transcription", "lora_summarization"]
            )
            
            # Store additional metadata in the post's platform_data
            post.platform_data.update({
                "video_metadata": result["analysis_info"],
                "transcription_file": result["analysis_info"]["transcription_file"],
                "audio_file": result["analysis_info"].get("audio_file"),
                "topics": result["content"]["topics"],
                "summary_file": result["analysis_info"].get("summary_file")  # Add this line
            })
            
            print(f"✅ YouTube video processed successfully")
            return processed
            
        except Exception as e:
            print(f"❌ Error processing YouTube video: {e}")
            raise e 