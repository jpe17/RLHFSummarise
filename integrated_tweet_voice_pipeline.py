import os
# Set MPS fallback to handle TTS model compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import time
from pathlib import Path
from datetime import datetime
import json
import re
import emoji

# Voice synthesis imports
from TTS.api import TTS

# Tweet summarization imports
import sys
sys.path.append("backend")
from backend.tweet_summarizer_pipeline_ppo import TweetSummarizerPipelinePPO

def clean_text_for_tts(text):
    """
    Clean text for better TTS synthesis by removing:
    - Hashtags (#hashtag)
    - Emojis and emoticons
    - URLs
    - Excessive punctuation
    - Twitter handles (@username)
    - RT (retweet indicators)
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text suitable for TTS
    """
    if not text:
        return ""
    
    # Remove emojis
    text = emoji.replace_emojis(text, replace='')
    
    # Remove hashtags (but keep the text part)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove Twitter handles
    text = re.sub(r'@\w+', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove RT indicators
    text = re.sub(r'\bRT\b', '', text, flags=re.IGNORECASE)
    
    # Clean up excessive punctuation
    # Replace multiple periods/commas with single ones
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r',{2,}', ',', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    
    # Remove emoticons like :), :(, :D, etc.
    text = re.sub(r':[)\-DdPpOo]', '', text)
    text = re.sub(r'[)\-DdPpOo]:', '', text)
    
    # Remove other common emoticons
    emoticon_patterns = [
        r'[;:]-?[)DdPpOo]',  # ;), ;D, etc.
        r'[)DdPpOo]-?[;:]',  # ):, D:, etc.
        r'[<>][;:]-?[)DdPpOo]',  # <3, >:), etc.
        r'[;:]-?[<>]',  # ;<, :>, etc.
    ]
    for pattern in emoticon_patterns:
        text = re.sub(pattern, '', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
    text = text.strip()
    
    # Remove leading/trailing punctuation
    text = re.sub(r'^[.,!?;:\s]+', '', text)
    text = re.sub(r'[.,!?;:\s]+$', '', text)
    
    # Ensure proper sentence endings
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    
    return text

class IntegratedTweetVoicePipeline:
    def __init__(self, 
                 model_id="Qwen/Qwen1.5-0.5B",
                 ppo_weights_path="simple_ppo_lora_final_20250716_130239.pt",
                 reward_model_path="qwen_reward_model.pt"):
        """
        Integrated pipeline for tweet summarization and voice synthesis.
        
        Args:
            model_id (str): Hugging Face model ID for summarization
            ppo_weights_path (str): Path to PPO-trained LoRA weights
            reward_model_path (str): Path to reward model weights
        """
        print("🚀 Initializing Integrated Tweet-to-Voice Pipeline...")
        
        # Initialize tweet summarizer
        self.tweet_pipeline = TweetSummarizerPipelinePPO(
            model_id=model_id,
            ppo_weights_path=ppo_weights_path,
            reward_model_path=reward_model_path
        )
        
        # Initialize TTS
        self.tts = None
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._setup_tts()
        
        print("✅ Pipeline initialized successfully!")
    
    def _setup_tts(self):
        """Setup text-to-speech model."""
        print("🎙️ Loading TTS model...")
        # Always use CPU for TTS to avoid MPS issues
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        print("✅ TTS model loaded on CPU!")
    
    def get_reference_audio(self, voice_name):
        """Get reference audio files for a voice."""
        voice_dir = Path(f"voices/{voice_name}")
        
        if not voice_dir.exists():
            print(f"❌ Voice directory not found: {voice_dir}")
            return None
        
        wav_files = list(voice_dir.glob("*.wav"))
        if not wav_files:
            print(f"❌ No .wav files found in {voice_dir}")
            return None
        
        # Sort by file size and take the largest (usually best quality)
        wav_files.sort(key=lambda x: x.stat().st_size, reverse=True)
        reference_wav = str(wav_files[0])
        
        print(f"🎭 Using reference audio: {Path(reference_wav).name}")
        return reference_wav
    
    def synthesize_voice(self, text, voice_name, output_path=None):
        """
        Convert text to speech using the specified voice.
        
        Args:
            text (str): Text to convert to speech
            voice_name (str): Name of the voice directory
            output_path (str): Optional output path for the audio file
            
        Returns:
            str: Path to the generated audio file
        """
        reference_wav = self.get_reference_audio(voice_name)
        if not reference_wav:
            return None
        
        # Clean text for better TTS synthesis
        original_text = text
        text = clean_text_for_tts(text)
        
        if text != original_text:
            print(f"🧹 Cleaned text for TTS:")
            print(f"   Original: {original_text}")
            print(f"   Cleaned:  {text}")
        
        print(f"🗣️ Generating speech with {voice_name} voice...")
        
        try:
            # Generate audio with optimized settings
            audio = self.tts.tts(
                text=text,
                speaker_wav=reference_wav,
                language="en",
                temperature=0.65,
                length_penalty=1.0,
                repetition_penalty=2.5,
                top_k=40,
                top_p=0.85,
                gpt_cond_len=12,
                gpt_cond_chunk_len=6,
                split_sentences=True
            )
            
            # Save audio
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"tweet_summary_{voice_name}_{timestamp}.wav"
            
            self.tts.synthesizer.save_wav(audio, output_path)
            print(f"💾 Audio saved: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Voice synthesis failed: {e}")
            return None
    
    def process_user_with_voice(self, username, voice_name, tweet_count=10):
        """
        Complete pipeline: scrape tweets, summarize, and convert to voice.
        
        Args:
            username (str): Twitter username (without @)
            voice_name (str): Name of the voice to use
            tweet_count (int): Number of tweets to fetch
            
        Returns:
            dict: Results containing summary, score, and audio path
        """
        print(f"\n{'='*60}")
        print(f"🎯 Processing @{username} with {voice_name} voice")
        print(f"{'='*60}")
        
        # Step 1: Get tweet summary
        summary_results = self.tweet_pipeline.process_user(username, tweet_count)
        
        if "error" in summary_results:
            print(f"❌ Tweet processing failed: {summary_results['error']}")
            return summary_results
        
        summary = summary_results.get("summary", "")
        if not summary:
            print("❌ No summary generated")
            return summary_results
        
        print(f"📝 Summary: {summary}")
        print(f"🏆 Quality Score: {summary_results.get('score', 0):.4f}")
        
        # Step 2: Convert to voice
        audio_path = self.synthesize_voice(summary, voice_name)
        
        # Add audio info to results
        summary_results["voice_name"] = voice_name
        summary_results["audio_path"] = audio_path
        summary_results["voice_generated"] = audio_path is not None
        
        return summary_results
    
    def run_batch_processing(self, user_voice_pairs):
        """
        Process multiple users with their corresponding voices.
        
        Args:
            user_voice_pairs (list): List of (username, voice_name) tuples
            
        Returns:
            list: List of results for each user
        """
        results = []
        
        for username, voice_name in user_voice_pairs:
            try:
                result = self.process_user_with_voice(username, voice_name)
                results.append(result)
                
                # Save individual results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"results_{username}_{voice_name}_{timestamp}.json"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                print(f"💾 Results saved: {filename}")
                
                # Small delay between users
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Error processing @{username}: {e}")
                results.append({
                    "username": username,
                    "voice_name": voice_name,
                    "error": str(e)
                })
        
        return results
    
    def close(self):
        """Clean up resources."""
        if self.tweet_pipeline:
            self.tweet_pipeline.close()
        print("🧹 Pipeline closed")

def main():
    """Demo the integrated pipeline."""
    # Configuration
    test_usernames = ["elonmusk", "BarackObama", "NASA"]
    voices = ["elonmusk", "freeman", "deniro"]
    
    # Create user-voice pairs
    user_voice_pairs = list(zip(test_usernames, voices))
    
    print("🎬 Starting Integrated Tweet-to-Voice Pipeline")
    print(f"📋 Processing: {user_voice_pairs}")
    
    # Initialize pipeline
    pipeline = IntegratedTweetVoicePipeline()
    
    try:
        # Process all users
        results = pipeline.run_batch_processing(user_voice_pairs)
        
        # Print summary
        print(f"\n{'='*80}")
        print("🎉 BATCH PROCESSING COMPLETE")
        print(f"{'='*80}")
        
        for result in results:
            if "error" in result:
                print(f"❌ @{result['username']} ({result['voice_name']}): {result['error']}")
            else:
                print(f"✅ @{result['username']} ({result['voice_name']}): {result.get('audio_path', 'No audio')}")
                print(f"   Score: {result.get('score', 0):.4f} | Summary: {result.get('summary', '')[:100]}...")
        
    finally:
        pipeline.close()

if __name__ == "__main__":
    main() 