import torch
import time
import os
from pathlib import Path
import numpy as np

# FOOLPROOF FIX: Set torch.load to use weights_only=False globally
# This is safe since we're loading from trusted Coqui TTS models
torch.serialization.add_safe_globals([
    'TTS.tts.configs.xtts_config.XttsConfig',
    'TTS.tts.models.xtts.XttsAudioConfig',
    'TTS.tts.configs.shared_configs.BaseDatasetConfig',
    'TTS.tts.configs.shared_configs.BaseAudioConfig',
    'TTS.tts.configs.shared_configs.BaseTrainingConfig',
    'TTS.tts.configs.shared_configs.BaseModelConfig',
    'TTS.vocoder.configs.hifigan_config.HifiganConfig',
    'TTS.vocoder.configs.shared_configs.BaseVocoderConfig',
    'TTS.encoder.configs.base_encoder_config.BaseEncoderConfig',
    'TTS.encoder.configs.speaker_encoder_config.SpeakerEncoderConfig',
])

# Alternative: Monkey patch torch.load to use weights_only=False
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

# Fix for GPT2InferenceModel compatibility
import transformers
if hasattr(transformers, 'GenerationMixin'):
    # Add generate method to GPT2InferenceModel if it doesn't exist
    def add_generate_method():
        try:
            from transformers import GPT2InferenceModel
            if not hasattr(GPT2InferenceModel, 'generate'):
                GPT2InferenceModel.generate = lambda self, *args, **kwargs: self.forward(*args, **kwargs)
        except ImportError:
            pass
    add_generate_method()

from TTS.api import TTS

# Setup
VOICE_DIR = Path("voices/elonmusk")
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Pre-written texts for testing - EDIT THESE TEXTS HERE
SAMPLE_TEXTS = [
    "I'm not a Trump fan, but I don't want to be a Trump fan. I don't want to get stuck in a Trump-verse. So, how can you tell if someone is an actual Trump supporter?"
]

class SimpleVoiceClone:
    def __init__(self):
        print(f"🚀 Loading TTS model...")
        
        # Load with explicit CPU first
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA to avoid conflicts
        
        self.tts = TTS(MODEL_NAME, gpu=False)  # Force CPU during loading
        
        # Now try to move to MPS if available
        current_device = DEVICE
        if current_device == "mps":
            try:
                self.tts = self.tts.to(current_device)
                print("✅ Model loaded on MPS!")
            except Exception as e:
                print(f"⚠️  MPS failed, using CPU: {e}")
                current_device = "cpu"
        else:
            print("✅ Model loaded on CPU!")
        
        # Quick warmup
        print("🔥 Warming up...")
        self._warmup()
        
    def _warmup(self):
        """Warmup the model"""
        try:
            # Get reference files for warmup
            reference_wavs = self.get_reference_audio()
            if reference_wavs:
                self.tts.tts("Test", speaker_wav=reference_wavs[0], language="en")
                print("✅ Warmup successful!")
            else:
                print("⚠️  Skipping warmup - no reference audio available")
        except Exception as e:
            print(f"Warmup failed: {e}")
        
    def get_reference_audio(self):
        """Get reference audio files - using multiple files for better quality"""
        wav_files = list(VOICE_DIR.glob("*.wav"))
        
        if len(wav_files) < 1:
            print(f"❌ No .wav files found in {VOICE_DIR}")
            print("Please add your actor's .wav files to the voices/elonmusk/ folder")
            return None
            
        print(f"📁 Found {len(wav_files)} audio files")
        
        # Filter out dummy.wav and use real audio files
        real_wav_files = [f for f in wav_files if f.name != "dummy.wav"]
        
        if not real_wav_files:
            print(f"❌ No valid audio files found in {VOICE_DIR}")
            print("Please add your actor's .wav files (not dummy.wav)")
            return None
            
        # Sort by file size (larger files usually have better quality)
        real_wav_files.sort(key=lambda x: x.stat().st_size, reverse=True)
        
        # Use multiple reference files for better voice consistency
        reference_wavs = [str(f) for f in real_wav_files[:3]]  # Use up to 3 best files
        
        print(f"🎭 Using {len(reference_wavs)} reference files:")
        for i, ref in enumerate(reference_wavs, 1):
            print(f"  {i}. {Path(ref).name}")
        
        return reference_wavs
    
    def speak(self, text, reference_wavs, quality_mode="high"):
        """Generate speech with cloned voice using optimized parameters"""
        if not reference_wavs:
            return None
            
        print(f"🗣️ Generating: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        start = time.time()
        
        try:
            # Use the best reference file (largest/first)
            primary_reference = reference_wavs[0]
            
            # Optimized parameters for better voice quality
            if quality_mode == "high":
                # High quality settings - slower but better
                kwargs = {
                    "temperature": 0.65,  # Lower = more consistent, higher = more varied
                    "length_penalty": 1.0,  # Balanced length
                    "repetition_penalty": 2.5,  # Reduce repetition and "uhhs"
                    "top_k": 40,  # More focused sampling
                    "top_p": 0.85,  # Nucleus sampling
                    "gpt_cond_len": 12,  # Use 12 seconds of reference audio
                    "gpt_cond_chunk_len": 6,  # 6 second chunks
                    "split_sentences": True,  # Better for longer texts
                }
            elif quality_mode == "balanced":
                # Balanced settings - good quality, reasonable speed
                kwargs = {
                    "temperature": 0.7,
                    "length_penalty": 1.0,
                    "repetition_penalty": 2.0,
                    "top_k": 50,
                    "top_p": 0.8,
                    "gpt_cond_len": 8,
                    "gpt_cond_chunk_len": 4,
                    "split_sentences": True,
                }
            else:  # fast
                # Fast settings - prioritize speed
                kwargs = {
                    "temperature": 0.75,
                    "repetition_penalty": 1.8,
                    "top_k": 50,
                    "top_p": 0.8,
                    "gpt_cond_len": 6,
                    "gpt_cond_chunk_len": 3,
                    "split_sentences": False,
                }
            
            # Generate audio with optimized XTTS v2 settings
            audio = self.tts.tts(
                text=text,
                speaker_wav=primary_reference,
                language="en",
                **kwargs
            )
            
            duration = time.time() - start
            print(f"⚡ Generated in {duration:.2f}s ({len(text)/duration:.1f} chars/sec)")
            
            return audio
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return None
    
    def save_audio(self, audio, output_path="output.wav"):
        """Save generated audio"""
        if audio is not None:
            self.tts.synthesizer.save_wav(audio, output_path)
            print(f"💾 Saved: {output_path}")
            return True
        return False
    
    def generate_sample_outputs(self, reference_wavs, quality_mode="high"):
        """Generate sample outputs using pre-written texts with different quality modes"""
        print(f"\n🎬 Generating outputs with {quality_mode} quality mode...")
        
        for i, text in enumerate(SAMPLE_TEXTS, 1):
            print(f"\n📝 Processing {i}/{len(SAMPLE_TEXTS)}")
            print(f"Text: {text}")
            
            audio = self.speak(text, reference_wavs, quality_mode)
            if audio is not None:
                output_path = f"output_{quality_mode}_{i:02d}.wav"
                self.save_audio(audio, output_path)
            
            # Small delay between generations
            time.sleep(0.5)
        
        print(f"\n✅ Generated {len(SAMPLE_TEXTS)} outputs in {quality_mode} mode!")

def main():
    # Create voice directory
    VOICE_DIR.mkdir(exist_ok=True)
    
    # Check for audio files
    if not list(VOICE_DIR.glob("*.wav")):
        print(f"📁 Please add your .wav files to: {VOICE_DIR}")
        print("Then run the script again.")
        return
    
    # Initialize
    try:
        cloner = SimpleVoiceClone()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Try: pip install --upgrade TTS torch")
        return
    
    # Get reference voices
    reference_wavs = cloner.get_reference_audio()
    if not reference_wavs:
        return
    
    # Generate outputs with high quality mode only
    print("\n🎭 Generating high-quality Elon Musk voice outputs...")
    cloner.generate_sample_outputs(reference_wavs, quality_mode="high")
    
    print("\n🎉 All done! Check the high-quality output files:")
    print("  - output_high_*.wav: High-quality Elon Musk voice outputs")

if __name__ == "__main__":
    main()