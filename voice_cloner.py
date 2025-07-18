#!/usr/bin/env python3
"""
Simple Voice Cloner Script - Streamlined for conversational mode
"""

import os
import sys
import argparse
from typing import List, Dict

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from implementations.voice_synthesizer import VoiceSynthesizerFactory


def get_available_voices(voices_dir: str = "voices") -> List[str]:
    """Get list of available voices from the voices directory."""
    voices = []
    
    if not os.path.exists(voices_dir):
        print(f"❌ Voices directory not found: {voices_dir}")
        return voices
        
    try:
        for item in os.listdir(voices_dir):
            voice_dir = os.path.join(voices_dir, item)
            if os.path.isdir(voice_dir):
                # Check if the voice directory contains audio files
                has_audio = any(
                    f.endswith(('.wav', '.mp3', '.flac'))
                    for f in os.listdir(voice_dir)
                )
                if has_audio:
                    voices.append(item)
                    
    except Exception as e:
        print(f"❌ Error scanning voices directory: {e}")
        
    return sorted(voices)


def get_supported_languages() -> Dict[str, str]:
    """Get supported languages and their descriptions."""
    return {
        "en": "English",
        "it": "Italian",
        "es": "Spanish", 
        "fr": "French",
        "de": "German",
        "pt": "Portuguese",
        "pl": "Polish",
        "tr": "Turkish",
        "ru": "Russian",
        "nl": "Dutch",
        "cs": "Czech",
        "ar": "Arabic",
        "zh-cn": "Chinese (Simplified)",
        "hu": "Hungarian",
        "ko": "Korean",
        "ja": "Japanese",
        "hi": "Hindi"
    }


def get_conversation_scripts() -> Dict[str, List[str]]:
    """Get pre-defined conversation scripts for different voices."""
    return {
        "bes": [
            "Today we're casually building Google from scratch. Should take about an hour.",
            "Now make me a karaoke app like SingStar! Perfect pitch detection by lunch!",
            "Crashed? PERFECT! That's the startup spirit right there!",
            "Listen carefully - every billion-dollar company is just three API calls in a trench coat.",
            "Remember: if it doesn't break spectacularly, you're not thinking big enough!"
            ],
        "barackobama": [
            "Hello there, how are you doing today?",
            "That's wonderful to hear.",
            "You know, technology like this reminds me of the importance of innovation.",
            "We must always strive to use new tools for the betterment of society.",
            "What do you think about the future of AI?",
            "I believe we can achieve great things when we work together.",
            "Keep pushing the boundaries of what's possible.",
            "Thank you for this interesting conversation."
        ],
        "elonmusk": [
            "Hey, this voice cloning tech is pretty incredible, right?",
            "We're literally living in the future.",
            "Mars colonization is next, but first we perfect AI on Earth.",
            "The neural networks behind this are fascinating.",
            "What's your take on the singularity?",
            "We need to be careful but also embrace the potential.",
            "Innovation is the key to human progress.",
            "Let's build something amazing together."
        ],
        "freeman": [
            "Hello, my friend. Welcome to this conversation.",
            "In a world of artificial intelligence, we find ourselves at a crossroads.",
            "The voice you hear is not truly mine, yet it carries my essence.",
            "Technology has given us the power to transcend our limitations.",
            "What wisdom would you seek in this digital age?",
            "Remember, with great power comes great responsibility.",
            "The future is what we make of it.",
            "Until we meet again, in whatever form that may be."
        ]
    }


def wait_for_user_response(prompt: str = "Press Enter to continue or type 'quit' to exit") -> bool:
    """Wait for user input to continue the conversation."""
    try:
        print(f"\n💬 {prompt}: ", end="")
        user_input = input().strip().lower()
        
        if user_input in ['quit', 'exit', 'q']:
            print("\n👋 Ending conversation. Goodbye!")
            return False
            
        return True
        
    except KeyboardInterrupt:
        print("\n\n👋 Conversation interrupted. Goodbye!")
        return False


def play_audio_with_system(audio_path: str) -> None:
    """Play audio file using system player."""
    try:
        import subprocess
        import platform
        
        system = platform.system()
        if system == "Darwin":  # macOS
            subprocess.run(["afplay", audio_path], check=False)
        elif system == "Windows":
            subprocess.run(["start", audio_path], shell=True, check=False)
        elif system == "Linux":
            subprocess.run(["aplay", audio_path], check=False)
    except Exception as e:
        print(f"🔇 Could not play audio: {e}")


def conversational_mode(voice_name: str, synthesizer, language: str = "en", voices_dir: str = "voices"):
    """Run the voice cloner in conversational mode."""
    print(f"\n🎭 Starting conversation with {voice_name} in {get_supported_languages().get(language, language)}...")
    print("=" * 60)
    print("💡 Tips:")
    print("   - Press Enter to start the conversation")
    print("   - Press Enter to continue to the next line")
    print("   - Type 'quit' to exit the conversation")
    print("   - Each line will be spoken with a pause for your response")
    print("=" * 60)
    
    # Get conversation script based on language
    # Always use English scripts regardless of language setting
    scripts = get_conversation_scripts()
    
    if voice_name in scripts:
        conversation_lines = scripts[voice_name]
        print(f"🎬 Using pre-written conversation for {voice_name} in {get_supported_languages().get(language, language)}")
        if language == "it":
            print("🇮🇹 Note: Using English text with Italian accent")
    else:
        print(f"🎬 No pre-written conversation found for {voice_name}")
        print("❌ Cannot continue without conversation script.")
        return
    
    print(f"\n🚀 Pre-generating all audio for instant playback...")
    print("   This will take a moment but ensures smooth conversation flow!")
    print("=" * 60)
    
    # Pre-generate all audio files
    pre_generated_audio = []
    for i, line in enumerate(conversation_lines, 1):
        print(f"🔊 Pre-generating audio {i}/{len(conversation_lines)}: {line[:50]}{'...' if len(line) > 50 else ''}")
        
        try:
            # Pass language as keyword argument to the internal method
            output_path = synthesizer.synthesize(line, voice_name, language=language)
            
            if output_path and os.path.exists(output_path):
                pre_generated_audio.append(output_path)
                print(f"✅ Audio {i} ready: {os.path.basename(output_path)}")
            else:
                print(f"❌ Failed to generate audio for line {i}")
                pre_generated_audio.append(None)
                
        except Exception as e:
            print(f"❌ Error generating audio {i}: {e}")
            pre_generated_audio.append(None)
    
    print(f"\n🎉 All audio pre-generated! Starting conversation...")
    print("=" * 60)
    
    # Wait for user to press Enter to start the conversation
    print("\n⏸️  Press Enter to start the conversation...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n\n👋 Conversation cancelled. Goodbye!")
        return
    
    # Run the conversation with pre-generated audio
    for i, (line, audio_path) in enumerate(zip(conversation_lines, pre_generated_audio), 1):
        print(f"\n🎭 {voice_name} (Line {i}/{len(conversation_lines)}): {line}")
        
        if audio_path and os.path.exists(audio_path):
            print("🎵 Playing pre-generated audio...")
            play_audio_with_system(audio_path)
            
            # Wait for user response
            if not wait_for_user_response(f"Your response to {voice_name}"):
                break
        else:
            print("❌ No audio available for this line.")
            if not wait_for_user_response("Continue anyway"):
                break
    
    print(f"\n🎭 Conversation with {voice_name} completed!")
    print("📁 All generated audio files are saved in the outputs/ directory.")


def main():
    """Main function to run the voice cloner."""
    parser = argparse.ArgumentParser(
        description="Simple Voice Cloner - Conversational mode"
    )
    
    parser.add_argument(
        "--voice", "-v",
        type=str,
        required=True,
        help="Voice name to use"
    )
    
    parser.add_argument(
        "--language", "-l",
        type=str,
        choices=list(get_supported_languages().keys()),
        default="en",
        help="Language/accent to use (default: en for English, it for Italian, etc.)"
    )
    
    parser.add_argument(
        "--conversation", "-c",
        action="store_true",
        help="Start conversational mode with breaks for responses"
    )
    
    parser.add_argument(
        "--voices-dir",
        type=str,
        default="voices",
        help="Directory containing voice models (default: voices)"
    )
    
    args = parser.parse_args()
    
    print("🎤 Voice Cloner - Conversational Mode")
    print("=" * 50)
    
    # Get available voices
    voices = get_available_voices(args.voices_dir)
    
    if not voices:
        print("❌ No voices found. Please check your voices directory.")
        return
    
    # Validate voice
    voice_name = args.voice
    if voice_name not in voices:
        print(f"❌ Voice '{voice_name}' not found.")
        print("Available voices:")
        for voice in voices:
            print(f"  - {voice}")
        return
    
    print(f"🎭 Using voice: {voice_name}")
    print(f"🌍 Using language/accent: {get_supported_languages().get(args.language, args.language)}")
    
    # Initialize voice synthesizer
    try:
        print("🚀 Initializing voice synthesizer...")
        synthesizer = VoiceSynthesizerFactory.create_synthesizer(
            "tts",
            voices_dir=args.voices_dir
        )
        
        # Run conversational mode
        conversational_mode(voice_name, synthesizer, args.language, args.voices_dir)
            
    except Exception as e:
        print(f"❌ Error during synthesis: {e}")
        print("💡 Make sure you have the required dependencies installed:")
        print("   pip install -r requirements.txt")


if __name__ == "__main__":
    main() 