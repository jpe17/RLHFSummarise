#!/usr/bin/env python3
"""
Demonstration script for bes voice with Italian accent.
"""

import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from implementations.voice_synthesizer import VoiceSynthesizerFactory


def demo_bes_italian_accent():
    """Demonstrate bes voice with Italian accent."""
    print("🎭 Bes Voice with Italian Accent Demo")
    print("=" * 50)
    
    # Initialize voice synthesizer
    try:
        print("🚀 Initializing voice synthesizer...")
        synthesizer = VoiceSynthesizerFactory.create_synthesizer("tts")
        
        # Bes voice startup-style lines
        bes_lines = [
            "Today we're casually building Google from scratch. Should take about an hour.",
            "Now make me a karaoke app like SingStar! Perfect pitch detection by lunch!",
            "Crashed? PERFECT! That's the startup spirit right there!",
            "Listen carefully - every billion-dollar company is just three API calls in a trench coat.",
            "Remember: if it doesn't break spectacularly, you're not thinking big enough!"
        ]
        
        print(f"\n🇮🇹 Bes voice will automatically use Italian accent!")
        print(f"📝 Will synthesize {len(bes_lines)} lines...")
        
        # Generate audio for each line
        for i, line in enumerate(bes_lines, 1):
            print(f"\n🔊 Line {i}/{len(bes_lines)}: {line}")
            
            # This will automatically apply Italian accent for bes voice
            output_path = synthesizer.synthesize(line, "bes", language="en")
            
            if output_path and os.path.exists(output_path):
                print(f"✅ Generated: {os.path.basename(output_path)}")
            else:
                print(f"❌ Failed to generate audio for line {i}")
        
        print(f"\n🎉 Demo completed!")
        print(f"📁 All audio files saved in outputs/ directory")
        print(f"🇮🇹 Bes voice used Italian accent automatically!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have the required dependencies installed:")
        print("   pip install -r requirements.txt")


def show_voice_info():
    """Show information about bes voice and language support."""
    print("\n🎤 Voice Information")
    print("=" * 30)
    
    try:
        synthesizer = VoiceSynthesizerFactory.create_synthesizer("tts")
        
        # Show voice preferences
        preferences = synthesizer.get_voice_language_preferences()
        languages = synthesizer.get_supported_languages()
        
        print("🎭 Voice language preferences:")
        for voice, lang in preferences.items():
            lang_name = languages.get(lang, lang)
            print(f"   {voice}: {lang} ({lang_name})")
        
        print(f"\n🇮🇹 Bes voice automatically uses Italian accent!")
        print(f"🌍 You can also manually specify other languages:")
        for code, name in list(languages.items())[:5]:  # Show first 5
            print(f"   - {code}: {name}")
        print(f"   ... and {len(languages)-5} more languages")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    show_voice_info()
    demo_bes_italian_accent() 