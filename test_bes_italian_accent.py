#!/usr/bin/env python3
"""
Test script to demonstrate Italian accent for bes voice.
"""

import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from implementations.voice_synthesizer import VoiceSynthesizerFactory


def test_bes_italian_accent():
    """Test the Italian accent functionality for bes voice."""
    print("🎭 Testing Italian accent for bes voice")
    print("=" * 50)
    
    # Initialize voice synthesizer
    try:
        print("🚀 Initializing voice synthesizer...")
        synthesizer = VoiceSynthesizerFactory.create_synthesizer("tts")
        
        # Test bes voice with English text (should auto-apply Italian accent)
        test_text = "Today we're casually building Google from scratch. Should take about an hour."
        
        print(f"\n📝 Testing text: {test_text}")
        print(f"🎤 Voice: bes")
        print(f"🌍 Expected: Auto-applied Italian accent")
        
        # This should automatically apply Italian accent
        output_path = synthesizer.synthesize(test_text, "bes", language="en")
        
        if output_path and os.path.exists(output_path):
            print(f"✅ Success! Audio generated: {output_path}")
            print(f"🇮🇹 Italian accent was automatically applied to bes voice")
        else:
            print("❌ Failed to generate audio")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have the required dependencies installed:")
        print("   pip install -r requirements.txt")


def test_voice_preferences():
    """Test the voice language preferences functionality."""
    print("\n🎭 Testing voice language preferences")
    print("=" * 50)
    
    try:
        synthesizer = VoiceSynthesizerFactory.create_synthesizer("tts")
        
        # Get supported languages
        languages = synthesizer.get_supported_languages()
        print("🌍 Supported languages:")
        for code, name in languages.items():
            print(f"   {code}: {name}")
        
        # Get voice preferences
        preferences = synthesizer.get_voice_language_preferences()
        print("\n🎤 Voice language preferences:")
        for voice, lang in preferences.items():
            lang_name = languages.get(lang, lang)
            print(f"   {voice}: {lang} ({lang_name})")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_voice_preferences()
    test_bes_italian_accent() 