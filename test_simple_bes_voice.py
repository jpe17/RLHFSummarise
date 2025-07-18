#!/usr/bin/env python3
"""
Simple test to verify bes voice works with Italian accent.
"""

import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from implementations.voice_synthesizer import VoiceSynthesizerFactory


def test_bes_voice():
    """Test that bes voice works with Italian accent."""
    print("🧪 Testing Bes Voice with Italian Accent")
    print("=" * 50)
    
    try:
        # Initialize voice synthesizer
        synthesizer = VoiceSynthesizerFactory.create_synthesizer("tts")
        
        # Test text
        test_text = "Today we're casually building Google from scratch. Should take about an hour."
        
        print(f"📝 Testing text: {test_text}")
        print(f"🎤 Voice: bes")
        print(f"🌍 Expected: Italian accent automatically applied")
        
        # This should automatically apply Italian accent for bes voice
        result = synthesizer.synthesize(test_text, "bes")
        
        if result and result.audio_path and os.path.exists(result.audio_path):
            print(f"✅ Success! Audio generated: {result.audio_path}")
            print(f"🇮🇹 Voice: {result.voice_name}")
            print(f"📄 Text: {result.text}")
            print(f"⏱️ Duration: {result.duration:.2f} seconds")
        else:
            print("❌ Failed to generate audio")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have the required dependencies installed:")
        print("   pip install -r requirements.txt")


if __name__ == "__main__":
    test_bes_voice() 