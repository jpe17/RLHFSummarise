#!/usr/bin/env python3
"""
Performance test script to measure optimization improvements.
"""

import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline_factory import create_preloaded_pipeline
from core.data_models import Platform

def test_performance():
    """Test the performance of the optimized pipeline."""
    print("🚀 Testing Performance Optimizations")
    print("=" * 50)
    
    # Initialize pipeline
    print("📥 Initializing pipeline...")
    start_time = time.time()
    pipeline = create_preloaded_pipeline()
    init_time = time.time() - start_time
    print(f"✅ Pipeline initialized in {init_time:.2f} seconds")
    
    # Test users
    test_users = [
        {"platform": "twitter", "username": "sama"},
        {"platform": "instagram", "username": "BarackObama"}
    ]
    
    # Test summary generation (without TTS)
    print("\n📝 Testing Summary Generation...")
    start_time = time.time()
    
    try:
        result = pipeline.run_pipeline_without_tts(
            users=test_users,
            selection_type="latest",
            count=3
        )
        
        summary_time = time.time() - start_time
        print(f"✅ Summary generated in {summary_time:.2f} seconds")
        print(f"📊 Summary length: {len(result.summary.content)} characters")
        print(f"📋 Summary preview: {result.summary.content[:100]}...")
        
    except Exception as e:
        print(f"❌ Summary generation failed: {e}")
        return
    
    # Test TTS generation
    print("\n🔊 Testing TTS Generation...")
    start_time = time.time()
    
    try:
        audio_path = pipeline.voice_synthesizer.synthesize(
            result.summary.content,
            "deniro"
        )
        
        tts_time = time.time() - start_time
        print(f"✅ TTS generated in {tts_time:.2f} seconds")
        print(f"🎵 Audio file: {audio_path}")
        
    except Exception as e:
        print(f"❌ TTS generation failed: {e}")
        return
    
    # Performance summary
    print("\n📊 Performance Summary:")
    print(f"  • Pipeline initialization: {init_time:.2f}s")
    print(f"  • Summary generation: {summary_time:.2f}s")
    print(f"  • TTS generation: {tts_time:.2f}s")
    print(f"  • Total processing time: {summary_time + tts_time:.2f}s")
    
    # Performance targets
    print("\n🎯 Performance Targets:")
    print(f"  • Summary generation: {'✅ GOOD' if summary_time < 30 else '⚠️ SLOW'} (target: <30s)")
    print(f"  • TTS generation: {'✅ GOOD' if tts_time < 60 else '⚠️ SLOW'} (target: <60s)")
    print(f"  • Total processing: {'✅ GOOD' if (summary_time + tts_time) < 90 else '⚠️ SLOW'} (target: <90s)")

if __name__ == "__main__":
    test_performance() 