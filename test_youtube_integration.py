#!/usr/bin/env python3
"""
Test script for YouTube integration
"""

import requests
import json
import time

def test_youtube_integration():
    """Test the YouTube integration endpoints."""
    
    base_url = "http://localhost:5464"
    
    print("🧪 Testing YouTube Integration...")
    
    # Test 1: Validate YouTube URL
    print("\n1. Testing URL validation...")
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://example.com/invalid",
        ""
    ]
    
    for url in test_urls:
        response = requests.post(f"{base_url}/api/youtube/validate", 
                               json={"youtube_url": url})
        if response.status_code == 200:
            data = response.json()
            print(f"   {url[:50]}... -> {'✅ Valid' if data.get('valid') else '❌ Invalid'}")
        else:
            print(f"   {url[:50]}... -> ❌ Error: {response.status_code}")
    
    # Test 2: Process a short YouTube video
    print("\n2. Testing YouTube processing...")
    test_video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll - short video
    
    try:
        response = requests.post(f"{base_url}/api/youtube/process", 
                               json={
                                   "youtube_url": test_video_url,
                                   "voice_name": "",  # No audio for test
                                   "whisper_model": "base",
                                   "use_gpu": False
                               })
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                job_id = data.get('job_id')
                print(f"   ✅ Processing started - Job ID: {job_id}")
                
                # Wait a bit and check status
                time.sleep(2)
                status_response = requests.get(f"{base_url}/api/status/{job_id}")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"   📊 Status: {status_data.get('status')} - {status_data.get('message', '')}")
            else:
                print(f"   ❌ Processing failed: {data.get('message')}")
        else:
            print(f"   ❌ Request failed: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("   ❌ Could not connect to server. Make sure the Flask app is running.")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n✅ YouTube integration test completed!")

if __name__ == "__main__":
    test_youtube_integration() 