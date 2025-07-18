#!/usr/bin/env python3
"""
Audio File Splitter

This script splits audio files into multiple segments of specified duration.
"""

import os
import sys
import argparse
from typing import List, Tuple
import librosa
import soundfile as sf


def split_audio_file(input_path: str, output_dir: str, segment_duration: float = 30.0, overlap: float = 0.0) -> List[str]:
    """
    Split an audio file into multiple segments.
    
    Args:
        input_path: Path to the input audio file
        output_dir: Directory to save the output segments
        segment_duration: Duration of each segment in seconds
        overlap: Overlap between segments in seconds
        
    Returns:
        List of paths to the generated audio segments
    """
    print(f"🎵 Loading audio file: {input_path}")
    
    # Load the audio file
    try:
        audio, sample_rate = librosa.load(input_path, sr=None)
        duration = len(audio) / sample_rate
        print(f"📊 Audio info: {duration:.2f}s duration, {sample_rate}Hz sample rate")
    except Exception as e:
        print(f"❌ Error loading audio file: {e}")
        return []
    
    # Calculate segment parameters
    samples_per_segment = int(segment_duration * sample_rate)
    samples_overlap = int(overlap * sample_rate)
    step_size = samples_per_segment - samples_overlap
    
    # Calculate number of segments
    num_segments = max(1, int((len(audio) - samples_overlap) / step_size))
    
    print(f"✂️  Splitting into {num_segments} segments of {segment_duration}s each")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    output_files = []
    
    # Split the audio
    for i in range(num_segments):
        start_sample = i * step_size
        end_sample = min(start_sample + samples_per_segment, len(audio))
        
        # Extract segment
        segment = audio[start_sample:end_sample]
        
        # Generate output filename
        output_filename = f"{base_name}_part{i+1:02d}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save segment
        try:
            sf.write(output_path, segment, sample_rate)
            segment_duration_actual = len(segment) / sample_rate
            print(f"✅ Segment {i+1}/{num_segments}: {output_filename} ({segment_duration_actual:.2f}s)")
            output_files.append(output_path)
        except Exception as e:
            print(f"❌ Error saving segment {i+1}: {e}")
    
    return output_files


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Split audio files into multiple segments"
    )
    
    parser.add_argument(
        "input_file",
        help="Path to the input audio file"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="split_audio",
        help="Output directory for segments (default: split_audio)"
    )
    
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=30.0,
        help="Duration of each segment in seconds (default: 30.0)"
    )
    
    parser.add_argument(
        "--overlap", "-l",
        type=float,
        default=0.0,
        help="Overlap between segments in seconds (default: 0.0)"
    )
    
    parser.add_argument(
        "--list-formats",
        action="store_true",
        help="List supported audio formats and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_formats:
        print("🎵 Supported audio formats:")
        print("   - WAV (.wav)")
        print("   - MP3 (.mp3)")
        print("   - FLAC (.flac)")
        print("   - OGG (.ogg)")
        print("   - M4A (.m4a)")
        print("   - And many more (via librosa)")
        return
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"❌ Input file not found: {args.input_file}")
        return
    
    print("🎵 Audio File Splitter")
    print("=" * 50)
    print(f"📁 Input: {args.input_file}")
    print(f"📁 Output: {args.output_dir}")
    print(f"⏱️  Segment duration: {args.duration}s")
    print(f"🔄 Overlap: {args.overlap}s")
    print("=" * 50)
    
    # Split the audio
    output_files = split_audio_file(
        args.input_file,
        args.output_dir,
        args.duration,
        args.overlap
    )
    
    if output_files:
        print(f"\n🎉 Successfully created {len(output_files)} audio segments!")
        print(f"📁 All files saved in: {args.output_dir}")
        
        # Show file sizes
        total_size = 0
        for file_path in output_files:
            size = os.path.getsize(file_path)
            total_size += size
            print(f"   {os.path.basename(file_path)}: {size/1024:.1f} KB")
        
        print(f"📊 Total size: {total_size/1024:.1f} KB")
    else:
        print("❌ No audio segments were created.")


if __name__ == "__main__":
    main() 