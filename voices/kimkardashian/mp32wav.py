#!/usr/bin/env python3
"""
MP3 to WAV converter script
Converts the first 30 seconds of all MP3 files in the specified directory to WAV format
"""

import os
import glob
from pydub import AudioSegment
import sys

def convert_mp3_to_wav(input_file, output_file=None, duration_seconds=30):
    """
    Convert the first N seconds of an MP3 file to WAV format
    
    Args:
        input_file (str): Path to the input MP3 file
        output_file (str): Path to the output WAV file (optional)
        duration_seconds (int): Duration in seconds to extract from the beginning
    """
    try:
        # Load the MP3 file
        audio = AudioSegment.from_mp3(input_file)
        
        # Extract only the first N seconds
        duration_ms = duration_seconds * 1000
        audio_clip = audio[:duration_ms]
        
        # Generate output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}.wav"
        
        # Export as WAV
        audio_clip.export(output_file, format="wav")
        print(f"✓ Converted first {duration_seconds}s: {input_file} -> {output_file}")
        return True
        
    except Exception as e:
        print(f"✗ Error converting {input_file}: {str(e)}")
        return False

def main():
    """Main function to convert the first 30 seconds of all MP3 files in the specified directory"""
    
    # Get directory from command line argument or use current directory
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        if not os.path.isdir(target_dir):
            print(f"Error: '{target_dir}' is not a valid directory.")
            sys.exit(1)
    else:
        target_dir = "."
    
    # Change to the target directory
    original_dir = os.getcwd()
    os.chdir(target_dir)
    
    # Get all MP3 files in the target directory
    mp3_files = glob.glob("*.mp3")
    
    if not mp3_files:
        print(f"No MP3 files found in '{target_dir}'.")
        os.chdir(original_dir)
        return
    
    print(f"Found {len(mp3_files)} MP3 file(s) to convert (first 30 seconds each):")
    for file in mp3_files:
        print(f"  - {file}")
    print()
    
    # Convert each MP3 file
    successful_conversions = 0
    total_files = len(mp3_files)
    
    for mp3_file in mp3_files:
        if convert_mp3_to_wav(mp3_file, duration_seconds=30):
            successful_conversions += 1
    
    print(f"\nConversion complete: {successful_conversions}/{total_files} files converted successfully.")
    
    # Change back to original directory
    os.chdir(original_dir)
    
    if successful_conversions < total_files:
        print("Some files failed to convert. Check the error messages above.")
        sys.exit(1)
    else:
        print("All files converted successfully!")

if __name__ == "__main__":
    main()