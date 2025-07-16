#!/usr/bin/env python3
"""
MP3 to WAV converter script
Converts all MP3 files in the current directory to WAV format
"""

import os
import glob
from pydub import AudioSegment
import sys

def convert_mp3_to_wav(input_file, output_file=None):
    """
    Convert an MP3 file to WAV format
    
    Args:
        input_file (str): Path to the input MP3 file
        output_file (str): Path to the output WAV file (optional)
    """
    try:
        # Load the MP3 file
        audio = AudioSegment.from_mp3(input_file)
        
        # Generate output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}.wav"
        
        # Export as WAV
        audio.export(output_file, format="wav")
        print(f"✓ Converted: {input_file} -> {output_file}")
        return True
        
    except Exception as e:
        print(f"✗ Error converting {input_file}: {str(e)}")
        return False

def main():
    """Main function to convert all MP3 files in the current directory"""
    
    # Get all MP3 files in the current directory
    mp3_files = glob.glob("*.mp3")
    
    if not mp3_files:
        print("No MP3 files found in the current directory.")
        return
    
    print(f"Found {len(mp3_files)} MP3 file(s) to convert:")
    for file in mp3_files:
        print(f"  - {file}")
    print()
    
    # Convert each MP3 file
    successful_conversions = 0
    total_files = len(mp3_files)
    
    for mp3_file in mp3_files:
        if convert_mp3_to_wav(mp3_file):
            successful_conversions += 1
    
    print(f"\nConversion complete: {successful_conversions}/{total_files} files converted successfully.")
    
    if successful_conversions < total_files:
        print("Some files failed to convert. Check the error messages above.")
        sys.exit(1)
    else:
        print("All files converted successfully!")

if __name__ == "__main__":
    main()