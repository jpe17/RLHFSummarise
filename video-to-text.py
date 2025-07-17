import yt_dlp
from moviepy.video.io.VideoFileClip import AudioFileClip
import sys
import os
import argparse
import json
from datetime import datetime
from openai import OpenAI
from typing import Dict, List, Tuple, Any
from path_utils import get_subdir
from faster_whisper import WhisperModel

# TODO - improve transcription error detection and correction

def get_openai_client():
    """Initialize OpenAI client with API key from environment variable."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("\nError: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key using one of these methods:")
        print("1. Export it in your terminal:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("2. Or add it to your ~/.zshrc or ~/.bashrc file:")
        print("   echo 'export OPENAI_API_KEY=\"your-api-key-here\"' >> ~/.zshrc")
        print("\nYou can get an API key from: https://platform.openai.com/api-keys")
        sys.exit(1)
    return OpenAI(api_key=api_key)

# Initialize OpenAI client
client = get_openai_client()

def detect_transcription_errors(transcribed_text: str) -> List[Dict]:
    """
    Detect potential errors in the transcribed text using OpenAI's API.
    
    Args:
        transcribed_text (str): The transcribed text to analyze
        
    Returns:
        List[Dict]: List of detected errors with their locations and descriptions
    """
    system_prompt = """You are an expert at detecting transcription errors in text. 
    Your task is to analyze the given transcribed text and identify potential errors such as:
    1. Misheard words or phrases
    2. Incorrect homophones
    3. Missing punctuation
    4. Grammatical inconsistencies
    5. Context-inappropriate words
    
    For each error found, provide:
    1. The error text
    2. The likely correct text
    3. A brief explanation of why it's probably an error
    4. Confidence level (high/medium/low)
    
    Format your response as a JSON array of objects, each containing:
    {
        "error_text": "the erroneous text",
        "likely_correction": "the probable correct text",
        "explanation": "why this is likely an error",
        "confidence": "high/medium/low"
    }
    
    If no errors are found, return an empty array.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this transcribed text for errors:\n\n{transcribed_text}"}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in error detection: {e}")
        return []

def extract_topics(text: str, max_topics: int = 10) -> List[Dict[str, any]]:
    """
    Extract and score the most relevant topics from the transcribed text.
    
    Args:
        text (str): The transcribed text to analyze
        max_topics (int): Maximum number of topics to extract (default: 10)
        
    Returns:
        List[Dict]: List of topics with their relevance scores and descriptions
    """
    system_prompt = """You are an expert at analyzing text and extracting key topics.
    Your task is to identify the most relevant topics from the given text, considering:
    1. Frequency of mention
    2. Centrality to the main discussion
    3. Specificity and uniqueness
    4. Contextual importance
    
    For each topic, provide:
    1. The topic name
    2. A relevance score (0-100)
    3. A brief description of why it's relevant
    4. Key terms or phrases associated with it
    
    Format your response as a JSON array of objects, each containing:
    {
        "topics": [
            {
                "topic": "the topic name",
                "relevance_score": number between 0-100,
                "description": "why this topic is relevant",
                "key_terms": ["term1", "term2", ...]
            }
        ]
    }
    
    Sort topics by relevance score in descending order.
    Limit the number of topics to the specified maximum.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract the top {max_topics} most relevant topics from this text:\n\n{text}"}
            ],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("topics", [])
    except Exception as e:
        print(f"Error in topic extraction: {e}")
        return []

def download_youtube_audio(youtube_url: str, output_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Download audio from a YouTube video.
    
    Args:
        youtube_url (str): The YouTube video URL
        output_path (str): Path to save the audio file
        
    Returns:
        Tuple[str, Dict[str, Any]]: Tuple containing (path to audio file, video metadata)
    """
    try:
        # Ensure output_path doesn't have .mp3 extension (yt-dlp will add it)
        if output_path.endswith('.mp3'):
            output_path = output_path[:-4]
            
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
            'keepvideo': False,  # Don't keep the original video file
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            print(f"Downloading: {info['title']}")
            
            # Extract metadata
            metadata = {
                'title': info.get('title'),
                'upload_date': info.get('upload_date'),  # YYYYMMDD format
                'timestamp': info.get('timestamp'),      # Unix timestamp
                'uploader': info.get('uploader')
            }
            
            # Return both the file path and metadata
            return f"{output_path}.mp3", metadata
    except Exception as e:
        print(f"Error downloading YouTube audio: {str(e)}")
        raise Exception(f"Failed to download YouTube audio: {str(e)}")

def transcribe_audio(audio_path: str, model_size: str = "base", use_gpu: bool = False) -> Dict[str, Any]:
    """
    Transcribe audio from a file using faster-whisper.
    
    Args:
        audio_path (str): Path to the audio file
        model_size (str): Size of the Whisper model to use ("tiny", "base", "small", "medium", "large")
        use_gpu (bool): Whether to use GPU for transcription
        
    Returns:
        Dict[str, Any]: Dictionary containing transcription and segments
    """
    try:
        # Ensure the audio file exists
        if not os.path.exists(audio_path):
            raise Exception(f"Audio file not found: {audio_path}")
            
        # Load the Whisper model with specified size
        print(f"Loading Whisper {model_size} model...")
        device = "cuda" if use_gpu else "cpu"
        model = WhisperModel(model_size, device=device)
        
        # Transcribe the audio
        print("Transcribing audio...")
        segments, info = model.transcribe(audio_path)
        
        # Process segments to match the expected format
        transcription_segments = []
        full_text = ""
        for segment in segments:
            transcription_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "avg_logprob": segment.avg_logprob
            })
            full_text += segment.text + " "
        
        return {
            "transcription": full_text.strip(),
            "segments": transcription_segments
        }
    except Exception as e:
        print(f"Error transcribing audio: {str(e)}")
        raise Exception(f"Failed to transcribe audio: {str(e)}")

def correct_transcription(transcribed_text: str, errors: List[Dict]) -> str:
    """
    Correct the detected errors in the transcribed text using OpenAI's API.
    
    Args:
        transcribed_text (str): The original transcribed text
        errors (List[Dict]): List of detected errors from detect_transcription_errors
        
    Returns:
        str: The corrected transcription
    """
    system_prompt = """You are an expert at correcting transcription errors in text.
    Your task is to take the original transcribed text and a list of detected errors,
    and produce a corrected version that:
    1. Fixes all identified errors
    2. Maintains the original meaning and context
    3. Sounds natural and fluent
    4. Preserves any correct parts of the original text
    
    Additionally:
    - Only make corrections that have high or medium confidence
    - Preserve the original text's style and tone
    - Maintain proper formatting and punctuation
    
    Return only the corrected text without any explanations or metadata.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""Original text:\n{transcribed_text}\n\nDetected errors:\n{json.dumps(errors, indent=2)}"""}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in correction: {e}")
        return transcribed_text

def save_to_json(transcription: str, topics: List[Dict], video_url: str, metadata: Dict[str, Any] = None, output_dir: str = None) -> str:
    """
    Save the transcription and topics to a JSON file.
    
    Args:
        transcription (str): The transcribed text
        topics (List[Dict]): List of topics with their relevance scores
        video_url (str): The YouTube video URL
        metadata (Dict[str, Any]): Video metadata from YouTube
        output_dir (str): Directory to save the JSON file (default: "transcriptions")
        
    Returns:
        str: Path to the saved JSON file
    """
    if output_dir is None:
        output_dir = get_subdir("transcriptions")
    else:
        output_dir = get_subdir(output_dir)
    
    # Generate filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"transcription_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Prepare the data structure
    metadata_dict = {
        "timestamp": timestamp,
        "video_url": video_url,
        "source": "YouTube",
        "processing_date": datetime.now().isoformat()
    }
    
    # Add video metadata if available
    if metadata:
        metadata_dict.update(metadata)
    
    data = {
        "metadata": metadata_dict,
        "content": {
            "transcription": transcription,
            "topics": topics
        }
    }
    
    # Write to JSON file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nTranscription saved to: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        return None

def process_video_audio(input_source: str, metadata: Dict[str, Any] = None, model_size: str = "base", use_gpu: bool = False) -> Dict[str, Any]:
    """
    Process video or audio content and return the processed data.
    
    Args:
        input_source (str): Path to the video/audio file or YouTube URL
        metadata (Dict[str, Any], optional): Additional metadata to include
        model_size (str): Size of the Whisper model to use ("tiny", "base", "small", "medium", "large")
        use_gpu (bool): Whether to use GPU for transcription
        
    Returns:
        Dict[str, Any]: Dictionary containing processed content and analysis info
    """
    try:
        # Create temp directory for audio files
        temp_dir = get_subdir("temp")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_audio = os.path.join(temp_dir, f"temp_audio_{timestamp}")
        
        # Download audio if it's a YouTube URL
        if input_source.startswith(('http://', 'https://')):
            print("Downloading audio from YouTube...")
            audio_file, video_metadata = download_youtube_audio(input_source, temp_audio)
            if metadata is None:
                metadata = video_metadata
            else:
                metadata.update(video_metadata)
        else:
            audio_file = input_source
            if metadata is None:
                metadata = {}
        
        # Step 1: Transcribe audio
        print("\nTranscribing audio...")
        transcription_data = transcribe_audio(audio_file, model_size=model_size, use_gpu=use_gpu)
        print("\nInitial transcription:")
        print(transcription_data["transcription"])
        
        # TODO - leaving it out for now as it leads to passing the entire description the second time to the LLM.
        # for the sake of cost savings
        # Step 2: Detect and correct errors
        # print("\nDetecting potential errors...")
        # errors = detect_transcription_errors(transcription_data["transcription"])
        errors = None
        
        if errors:
            print("\nDetected errors:")
            print(json.dumps(errors, indent=2))
            
            print("\nApplying corrections...")
            corrected_text = correct_transcription(transcription_data["transcription"], errors)
            print("\nCorrected transcription:")
            print(corrected_text)
        else:
            print("\nNo errors detected in the transcription.")
            corrected_text = transcription_data["transcription"]
        
        # Step 3: Extract topics
        print("\nExtracting main topics...")
        topics = extract_topics(corrected_text)
        if topics:
            print("\nMain topics identified:")
            for topic in topics:
                print(f"\nTopic: {topic['topic']}")
                print(f"Relevance Score: {topic['relevance_score']}")
                print(f"Description: {topic['description']}")
                print(f"Key Terms: {', '.join(topic['key_terms'])}")
        
        # Step 4: Save transcription
        transcription_file = save_to_json(
            corrected_text,
            topics,
            input_source,
            metadata
        )
        print(f"\n✅ Transcription saved to: {transcription_file}")
        
        # Step 5: Save audio file permanently if it's a YouTube URL
        if input_source.startswith(('http://', 'https://')):
            try:
                audio_dir = get_subdir("audio_content")
                permanent_audio = os.path.join(audio_dir, f"audio_{timestamp}.mp3")
                import shutil
                shutil.copy2(audio_file, permanent_audio)
                print(f"✅ Audio file saved to: {permanent_audio}")
                metadata["audio_file"] = permanent_audio
            except Exception as e:
                print(f"Warning: Could not save audio file permanently: {str(e)}")
        
        # Step 6: Return processed data
        return {
            "content": {
                "transcription": corrected_text,
                "topics": topics,
                "errors": errors
            },
            "analysis_info": {
                "source_file": input_source,
                "content_type": "video" if input_source.startswith(('http://', 'https://')) else "audio",
                "processed_at": datetime.now().isoformat(),
                "transcription_file": transcription_file,
                **metadata
            }
        }
        
    except Exception as e:
        raise Exception(f"Error processing video/audio content: {str(e)}")

def main():
    """Main function for direct script execution."""
    parser = argparse.ArgumentParser(description='Transcribe YouTube video audio to text')
    parser.add_argument('-u', '--url', required=True, help='YouTube video URL (use quotes around the URL)')
    args = parser.parse_args()

    try:
        # Process the video
        result = process_video_audio(args.url)
        
        # Save the result to processed_content directory
        output_dir = get_subdir("processed_content")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f"content_{timestamp}.json")
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✅ Processed content saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

    # python crypto_insights/video_to_text/video_to_text.py --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ"