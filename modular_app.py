#!/usr/bin/env python3
"""
Modular Pipeline Web Application

This is the new Flask app that uses the modular pipeline architecture.
"""

import os
import json
import time
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit

# Set MPS fallback for TTS compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from pipeline_factory import PipelineFactory, create_preloaded_pipeline
from core.data_models import Platform

# Import YouTube processing functionality
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("video_to_text_lora", "video-to-text-lora.py")
    if spec and spec.loader:
        video_to_text_lora = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(video_to_text_lora)
        LoRASummarizer = video_to_text_lora.LoRASummarizer
        process_video_audio = video_to_text_lora.process_video_audio
    else:
        LoRASummarizer = None
        process_video_audio = None
except Exception as e:
    print(f"Warning: Could not import video-to-text-lora: {e}")
    LoRASummarizer = None
    process_video_audio = None

# Fix PyTorch 2.6+ compatibility with TTS
try:
    import fix_torch_compatibility
except ImportError:
    print("⚠️ Could not import torch compatibility fix")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Global Pipeline Initialization ---
# Initialize the pipeline here, in the main thread, before the app runs.
# This ensures all models, including the problematic TTS model, are loaded once
# in the correct context.

print("🚀 Initializing AI pipeline in main thread...")
try:
    pipeline = create_preloaded_pipeline()
    if hasattr(pipeline.voice_synthesizer, '_get_tts'):
        print("🔊 Pre-loading TTS model in main thread...")
        pipeline.voice_synthesizer._get_tts()
    print("✅ AI pipeline initialized successfully.")
except Exception as e:
    print(f"❌ FATAL: Could not initialize AI pipeline: {e}")
    pipeline = None

# Initialize LoRA summarizer for YouTube processing
if LoRASummarizer is not None:
    print("🎥 Initializing LoRA summarizer for YouTube processing...")
    try:
        lora_summarizer = LoRASummarizer()
        print("✅ LoRA summarizer initialized successfully.")
    except Exception as e:
        print(f"❌ FATAL: Could not initialize LoRA summarizer: {e}")
        lora_summarizer = None
else:
    print("⚠️ LoRA summarizer not available - YouTube processing will be disabled")
    lora_summarizer = None
# ------------------------------------

processing_status = {}

@app.route('/')
def index():
    """Main page with social media UI."""
    return render_template('index.html')

@app.route('/api/initialize', methods=['POST'])
def initialize_pipeline():
    """Check pipeline initialization status."""
    global pipeline
    
    if pipeline:
        return jsonify({
            'success': True,
            'message': '✅ Modular pipeline is initialized and ready.',
            'config': pipeline.validate_configuration()
        })
    else:
        return jsonify({
            'success': False,
            'message': '❌ Pipeline failed to initialize. Check server logs.'
        })

@app.route('/api/platforms')
def get_platforms():
    """Get available platforms."""
    return jsonify({
        'success': True,
        'platforms': [
            {'value': 'twitter', 'label': 'Twitter'},
            {'value': 'instagram', 'label': 'Instagram'},
            {'value': 'tiktok', 'label': 'TikTok (Coming Soon)'}
        ]
    })

@app.route('/api/users')
def get_users_default():
    """Get available users for default platform (Twitter)."""
    return get_users('twitter')

@app.route('/api/users/<platform>')
def get_users(platform):
    """Get available users for a platform."""
    try:
        if pipeline is None:
            return jsonify({'success': False, 'message': 'Pipeline not initialized'})
        
        platform_enum = Platform(platform)
        users = pipeline.get_available_users(platform_enum)
        
        return jsonify({
            'success': True,
            'users': users,
            'platform': platform
        })
        
    except ValueError:
        return jsonify({
            'success': False,
            'message': f'Invalid platform: {platform}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting users: {str(e)}'
        })

@app.route('/api/voices')
def get_voices():
    """Get available voices."""
    try:
        if pipeline is None:
            return jsonify({'success': False, 'message': 'Pipeline not initialized'})
        
        voices = pipeline.get_available_voices()
        
        return jsonify({
            'success': True,
            'voices': voices
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting voices: {str(e)}'
        })

@app.route('/api/preview-posts', methods=['POST'])
def preview_posts():
    """Preview posts for Twitter and Instagram handles."""
    global pipeline
    
    data = request.json
    twitter_handle = data.get('twitter_handle')
    instagram_handle = data.get('instagram_handle')
    count = data.get('count', 10)
    selection_type = data.get('selection_type', 'latest')
    
    if not pipeline:
        return jsonify({'success': False, 'message': 'Pipeline not initialized'})
    
    try:
        all_posts = []
        
        # Fetch Twitter posts
        if twitter_handle:
            twitter_posts = pipeline.load_posts(Platform.TWITTER, twitter_handle, limit=count)
            all_posts.extend(twitter_posts)
            
        # Fetch Instagram posts
        if instagram_handle:
            insta_posts = pipeline.load_posts(Platform.INSTAGRAM, instagram_handle, limit=count)
            all_posts.extend(insta_posts)

        if not all_posts:
            return jsonify({'success': False, 'message': 'No posts found for the given handles.'})

        # The 'select_posts' might be redundant if we already limited by count, but good for sorting
        selected_posts = pipeline.select_posts(all_posts, selection_type, count * 2) # Ensure we have enough for both feeds
        
        posts_data = [post.to_dict() for post in selected_posts]
        
        return jsonify({'success': True, 'posts': posts_data})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error previewing posts: {str(e)}'})

@app.route('/api/process', methods=['POST'])
def process_user():
    """
    Process multiple users from different platforms.
    
    Expected JSON payload:
    {
        "users": [{"platform": "twitter", "username": "sama"}, {"platform": "instagram", "username": "BarackObama"}],
        "voice_name": "deniro",
        "count": 3,  # Number of posts PER USER (not total)
        "selection_type": "latest"  # "latest", "top", "diverse", or "random"
    }
    """
    global pipeline, processing_status
    
    data = request.json
    users = data.get('users', [])
    voice_name = data.get('voice_name')
    count = data.get('count', 5)
    selection_type = data.get('selection_type', 'latest')
    
    if not pipeline:
        return jsonify({'success': False, 'message': 'Pipeline not initialized'})
        
    if not users:
        return jsonify({'success': False, 'message': 'No users specified for processing'})

    job_id = f"job_{int(time.time())}"
    processing_status[job_id] = {'status': 'starting', 'progress': 0, 'message': 'Starting...'}

    def process_in_background():
        try:
            # Step 1: Run pipeline to get summary
            result = pipeline.run_pipeline_without_tts(
                users=users,
                selection_type=selection_type,
                count=count
            )
            
            # Emit summary as soon as it's ready
            summary_data = result.to_dict()
            socketio.emit('summary_ready', {'job_id': job_id, 'result': summary_data})

            # Step 2: Synthesize audio
            try:
                pipeline.set_progress_callback(lambda p, m: socketio.emit('progress_update', {'job_id': job_id, 'progress': p, 'message': m}))
                pipeline._update_progress(90, f'🔊 Synthesizing audio for {voice_name}...')
                
                audio_path = pipeline.voice_synthesizer.synthesize(result.summary.content, voice_name)
                
                from core.data_models import VoiceOutput
                result.voice_output = VoiceOutput(
                    audio_path=audio_path,
                    voice_name=voice_name,
                    text=result.summary.content
                )
                pipeline._update_progress(100, '✅ Processing complete!')

                # Emit audio when it's ready
                audio_data = {
                    'job_id': job_id,
                    'audio_path': audio_path,
                    'voice_name': voice_name
                }
                socketio.emit('audio_ready', audio_data)

            except Exception as e:
                error_message = f'❌ Error synthesizing voice: {str(e)}'
                pipeline._update_progress(100, error_message)
                socketio.emit('processing_error', {'job_id': job_id, 'message': error_message})
                return

            # Final completion update
            processing_status[job_id] = {'status': 'complete', 'result': result.to_dict()}
            socketio.emit('processing_complete', {'job_id': job_id, 'status': 'complete'})

        except Exception as e:
            error_message = f'Processing error: {str(e)}'
            processing_status[job_id] = {'status': 'error', 'message': error_message}
            socketio.emit('processing_complete', {'job_id': job_id, 'status': 'error', 'message': error_message})

    socketio.start_background_task(target=process_in_background)
    
    return jsonify({'success': True, 'job_id': job_id, 'message': 'Processing started'})

@app.route('/api/crawl', methods=['POST'])
def crawl_user():
    """Crawl new posts for a user."""
    global pipeline
    
    data = request.json
    platform = data.get('platform', 'twitter')
    username = data.get('username')
    count = data.get('count', 10)
    
    if not pipeline:
        return jsonify({
            'success': False,
            'message': 'Pipeline not initialized'
        })
    
    try:
        platform_enum = Platform(platform)
        
        # Crawl new posts
        posts = pipeline.crawl_user(platform_enum, username, count)
        
        if not posts:
            return jsonify({
                'success': False,
                'message': f'No posts found for @{username} on {platform}'
            })
        
        # Store the crawled posts
        pipeline.storage.store_posts(posts)
        
        return jsonify({
            'success': True,
            'message': f'Successfully crawled {len(posts)} posts from @{username}',
            'post_count': len(posts),
            'platform': platform
        })
        
    except ValueError:
        return jsonify({
            'success': False,
            'message': f'Invalid platform: {platform}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error crawling posts: {str(e)}'
        })

@app.route('/api/audio/<path:filename>')
def serve_audio(filename):
    """Serve audio files."""
    return send_file(filename, mimetype='audio/wav')

@app.route('/data/posts/instagram/<username>/<filename>')
def serve_instagram_image(username, filename):
    """Serve Instagram images."""
    image_path = os.path.join('data', 'posts', 'instagram', username, filename)
    return send_file(image_path)

@app.route('/api/status/<job_id>')
def get_status(job_id):
    """Get processing status for a job."""
    status = processing_status.get(job_id, {
        'status': 'not_found',
        'message': 'Job not found'
    })
    return jsonify(status)

@app.route('/api/config')
def get_config():
    """Get current pipeline configuration."""
    try:
        if pipeline is None:
            return jsonify({'success': False, 'message': 'Pipeline not initialized'})
        
        validation = pipeline.validate_configuration()
        
        return jsonify({
            'success': True,
            'config': validation
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting config: {str(e)}'
        })

@app.route('/api/stats/<platform>/<username>')
def get_user_stats(platform, username):
    """Get statistics for a user on a platform."""
    try:
        if pipeline is None:
            return jsonify({'success': False, 'message': 'Pipeline not initialized'})
        
        platform_enum = Platform(platform)
        
        # This would require extending the storage interface
        # For now, just return basic info
        posts = pipeline.load_posts(platform_enum, username)
        
        if not posts:
            return jsonify({
                'success': False,
                'message': f'No data found for @{username} on {platform}'
            })
        
        total_engagement = sum(post.engagement_total for post in posts)
        avg_engagement = total_engagement / len(posts)
        
        stats = {
            'username': username,
            'platform': platform,
            'total_posts': len(posts),
            'total_engagement': total_engagement,
            'average_engagement': avg_engagement,
            'latest_post': posts[0].timestamp.isoformat() if posts else None,
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except ValueError:
        return jsonify({
            'success': False,
            'message': f'Invalid platform: {platform}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting stats: {str(e)}'
        })

@app.route('/api/youtube/process', methods=['POST'])
def process_youtube():
    """Process YouTube video and generate summary."""
    global lora_summarizer
    
    data = request.json
    youtube_url = data.get('youtube_url')
    voice_name = data.get('voice_name')
    whisper_model = data.get('whisper_model', 'base')
    use_gpu = data.get('use_gpu', False)
    
    if not lora_summarizer:
        return jsonify({'success': False, 'message': 'LoRA summarizer not initialized'})
    
    if not youtube_url:
        return jsonify({'success': False, 'message': 'YouTube URL is required'})
    
    job_id = f"youtube_job_{int(time.time())}"
    processing_status[job_id] = {'status': 'starting', 'progress': 0, 'message': 'Starting YouTube processing...'}
    
    def process_youtube_in_background():
        try:
            # Update progress
            socketio.emit('progress_update', {'job_id': job_id, 'progress': 10, 'message': '🎥 Downloading YouTube audio...'})
            
            # Process the YouTube video
            result = process_video_audio(
                input_source=youtube_url,
                model_size=whisper_model,
                use_gpu=use_gpu
            )
            
            socketio.emit('progress_update', {'job_id': job_id, 'progress': 80, 'message': '✅ YouTube processing complete!'})
            
            # Emit the result
            socketio.emit('youtube_summary_ready', {
                'job_id': job_id, 
                'result': result
            })
            
            # If voice synthesis is requested, generate audio
            if voice_name and pipeline and pipeline.voice_synthesizer:
                try:
                    socketio.emit('progress_update', {'job_id': job_id, 'progress': 85, 'message': f'🔊 Synthesizing audio for {voice_name}...'})
                    
                    summary_text = result['content']['summary']
                    audio_path = pipeline.voice_synthesizer.synthesize(summary_text, voice_name)
                    
                    socketio.emit('progress_update', {'job_id': job_id, 'progress': 100, 'message': '✅ Audio synthesis complete!'})
                    
                    # Emit audio when it's ready
                    audio_data = {
                        'job_id': job_id,
                        'audio_path': audio_path,
                        'voice_name': voice_name,
                        'text': summary_text
                    }
                    socketio.emit('youtube_audio_ready', audio_data)
                    
                except Exception as e:
                    error_message = f'❌ Error synthesizing voice: {str(e)}'
                    socketio.emit('processing_error', {'job_id': job_id, 'message': error_message})
            
            # Final completion update
            processing_status[job_id] = {'status': 'complete', 'result': result}
            socketio.emit('processing_complete', {'job_id': job_id, 'status': 'complete'})
            
        except Exception as e:
            error_message = f'YouTube processing error: {str(e)}'
            processing_status[job_id] = {'status': 'error', 'message': error_message}
            socketio.emit('processing_complete', {'job_id': job_id, 'status': 'error', 'message': error_message})
    
    socketio.start_background_task(target=process_youtube_in_background)
    
    return jsonify({'success': True, 'job_id': job_id, 'message': 'YouTube processing started'})

@app.route('/api/youtube/validate', methods=['POST'])
def validate_youtube_url():
    """Validate YouTube URL format."""
    data = request.json
    youtube_url = data.get('youtube_url', '')
    
    # Simple YouTube URL validation
    valid_patterns = [
        r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'https?://youtu\.be/[\w-]+',
        r'https?://(?:www\.)?youtube\.com/embed/[\w-]+'
    ]
    
    import re
    for pattern in valid_patterns:
        if re.match(pattern, youtube_url):
            return jsonify({'success': True, 'valid': True})
    
    return jsonify({'success': True, 'valid': False, 'message': 'Invalid YouTube URL format'})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5464)  # Different port to avoid conflicts 