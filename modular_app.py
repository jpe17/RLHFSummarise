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
    print("✅ AI pipeline initialized successfully.")
    print("🎭 bes voice will automatically use Italian accent when selected")
    print("🔊 TTS model will load on-demand when first voice synthesis is requested")
except Exception as e:
    print(f"❌ FATAL: Could not initialize AI pipeline: {e}")
    pipeline = None
# ------------------------------------

processing_status = {}

@app.route('/')
def index():
    """Main page with social media UI."""
    return render_template('index_with_youtube.html')

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
            {'value': 'tiktok', 'label': 'TikTok (Coming Soon)'},
            {'value': 'youtube', 'label': 'YouTube'}
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
            'voices': voices,
            'message': '🎭 bes voice automatically uses Italian accent when selected'
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
                
                # Synthesize voice (bes will automatically use Italian accent)
                voice_output = pipeline.voice_synthesizer.synthesize(result.summary.content, voice_name)
                
                # Check if voice synthesis was successful
                if not voice_output or not voice_output.audio_path or not voice_output.audio_path.strip():
                    raise Exception("Voice synthesis failed - no audio file generated")
                
                # Use the voice output from synthesis
                result.voice_output = voice_output
                pipeline._update_progress(100, '✅ Processing complete!')

                # Emit audio when it's ready
                audio_data = {
                    'job_id': job_id,
                    'audio_path': voice_output.audio_path,
                    'voice_name': voice_name
                }
                socketio.emit('audio_ready', audio_data)

            except Exception as e:
                error_message = f'❌ Error synthesizing voice: {str(e)}'
                print(f"Voice synthesis error details: {e}")  # Log to console
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

@app.route('/api/process-youtube', methods=['POST'])
def process_youtube():
    """Process YouTube video and generate summary."""
    global pipeline, processing_status
    
    data = request.json
    youtube_url = data.get('youtube_url')
    voice_name = data.get('voice_name')
    
    if not pipeline:
        return jsonify({'success': False, 'message': 'Pipeline not initialized'})
        
    if not youtube_url:
        return jsonify({'success': False, 'message': 'YouTube URL is required'})

    job_id = f"youtube_{int(time.time())}"
    processing_status[job_id] = {'status': 'starting', 'progress': 0, 'message': 'Starting YouTube processing...'}

    def process_in_background():
        try:
            # Step 1: Process YouTube video
            pipeline._update_progress(10, '🎬 Downloading YouTube video...')
            
            # Use the YouTube processor directly
            from implementations.youtube_processor import YouTubeProcessor
            from core.data_models import SocialPost, Platform
            from datetime import datetime
            
            youtube_processor = YouTubeProcessor()
            
            # Create a SocialPost object for the YouTube URL
            youtube_post = SocialPost(
                id=f"youtube_{int(time.time())}",
                platform=Platform.YOUTUBE,
                username="YouTube Video",
                content=youtube_url,
                timestamp=datetime.now()
            )
            
            # Process the video with progress callback and streaming support
            def progress_callback(progress, message):
                pipeline._update_progress(progress, message)
                # Emit streaming updates for summary generation
                if "Generating summary" in message:
                    socketio.emit('summary_stream', {
                        'job_id': job_id, 
                        'progress': progress, 
                        'message': message
                    })
            
            processed_content = youtube_processor.process(youtube_post, progress_callback=progress_callback)
            
            # Check if this was a cached result and show appropriate progress
            is_cached = processed_content.original_posts[0].platform_data.get('video_metadata', {}).get('cached', False)
            
            if is_cached:
                pipeline._update_progress(80, '📝 Loading cached summary...')
            else:
                pipeline._update_progress(80, '📝 Preparing summary...')
            
            # Create result object
            from core.data_models import PipelineResult, Summary, VoiceOutput
            from datetime import datetime
            
            # Create a proper Summary object from the processed content
            summary = Summary(
                content=processed_content.processed_text,  # The summary is in processed_text
                score=0.8,  # Default score for YouTube summaries
                original_content=processed_content,
                model_name="LoRA YouTube Summarizer"
            )
            
            result = PipelineResult(
                platform=Platform.YOUTUBE,
                username="YouTube Video",
                posts=[],  # No posts for YouTube
                selection_type="video",
                processed_content=processed_content,
                summary=summary,  # Now using the proper Summary object
                voice_output=None,
                total_duration=0  # We don't track processing time yet
            )
            
            # Emit summary as soon as it's ready
            summary_data = result.to_dict()
            print(f"🔍 Debug: Emitting YouTube summary data with keys: {list(summary_data.keys())}")
            print(f"🔍 Debug: Summary content: {summary_data.get('summary', {}).get('content', 'NO SUMMARY')[:100]}...")
            socketio.emit('summary_ready', {'job_id': job_id, 'result': summary_data})

            # Step 2: Synthesize audio if voice is selected
            if voice_name:
                try:
                    pipeline._update_progress(90, f'🔊 Synthesizing audio for {voice_name}...')
                    
                    # Synthesize voice
                    voice_output = pipeline.voice_synthesizer.synthesize(result.summary.content, voice_name)
                    
                    # Check if voice synthesis was successful
                    if not voice_output or not voice_output.audio_path or not voice_output.audio_path.strip():
                        raise Exception("Voice synthesis failed - no audio file generated")
                    
                    # Use the voice output from synthesis
                    result.voice_output = voice_output
                    pipeline._update_progress(100, '✅ Processing complete!')

                    # Emit audio when it's ready
                    audio_data = {
                        'job_id': job_id,
                        'audio_path': voice_output.audio_path,
                        'voice_name': voice_name
                    }
                    socketio.emit('audio_ready', audio_data)

                except Exception as e:
                    error_message = f'❌ Error synthesizing voice: {str(e)}'
                    print(f"YouTube voice synthesis error details: {e}")
                    pipeline._update_progress(100, error_message)
                    socketio.emit('processing_error', {'job_id': job_id, 'message': error_message})
                    return
            else:
                pipeline._update_progress(100, '✅ Processing complete!')

            # Final completion update
            processing_status[job_id] = {'status': 'complete', 'result': result.to_dict()}
            socketio.emit('processing_complete', {'job_id': job_id, 'status': 'complete'})

        except Exception as e:
            error_message = f'YouTube processing error: {str(e)}'
            print(f"YouTube processing error details: {e}")
            import traceback
            traceback.print_exc()
            processing_status[job_id] = {'status': 'error', 'message': error_message}
            socketio.emit('processing_complete', {'job_id': job_id, 'status': 'error', 'message': error_message})

    socketio.start_background_task(target=process_in_background)
    
    return jsonify({'success': True, 'job_id': job_id, 'message': 'YouTube processing started'})

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

@app.route('/api/youtube/summary/<job_id>')
def get_youtube_summary(job_id):
    """Get the summary for a YouTube processing job."""
    try:
        # Look for the most recent summary file
        summaries_dir = "summaries"
        if not os.path.exists(summaries_dir):
            return jsonify({'success': False, 'message': 'No summaries directory found'})
        
        # Get all summary files and sort by modification time (newest first)
        summary_files = []
        for filename in os.listdir(summaries_dir):
            if filename.startswith('summary_') and filename.endswith('.txt'):
                filepath = os.path.join(summaries_dir, filename)
                mod_time = os.path.getmtime(filepath)
                summary_files.append((filepath, mod_time))
        
        if not summary_files:
            return jsonify({'success': False, 'message': 'No summary files found'})
        
        # Get the most recent summary file
        summary_files.sort(key=lambda x: x[1], reverse=True)
        latest_summary_file = summary_files[0][0]
        
        # Read the summary file
        with open(latest_summary_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract the summary part (everything after "SUMMARY:")
        summary_start = content.find("SUMMARY:")
        if summary_start != -1:
            summary_content = content[summary_start:].replace("SUMMARY:\n", "").replace("-" * 40 + "\n", "").strip()
        else:
            summary_content = content
        
        return jsonify({
            'success': True,
            'summary': summary_content,
            'file_path': latest_summary_file,
            'job_id': job_id
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error reading summary: {str(e)}'
        })

@app.route('/api/youtube/cache/stats')
def get_youtube_cache_stats():
    """Get YouTube cache statistics."""
    try:
        from implementations.youtube_processor import youtube_cache
        stats = youtube_cache.get_cache_stats()
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting cache stats: {str(e)}'
        })

@app.route('/api/youtube/cache/clear')
def clear_youtube_cache():
    """Clear the YouTube cache."""
    try:
        from implementations.youtube_processor import youtube_cache
        youtube_cache.cache_data = {"videos": {}, "metadata": {"created": datetime.now().isoformat()}}
        youtube_cache._save_cache()
        
        return jsonify({
            'success': True,
            'message': 'YouTube cache cleared successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error clearing cache: {str(e)}'
        })

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5464)  # Different port to avoid conflicts 