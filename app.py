#!/usr/bin/env python3
"""
Twitter-like Web UI for Tweet Summarization and Voice Synthesis
"""

import os
import json
import time
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from pipeline_twitter.integrated_db_pipeline import IntegratedJSONPipeline

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global pipeline instance
pipeline = None
processing_status = {}

@app.route('/')
def index():
    """Main page with Twitter-like UI."""
    return render_template('index.html')

@app.route('/api/initialize', methods=['POST'])
def initialize_pipeline():
    """Initialize the pipeline."""
    global pipeline
    
    try:
        # Use the correct path to the JSON files
        json_dir = os.path.join(project_root, "pipeline_twitter", "data", "json_tweets")
        pipeline = IntegratedJSONPipeline(json_dir=json_dir)
        success = pipeline.initialize_pipelines()
        
        if success:
            return jsonify({
                'success': True,
                'message': '✅ Pipelines initialized successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': '❌ Failed to initialize pipelines'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'❌ Error: {str(e)}'
        })

@app.route('/api/users')
def get_users():
    """Get available users from JSON files."""
    try:
        if pipeline is None:
            return jsonify({'success': False, 'message': 'Pipeline not initialized'})
        
        users = pipeline.selector.get_available_users()
        return jsonify({
            'success': True,
            'users': users
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting users: {str(e)}'
        })

@app.route('/api/voices')
def get_voices():
    """Get available voices."""
    voices = [
        "christina", "elonmusk", "barackobama", "freeman", 
        "angie", "daniel", "emma", "halle", "jlaw", "weaver"
    ]
    return jsonify({
        'success': True,
        'voices': voices
    })

@app.route('/api/preview-tweets', methods=['POST'])
def preview_tweets():
    """Preview tweets for a user with selected filter."""
    global pipeline
    
    data = request.json
    username = data.get('username')
    selection_type = data.get('selection_type', 'top')
    count = data.get('count', 5)
    
    if not pipeline:
        return jsonify({
            'success': False,
            'message': 'Pipeline not initialized'
        })
    
    if not username:
        return jsonify({
            'success': False,
            'message': 'Username is required'
        })
    
    try:
        tweets = pipeline.select_tweets(username, selection_type, count)
        
        if not tweets:
            return jsonify({
                'success': False,
                'message': f'No tweets found for @{username}'
            })
        
        return jsonify({
            'success': True,
            'tweets': tweets,
            'count': len(tweets),
            'selection_type': selection_type
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error previewing tweets: {str(e)}'
        })

@app.route('/api/process', methods=['POST'])
def process_user():
    """Process a user with selected parameters."""
    global pipeline, processing_status
    
    data = request.json
    username = data.get('username')
    voice_name = data.get('voice_name')
    selection_type = data.get('selection_type', 'top')
    count = data.get('count', 5)
    
    if not pipeline:
        return jsonify({
            'success': False,
            'message': 'Pipeline not initialized'
        })
    
    # Create a unique job ID
    job_id = f"{username}_{voice_name}_{int(time.time())}"
    processing_status[job_id] = {
        'status': 'processing',
        'progress': 0,
        'message': 'Starting processing...'
    }
    
    # Start processing in background thread
    def process_in_background():
        try:
            processing_status[job_id]['progress'] = 10
            processing_status[job_id]['message'] = 'Selecting tweets...'
            socketio.emit('progress_update', {
                'job_id': job_id,
                'progress': 10,
                'message': 'Selecting tweets...'
            })
            
            # Select tweets
            tweets = pipeline.select_tweets(username, selection_type, count)
            
            if not tweets:
                processing_status[job_id] = {
                    'status': 'error',
                    'message': f'No tweets found for @{username}'
                }
                socketio.emit('processing_complete', {
                    'job_id': job_id,
                    'status': 'error',
                    'message': f'No tweets found for @{username}'
                })
                return
            
            processing_status[job_id]['progress'] = 30
            processing_status[job_id]['message'] = 'Generating summary...'
            socketio.emit('progress_update', {
                'job_id': job_id,
                'progress': 30,
                'message': 'Generating summary...'
            })
            
            # Generate summary
            combined_text = "\n\n".join([tweet['content'] for tweet in tweets])
            summary = pipeline.summarizer_pipeline.generate_summary(combined_text, max_length=200)
            
            if not summary:
                processing_status[job_id] = {
                    'status': 'error',
                    'message': 'Failed to generate summary'
                }
                socketio.emit('processing_complete', {
                    'job_id': job_id,
                    'status': 'error',
                    'message': 'Failed to generate summary'
                })
                return
            
            processing_status[job_id]['progress'] = 60
            processing_status[job_id]['message'] = 'Generating voice...'
            socketio.emit('progress_update', {
                'job_id': job_id,
                'progress': 60,
                'message': 'Generating voice...'
            })
            
            # Generate voice with progress updates
            audio_path = pipeline.voice_pipeline.synthesize_voice(summary, voice_name)
            
            # Update progress during voice generation
            processing_status[job_id]['progress'] = 80
            processing_status[job_id]['message'] = 'Finalizing audio...'
            socketio.emit('progress_update', {
                'job_id': job_id,
                'progress': 80,
                'message': 'Finalizing audio...'
            })
            
            if not audio_path:
                processing_status[job_id] = {
                    'status': 'error',
                    'message': 'Failed to generate voice'
                }
                socketio.emit('processing_complete', {
                    'job_id': job_id,
                    'status': 'error',
                    'message': 'Failed to generate voice'
                })
                return
            
            processing_status[job_id]['progress'] = 100
            processing_status[job_id]['message'] = 'Complete!'
            processing_status[job_id]['status'] = 'complete'
            
            # Score the summary
            score = pipeline.summarizer_pipeline.score_summary(combined_text, summary)
            
            result = {
                'username': username,
                'voice_name': voice_name,
                'selection_type': selection_type,
                'tweet_count': len(tweets),
                'summary': summary,
                'score': score,
                'audio_path': audio_path,
                'tweets': tweets
            }
            
            processing_status[job_id]['result'] = result
            
            socketio.emit('processing_complete', {
                'job_id': job_id,
                'status': 'complete',
                'result': result
            })
            
        except Exception as e:
            processing_status[job_id] = {
                'status': 'error',
                'message': f'Processing error: {str(e)}'
            }
            socketio.emit('processing_complete', {
                'job_id': job_id,
                'status': 'error',
                'message': f'Processing error: {str(e)}'
            })
    
    # Start background thread
    thread = threading.Thread(target=process_in_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'job_id': job_id,
        'message': 'Processing started'
    })

@app.route('/api/audio/<path:filename>')
def serve_audio(filename):
    """Serve audio files."""
    return send_file(filename, mimetype='audio/wav')

@app.route('/api/status/<job_id>')
def get_status(job_id):
    """Get processing status for a job."""
    status = processing_status.get(job_id, {
        'status': 'not_found',
        'message': 'Job not found'
    })
    return jsonify(status)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5463) 