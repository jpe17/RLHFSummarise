#!/usr/bin/env python3
"""
Run the Twitter-like Web UI for Tweet Summarization
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import flask
        import flask_socketio
        print("✅ Flask and Flask-SocketIO are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return False

def check_json_files():
    """Check if JSON tweet files exist."""
    json_dir = "pipeline_twitter/data/json_tweets"
    if not os.path.exists(json_dir):
        print(f"❌ JSON directory not found: {json_dir}")
        print("💡 Run the JSON converter first to create tweet files")
        print("💡 Run: python run_json_converter.py")
        return False
    
    json_files = [f for f in os.listdir(json_dir) if f.endswith('_tweets.json')]
    if not json_files:
        print(f"❌ No JSON tweet files found in {json_dir}")
        print("💡 Run the JSON converter first to create tweet files")
        print("💡 Run: python run_json_converter.py")
        return False
    
    print(f"✅ Found {len(json_files)} JSON tweet files")
    return True

def main():
    """Main function to run the web application."""
    print("🚀 Starting Tweet Summarizer Web UI")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check JSON files
    if not check_json_files():
        sys.exit(1)
    
    print("\n🌐 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:5463")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Run the Flask app
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n⚠️ Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running web app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
