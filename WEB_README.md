# 🐦 Tweet Summarizer Web UI

A Twitter-like web interface for AI-powered tweet summarization and voice synthesis.

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Web Application**
   ```bash
   python run_web_app.py
   ```

3. **Open Your Browser**
   Navigate to: http://localhost:5000

## 🎯 Features

- **Twitter-like UI**: Modern, responsive interface inspired by Twitter's design
- **Real-time Processing**: Live progress updates with WebSocket connections
- **Multiple Voices**: Choose from various AI-generated voices
- **Flexible Tweet Selection**: Top, latest, or random tweet filtering
- **Audio Playback**: Listen to generated summaries directly in the browser

## 📋 Prerequisites

Before running the web app, ensure you have:

1. **JSON Tweet Files**: Run the JSON converter first
   ```bash
   python run_json_converter.py
   ```

2. **AI Models**: The following model files should be present:
   - `simple_ppo_lora_final_20250716_130239.pt`
   - `qwen_reward_model.pt`

## 🎮 How to Use

1. **Initialize Pipeline**: Click "Initialize Pipeline" to load AI models
2. **Select User**: Choose from available Twitter users
3. **Choose Voice**: Pick an AI voice for synthesis
4. **Set Filter**: Select tweet filtering method (top/latest/random)
5. **Process**: Click "Process User" to generate summary and voice
6. **Listen**: Play the generated audio directly in the browser

## 🎨 UI Components

- **Initialization Section**: Load AI models
- **Form Section**: Configure processing parameters
- **Progress Section**: Real-time processing updates
- **Result Section**: Display summaries and audio players

## 🔧 Technical Details

- **Backend**: Flask with SocketIO for real-time updates
- **Frontend**: Vanilla JavaScript with Twitter-inspired CSS
- **Audio**: Served via Flask static file handling
- **Processing**: Background threads for non-blocking operations

## 🐛 Troubleshooting

### Common Issues

1. **"Pipeline not initialized"**
   - Click "Initialize Pipeline" first
   - Check that model files exist

2. **"No JSON tweet files found"**
   - Run `python run_json_converter.py` first
   - Ensure `json_tweets/` directory exists

3. **Audio not playing**
   - Check browser console for errors
   - Ensure audio file was generated successfully

4. **Processing fails**
   - Check terminal for error messages
   - Verify all dependencies are installed

### Debug Mode

Run with debug output:
```bash
python app.py
```

## 📁 File Structure

```
RLHFSummarise/
├── app.py                    # Flask web application
├── templates/
│   └── index.html           # Twitter-like UI template
├── run_web_app.py           # Web app launcher
├── integrated_db_pipeline.py # Core pipeline logic
├── json_tweets/             # Tweet data files
└── *.wav                    # Generated audio files
```

## 🎵 Available Voices

- christina
- elonmusk
- barackobama
- freeman
- angie
- daniel
- emma
- halle
- jlaw
- weaver

## 📊 Available Users

Users are loaded from JSON files in `json_tweets/` directory:
- elonmusk
- BarackObama
- AOC
- NASA
- dril
- horse_ebooks
- sama
- realDonaldTrump

## 🔄 Processing Flow

1. **Tweet Selection**: Load tweets from JSON files
2. **Summarization**: Generate AI summary using PPO model
3. **Voice Synthesis**: Convert summary to speech
4. **Audio Delivery**: Serve audio file to browser
5. **Playback**: User listens to generated summary

## 🚀 Performance Tips

- Initialize pipeline once at startup
- Process one user at a time for best performance
- Close browser tabs to free up memory
- Restart server if processing becomes slow

## 📝 Development

To modify the web interface:

1. Edit `templates/index.html` for UI changes
2. Modify `app.py` for backend logic
3. Update CSS in the HTML file for styling
4. Test changes by restarting the server

## 🎉 Enjoy!

The web interface provides a seamless way to explore AI-powered tweet summarization with voice synthesis. The Twitter-like design makes it familiar and easy to use! 