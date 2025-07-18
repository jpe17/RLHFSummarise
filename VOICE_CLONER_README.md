# Voice Cloner Script

A simple standalone script to clone voices using text input. This script uses your existing voice synthesis system to generate speech from text using any of the available voice models.

## Features

- 🎭 **33 Available Voices**: Choose from a wide variety of voice models
- 🎯 **Interactive Mode**: User-friendly interface for voice selection and text input
- 🤖 **Command Line Mode**: Batch processing with command line arguments
- 🎬 **Conversational Mode**: Interactive conversations with breaks for responses
- 🔊 **Auto-Play**: Automatically plays generated audio on supported systems
- 📁 **File Output**: Saves generated audio to the `outputs/` directory

## Available Voices

The script includes 33 different voices:

**Popular Voices:**
- `barackobama` - Barack Obama's voice
- `elonmusk` - Elon Musk's voice
- `sama` - Sam Altman's voice
- `deniro` - Robert De Niro's voice
- `freeman` - Morgan Freeman's voice
- `emma` - Emma Watson's voice
- `geralt` - Geralt of Rivia's voice
- `halle` - Halle Berry's voice
- `jlaw` - Jennifer Lawrence's voice
- `bes` - Bes voice (with pre-written conversation)

**And many more**: angie, applejack, daniel, lj, mol, myself, pat, pat2, rainbow, snakes, tim_reynolds, tom, weaver, william, plus various training voices.

## Usage

### Quick Start - Interactive Mode

```bash
python voice_cloner.py
```

This will:
1. Show you all available voices
2. Let you select a voice interactively
3. Ask for the text you want to synthesize
4. Generate the audio file

### 🎬 NEW: Conversational Mode

Have interactive conversations with breaks for your responses!

```bash
# Start conversation with Bes (includes pre-written course conversation)
python voice_cloner.py -v bes --conversation

# Start conversation with Barack Obama
python voice_cloner.py -v barackobama --conversation

# Start conversation with Elon Musk
python voice_cloner.py -v elonmusk --conversation

# Start conversation with Morgan Freeman
python voice_cloner.py -v freeman --conversation
```

**What happens in conversational mode:**
1. **Pre-generation phase**: All conversation audio is generated upfront for instant playback
2. **Conversation phase**: The AI speaks a line from the conversation
3. Audio is **instantly played** (no waiting for generation!)
4. **You get a pause to respond** (speak out loud, think, or type)
5. Press Enter to continue to the next line
6. Type 'quit' to exit the conversation

### Pre-written Conversations

**Bes Conversation** (Perfect for course interactions):
- "Hi Joao! How did you find this course?"
- "Well I appreciate that, it was fun to teach."
- "The voice cloning technology is quite impressive, isn't it?"
- "I hope you'll use this knowledge responsibly."
- "Any questions about the implementation?"
- "Great! Feel free to experiment with different voices."
- "Remember, practice makes perfect with AI technologies."
- "Thanks for being such an engaged student!"

**Barack Obama Conversation**:
- Inspiring talk about technology and society
- Questions about the future of AI
- Motivational closing remarks

**Elon Musk Conversation**:
- Discussion about voice cloning technology
- Mars colonization and AI topics
- Innovation and the future

**Morgan Freeman Conversation**:
- Philosophical discussion about AI
- Wisdom about technology and responsibility
- Poetic reflections on the future

### Command Line Mode

```bash
# Generate audio with specific voice and text
python voice_cloner.py --voice barackobama --text "Hello, this is a test of voice cloning technology."

# Short form
python voice_cloner.py -v elonmusk -t "We're going to Mars!"

# Start conversation mode
python voice_cloner.py -v bes -c
```

### List Available Voices

```bash
python voice_cloner.py --list-voices
```

### Advanced Usage

```bash
# Use custom voices directory
python voice_cloner.py --voices-dir /path/to/custom/voices

# Combine with other options
python voice_cloner.py --voices-dir custom_voices --voice custom_voice --conversation
```

## Command Line Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--text` | `-t` | Text to synthesize (optional - will prompt if not provided) |
| `--voice` | `-v` | Voice name to use (optional - will show selection menu if not provided) |
| `--conversation` | `-c` | **NEW!** Start conversational mode with breaks for responses |
| `--list-voices` | `-l` | List available voices and exit |
| `--voices-dir` | | Directory containing voice models (default: `voices`) |

## Examples

### Example 1: Regular Voice Synthesis
```bash
python voice_cloner.py -v barackobama -t "My fellow Americans, today we're testing voice cloning technology."
```

### Example 2: Interactive Conversation with Bes
```bash
python voice_cloner.py -v bes --conversation
```

**What you'll see:**
```
🎭 Starting conversation with bes...
💡 Tips:
   - Press Enter to continue to the next line
   - Type 'quit' to exit the conversation
   - Each line will be spoken with a pause for your response

🚀 Pre-generating all audio for instant playback...
   This will take a moment but ensures smooth conversation flow!
🔊 Pre-generating audio 1/8: Hi Joao! How did you find this course?
✅ Audio 1 ready: synthesis_bes_20241217_143022.wav
🔊 Pre-generating audio 2/8: Well I appreciate that, it was fun to teach.
✅ Audio 2 ready: synthesis_bes_20241217_143045.wav
...
🎉 All audio pre-generated! Starting conversation...

🎭 bes (Line 1/8): Hi Joao! How did you find this course?
🎵 Playing pre-generated audio...

💬 Your response to bes: [You respond here, then press Enter]
```

### Example 3: Conversation with Elon Musk
```bash
python voice_cloner.py -v elonmusk -c
```

### Example 4: Custom Conversation
```bash
# For voices without pre-written conversations
python voice_cloner.py -v daniel --conversation
```

## Output

Generated audio files are saved in the `outputs/` directory with the following naming pattern:
```
synthesis_{voice_name}_{timestamp}.wav
```

Example: `synthesis_bes_20241217_143022.wav`

## Requirements

Make sure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```

The script uses the existing TTS system which requires:
- TTS library (XTTS v2 model)
- PyTorch
- Other dependencies from `requirements.txt`

## Features & Optimizations

- **Smart Text Processing**: Automatically cleans and optimizes text for better speech synthesis
- **Intelligent Chunking**: Splits long text into manageable chunks for processing
- **Conversational Breaks**: Pauses between lines for natural conversation flow
- **Pre-generated Audio**: All conversation audio is generated upfront for instant playback
- **Pre-written Scripts**: Engaging conversations for popular voices
- **Custom Conversations**: Create your own conversation scripts on the fly
- **Caching**: Caches generated audio to avoid regenerating the same content
- **Speed Optimizations**: Uses optimized settings for faster generation
- **Error Handling**: Graceful error handling with helpful error messages
- **Cross-Platform Audio**: Automatically plays generated audio on macOS, Windows, and Linux

## Conversational Mode Tips

- 🎯 **Respond naturally**: Take your time to respond out loud as if having a real conversation
- 🔄 **Practice conversations**: Use the same conversation multiple times to practice
- 🎭 **Try different voices**: Each voice has its own personality and conversation style
- 📝 **Create custom scripts**: For voices without pre-written conversations, you can create your own
- 🚪 **Easy exit**: Type 'quit' at any pause to exit the conversation
- 📁 **Audio library**: All generated audio is saved, so you can replay conversations later

## Troubleshooting

### Common Issues

1. **No voices found**: Make sure the `voices/` directory exists and contains voice folders with audio files
2. **TTS model loading error**: Ensure PyTorch is properly installed and compatible with your system
3. **Audio playback issues**: Audio file is still saved even if auto-play fails
4. **Conversation interruption**: Use Ctrl+C to gracefully exit at any time

### Performance Tips

- First run may take longer as it downloads and initializes the TTS model
- Subsequent runs are much faster due to model caching
- **Conversational mode pre-generates all audio upfront for instant playback during conversation**
- The pre-generation phase takes a moment but ensures smooth conversation flow
- Shorter lines generate faster than longer lines
- The script automatically truncates very long text for optimal performance

## Fun Use Cases

- 🎓 **Course conversations**: Have Bes walk you through course material
- 🎬 **Character roleplay**: Have conversations with different personality voices
- 📚 **Educational content**: Create interactive learning experiences
- 🎪 **Entertainment**: Fun conversations with famous voice clones
- 🧑‍🏫 **Teaching tool**: Demonstrate voice cloning technology interactively

## License

This script uses the existing voice synthesis system and inherits its licensing terms. 