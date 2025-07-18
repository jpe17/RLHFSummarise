# Bes Voice with Italian Accent

This feature automatically applies an Italian accent to the "bes" voice, making it sound more authentic and engaging.

## 🎭 How It Works

When you use the "bes" voice with the voice synthesizer, it automatically detects that this voice should use an Italian accent and applies it to the speech synthesis.

### Automatic Detection

The system automatically applies Italian accent when:
- Voice name is "bes" (case-insensitive)
- Language is set to "en" (English)
- The system will automatically switch to "it" (Italian) for the accent

### Example Usage

```python
from implementations.voice_synthesizer import VoiceSynthesizerFactory

# Initialize the synthesizer
synthesizer = VoiceSynthesizerFactory.create_synthesizer("tts")

# This will automatically use Italian accent for bes voice
output_path = synthesizer.synthesize(
    "Today we're casually building Google from scratch. Should take about an hour.",
    "bes", 
    language="en"  # Will auto-switch to Italian accent
)
```

## 🚀 Quick Demo

Run the demonstration script to see the Italian accent in action:

```bash
python demo_bes_italian.py
```

This will:
1. Show voice information and language preferences
2. Generate audio for bes voice with Italian accent
3. Save all audio files in the `outputs/` directory

## 🎤 Voice Cloner Integration

You can also use the voice cloner script with bes voice:

```bash
python voice_cloner.py --voice bes --language en
```

Even though you specify English (`en`), the system will automatically apply Italian accent for the bes voice.

## 🌍 Supported Languages

The voice synthesizer supports many languages and accents:

- **en**: English
- **it**: Italian (used for bes voice)
- **es**: Spanish
- **fr**: French
- **de**: German
- **pt**: Portuguese
- **pl**: Polish
- **tr**: Turkish
- **ru**: Russian
- **nl**: Dutch
- **cs**: Czech
- **ar**: Arabic
- **zh-cn**: Chinese (Simplified)
- **hu**: Hungarian
- **ko**: Korean
- **ja**: Japanese
- **hi**: Hindi

## 🎭 Voice Preferences

The system has built-in preferences for certain voices:

- **bes**: Italian accent (it)
- **barackobama**: English (en)
- **elonmusk**: English (en)
- **freeman**: English (en)

## 🔧 Technical Details

### Auto-Detection Logic

The system checks for voice preferences in the `synthesize()` method:

```python
# Auto-detect language preferences for specific voices
voice_preferences = self.get_voice_language_preferences()
if voice_name.lower() in voice_preferences and language == "en":
    preferred_language = voice_preferences[voice_name.lower()]
    if preferred_language != "en":
        language = preferred_language
        language_name = self.get_supported_languages().get(language, language)
        print(f"🌍 Auto-applying {language_name} accent for {voice_name} voice")
```

### Cache Key Generation

The cache system includes language in the key to ensure different accents are cached separately:

```python
def _get_cache_key(self, text: str, voice_name: str, language: str = "en") -> str:
    content = f"{text}:{voice_name}:{language}"
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
```

## 🎵 Sample Output

When you run the demo, you'll see output like:

```
🎭 Bes Voice with Italian Accent Demo
==================================================
🚀 Initializing voice synthesizer...
🌍 Auto-applying Italian accent for bes voice
🔊 Line 1/5: Today we're casually building Google from scratch...
✅ Generated: synthesis_bes_it_20241201_143022.wav
```

## 🎯 Use Cases

The Italian accent for bes voice is perfect for:

- **Startup presentations** with authentic Italian flair
- **Tech demos** with personality
- **Educational content** with engaging accents
- **Entertainment applications** with character voices

## 🔄 Extending to Other Voices

You can easily add more voice-language preferences by modifying the `get_voice_language_preferences()` method:

```python
def get_voice_language_preferences(self) -> Dict[str, str]:
    return {
        "bes": "it",  # Italian accent
        "newvoice": "es",  # Spanish accent
        "anothervoice": "fr",  # French accent
        # ... more preferences
    }
```

## 🎉 Enjoy!

The bes voice with Italian accent brings a unique, authentic character to your voice synthesis projects. The automatic detection makes it seamless to use while maintaining the flexibility to override with other languages when needed. 