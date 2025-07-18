# Installation Guide for RLHFSummarise

## Quick Start (Recommended)

### Option 1: Minimal Installation (No TTS compilation issues)
```bash
# Install minimal dependencies (uses mock voice synthesizer)
uv add -r requirements-minimal.txt
```

### Option 2: Full Installation with TTS
```bash
# Run the macOS installation script
./install-macos.sh
```

### Option 3: Manual Installation
```bash
# Install numpy first (pre-compiled)
uv add numpy>=1.24.0,<2.0.0

# Install PyTorch with pre-compiled wheels
uv add --find-links https://download.pytorch.org/whl/torch_stable.html torch>=2.0.0 torchaudio>=2.0.0 torchvision>=0.15.0

# Install remaining dependencies
uv add -r requirements.txt
```

## Troubleshooting

### TTS Compilation Issues
If you encounter TTS compilation issues on macOS:

1. **Use the mock voice synthesizer** (recommended for development):
   - The app will use simulated audio output
   - No compilation required
   - Use `requirements-minimal.txt`

2. **Install Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```

3. **Use a virtual environment with conda**:
   ```bash
   conda create -n rlhf python=3.10
   conda activate rlhf
   pip install -r requirements.txt
   ```

### Common Issues

1. **numpy compilation fails**: Use `requirements-minimal.txt` or install numpy separately first
2. **TTS fails to load**: The app will automatically fall back to mock voice synthesizer
3. **PyTorch issues**: Use the pre-compiled wheels from PyTorch's official repository

## Running the Application

After installation:

```bash
# Start the web application
python modular_app.py
```

The app will be available at `http://localhost:5464`

## Voice Synthesizer Options

- **Mock**: Simulated audio (no compilation required)
- **TTS**: Real voice synthesis (requires compilation)

To switch between them, modify the configuration in `pipeline_factory.py`:
```python
"voice_synthesizer_type": "mock"  # or "tts"
``` 