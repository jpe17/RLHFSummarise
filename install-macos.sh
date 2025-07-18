#!/bin/bash

# macOS-specific installation script for RLHFSummarise
# This script handles dependency installation with proper flags to avoid compilation issues

echo "🚀 Installing RLHFSummarise dependencies for macOS..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is for macOS only"
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "📦 Installing dependencies with uv..."

# First, install numpy separately to ensure we get a pre-compiled version
echo "🔧 Installing numpy first..."
uv add numpy>=1.24.0,<2.0.0

# Install PyTorch with pre-compiled wheels
echo "🔥 Installing PyTorch..."
uv add --find-links https://download.pytorch.org/whl/torch_stable.html torch>=2.0.0 torchaudio>=2.0.0 torchvision>=0.15.0

# Install other core dependencies
echo "📚 Installing core ML dependencies..."
uv add transformers>=4.35.0 accelerate>=0.20.0

# Install TTS (this should now work with the newer numpy)
echo "🎤 Installing TTS..."
uv add TTS>=0.22.0

# Install remaining dependencies
echo "🔧 Installing remaining dependencies..."
uv add -r requirements.txt

echo "✅ Installation complete!"
echo ""
echo "🎉 You can now run the application with:"
echo "   python modular_app.py"
echo ""
echo "📝 Note: If you encounter any issues with TTS, you may need to:"
echo "   1. Install Xcode Command Line Tools: xcode-select --install"
echo "   2. Or use the mock voice synthesizer for testing" 