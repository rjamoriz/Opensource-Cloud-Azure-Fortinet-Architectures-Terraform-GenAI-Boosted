#!/bin/bash

# Setup script for Enhanced Voice Chat functionality
# FortiGate Azure Chatbot - Voice Processing Dependencies

echo "🎤 Setting up Enhanced Voice Chat for FortiGate Azure Chatbot..."
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not installed."
    exit 1
fi

echo "✅ Python and pip found"

# Install voice processing dependencies
echo "📦 Installing voice processing dependencies..."

# Core OpenAI and audio processing
pip3 install openai>=1.0.0
pip3 install pydub>=0.25.1
pip3 install soundfile>=0.12.1

# Audio visualization
pip3 install matplotlib>=3.7.0
pip3 install numpy>=1.24.0

# Streamlit components
pip3 install streamlit-components-v1>=0.0.1

echo "🎵 Installing optional audio dependencies..."

# Optional: Real-time audio streaming
pip3 install streamlit-webrtc>=0.45.0 || echo "⚠️ streamlit-webrtc installation failed (optional)"

# Optional: Advanced audio processing
pip3 install librosa>=0.10.0 || echo "⚠️ librosa installation failed (optional)"

# Optional: Speech recognition fallback
pip3 install SpeechRecognition>=3.10.0 || echo "⚠️ SpeechRecognition installation failed (optional)"

# Optional: Text-to-speech alternatives
pip3 install gTTS>=2.3.0 || echo "⚠️ gTTS installation failed (optional)"
pip3 install pyttsx3>=2.90 || echo "⚠️ pyttsx3 installation failed (optional)"

echo "🔧 Checking system audio dependencies..."

# Check for system audio dependencies (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 macOS detected"
    
    # Check if Homebrew is available for ffmpeg
    if command -v brew &> /dev/null; then
        echo "🍺 Installing ffmpeg via Homebrew..."
        brew install ffmpeg || echo "⚠️ ffmpeg installation failed (optional)"
    else
        echo "⚠️ Homebrew not found. Please install ffmpeg manually for full audio support."
        echo "   Install Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo "   Then run: brew install ffmpeg"
    fi
    
    # Check for portaudio (needed for pyaudio)
    brew install portaudio || echo "⚠️ portaudio installation failed (optional)"
    pip3 install pyaudio>=0.2.11 || echo "⚠️ pyaudio installation failed (optional)"

elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 Linux detected"
    
    # Ubuntu/Debian
    if command -v apt-get &> /dev/null; then
        echo "📦 Installing system audio dependencies..."
        sudo apt-get update
        sudo apt-get install -y ffmpeg portaudio19-dev python3-pyaudio || echo "⚠️ System audio dependencies installation failed (optional)"
        pip3 install pyaudio>=0.2.11 || echo "⚠️ pyaudio installation failed (optional)"
    
    # CentOS/RHEL/Fedora
    elif command -v yum &> /dev/null; then
        echo "📦 Installing system audio dependencies..."
        sudo yum install -y ffmpeg portaudio-devel || echo "⚠️ System audio dependencies installation failed (optional)"
        pip3 install pyaudio>=0.2.11 || echo "⚠️ pyaudio installation failed (optional)"
    fi

else
    echo "⚠️ Unknown operating system. Please install ffmpeg and portaudio manually."
fi

echo ""
echo "🎉 Voice chat setup completed!"
echo "=================================================="
echo ""
echo "📋 Next steps:"
echo "1. Set your OpenAI API key:"
echo "   export OPENAI_API_KEY='your-api-key-here'"
echo ""
echo "2. Restart your Streamlit app:"
echo "   streamlit run src/app.py"
echo ""
echo "3. Navigate to the 'Enhanced Voice Chat' tab"
echo ""
echo "🔍 Troubleshooting:"
echo "• If audio recording doesn't work, check microphone permissions"
echo "• For browser audio issues, use HTTPS or localhost"
echo "• Some features require a modern web browser with WebRTC support"
echo ""
echo "✅ Setup complete! Enjoy your enhanced voice chat experience!"
