# 🎤 Enhanced Voice Chat with AI Models

## Overview

The Enhanced Voice Chat feature allows you to interact with multiple AI models using voice input and output, combining the power of OpenAI's standard models (GPT-4o, GPT-4) with your custom fine-tuned models trained on FortiGate and Azure-specific data.

## 🌟 Key Features

### 🎯 Multi-Model Support
- **GPT-4o**: Latest multimodal model with vision capabilities
- **GPT-4**: Advanced reasoning for complex tasks
- **GPT-3.5 Turbo**: Fast and efficient responses
- **Fine-tuned Models**: Your custom models trained with FortiGate/Azure data

### 🎤 Voice Input
- Real-time speech-to-text using OpenAI Whisper
- Multiple language support
- Audio visualization during recording
- Automatic silence detection
- Fallback text input option

### 🔊 Voice Output
- High-quality text-to-speech using OpenAI TTS
- Multiple voice options (Alloy, Echo, Fable, Onyx, Nova, Shimmer)
- Adjustable speech speed and volume
- Audio playback controls

### 💬 Advanced Chat Features
- Conversation history with context awareness
- Model comparison mode
- Voice response playback for each message
- Conversation export and management

## 🚀 Quick Start

### 1. Installation

Run the automated setup script:
```bash
./setup_voice_chat.sh
```

Or install manually:
```bash
# Core dependencies
pip install openai>=1.0.0 pydub>=0.25.1 soundfile>=0.12.1

# Optional advanced features
pip install streamlit-webrtc>=0.45.0 librosa>=0.10.0
```

### 2. Configuration

Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

### 3. Launch

Start the Streamlit app and navigate to the "Enhanced Voice Chat" tab:
```bash
streamlit run src/app.py
```

## 🎯 How to Use

### Basic Voice Chat

1. **Select a Model**: Choose from GPT-4o, GPT-4, GPT-3.5, or your fine-tuned models
2. **Record Your Question**: Click the microphone button and speak
3. **Get AI Response**: The selected model will process and respond
4. **Listen to Response**: Enable voice output to hear the AI's response

### Model Comparison

1. Click "Compare Models" to enter comparison mode
2. Enter a question to ask all available models
3. Compare responses from different models side-by-side
4. Listen to each model's response individually

### Voice Settings

Customize your voice experience:
- **Input Language**: Choose your preferred language
- **Recording Sensitivity**: Adjust microphone sensitivity
- **Voice Type**: Select from Neural, Standard, or Premium voices
- **Speech Speed**: Control playback speed (0.5x to 2.0x)
- **Voice Volume**: Adjust output volume

## 🔧 Technical Architecture

### Components

1. **EnhancedVoiceChatManager**: Core chat logic and model management
2. **VoiceProcessor**: Speech-to-text and text-to-speech processing
3. **Audio Components**: Custom HTML/JavaScript for real-time audio
4. **Model Integration**: Seamless switching between standard and fine-tuned models

### Audio Processing Pipeline

```
Voice Input → Whisper STT → Model Processing → TTS Generation → Audio Output
     ↓              ↓              ↓              ↓              ↓
  Recording    Transcription   AI Response   Audio Synthesis  Playback
```

### Model Selection Logic

```python
# Standard models
"gpt-4o": Latest multimodal capabilities
"gpt-4": Advanced reasoning
"gpt-3.5-turbo": Fast responses

# Fine-tuned models (auto-detected)
"ft:gpt-3.5-turbo:your-org:model-name": Custom FortiGate/Azure expertise
```

## 🎨 User Interface

### Main Interface
- **Model Selection**: Visual buttons with icons and descriptions
- **Voice Recorder**: Interactive recording with audio visualization
- **Chat History**: Conversation display with model attribution
- **Voice Controls**: Play/pause, volume, and speed controls

### Advanced Features
- **Real-time Audio Visualization**: See audio levels during recording
- **Conversation Context**: Models remember previous exchanges
- **Export Options**: Save conversations for later reference
- **Error Handling**: Graceful fallbacks for audio issues

## 🔍 Troubleshooting

### Common Issues

**Audio Recording Not Working**
- Check microphone permissions in your browser
- Ensure you're using HTTPS or localhost
- Try refreshing the page and allowing microphone access

**Voice Output Not Playing**
- Verify your OpenAI API key is set correctly
- Check browser audio settings
- Try different voice options in settings

**Model Not Available**
- Ensure your fine-tuned models are properly deployed
- Check OpenAI API key permissions
- Verify model IDs are correct

**Poor Audio Quality**
- Adjust recording sensitivity in settings
- Check microphone quality and positioning
- Try different audio quality settings

### Browser Compatibility

**Recommended Browsers**:
- Chrome 70+ (best WebRTC support)
- Firefox 65+
- Safari 14+
- Edge 79+

**Required Features**:
- WebRTC for real-time audio
- MediaRecorder API for recording
- Web Audio API for visualization

## 🔐 Security & Privacy

### Data Handling
- Audio data is processed locally when possible
- OpenAI API calls follow their privacy policy
- No audio data is stored permanently
- Conversation history is session-based

### API Key Security
- Store API keys as environment variables
- Never commit keys to version control
- Use least-privilege API key permissions
- Monitor API usage regularly

## 🚀 Advanced Usage

### Custom Voice Prompts

Fine-tuned models can be optimized with specialized system prompts:

```python
# FortiGate-specific prompt
system_msg = """You are a specialized FortiGate Azure deployment assistant. 
You have been fine-tuned with specific knowledge about FortiGate configurations, 
Azure infrastructure, and best practices. Provide detailed, accurate responses 
based on your specialized training."""
```

### Batch Processing

Process multiple questions efficiently:

```python
# Compare responses across models
questions = [
    "How do I configure FortiGate HA?",
    "What are Azure VNET best practices?",
    "How to troubleshoot FortiGate connectivity?"
]

for question in questions:
    for model in available_models:
        response = get_model_response(question, model)
        # Process and compare responses
```

### Integration with Existing Workflows

The voice chat can be integrated with:
- Terraform deployment scripts
- Azure resource management
- FortiGate configuration automation
- Documentation generation

## 📊 Performance Optimization

### Response Times
- GPT-3.5 Turbo: ~2-3 seconds
- GPT-4: ~5-8 seconds
- GPT-4o: ~3-5 seconds
- Fine-tuned models: ~2-4 seconds

### Audio Processing
- Speech-to-text: ~1-2 seconds
- Text-to-speech: ~2-3 seconds
- Total latency: ~5-10 seconds end-to-end

### Cost Optimization
- Use GPT-3.5 for simple queries
- Reserve GPT-4o for complex multimodal tasks
- Fine-tuned models for specialized knowledge
- Monitor token usage and optimize prompts

## 🔄 Updates and Maintenance

### Regular Updates
- Keep OpenAI library updated for latest features
- Monitor model availability and deprecations
- Update voice processing dependencies
- Test browser compatibility regularly

### Model Management
- Regularly retrain fine-tuned models with new data
- Monitor model performance and accuracy
- Archive old model versions
- Document model changes and improvements

## 🤝 Contributing

To contribute to the Enhanced Voice Chat feature:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

### Development Setup

```bash
# Clone repository
git clone [repository-url]

# Install development dependencies
pip install -r requirements_voice.txt
pip install -r requirements_dev.txt

# Run tests
python -m pytest tests/voice_chat/

# Start development server
streamlit run src/app.py --server.port=8514
```

## 📞 Support

For issues and questions:
- Check the troubleshooting section above
- Review browser console for error messages
- Test with different browsers and devices
- Verify API key permissions and quotas

## 🎉 Conclusion

The Enhanced Voice Chat feature transforms your FortiGate Azure Chatbot into a powerful voice-enabled assistant that can leverage both standard AI models and your custom fine-tuned expertise. Whether you're configuring complex FortiGate deployments or managing Azure infrastructure, you can now interact naturally using voice commands and receive intelligent, contextual responses.

Enjoy your enhanced voice-powered FortiGate Azure assistant! 🚀🎤
