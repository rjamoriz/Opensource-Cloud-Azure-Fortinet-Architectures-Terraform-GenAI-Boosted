"""
Enhanced Voice Processing System
Real-time voice processing with model routing and multi-provider TTS
"""

import streamlit as st
import logging
import asyncio
import time
import io
import base64
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class VoiceProvider(Enum):
    OPENAI = "openai"
    CARTESIA = "cartesia"
    ELEVENLABS = "elevenlabs"

class ModelType(Enum):
    GPT4 = "gpt-4"
    GPT4O = "gpt-4o"
    FINE_TUNED = "fine-tuned"
    LLAMA = "llama"

@dataclass
class VoiceConfig:
    """Voice processing configuration"""
    provider: VoiceProvider
    voice_id: str
    speed: float = 1.0
    pitch: float = 1.0
    stability: float = 0.5
    clarity: float = 0.75

@dataclass
class ModelConfig:
    """Model configuration for voice processing"""
    model_type: ModelType
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 500
    system_prompt: str = ""

class EnhancedVoiceProcessor:
    """Enhanced voice processor with real-time capabilities"""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_providers()
    
    def initialize_session_state(self):
        """Initialize session state for voice processing"""
        if "voice_conversation_history" not in st.session_state:
            st.session_state.voice_conversation_history = []
        
        if "voice_config" not in st.session_state:
            st.session_state.voice_config = VoiceConfig(
                provider=VoiceProvider.OPENAI,
                voice_id="alloy"
            )
        
        if "model_config" not in st.session_state:
            st.session_state.model_config = ModelConfig(
                model_type=ModelType.GPT4,
                model_name="gpt-4",
                system_prompt="You are a helpful FortiGate Azure deployment assistant."
            )
        
        if "voice_processing_active" not in st.session_state:
            st.session_state.voice_processing_active = False
    
    def setup_providers(self):
        """Setup voice providers"""
        self.providers = {
            VoiceProvider.OPENAI: self._setup_openai,
            VoiceProvider.CARTESIA: self._setup_cartesia,
            VoiceProvider.ELEVENLABS: self._setup_elevenlabs
        }
    
    def _setup_openai(self):
        """Setup OpenAI provider"""
        try:
            import openai
            api_key = st.session_state.get('openai_api_key') or st.secrets.get('openai', {}).get('api_key')
            if api_key:
                return openai.OpenAI(api_key=api_key)
        except ImportError:
            pass
        return None
    
    def _setup_cartesia(self):
        """Setup Cartesia provider"""
        # Placeholder for Cartesia integration
        return None
    
    def _setup_elevenlabs(self):
        """Setup ElevenLabs provider"""
        # Placeholder for ElevenLabs integration
        return None
    
    def display(self):
        """Display the enhanced voice interface"""
        st.markdown("### 🎤 Enhanced Voice Processing")
        
        # Status and configuration
        self._display_status()
        
        # Main interface tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎙️ Voice Chat",
            "⚙️ Voice Settings", 
            "🤖 Model Selection",
            "📊 Analytics"
        ])
        
        with tab1:
            self._display_voice_chat()
        
        with tab2:
            self._display_voice_settings()
        
        with tab3:
            self._display_model_selection()
        
        with tab4:
            self._display_analytics()
    
    def _display_status(self):
        """Display system status"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            openai_client = self._setup_openai()
            if openai_client:
                st.success("🔊 OpenAI: ✅")
            else:
                st.error("🔊 OpenAI: ❌")
        
        with col2:
            provider = st.session_state.voice_config.provider.value
            st.info(f"🎵 Provider: {provider.title()}")
        
        with col3:
            model = st.session_state.model_config.model_type.value
            st.info(f"🤖 Model: {model.upper()}")
        
        with col4:
            conversations = len(st.session_state.voice_conversation_history)
            st.metric("💬 Conversations", conversations)
    
    def _display_voice_chat(self):
        """Display voice chat interface"""
        st.markdown("#### 🎙️ Real-Time Voice Chat")
        
        # Voice input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Audio recorder placeholder (would use streamlit-audiorecorder in production)
            st.markdown("**🎤 Voice Input:**")
            uploaded_audio = st.file_uploader(
                "Upload audio file or use microphone",
                type=['wav', 'mp3', 'm4a'],
                help="Upload an audio file to process"
            )
            
            if uploaded_audio:
                st.audio(uploaded_audio, format='audio/wav')
                
                if st.button("🚀 Process Voice", type="primary", key="voice_process_button"):
                    self._process_voice_input(uploaded_audio)
        
        with col2:
            st.markdown("**⚡ Quick Actions:**")
            if st.button("🎙️ Start Recording", key="voice_start_recording"):
                st.info("Recording feature coming soon!")
            
            if st.button("⏹️ Stop Recording", key="voice_stop_recording"):
                st.info("Recording feature coming soon!")
            
            if st.button("🗑️ Clear History", key="voice_clear_history"):
                st.session_state.voice_conversation_history = []
                st.rerun()
        
        # Conversation history
        self._display_conversation_history()
        
        # Text input fallback
        st.markdown("---")
        st.markdown("**💬 Text Input (Fallback):**")
        text_input = st.text_area(
            "Type your message:",
            placeholder="Enter your message here...",
            height=100
        )
        
        if st.button("📝 Send Text Message", key="voice_send_text"):
            if text_input.strip():
                self._process_text_input(text_input)
    
    def _display_conversation_history(self):
        """Display conversation history"""
        if st.session_state.voice_conversation_history:
            st.markdown("#### 📜 Conversation History")
            
            for i, exchange in enumerate(st.session_state.voice_conversation_history):
                with st.container():
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown(f"**👤 You ({exchange.get('timestamp', 'Unknown')}):**")
                        st.markdown(exchange.get('user_text', 'No transcription'))
                        
                        if exchange.get('user_audio'):
                            st.audio(exchange['user_audio'], format='audio/wav')
                    
                    with col2:
                        st.markdown(f"**🤖 Assistant ({exchange.get('model', 'Unknown')}):**")
                        st.markdown(exchange.get('assistant_text', 'No response'))
                        
                        if exchange.get('assistant_audio'):
                            st.audio(exchange['assistant_audio'], format='audio/wav')
                    
                    st.markdown("---")
    
    def _display_voice_settings(self):
        """Display voice configuration settings"""
        st.markdown("#### ⚙️ Voice Provider Settings")
        
        # Provider selection
        provider = st.selectbox(
            "Voice Provider:",
            options=[p.value for p in VoiceProvider],
            index=list(VoiceProvider).index(st.session_state.voice_config.provider),
            format_func=lambda x: x.title()
        )
        
        # Update provider
        if provider != st.session_state.voice_config.provider.value:
            st.session_state.voice_config.provider = VoiceProvider(provider)
        
        # Provider-specific settings
        if st.session_state.voice_config.provider == VoiceProvider.OPENAI:
            self._display_openai_settings()
        elif st.session_state.voice_config.provider == VoiceProvider.CARTESIA:
            self._display_cartesia_settings()
        elif st.session_state.voice_config.provider == VoiceProvider.ELEVENLABS:
            self._display_elevenlabs_settings()
        
        # Global voice settings
        st.markdown("#### 🎛️ Voice Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            speed = st.slider(
                "Speech Speed:",
                min_value=0.5,
                max_value=2.0,
                value=st.session_state.voice_config.speed,
                step=0.1
            )
            st.session_state.voice_config.speed = speed
        
        with col2:
            pitch = st.slider(
                "Pitch:",
                min_value=0.5,
                max_value=2.0,
                value=st.session_state.voice_config.pitch,
                step=0.1
            )
            st.session_state.voice_config.pitch = pitch
    
    def _display_openai_settings(self):
        """Display OpenAI-specific settings"""
        st.markdown("**🤖 OpenAI Voice Settings**")
        
        voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        voice = st.selectbox(
            "Voice:",
            options=voices,
            index=voices.index(st.session_state.voice_config.voice_id) if st.session_state.voice_config.voice_id in voices else 0
        )
        st.session_state.voice_config.voice_id = voice
        
        # API Key configuration
        api_key = st.text_input(
            "OpenAI API Key:",
            value=st.session_state.get('openai_api_key', ''),
            type="password",
            help="Your OpenAI API key"
        )
        
        if api_key:
            st.session_state.openai_api_key = api_key
    
    def _display_cartesia_settings(self):
        """Display Cartesia-specific settings"""
        st.info("🚧 Cartesia integration coming soon!")
    
    def _display_elevenlabs_settings(self):
        """Display ElevenLabs-specific settings"""
        st.info("🚧 ElevenLabs integration coming soon!")
    
    def _display_model_selection(self):
        """Display model selection interface"""
        st.markdown("#### 🤖 AI Model Configuration")
        
        # Model type selection
        model_type = st.selectbox(
            "Model Type:",
            options=[m.value for m in ModelType],
            index=list(ModelType).index(st.session_state.model_config.model_type),
            format_func=lambda x: x.upper().replace('_', ' ')
        )
        
        if model_type != st.session_state.model_config.model_type.value:
            st.session_state.model_config.model_type = ModelType(model_type)
        
        # Model-specific configuration
        if st.session_state.model_config.model_type == ModelType.GPT4:
            model_name = st.selectbox(
                "GPT-4 Model:",
                options=["gpt-4", "gpt-4-turbo", "gpt-4-turbo-preview"],
                index=0
            )
        elif st.session_state.model_config.model_type == ModelType.GPT4O:
            model_name = st.selectbox(
                "GPT-4o Model:",
                options=["gpt-4o", "gpt-4o-mini"],
                index=0
            )
        elif st.session_state.model_config.model_type == ModelType.FINE_TUNED:
            model_name = st.text_input(
                "Fine-tuned Model ID:",
                value=st.session_state.model_config.model_name,
                placeholder="ft:gpt-3.5-turbo:..."
            )
        else:  # LLAMA
            model_name = st.selectbox(
                "Llama Model:",
                options=["llama-7b", "llama-13b", "llama-70b"],
                index=0
            )
        
        st.session_state.model_config.model_name = model_name
        
        # Model parameters
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider(
                "Temperature:",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.model_config.temperature,
                step=0.1
            )
            st.session_state.model_config.temperature = temperature
        
        with col2:
            max_tokens = st.number_input(
                "Max Tokens:",
                min_value=50,
                max_value=2000,
                value=st.session_state.model_config.max_tokens,
                step=50
            )
            st.session_state.model_config.max_tokens = max_tokens
        
        # System prompt
        system_prompt = st.text_area(
            "System Prompt:",
            value=st.session_state.model_config.system_prompt,
            height=100,
            help="Instructions for the AI model"
        )
        st.session_state.model_config.system_prompt = system_prompt
    
    def _display_analytics(self):
        """Display voice processing analytics"""
        st.markdown("#### 📊 Voice Processing Analytics")
        
        if not st.session_state.voice_conversation_history:
            st.info("No conversation data available yet.")
            return
        
        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_conversations = len(st.session_state.voice_conversation_history)
            st.metric("Total Conversations", total_conversations)
        
        with col2:
            avg_response_time = 1.2  # Placeholder
            st.metric("Avg Response Time", f"{avg_response_time:.1f}s")
        
        with col3:
            success_rate = 95.5  # Placeholder
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        with col4:
            total_audio_time = 45.3  # Placeholder
            st.metric("Total Audio Time", f"{total_audio_time:.1f}s")
        
        # Recent activity
        st.markdown("#### 📈 Recent Activity")
        
        recent_conversations = st.session_state.voice_conversation_history[-5:]
        for i, conv in enumerate(recent_conversations):
            with st.expander(f"Conversation {len(st.session_state.voice_conversation_history) - len(recent_conversations) + i + 1}"):
                st.json({
                    "timestamp": conv.get('timestamp', 'Unknown'),
                    "model": conv.get('model', 'Unknown'),
                    "provider": conv.get('provider', 'Unknown'),
                    "user_text_length": len(conv.get('user_text', '')),
                    "assistant_text_length": len(conv.get('assistant_text', ''))
                })
    
    def _process_voice_input(self, audio_file):
        """Process voice input"""
        try:
            with st.spinner("🎤 Processing voice input..."):
                # Speech-to-text
                transcription = self._speech_to_text(audio_file)
                
                if transcription:
                    # Generate response
                    response = self._generate_response(transcription)
                    
                    # Text-to-speech
                    audio_response = self._text_to_speech(response)
                    
                    # Store conversation
                    self._store_conversation(
                        user_text=transcription,
                        assistant_text=response,
                        user_audio=audio_file,
                        assistant_audio=audio_response
                    )
                    
                    st.rerun()
                
        except Exception as e:
            st.error(f"Error processing voice input: {e}")
    
    def _process_text_input(self, text):
        """Process text input"""
        try:
            with st.spinner("💭 Generating response..."):
                # Generate response
                response = self._generate_response(text)
                
                # Text-to-speech
                audio_response = self._text_to_speech(response)
                
                # Store conversation
                self._store_conversation(
                    user_text=text,
                    assistant_text=response,
                    assistant_audio=audio_response
                )
                
                st.rerun()
                
        except Exception as e:
            st.error(f"Error processing text input: {e}")
    
    def _speech_to_text(self, audio_file) -> str:
        """Convert speech to text"""
        try:
            client = self._setup_openai()
            if not client:
                return "Error: OpenAI client not available"
            
            # Convert audio file to the right format
            audio_bytes = audio_file.read()
            audio_file.seek(0)  # Reset file pointer
            
            # Create a file-like object
            audio_io = io.BytesIO(audio_bytes)
            audio_io.name = "audio.wav"
            
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_io
            )
            
            return transcript.text
            
        except Exception as e:
            logger.error(f"Speech-to-text error: {e}")
            return f"Error: {str(e)}"
    
    def _generate_response(self, text: str) -> str:
        """Generate AI response"""
        try:
            model_config = st.session_state.model_config
            
            if model_config.model_type in [ModelType.GPT4, ModelType.GPT4O, ModelType.FINE_TUNED]:
                return self._generate_openai_response(text)
            elif model_config.model_type == ModelType.LLAMA:
                return self._generate_llama_response(text)
            else:
                return "Error: Unsupported model type"
                
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return f"Error generating response: {str(e)}"
    
    def _generate_openai_response(self, text: str) -> str:
        """Generate OpenAI response"""
        try:
            client = self._setup_openai()
            if not client:
                return "Error: OpenAI client not available"
            
            model_config = st.session_state.model_config
            
            response = client.chat.completions.create(
                model=model_config.model_name,
                messages=[
                    {"role": "system", "content": model_config.system_prompt},
                    {"role": "user", "content": text}
                ],
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _generate_llama_response(self, text: str) -> str:
        """Generate Llama response"""
        # Placeholder for Llama integration
        return "Llama response generation coming soon!"
    
    def _text_to_speech(self, text: str) -> Optional[bytes]:
        """Convert text to speech"""
        try:
            provider = st.session_state.voice_config.provider
            
            if provider == VoiceProvider.OPENAI:
                return self._openai_text_to_speech(text)
            elif provider == VoiceProvider.CARTESIA:
                return self._cartesia_text_to_speech(text)
            elif provider == VoiceProvider.ELEVENLABS:
                return self._elevenlabs_text_to_speech(text)
            
        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")
            return None
    
    def _openai_text_to_speech(self, text: str) -> Optional[bytes]:
        """OpenAI text-to-speech"""
        try:
            client = self._setup_openai()
            if not client:
                return None
            
            voice_config = st.session_state.voice_config
            
            response = client.audio.speech.create(
                model="tts-1",
                voice=voice_config.voice_id,
                input=text,
                speed=voice_config.speed
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"OpenAI TTS error: {e}")
            return None
    
    def _cartesia_text_to_speech(self, text: str) -> Optional[bytes]:
        """Cartesia text-to-speech"""
        # Placeholder for Cartesia integration
        return None
    
    def _elevenlabs_text_to_speech(self, text: str) -> Optional[bytes]:
        """ElevenLabs text-to-speech"""
        # Placeholder for ElevenLabs integration
        return None
    
    def _store_conversation(self, user_text: str, assistant_text: str, 
                          user_audio=None, assistant_audio=None):
        """Store conversation in session state"""
        conversation = {
            "timestamp": datetime.now().isoformat(),
            "user_text": user_text,
            "assistant_text": assistant_text,
            "model": st.session_state.model_config.model_name,
            "provider": st.session_state.voice_config.provider.value,
            "user_audio": user_audio,
            "assistant_audio": assistant_audio
        }
        
        st.session_state.voice_conversation_history.append(conversation)

def get_enhanced_voice_processor():
    """Factory function to get enhanced voice processor"""
    return EnhancedVoiceProcessor()
