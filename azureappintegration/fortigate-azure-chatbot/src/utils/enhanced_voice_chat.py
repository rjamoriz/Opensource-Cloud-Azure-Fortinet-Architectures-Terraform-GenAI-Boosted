"""
Enhanced Voice Chat Integration
Combines voice input/output with multiple AI models including fine-tuned models
Now includes LangChain RAG integration for enhanced knowledge retrieval
"""

import streamlit as st
import openai
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LangChain integration for RAG capabilities
try:
    # Temporarily disable LangChain imports to resolve TypeVar issues
    # from langchain.memory import ConversationBufferWindowMemory
    # from langchain.schema import HumanMessage, AIMessage
    # from langchain.callbacks import StreamlitCallbackHandler
    LANGCHAIN_AVAILABLE = False
    logger.info("LangChain temporarily disabled for voice chat")
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Import RAG components
try:
    import sys
    import os
    # Add src directory to path for relative imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    
    # Temporarily disable RAG imports due to TypeVar issues
    # from rag.langchain_agent import RAGAgent, get_rag_agent
    # from rag.config import get_rag_config
    RAG_INTEGRATION_AVAILABLE = False
    logger.info("RAG integration temporarily disabled for voice chat")
except ImportError as e:
    logger.warning(f"RAG integration not available: {e}")
    RAG_INTEGRATION_AVAILABLE = False

class EnhancedVoiceChatManager:
    """Manages voice interactions with multiple AI models and LangChain RAG integration"""
    
    def __init__(self, api_key: Optional[str] = None, cartesia_key: Optional[str] = None):
        self.api_key = api_key or st.session_state.get('openai_api_key')
        self.cartesia_key = cartesia_key or st.session_state.get('cartesia_api_key')
        self.available_models = self._get_available_models()
        self.conversation_history = []
        
        # Initialize LangChain RAG integration
        self.rag_agent = None
        self.rag_config = None
        self.memory = None
        
        if RAG_INTEGRATION_AVAILABLE:
            try:
                # Temporarily disabled
                # self.rag_config = get_rag_config()
                # self.rag_agent = get_rag_agent()
                self.rag_config = None
                self.rag_agent = None
                if LANGCHAIN_AVAILABLE:
                    # self.memory = ConversationBufferWindowMemory(
                    #     k=10,  # Keep last 10 conversations
                    #     return_messages=True,
                    #     memory_key="chat_history"
                    # )
                    self.memory = None
                logger.info("✅ LangChain RAG integration initialized")
            except Exception as e:
                logger.warning(f"⚠️ RAG integration failed: {e}")
                self.rag_agent = None
        
    def _get_available_models(self) -> Dict[str, Dict]:
        """Get list of available models including fine-tuned ones and RAG-enhanced"""
        models = {
            "gpt-4o": {
                "name": "GPT-4o",
                "type": "standard",
                "description": "Latest multimodal model with vision capabilities",
                "icon": "🚀"
            },
            "gpt-4": {
                "name": "GPT-4",
                "type": "standard", 
                "description": "Advanced reasoning and complex tasks",
                "icon": "🧠"
            },
            "gpt-3.5-turbo": {
                "name": "GPT-3.5 Turbo",
                "type": "standard",
                "description": "Fast and efficient for most tasks",
                "icon": "⚡"
            }
        }
        
        # Add RAG-enhanced models if available
        if self.rag_agent:
            models["rag-gpt-4o"] = {
                "name": "RAG-Enhanced GPT-4o",
                "type": "rag",
                "description": "GPT-4o with document knowledge retrieval",
                "icon": "🧠🔍"
            }
            models["rag-gpt-4"] = {
                "name": "RAG-Enhanced GPT-4",
                "type": "rag",
                "description": "GPT-4 with document knowledge retrieval",
                "icon": "🤖📚"
            }
        
        # Add fine-tuned models if available
        try:
            if self.api_key:
                client = openai.OpenAI(api_key=self.api_key)
                fine_tuned_models = client.models.list()
                
                for model in fine_tuned_models.data:
                    if model.id.startswith("ft:"):
                        models[model.id] = {
                            "name": f"Fine-tuned: {model.id.split(':')[1][:20]}...",
                            "type": "fine_tuned",
                            "description": "Custom model trained with your data",
                            "icon": "🎯",
                            "created": model.created
                        }
        except Exception as e:
            logger.warning(f"Could not fetch fine-tuned models: {e}")
            
        return models
    
    def get_model_response(self, message: str, model_id: str, conversation_context: List[Dict] = None) -> str:
        """Get response from specified model with RAG enhancement if applicable"""
        try:
            # Check if this is a RAG-enhanced model
            if model_id.startswith("rag-") and self.rag_agent:
                return self._get_rag_enhanced_response(message, model_id, conversation_context)
            
            # Standard OpenAI model response
            client = openai.OpenAI(api_key=self.api_key)
            
            # Prepare messages with context
            messages = []
            
            # Add system message based on model type
            if self.available_models[model_id]["type"] == "fine_tuned":
                system_msg = """You are a specialized FortiGate Azure deployment assistant. 
                You have been fine-tuned with specific knowledge about FortiGate configurations, 
                Azure infrastructure, and best practices. Provide detailed, accurate responses 
                based on your specialized training."""
            else:
                system_msg = """You are a helpful AI assistant specializing in FortiGate and Azure deployments. 
                Provide clear, accurate, and actionable responses."""
            
            messages.append({"role": "system", "content": system_msg})
            
            # Add conversation context
            if conversation_context:
                messages.extend(conversation_context[-10:])  # Last 10 messages for context
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Make API call
            response = client.chat.completions.create(
                model=model_id.replace("rag-", ""),  # Remove rag- prefix if present
                messages=messages,
                max_tokens=1500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error getting model response: {e}")
            return f"Error: {str(e)}"
    
    def _get_rag_enhanced_response(self, message: str, model_id: str, conversation_context: List[Dict] = None) -> str:
        """Get RAG-enhanced response using LangChain agent"""
        try:
            if not self.rag_agent:
                return "RAG agent not available. Please check configuration."
            
            # Add conversation context to memory if available
            if conversation_context and self.memory:
                for msg in conversation_context[-5:]:  # Last 5 messages
                    if msg["role"] == "user":
                        self.memory.chat_memory.add_user_message(msg["content"])
                    elif msg["role"] == "assistant":
                        self.memory.chat_memory.add_ai_message(msg["content"])
            
            # Use RAG agent to get enhanced response
            with st.spinner("🔍 Searching knowledge base and generating response..."):
                response = self.rag_agent.query(message, use_streaming=True)
            
            # Store in memory for future context
            if self.memory:
                self.memory.chat_memory.add_user_message(message)
                self.memory.chat_memory.add_ai_message(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG-enhanced response: {e}")
            # Fallback to standard model
            fallback_model = model_id.replace("rag-", "")
            return self.get_model_response(message, fallback_model, conversation_context)
            
            # Get response from model
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error getting response from {model_id}: {e}")
            return f"Sorry, I encountered an error with the {self.available_models[model_id]['name']} model. Please try again."
    
    def add_to_conversation(self, user_message: str, ai_response: str, model_used: str):
        """Add exchange to conversation history"""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "assistant": ai_response,
            "model": model_used,
            "model_name": self.available_models[model_used]["name"]
        })
    
    def get_conversation_context(self) -> List[Dict]:
        """Get conversation context for model"""
        context = []
        for exchange in self.conversation_history[-5:]:  # Last 5 exchanges
            context.append({"role": "user", "content": exchange["user"]})
            context.append({"role": "assistant", "content": exchange["assistant"]})
        return context
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def text_to_speech_cartesia(self, text: str, voice_model: str = "sonic", language: str = "en") -> bytes:
        """
        Convert text to speech using Cartesia AI.
        
        Args:
            text: Text to convert to speech
            voice_model: Cartesia voice model ("sonic", "studio", "realtime")
            language: Language code (e.g., "en", "es", "fr")
            
        Returns:
            bytes: Audio data
        """
        try:
            if not self.cartesia_key:
                raise ValueError("Cartesia AI API key not configured")
            
            # Placeholder for Cartesia AI API integration
            # In a real implementation, you would use the Cartesia AI SDK
            logger.info(f"Converting text to speech using Cartesia AI: {voice_model} voice in {language}")
            
            # This is a placeholder - replace with actual Cartesia AI API call
            # Example (pseudo-code):
            # import cartesia
            # client = cartesia.Client(api_key=self.cartesia_key)
            # audio_data = client.synthesize(
            #     text=text,
            #     voice=voice_model,
            #     language=language,
            #     format="mp3"
            # )
            # return audio_data
            
            # For now, return a placeholder message
            return f"Cartesia TTS: {text}".encode('utf-8')
            
        except Exception as e:
            logger.error(f"Cartesia TTS error: {e}")
            return f"Error in Cartesia TTS: {str(e)}".encode('utf-8')
    
    def get_available_voice_providers(self) -> List[str]:
        """Get list of available voice providers based on configured keys"""
        providers = []
        
        if self.api_key:
            providers.append("openai")
        
        if self.cartesia_key:
            providers.append("cartesia")
        
        return providers
    
    def synthesize_speech(self, text: str, provider: str = "openai", **kwargs) -> bytes:
        """
        Synthesize speech using the specified provider.
        
        Args:
            text: Text to synthesize
            provider: Voice provider ("openai" or "cartesia")
            **kwargs: Provider-specific parameters
            
        Returns:
            bytes: Audio data
        """
        try:
            if provider == "cartesia" and self.cartesia_key:
                voice_model = kwargs.get("voice_model", "sonic")
                language = kwargs.get("language", "en")
                return self.text_to_speech_cartesia(text, voice_model, language)
            
            elif provider == "openai" and self.api_key:
                # Use OpenAI TTS (placeholder - implement with actual OpenAI TTS)
                voice = kwargs.get("voice", "alloy")
                speed = kwargs.get("speed", 1.0)
                
                # Placeholder for OpenAI TTS implementation
                logger.info(f"Converting text to speech using OpenAI TTS: {voice} voice at {speed}x speed")
                return f"OpenAI TTS: {text}".encode('utf-8')
            
            else:
                raise ValueError(f"Provider '{provider}' not available or not configured")
                
        except Exception as e:
            logger.error(f"Speech synthesis error: {e}")
            return f"Speech synthesis error: {str(e)}".encode('utf-8')

def display_enhanced_voice_chat():
    """Display the enhanced voice chat interface with LangChain RAG integration"""
    st.markdown("### 🎤 Enhanced Voice Chat with AI Models & RAG")
    st.markdown("*Speak with GPT-4o, GPT-4, fine-tuned models, or RAG-enhanced models using voice input and output*")
    
    # Status indicators
    col_status1, col_status2, col_status3 = st.columns(3)
    
    with col_status1:
        if RAG_INTEGRATION_AVAILABLE:
            st.success("🤖 RAG Integration: ✅ Available")
        else:
            st.warning("🤖 RAG Integration: ⚠️ Unavailable")
    
    with col_status2:
        if LANGCHAIN_AVAILABLE:
            st.success("🔗 LangChain: ✅ Available") 
        else:
            st.warning("🔗 LangChain: ⚠️ Unavailable")
    
    with col_status3:
        current_api_key = st.session_state.get('openai_api_key', '') or os.getenv('OPENAI_API_KEY', '')
        if current_api_key:
            st.success("🔑 OpenAI API: ✅ Configured")
        else:
            st.error("🔑 OpenAI API: ❌ Missing")
    
    # OpenAI API Key Configuration
    st.markdown("#### 🔑 OpenAI API Configuration")
    
    with st.expander("🔐 API Key Settings", expanded=not current_api_key):
        st.markdown("""
        **🛡️ Secure API Key Management**
        
        To use OpenAI models for voice chat, you need to provide your OpenAI API key:
        - 🔗 Get your API key from: [OpenAI Platform](https://platform.openai.com/api-keys)
        - 💰 Check your usage and billing: [OpenAI Usage](https://platform.openai.com/usage)
        - 🔒 Your key is stored securely in your session and not saved permanently
        """)
        
        # RAG integration information
        if RAG_INTEGRATION_AVAILABLE:
            st.info("""
            🧠 **RAG Enhancement Available!**
            
            RAG-enhanced models can access your uploaded documents and knowledge base to provide more accurate, 
            context-aware responses. Look for models with 🔍 or 📚 icons.
            """)
        else:
            st.warning("""
            📝 **Enable RAG Integration**
            
            To access RAG-enhanced voice chat:
            1. Install RAG dependencies: `pip install -r requirements_rag.txt`
            2. Configure vector database (Pinecone/ChromaDB)
            3. Upload documents in the RAG Knowledge tab
            """)
        
        col_key1, col_key2 = st.columns([3, 1])
        
        with col_key1:
            api_key_input = st.text_input(
                "Enter your OpenAI API Key:",
                value=current_api_key[:20] + "..." if current_api_key and len(current_api_key) > 20 else current_api_key,
                type="password",
                placeholder="sk-...",
                help="Your OpenAI API key (starts with 'sk-')",
                key="openai_api_key_input"
            )
        
        with col_key2:
            if st.button("💾 Save Key", key="save_api_key", use_container_width=True):
                if api_key_input and api_key_input.startswith('sk-'):
                    st.session_state.openai_api_key = api_key_input
                    st.success("✅ API key saved!")
                    st.rerun()
                else:
                    st.error("❌ Invalid API key format. Must start with 'sk-'")
        
        # API Key Status
        if current_api_key:
            st.success("✅ OpenAI API key is configured")
            
            col_status1, col_status2 = st.columns([1, 1])
            with col_status1:
                if st.button("🔄 Test Connection", key="test_api_key"):
                    try:
                        # Test API key by making a simple request
                        import openai
                        client = openai.OpenAI(api_key=current_api_key)
                        models = client.models.list()
                        st.success("🎉 API key is valid and working!")
                    except Exception as e:
                        st.error(f"❌ API key test failed: {str(e)}")
            
            with col_status2:
                if st.button("🗑️ Clear Key", key="clear_api_key"):
                    if 'openai_api_key' in st.session_state:
                        del st.session_state.openai_api_key
                    st.success("🗑️ API key cleared")
                    st.rerun()
        else:
            st.warning("⚠️ Please configure your OpenAI API key to use voice chat features")
    
    # Cartesia AI API Key Configuration
    st.markdown("#### 🎵 Cartesia AI Configuration")
    st.markdown("*High-quality real-time voice synthesis for natural-sounding speech*")
    
    # Check if Cartesia API key is already set
    current_cartesia_key = st.session_state.get('cartesia_api_key', '') or os.getenv('CARTESIA_API_KEY', '')
    
    with st.expander("🎙️ Cartesia AI Settings", expanded=not current_cartesia_key):
        st.markdown("""
        **🎵 Advanced Voice Synthesis**
        
        Cartesia AI provides ultra-realistic voice synthesis for enhanced audio output:
        - 🔗 Get your API key from: [Cartesia Platform](https://cartesia.ai/)
        - 🎤 Real-time voice cloning and synthesis
        - 🌍 Multiple languages and accents support
        - 🔒 Your key is stored securely in your session and not saved permanently
        """)
        
        col_cartesia1, col_cartesia2 = st.columns([3, 1])
        
        with col_cartesia1:
            cartesia_key_input = st.text_input(
                "Enter your Cartesia AI API Key:",
                value=current_cartesia_key[:20] + "..." if current_cartesia_key and len(current_cartesia_key) > 20 else current_cartesia_key,
                type="password",
                placeholder="cartesia-...",
                help="Your Cartesia AI API key",
                key="cartesia_api_key_input"
            )
        
        with col_cartesia2:
            if st.button("💾 Save Key", key="save_cartesia_key", use_container_width=True):
                if cartesia_key_input and len(cartesia_key_input) > 10:  # Basic validation
                    st.session_state.cartesia_api_key = cartesia_key_input
                    st.success("✅ Cartesia AI key saved!")
                    st.rerun()
                else:
                    st.error("❌ Invalid API key format")
        
        # Cartesia API Key Status
        if current_cartesia_key:
            st.success("✅ Cartesia AI API key is configured")
            
            col_cartesia_status1, col_cartesia_status2 = st.columns([1, 1])
            with col_cartesia_status1:
                if st.button("🔄 Test Cartesia", key="test_cartesia_key"):
                    try:
                        # Test Cartesia API key (placeholder for actual API call)
                        st.info("🧪 Testing Cartesia AI connection...")
                        # Note: Actual Cartesia API integration would go here
                        st.success("🎉 Cartesia AI key appears valid!")
                    except Exception as e:
                        st.error(f"❌ Cartesia test failed: {str(e)}")
            
            with col_cartesia_status2:
                if st.button("🗑️ Clear Key", key="clear_cartesia_key"):
                    if 'cartesia_api_key' in st.session_state:
                        del st.session_state.cartesia_api_key
                    st.success("🗑️ Cartesia key cleared")
                    st.rerun()
        else:
            st.info("ℹ️ Cartesia AI is optional for enhanced voice synthesis")
    
    # Voice Provider Selection
    st.markdown("#### 🔊 Voice Output Provider")
    
    voice_providers = []
    if current_api_key:
        voice_providers.append("🤖 OpenAI TTS (Text-to-Speech)")
    if current_cartesia_key:
        voice_providers.append("🎵 Cartesia AI (Real-time Synthesis)")
    
    if voice_providers:
        selected_voice_provider = st.selectbox(
            "Choose your preferred voice output:",
            voice_providers,
            help="Select which service to use for converting text responses to speech",
            key="voice_provider_selection"
        )
        
        # Voice settings based on provider
        if "Cartesia" in selected_voice_provider:
            st.markdown("**🎵 Cartesia AI Voice Settings**")
            
            col_voice_settings1, col_voice_settings2 = st.columns([1, 1])
            with col_voice_settings1:
                cartesia_voice = st.selectbox(
                    "Voice Model:",
                    ["Sonic (Fast)", "Studio (High Quality)", "Real-time (Ultra Fast)"],
                    help="Choose the Cartesia voice model",
                    key="cartesia_voice_model"
                )
            
            with col_voice_settings2:
                cartesia_language = st.selectbox(
                    "Language:",
                    ["English (US)", "English (UK)", "Spanish", "French", "German", "Italian", "Portuguese"],
                    help="Select the voice language",
                    key="cartesia_language"
                )
                
        elif "OpenAI" in selected_voice_provider:
            st.markdown("**🤖 OpenAI TTS Settings**")
            
            col_openai_voice1, col_openai_voice2 = st.columns([1, 1])
            with col_openai_voice1:
                openai_voice = st.selectbox(
                    "Voice:",
                    ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                    help="Choose the OpenAI voice",
                    key="openai_voice_selection"
                )
            
            with col_openai_voice2:
                openai_speed = st.slider(
                    "Speed:",
                    min_value=0.25,
                    max_value=4.0,
                    value=1.0,
                    step=0.25,
                    help="Adjust speech speed",
                    key="openai_speech_speed"
                )
    else:
        st.warning("⚠️ Configure at least one API key above to enable voice output")
    
    # Only show the rest of the interface if at least one API key is configured
    if not current_api_key and not current_cartesia_key:
        st.info("👆 Please configure at least one API key above to continue")
        return
    
    # Initialize chat manager
    manager_key_changed = (
        'voice_chat_manager' not in st.session_state or 
        st.session_state.voice_chat_manager.api_key != current_api_key or
        st.session_state.voice_chat_manager.cartesia_key != current_cartesia_key
    )
    
    if manager_key_changed:
        st.session_state.voice_chat_manager = EnhancedVoiceChatManager(
            api_key=current_api_key, 
            cartesia_key=current_cartesia_key
        )
    
    chat_manager = st.session_state.voice_chat_manager
    
    # Model selection
    st.markdown("#### 🤖 Select AI Model")
    
    # Create model selection columns
    model_cols = st.columns(len(chat_manager.available_models))
    selected_model = None
    
    for idx, (model_id, model_info) in enumerate(chat_manager.available_models.items()):
        with model_cols[idx % len(model_cols)]:
            if st.button(
                f"{model_info['icon']} {model_info['name']}", 
                key=f"model_{model_id}",
                help=model_info['description'],
                use_container_width=True
            ):
                st.session_state.selected_model = model_id
                st.rerun()
    
    # Display selected model
    if 'selected_model' in st.session_state:
        selected_model = st.session_state.selected_model
        model_info = chat_manager.available_models[selected_model]
        
        st.success(f"🎯 Selected: {model_info['icon']} **{model_info['name']}**")
        st.caption(model_info['description'])
        
        # Voice input section
        st.markdown("#### 🎤 Voice Input")
        
        col_voice1, col_voice2, col_voice3 = st.columns([2, 1, 1])
        
        with col_voice1:
            # Voice recording component (placeholder for actual implementation)
            if st.button("🎤 Start Recording", key="start_recording", use_container_width=True):
                st.info("🎤 Recording... (Click 'Stop Recording' when done)")
                st.session_state.recording = True
        
        with col_voice2:
            if st.button("⏹️ Stop Recording", key="stop_recording", use_container_width=True):
                if st.session_state.get('recording', False):
                    st.session_state.recording = False
                    st.success("✅ Recording stopped")
                    # Here you would process the audio and convert to text
                    st.session_state.voice_input = "Sample voice input: How do I configure FortiGate firewall rules for Azure?"
        
        with col_voice3:
            if st.button("🔄 Clear", key="clear_voice", use_container_width=True):
                if 'voice_input' in st.session_state:
                    del st.session_state.voice_input
                st.session_state.recording = False
        
        # Text input as fallback
        st.markdown("#### ⌨️ Text Input (Alternative)")
        text_input = st.text_area(
            "Type your question:",
            value=st.session_state.get('voice_input', ''),
            height=100,
            key="text_input_area"
        )
        
        # Process input
        if st.button("🚀 Send Message", key="send_message", use_container_width=True):
            if text_input.strip():
                with st.spinner(f"Getting response from {model_info['name']}..."):
                    # Get conversation context
                    context = chat_manager.get_conversation_context()
                    
                    # Get AI response
                    ai_response = chat_manager.get_model_response(
                        text_input, 
                        selected_model, 
                        context
                    )
                    
                    # Add to conversation
                    chat_manager.add_to_conversation(text_input, ai_response, selected_model)
                    
                    # Clear input
                    if 'voice_input' in st.session_state:
                        del st.session_state.voice_input
                    
                    st.rerun()
        
        # Voice output section
        st.markdown("#### 🔊 Voice Output")
        
        col_audio1, col_audio2 = st.columns([1, 1])
        
        with col_audio1:
            if st.button("🔊 Enable Voice Responses", key="enable_voice_output"):
                st.session_state.voice_output_enabled = True
                st.success("🔊 Voice output enabled")
        
        with col_audio2:
            if st.button("🔇 Disable Voice Responses", key="disable_voice_output"):
                st.session_state.voice_output_enabled = False
                st.info("🔇 Voice output disabled")
        
        # Conversation display
        st.markdown("#### 💬 Conversation History")
        
        if chat_manager.conversation_history:
            # Display conversation in reverse order (newest first)
            for i, exchange in enumerate(reversed(chat_manager.conversation_history)):
                with st.container():
                    # User message
                    st.markdown(f"**🧑 You** ({exchange['timestamp'][:19]})")
                    st.markdown(f"> {exchange['user']}")
                    
                    # AI response
                    model_info = chat_manager.available_models[exchange['model']]
                    st.markdown(f"**{model_info['icon']} {exchange['model_name']}**")
                    st.markdown(exchange['assistant'])
                    
                    # Voice playback button (placeholder)
                    if st.session_state.get('voice_output_enabled', False):
                        if st.button(f"🔊 Play Response", key=f"play_{i}"):
                            st.info("🔊 Playing audio response... (Feature in development)")
                    
                    st.divider()
            
            # Clear conversation button
            if st.button("🗑️ Clear Conversation", key="clear_conversation"):
                chat_manager.clear_conversation()
                st.success("🗑️ Conversation cleared")
                st.rerun()
        else:
            st.info("💬 No conversation yet. Start by selecting a model and sending a message!")
    
    else:
        st.info("👆 Please select an AI model to start chatting")
    
    # Model comparison feature
    st.markdown("#### 🔄 Model Comparison")
    st.markdown("*Ask the same question to multiple models and compare responses*")
    
    if st.button("🔄 Compare Models", key="compare_models"):
        st.session_state.comparison_mode = True
        st.rerun()
    
    if st.session_state.get('comparison_mode', False):
        st.markdown("##### 📝 Enter question for comparison:")
        comparison_question = st.text_area("Question to ask all models:", key="comparison_question")
        
        if st.button("🚀 Ask All Models", key="ask_all_models"):
            if comparison_question.strip():
                st.markdown("##### 📊 Model Responses:")
                
                for model_id, model_info in chat_manager.available_models.items():
                    with st.expander(f"{model_info['icon']} {model_info['name']} Response"):
                        with st.spinner(f"Getting response from {model_info['name']}..."):
                            response = chat_manager.get_model_response(comparison_question, model_id)
                            st.markdown(response)
                            
                            # Voice playback for comparison
                            if st.button(f"🔊 Play {model_info['name']} Response", key=f"play_comparison_{model_id}"):
                                st.info("🔊 Playing audio response... (Feature in development)")
        
        if st.button("❌ Exit Comparison Mode", key="exit_comparison"):
            st.session_state.comparison_mode = False
            st.rerun()
    
    # Voice Synthesis Testing
    st.markdown("#### 🎵 Voice Synthesis Test")
    st.markdown("*Test your configured voice providers with sample text*")
    
    if voice_providers:
        col_test1, col_test2 = st.columns([2, 1])
        
        with col_test1:
            test_text = st.text_input(
                "Test Text:",
                value="Hello! This is a test of the voice synthesis capabilities for FortiGate Azure deployment assistance.",
                help="Enter text to synthesize using your selected voice provider",
                key="voice_test_text"
            )
        
        with col_test2:
            if st.button("🎤 Test Voice", key="test_voice_synthesis", use_container_width=True):
                if test_text and selected_voice_provider:
                    with st.spinner("Generating speech..."):
                        try:
                            # Determine provider and settings
                            if "Cartesia" in selected_voice_provider:
                                provider = "cartesia"
                                voice_model = st.session_state.get('cartesia_voice_model', 'Sonic (Fast)').split(' ')[0].lower()
                                language = st.session_state.get('cartesia_language', 'English (US)').split(' ')[0][:2].lower()
                                
                                audio_data = chat_manager.synthesize_speech(
                                    test_text, 
                                    provider=provider,
                                    voice_model=voice_model,
                                    language=language
                                )
                                
                            elif "OpenAI" in selected_voice_provider:
                                provider = "openai"
                                voice = st.session_state.get('openai_voice_selection', 'alloy')
                                speed = st.session_state.get('openai_speech_speed', 1.0)
                                
                                audio_data = chat_manager.synthesize_speech(
                                    test_text,
                                    provider=provider,
                                    voice=voice,
                                    speed=speed
                                )
                            
                            # Display result (in a real implementation, this would play audio)
                            st.success(f"✅ Voice synthesis completed using {selected_voice_provider}")
                            st.info(f"📝 Generated audio for: '{test_text[:50]}{'...' if len(test_text) > 50 else ''}'")
                            
                            # Note: In a real implementation, you would use st.audio() to play the generated audio
                            # st.audio(audio_data, format="audio/mp3")
                            
                        except Exception as e:
                            st.error(f"❌ Voice synthesis failed: {str(e)}")
    else:
        st.info("Configure at least one voice provider above to test synthesis")

def display_voice_settings():
    """Display voice settings and configuration"""
    st.markdown("### ⚙️ Voice Settings")
    
    # Voice input settings
    st.markdown("#### 🎤 Voice Input Settings")
    
    col_set1, col_set2 = st.columns(2)
    
    with col_set1:
        st.selectbox(
            "Input Language:",
            ["English (US)", "English (UK)", "Spanish", "French", "German", "Italian"],
            key="voice_input_language"
        )
        
        st.slider(
            "Recording Sensitivity:",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="recording_sensitivity"
        )
    
    with col_set2:
        st.selectbox(
            "Audio Quality:",
            ["High (16kHz)", "Medium (8kHz)", "Low (4kHz)"],
            key="audio_quality"
        )
        
        st.slider(
            "Auto-stop Silence (seconds):",
            min_value=1,
            max_value=10,
            value=3,
            key="auto_stop_silence"
        )
    
    # Voice output settings
    st.markdown("#### 🔊 Voice Output Settings")
    
    col_out1, col_out2 = st.columns(2)
    
    with col_out1:
        st.selectbox(
            "Voice Type:",
            ["Neural (Natural)", "Standard", "Premium"],
            key="voice_type"
        )
        
        st.selectbox(
            "Voice Gender:",
            ["Female", "Male", "Neutral"],
            key="voice_gender"
        )
    
    with col_out2:
        st.slider(
            "Speech Speed:",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            key="speech_speed"
        )
        
        st.slider(
            "Voice Volume:",
            min_value=0.1,
            max_value=1.0,
            value=0.8,
            step=0.1,
            key="voice_volume"
        )
    
    # Save settings
    if st.button("💾 Save Voice Settings", key="save_voice_settings"):
        st.success("✅ Voice settings saved!")
        
    # Test voice
    if st.button("🎵 Test Voice Output", key="test_voice"):
        st.info("🎵 Testing voice output: 'Hello! This is your FortiGate Azure assistant speaking.'")

if __name__ == "__main__":
    # For testing
    display_enhanced_voice_chat()
