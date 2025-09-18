import streamlit as st
from openai import OpenAI
import speech_recognition as sr
import io
import tempfile
import os
from typing import Optional, Tuple
import base64
from audiorecorder import audiorecorder

class VoiceIntegration:
    """
    OpenAI multimodal voice integration for the FortiGate Azure Chatbot.
    Supports speech-to-text and text-to-speech functionality.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the voice integration with OpenAI API key."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        self.recognizer = sr.Recognizer()
        
    def speech_to_text_openai(self, audio_file) -> str:
        """Convert speech to text using OpenAI Whisper API."""
        try:
            if not self.client:
                return "OpenAI API key not configured"
            
            # Use OpenAI Whisper API for transcription
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            return transcript.text
        except Exception as e:
            return f"Error in speech-to-text: {str(e)}"
    
    def text_to_speech_openai(self, text: str, voice: str = "alloy") -> bytes:
        """
        Convert text to speech using OpenAI TTS API.
        
        Args:
            text: Text to convert to speech
            voice: Voice model to use (alloy, echo, fable, onyx, nova, shimmer)
            
        Returns:
            bytes: Audio data
        """
        try:
            if not self.client:
                raise Exception("OpenAI API key not configured")
                
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )
            return response.content
        except Exception as e:
            st.error(f"Error in text-to-speech: {str(e)}")
            return b""
    
    def process_voice_command(self, audio_file) -> Tuple[str, str]:
        """
        Process voice command and return both transcription and AI response.
        
        Args:
            audio_file: Audio file from user
            
        Returns:
            Tuple[str, str]: (transcribed_text, ai_response)
        """
        try:
            # Convert speech to text
            transcribed_text = self.speech_to_text_openai(audio_file)
            
            if transcribed_text.startswith("Error"):
                return transcribed_text, ""
            
            # Generate AI response for FortiGate deployment context
            ai_response = self.generate_fortigate_response(transcribed_text)
            
            return transcribed_text, ai_response
            
        except Exception as e:
            error_msg = f"Error processing voice command: {str(e)}"
            return error_msg, ""
    
    def generate_fortigate_response(self, user_input: str) -> str:
        """
        Generate contextual response for FortiGate Azure deployment.
        
        Args:
            user_input: User's transcribed input
            
        Returns:
            str: AI-generated response
        """
        try:
            if not self.client:
                return "AI response unavailable - API key not configured"
            
            # Create a FortiGate-specific prompt
            system_prompt = """
            You are a FortiGate Azure deployment assistant. Help users deploy FortiGate-VM on Azure using Terraform.
            
            Available deployment options:
            - HA cluster (high availability)
            - Single FortiGate-VM
            - Active-passive cluster
            - Active-active cluster
            - Azure service integrations (Firewall, Load Balancer, Application Gateway, Bastion, VPN Gateway, ExpressRoute, Private Link)
            
            Provide clear, concise guidance for Azure FortiGate deployments.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

def create_audio_player(audio_bytes: bytes, format: str = "audio/mp3") -> str:
    """
    Create an HTML audio player for Streamlit.
    
    Args:
        audio_bytes: Audio data
        format: Audio format
        
    Returns:
        str: HTML audio player
    """
    if not audio_bytes:
        return ""
        
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <audio controls>
        <source src="data:{format};base64,{audio_base64}" type="{format}">
        Your browser does not support the audio element.
    </audio>
    """
    return audio_html

def display_voice_interface():
    """
    Display the voice interaction interface in Streamlit.
    """
    st.subheader("🎤 Voice Interaction")
    
    # Real-time microphone recording section
    st.write("### 🎙️ Real-time Voice Chat")
    st.info("Click the microphone button below to start recording. Click again to stop and process your voice command.")
    
    # Audio recorder widget
    audio = audiorecorder(
        start_prompt="🎤 Start Recording",
        stop_prompt="⏹️ Stop Recording",
        pause_prompt="⏸️ Pause Recording",
        key="voice_recorder"
    )
    
    if len(audio) > 0:
        # Display audio player
        st.audio(audio.export().read())
        
        if st.button("🚀 Process Voice Command", type="primary"):
            voice_integration = VoiceIntegration()
            
            with st.spinner("🔄 Processing your voice command..."):
                # Save audio to temporary file for processing
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    audio.export(tmp_file.name, format="wav")
                    
                    # Process the audio file
                    with open(tmp_file.name, "rb") as audio_file:
                        transcribed_text = voice_integration.speech_to_text_openai(audio_file)
                    
                    # Clean up temporary file
                    os.unlink(tmp_file.name)
            
            # Display transcription
            st.success("✅ Voice command processed!")
            st.write("**🗣️ What you said:**")
            st.info(transcribed_text)
            
            # Generate AI response if transcription was successful
            if transcribed_text and "Error" not in transcribed_text:
                with st.spinner("🤖 Generating AI response..."):
                    ai_response = voice_integration.generate_fortigate_response(transcribed_text)
                
                st.write("**🤖 AI Response:**")
                st.success(ai_response)
                
                # Generate audio response
                if st.button("🔊 Hear AI Response", key="tts_response"):
                    with st.spinner("🎵 Generating audio response..."):
                        audio_bytes = voice_integration.text_to_speech_openai(ai_response)
                    
                    if audio_bytes:
                        st.write("**🎧 Audio Response:**")
                        audio_html = create_audio_player(audio_bytes)
                        st.markdown(audio_html, unsafe_allow_html=True)
                        
                        # Download option
                        st.download_button(
                            label="📥 Download Audio Response",
                            data=audio_bytes,
                            file_name="fortigate_ai_response.mp3",
                            mime="audio/mp3"
                        )
    
    st.divider()
    
    # File upload section (keep as alternative)
    st.write("### 📁 Upload Audio File")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Alternative: Upload Audio File**")
        uploaded_audio = st.file_uploader(
            "Upload audio file", 
            type=['wav', 'mp3', 'm4a', 'ogg'],
            help="Upload an audio file to transcribe"
        )
        
        if uploaded_audio is not None:
            st.audio(uploaded_audio, format='audio/wav')
            
            if st.button("🎯 Process Voice Command"):
                voice_integration = VoiceIntegration()
                
                with st.spinner("Processing voice command..."):
                    transcribed_text, ai_response = voice_integration.process_voice_command(uploaded_audio)
                
                st.success("✅ Voice command processed!")
                
                # Display results
                st.write("**Transcribed Text:**")
                st.info(transcribed_text)
                
                if ai_response:
                    st.write("**AI Response:**")
                    st.success(ai_response)
                    
                    # Generate audio response
                    if st.button("🔊 Generate Audio Response"):
                        with st.spinner("Generating audio response..."):
                            audio_bytes = voice_integration.text_to_speech_openai(ai_response)
                            
                        if audio_bytes:
                            st.write("**Audio Response:**")
                            audio_html = create_audio_player(audio_bytes)
                            st.markdown(audio_html, unsafe_allow_html=True)
    
    with col2:
        st.write("**Text-to-Speech**")
        text_input = st.text_area(
            "Enter text to convert to speech:",
            placeholder="Enter FortiGate deployment instructions...",
            height=100
        )
        
        voice_option = st.selectbox(
            "Select voice:",
            ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            help="Choose the voice for text-to-speech"
        )
        
        if st.button("🎵 Generate Speech") and text_input:
            voice_integration = VoiceIntegration()
            
            with st.spinner("Generating speech..."):
                audio_bytes = voice_integration.text_to_speech_openai(text_input, voice_option)
            
            if audio_bytes:
                st.success("✅ Speech generated!")
                audio_html = create_audio_player(audio_bytes)
                st.markdown(audio_html, unsafe_allow_html=True)
                
                # Option to download audio
                st.download_button(
                    label="📥 Download Audio",
                    data=audio_bytes,
                    file_name="fortigate_response.mp3",
                    mime="audio/mp3"
                )
