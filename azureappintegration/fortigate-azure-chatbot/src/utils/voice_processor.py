"""
Voice Processing Utilities
Handles speech-to-text and text-to-speech functionality
"""

import streamlit as st
import openai
import io
import base64
import json
import tempfile
import os
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceProcessor:
    """Handles voice input and output processing"""
    
    def __init__(self):
        self.client = openai.OpenAI() if os.getenv('OPENAI_API_KEY') else None
        
    def speech_to_text(self, audio_data: bytes, language: str = "en") -> Optional[str]:
        """Convert speech to text using OpenAI Whisper"""
        if not self.client:
            logger.error("OpenAI client not initialized - API key missing")
            return None
            
        try:
            # Create temporary file for audio data
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file.flush()
                
                # Use OpenAI Whisper for transcription
                with open(temp_file.name, "rb") as audio_file:
                    transcript = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=language
                    )
                
                # Clean up temporary file
                os.unlink(temp_file.name)
                
                return transcript.text
                
        except Exception as e:
            logger.error(f"Error in speech-to-text conversion: {e}")
            return None
    
    def text_to_speech(self, text: str, voice: str = "alloy", speed: float = 1.0) -> Optional[bytes]:
        """Convert text to speech using OpenAI TTS"""
        if not self.client:
            logger.error("OpenAI client not initialized - API key missing")
            return None
            
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=voice,  # alloy, echo, fable, onyx, nova, shimmer
                input=text,
                speed=speed
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error in text-to-speech conversion: {e}")
            return None
    
    def get_available_voices(self) -> Dict[str, str]:
        """Get available TTS voices"""
        return {
            "alloy": "Alloy (Neutral)",
            "echo": "Echo (Male)",
            "fable": "Fable (British Male)",
            "onyx": "Onyx (Deep Male)",
            "nova": "Nova (Female)",
            "shimmer": "Shimmer (Soft Female)"
        }

def create_audio_recorder_component():
    """Create a custom audio recorder component using HTML/JavaScript"""
    
    audio_recorder_html = """
    <div id="audio-recorder">
        <style>
            .recorder-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 20px;
                border: 2px dashed #ccc;
                border-radius: 10px;
                margin: 10px 0;
                background-color: #f9f9f9;
            }
            .recorder-button {
                background-color: #ff4b4b;
                color: white;
                border: none;
                border-radius: 50%;
                width: 80px;
                height: 80px;
                font-size: 24px;
                cursor: pointer;
                margin: 10px;
                transition: all 0.3s ease;
            }
            .recorder-button:hover {
                background-color: #ff6b6b;
                transform: scale(1.1);
            }
            .recorder-button.recording {
                background-color: #ff0000;
                animation: pulse 1s infinite;
            }
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.1); }
                100% { transform: scale(1); }
            }
            .status-text {
                margin: 10px 0;
                font-weight: bold;
                color: #333;
            }
            .audio-visualizer {
                width: 200px;
                height: 50px;
                background-color: #000;
                margin: 10px 0;
                border-radius: 5px;
                display: none;
            }
        </style>
        
        <div class="recorder-container">
            <button id="recordButton" class="recorder-button" onclick="toggleRecording()">
                🎤
            </button>
            <div id="status" class="status-text">Click to start recording</div>
            <canvas id="visualizer" class="audio-visualizer"></canvas>
            <audio id="audioPlayback" controls style="display: none;"></audio>
        </div>
    </div>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        let audioContext;
        let analyser;
        let microphone;
        let dataArray;
        let canvas;
        let canvasContext;

        function toggleRecording() {
            if (!isRecording) {
                startRecording();
            } else {
                stopRecording();
            }
        }

        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    
                    // Display audio player
                    const audioPlayback = document.getElementById('audioPlayback');
                    audioPlayback.src = audioUrl;
                    audioPlayback.style.display = 'block';
                    
                    // Convert to base64 and send to Streamlit
                    const reader = new FileReader();
                    reader.onloadend = function() {
                        const base64Audio = reader.result.split(',')[1];
                        window.parent.postMessage({
                            type: 'audio_recorded',
                            data: base64Audio
                        }, '*');
                    };
                    reader.readAsDataURL(audioBlob);
                };
                
                mediaRecorder.start();
                isRecording = true;
                
                // Update UI
                const button = document.getElementById('recordButton');
                const status = document.getElementById('status');
                button.classList.add('recording');
                button.innerHTML = '⏹️';
                status.textContent = 'Recording... Click to stop';
                
                // Start audio visualization
                setupAudioVisualization(stream);
                
            } catch (error) {
                console.error('Error accessing microphone:', error);
                document.getElementById('status').textContent = 'Error: Could not access microphone';
            }
        }

        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                isRecording = false;
                
                // Update UI
                const button = document.getElementById('recordButton');
                const status = document.getElementById('status');
                button.classList.remove('recording');
                button.innerHTML = '🎤';
                status.textContent = 'Processing audio...';
                
                // Stop audio visualization
                if (audioContext) {
                    audioContext.close();
                }
                document.getElementById('visualizer').style.display = 'none';
                
                // Stop all tracks
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }
        }

        function setupAudioVisualization(stream) {
            audioContext = new AudioContext();
            analyser = audioContext.createAnalyser();
            microphone = audioContext.createMediaStreamSource(stream);
            
            analyser.fftSize = 256;
            const bufferLength = analyser.frequencyBinCount;
            dataArray = new Uint8Array(bufferLength);
            
            microphone.connect(analyser);
            
            canvas = document.getElementById('visualizer');
            canvasContext = canvas.getContext('2d');
            canvas.style.display = 'block';
            
            draw();
        }

        function draw() {
            if (!isRecording) return;
            
            requestAnimationFrame(draw);
            
            analyser.getByteFrequencyData(dataArray);
            
            canvasContext.fillStyle = '#000';
            canvasContext.fillRect(0, 0, canvas.width, canvas.height);
            
            const barWidth = (canvas.width / dataArray.length) * 2.5;
            let barHeight;
            let x = 0;
            
            for (let i = 0; i < dataArray.length; i++) {
                barHeight = (dataArray[i] / 255) * canvas.height;
                
                const r = barHeight + 25 * (i / dataArray.length);
                const g = 250 * (i / dataArray.length);
                const b = 50;
                
                canvasContext.fillStyle = `rgb(${r},${g},${b})`;
                canvasContext.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
                
                x += barWidth + 1;
            }
        }
    </script>
    """
    
    return audio_recorder_html

def create_audio_player_component(audio_base64: str):
    """Create an audio player component for TTS output"""
    
    audio_player_html = f"""
    <div id="audio-player">
        <style>
            .player-container {{
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 8px;
                margin: 10px 0;
                background-color: #f8f9fa;
            }}
            .play-button {{
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 50%;
                width: 60px;
                height: 60px;
                font-size: 20px;
                cursor: pointer;
                margin: 10px;
                transition: all 0.3s ease;
            }}
            .play-button:hover {{
                background-color: #218838;
                transform: scale(1.1);
            }}
            .volume-control {{
                margin: 10px 0;
                width: 200px;
            }}
        </style>
        
        <div class="player-container">
            <button id="playButton" class="play-button" onclick="togglePlayback()">
                🔊
            </button>
            <div>AI Response Audio</div>
            <input type="range" id="volumeControl" class="volume-control" 
                   min="0" max="100" value="80" onchange="setVolume(this.value)">
            <audio id="audioElement" preload="auto">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
        </div>
    </div>

    <script>
        const audio = document.getElementById('audioElement');
        const playButton = document.getElementById('playButton');
        
        function togglePlayback() {{
            if (audio.paused) {{
                audio.play();
                playButton.innerHTML = '⏸️';
            }} else {{
                audio.pause();
                playButton.innerHTML = '🔊';
            }}
        }}
        
        function setVolume(value) {{
            audio.volume = value / 100;
        }}
        
        audio.onended = function() {{
            playButton.innerHTML = '🔊';
        }};
        
        // Set initial volume
        audio.volume = 0.8;
    </script>
    """
    
    return audio_player_html

def display_voice_recorder():
    """Display the voice recorder interface"""
    st.markdown("#### 🎤 Voice Recorder")
    
    # Display the audio recorder component
    recorder_html = create_audio_recorder_component()
    st.components.v1.html(recorder_html, height=300)
    
    # Handle recorded audio (this would be implemented with proper message handling)
    if st.button("🔄 Process Last Recording", key="process_recording"):
        st.info("🎤 Processing audio... (Feature in development)")
        # Here you would process the base64 audio data received from JavaScript
        st.session_state.voice_input = "Sample transcription: How do I configure FortiGate for Azure?"

def display_voice_player(text: str, voice_settings: Dict[str, Any] = None):
    """Display the voice player for TTS output"""
    if not text:
        return
        
    st.markdown("#### 🔊 AI Response Audio")
    
    # Initialize voice processor
    if 'voice_processor' not in st.session_state:
        st.session_state.voice_processor = VoiceProcessor()
    
    processor = st.session_state.voice_processor
    
    # Get voice settings
    voice = voice_settings.get('voice', 'alloy') if voice_settings else 'alloy'
    speed = voice_settings.get('speed', 1.0) if voice_settings else 1.0
    
    # Generate audio
    if st.button("🎵 Generate Audio", key="generate_audio"):
        with st.spinner("🎵 Generating audio response..."):
            audio_data = processor.text_to_speech(text, voice=voice, speed=speed)
            
            if audio_data:
                # Convert to base64 for HTML player
                import base64
                audio_base64 = base64.b64encode(audio_data).decode()
                
                # Display audio player
                player_html = create_audio_player_component(audio_base64)
                st.components.v1.html(player_html, height=200)
                
                st.success("🎵 Audio generated successfully!")
            else:
                st.error("❌ Failed to generate audio. Please check your OpenAI API key.")

if __name__ == "__main__":
    # For testing
    st.title("Voice Processing Test")
    display_voice_recorder()
    display_voice_player("Hello! This is a test of the text-to-speech functionality.")
