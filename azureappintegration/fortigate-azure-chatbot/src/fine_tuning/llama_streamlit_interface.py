"""
Streamlit Interface for Llama 7B Fine-Tuning
Advanced UI for loading models, uploading data, and fine-tuning
"""

import streamlit as st
import json
import os
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any
import time
from datetime import datetime
from datetime import timedelta

try:
    from .llama_fine_tuner import LlamaFineTuner, get_system_requirements, DEPENDENCIES_AVAILABLE
    from .visualization_charts import FineTuningVisualizer, display_visualization_dashboard, get_visualizer
except ImportError:
    try:
        from llama_fine_tuner import LlamaFineTuner, get_system_requirements, DEPENDENCIES_AVAILABLE
        from visualization_charts import FineTuningVisualizer, display_visualization_dashboard, get_visualizer
    except ImportError as e:
        print(f"Warning: Llama fine-tuning dependencies not available: {e}")
        LlamaFineTuner = None
        DEPENDENCIES_AVAILABLE = False
        
        # Fallback visualization classes
        class FineTuningVisualizer:
            def __init__(self):
                pass
            def start_system_monitoring(self):
                pass
            def stop_system_monitoring(self):
                pass
            def add_training_metric(self, metric):
                pass
            def add_performance_metric(self, metric):
                pass
        
        def display_visualization_dashboard(visualizer):
            st.warning("📊 Visualization features require additional dependencies. Install streamlit-echarts for advanced charts.")
        
        def get_visualizer():
            return FineTuningVisualizer()
        
        def get_system_requirements():
            return {
                "minimum_requirements": {
                    "python_version": "3.8+",
                    "ram": "16 GB",
                    "disk_space": "50 GB",
                    "gpu": "CUDA-compatible (recommended)"
                },
                "recommended_requirements": {
                    "python_version": "3.9+",
                    "ram": "32 GB",
                    "disk_space": "100 GB",
                    "gpu": "NVIDIA GPU with 12+ GB VRAM"
                },
                "dependencies": [
                    "torch>=2.0.0", "transformers>=4.35.0", "peft>=0.6.0",
                    "bitsandbytes>=0.41.0", "datasets>=2.14.0", "accelerate>=0.24.0"
                ]
            }

def display_system_requirements():
    """Display system requirements for Llama fine-tuning"""
    st.subheader("🖥️ System Requirements")
    
    requirements = get_system_requirements()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Minimum Requirements:**")
        for key, value in requirements["minimum_requirements"].items():
            st.write(f"• **{key.replace('_', ' ').title()}**: {value}")
    
    with col2:
        st.write("**Recommended Requirements:**")
        for key, value in requirements["recommended_requirements"].items():
            st.write(f"• **{key.replace('_', ' ').title()}**: {value}")
    
    st.write("**Required Dependencies:**")
    deps_text = ", ".join(requirements["dependencies"])
    st.code(f"pip install {deps_text}", language="bash")

def display_dependency_status():
    """Display current dependency and system status"""
    st.subheader("📊 System Status")
    
    if not DEPENDENCIES_AVAILABLE:
        st.error("❌ Required dependencies not installed")
        st.write("Install dependencies:")
        st.code("""
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft bitsandbytes datasets accelerate
pip install huggingface_hub
        """, language="bash")
        return False
    
    if LlamaFineTuner is None:
        st.error("❌ LlamaFineTuner class not available")
        return False
    
    # Check system status
    fine_tuner = LlamaFineTuner()
    status = fine_tuner.check_dependencies()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if status["cuda_available"]:
            st.success(f"✅ GPU Available ({status['gpu_count']} GPUs)")
            for gpu_info in status["gpu_memory"]:
                st.write(f"🔹 {gpu_info}")
        else:
            st.warning("⚠️ No GPU detected - Training will be very slow")
    
    with col2:
        st.write(f"**PyTorch**: {status['torch_version']}")
        st.write(f"**Dependencies**: {'✅ Available' if status['dependencies_available'] else '❌ Missing'}")
    
    with col3:
        if status["huggingface_token"]:
            st.success("✅ HuggingFace Token Set")
        else:
            st.warning("⚠️ HuggingFace Token Missing")
            st.write("Set token for Llama access:")
            st.code("export HUGGINGFACE_TOKEN='your-token'", language="bash")
    
    return status["dependencies_available"] and status["cuda_available"]

def display_model_loader():
    """Display model loading interface"""
    st.subheader("🦙 Load Llama 7B Base Model")
    
    # Model selection
    model_options = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-7b-hf",
        "microsoft/DialoGPT-medium",
        "huggyllama/llama-7b"
    ]
    
    selected_model = st.selectbox(
        "Select Base Model:",
        model_options,
        help="Choose the base Llama model for fine-tuning"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Load Base Model", type="primary"):
            if 'llama_fine_tuner' not in st.session_state:
                st.session_state.llama_fine_tuner = LlamaFineTuner(selected_model)
            
            with st.spinner("Loading Llama model... This may take several minutes..."):
                success = st.session_state.llama_fine_tuner.load_base_model()
                
                if success:
                    st.success("✅ Llama model loaded successfully!")
                    st.session_state.model_loaded = True
                else:
                    st.error("❌ Failed to load model. Check logs for details.")
                    st.session_state.model_loaded = False
    
    with col2:
        if st.session_state.get('model_loaded', False):
            st.success("🦙 Model Ready")
            model_info = st.session_state.llama_fine_tuner.get_model_info()
            st.json(model_info)
        else:
            st.info("📥 Model not loaded")

def display_data_upload():
    """Display data upload interface"""
    st.subheader("📁 Upload Training Data")
    
    st.write("Upload your FortiGate Azure training data in various formats:")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose training data files",
        accept_multiple_files=True,
        type=['json', 'jsonl', 'txt', 'csv'],
        help="Supported formats: JSON, JSONL, TXT, CSV"
    )
    
    if uploaded_files:
        st.write(f"📊 **{len(uploaded_files)} files uploaded:**")
        
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size} bytes)")
        
        # Data format examples
        with st.expander("📋 Data Format Examples"):
            st.write("**JSON Format (OpenAI style):**")
            st.code("""
{
  "messages": [
    {"role": "system", "content": "You are a FortiGate expert..."},
    {"role": "user", "content": "How to deploy FortiGate HA?"},
    {"role": "assistant", "content": "To deploy FortiGate HA..."}
  ]
}
            """, language="json")
            
            st.write("**JSONL Format (Alpaca style):**")
            st.code("""
{"instruction": "Explain FortiGate HA", "input": "", "output": "FortiGate HA..."}
{"instruction": "Configure Azure networking", "input": "VNET setup", "output": "Steps..."}
            """, language="json")
        
        # Process uploaded data
        if st.button("🔄 Process Training Data"):
            if 'llama_fine_tuner' not in st.session_state:
                st.error("❌ Please load the base model first")
                return
            
            with st.spinner("Processing uploaded data..."):
                success = st.session_state.llama_fine_tuner.process_uploaded_data(uploaded_files)
                
                if success:
                    st.success(f"✅ Processed {len(st.session_state.llama_fine_tuner.training_data)} training examples")
                    st.session_state.data_processed = True
                    
                    # Show data preview
                    if st.session_state.llama_fine_tuner.training_data:
                        st.write("**Data Preview:**")
                        preview_data = st.session_state.llama_fine_tuner.training_data[:3]
                        for i, example in enumerate(preview_data):
                            with st.expander(f"Example {i+1}"):
                                st.json(example)
                else:
                    st.error("❌ Failed to process training data")
                    st.session_state.data_processed = False

def display_fine_tuning_controls():
    """Display fine-tuning configuration and controls"""
    st.subheader("🔥 Fine-Tuning Configuration")
    
    if not st.session_state.get('model_loaded', False):
        st.warning("⚠️ Please load the base model first")
        return
    
    if not st.session_state.get('data_processed', False):
        st.warning("⚠️ Please upload and process training data first")
        return
    
    # Training parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        epochs = st.slider("Training Epochs", 1, 10, 3, help="Number of training epochs")
        learning_rate = st.select_slider(
            "Learning Rate", 
            options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4],
            value=2e-4,
            format_func=lambda x: f"{x:.0e}"
        )
    
    with col2:
        batch_size = st.selectbox("Batch Size", [1, 2, 4, 8], index=2)
        gradient_accumulation = st.selectbox("Gradient Accumulation Steps", [2, 4, 8, 16], index=1)
    
    with col3:
        lora_rank = st.slider("LoRA Rank", 8, 64, 16, step=8, help="LoRA rank parameter")
        lora_alpha = st.slider("LoRA Alpha", 16, 128, 32, step=16, help="LoRA alpha parameter")
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        warmup_ratio = st.slider("Warmup Ratio", 0.0, 0.1, 0.03, step=0.01)
        weight_decay = st.slider("Weight Decay", 0.0, 0.01, 0.001, step=0.001)
        max_grad_norm = st.slider("Max Gradient Norm", 0.1, 1.0, 0.3, step=0.1)
    
    # Estimated training time and cost
    st.write("**📊 Training Estimates:**")
    training_examples = len(st.session_state.llama_fine_tuner.training_data)
    estimated_time = (training_examples * epochs * batch_size) / 100  # Rough estimate
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Examples", training_examples)
    with col2:
        st.metric("Estimated Time", f"{estimated_time:.1f} min")
    with col3:
        st.metric("GPU Memory", "~12 GB")
    
    # Start fine-tuning
    if st.button("🚀 Start Fine-Tuning", type="primary"):
        if not st.session_state.get('model_loaded') or not st.session_state.get('data_processed'):
            st.error("❌ Please load model and process data first")
            return
        
        # Confirmation dialog
        st.warning("⚠️ Fine-tuning will start intensive GPU training. This process cannot be interrupted safely.")
        
        if st.button("✅ Confirm and Start Training"):
            with st.spinner("🔥 Fine-tuning in progress... This will take several minutes..."):
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Add initial training metrics
                start_time = datetime.now()
                st.session_state.visualizer.add_training_metric({
                    'epoch': 0,
                    'train_loss': 0.0,
                    'val_loss': 0.0,
                    'learning_rate': learning_rate,
                    'timestamp': start_time.strftime('%H:%M:%S')
                })
                
                # Start fine-tuning
                success = st.session_state.llama_fine_tuner.start_fine_tuning(
                    epochs=epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size
                )
                
                if success:
                    progress_bar.progress(100)
                    status_text.success("✅ Fine-tuning completed successfully!")
                    st.session_state.fine_tuning_complete = True
                    
                    # Add completion metrics
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    # Add final performance metrics
                    st.session_state.visualizer.add_performance_metric({
                        'accuracy': 0.85,  # Mock data - replace with actual metrics
                        'f1_score': 0.82,
                        'precision': 0.87,
                        'recall': 0.78,
                        'perplexity': 2.1,
                        'training_time': duration,
                        'timestamp': end_time.strftime('%H:%M:%S')
                    })
                    
                    # Add model comparison data
                    st.session_state.visualizer.add_comparison_data({
                        'model_name': f'Llama-7B-FortiGate-{datetime.now().strftime("%Y%m%d")}',
                        'accuracy': 0.85,
                        'f1_score': 0.82,
                        'training_time': duration,
                        'inference_speed': 95.0,
                        'parameters': '7B',
                        'dataset_size': len(st.session_state.llama_fine_tuner.training_data)
                    })
                    
                    st.balloons()
                    
                    # Show model info
                    model_info = st.session_state.llama_fine_tuner.get_model_info()
                    st.json(model_info)
                else:
                    status_text.error("❌ Fine-tuning failed. Check logs for details.")

def display_model_testing():
    """Display model testing interface"""
    st.subheader("🧪 Test Fine-Tuned Model")
    
    if not st.session_state.get('fine_tuning_complete', False):
        st.info("ℹ️ Complete fine-tuning first to test the model")
        return
    
    # Load fine-tuned model if not already loaded
    if 'fine_tuned_loaded' not in st.session_state:
        with st.spinner("Loading fine-tuned model..."):
            model_path = "models/llama_fine_tuned"
            success = st.session_state.llama_fine_tuner.load_fine_tuned_model(model_path)
            st.session_state.fine_tuned_loaded = success
    
    if not st.session_state.get('fine_tuned_loaded', False):
        st.error("❌ Failed to load fine-tuned model")
        return
    
    st.success("✅ Fine-tuned Llama model ready for testing")
    
    # Test interface
    test_prompt = st.text_area(
        "Enter your FortiGate Azure question:",
        placeholder="How do I configure FortiGate HA with Azure Load Balancer?",
        height=100
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_length = st.slider("Max Response Length", 50, 1000, 512)
    
    with col2:
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7, step=0.1)
    
    if st.button("🎯 Generate Response") and test_prompt:
        with st.spinner("Generating response..."):
            response = st.session_state.llama_fine_tuner.generate_response(
                test_prompt, 
                max_length=max_length
            )
            
            st.write("**🦙 Llama Response:**")
            st.write(response)
            
            # Save response for comparison
            if 'test_responses' not in st.session_state:
                st.session_state.test_responses = []
            
            st.session_state.test_responses.append({
                "prompt": test_prompt,
                "response": response,
                "timestamp": time.time()
            })

def display_llama_fine_tuning_interface():
    """Main Llama fine-tuning interface"""
    st.header("🦙 Llama 7B Fine-Tuning")
    
    # Initialize session state
    if 'llama_fine_tuner' not in st.session_state:
        st.session_state.llama_fine_tuner = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'fine_tuning_complete' not in st.session_state:
        st.session_state.fine_tuning_complete = False
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = get_visualizer()
    
    # Check dependencies first
    if not DEPENDENCIES_AVAILABLE:
        st.error("❌ Required dependencies not available")
        display_system_requirements()
        return
    
    # Main interface tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🖥️ System Check", 
        "🦙 Load Model", 
        "📁 Upload Data", 
        "🔥 Fine-Tune", 
        "🧪 Test Model",
        "📊 Performance Dashboard"
    ])
    
    with tab1:
        display_system_requirements()
        st.divider()
        system_ready = display_dependency_status()
        
        if system_ready:
            st.success("✅ System ready for Llama fine-tuning!")
        else:
            st.warning("⚠️ System requirements not met")
    
    with tab2:
        display_model_loader()
    
    with tab3:
        display_data_upload()
    
    with tab4:
        display_fine_tuning_controls()
    
    with tab5:
        display_model_testing()
    
    with tab6:
        st.markdown("### 📊 Real-Time Performance Dashboard")
        
        # Check if visualization dependencies are available
        try:
            import streamlit_echarts
            viz_available = True
        except ImportError:
            viz_available = False
        
        if not viz_available:
            st.warning("📊 Visualization dependencies not installed")
            col_warn1, col_warn2 = st.columns([1, 2])
            with col_warn1:
                if st.button("🎨 Quick Install", key="quick_install_llama"):
                    with st.spinner("Installing visualization dependencies..."):
                        import subprocess
                        try:
                            result = subprocess.run(
                                ["pip", "install", "streamlit-echarts", "psutil"],
                                capture_output=True,
                                text=True,
                                timeout=120
                            )
                            if result.returncode == 0:
                                st.success("✅ Installed! Please refresh the page.")
                                st.rerun()
                            else:
                                st.error("❌ Installation failed")
                                st.code(result.stderr, language="bash")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            with col_warn2:
                st.info("Or run: `./setup_visualization.sh` in terminal")
            return
        
        # Dashboard controls
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🚀 Start Monitoring"):
                st.session_state.visualizer.start_system_monitoring()
                st.success("System monitoring started!")
        
        with col2:
            if st.button("⏹️ Stop Monitoring"):
                st.session_state.visualizer.stop_system_monitoring()
                st.info("System monitoring stopped.")
        
        with col3:
            auto_refresh = st.checkbox("🔄 Auto Refresh", value=True)
        
        # Display the visualization dashboard
        display_visualization_dashboard(st.session_state.visualizer)
        
        # Auto-refresh functionality
        if auto_refresh:
            time.sleep(2)
            st.rerun()
    
    # Status sidebar
    with st.sidebar:
        st.subheader("🔄 Process Status")
        
        # Model status
        if st.session_state.get('model_loaded', False):
            st.success("✅ Model Loaded")
        else:
            st.error("❌ Model Not Loaded")
        
        # Data status
        if st.session_state.get('data_processed', False):
            st.success("✅ Data Processed")
            if st.session_state.llama_fine_tuner:
                st.write(f"📊 {len(st.session_state.llama_fine_tuner.training_data)} examples")
        else:
            st.error("❌ No Data Processed")
        
        # Fine-tuning status
        if st.session_state.get('fine_tuning_complete', False):
            st.success("✅ Fine-Tuning Complete")
        else:
            st.error("❌ Fine-Tuning Pending")
        
        # Quick actions
        st.divider()
        st.subheader("🚀 Quick Actions")
        
        if st.button("🔄 Reset All"):
            for key in ['llama_fine_tuner', 'model_loaded', 'data_processed', 'fine_tuning_complete']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        if st.button("📋 Show Logs"):
            st.code("Check console for detailed logs", language="bash")

if __name__ == "__main__":
    # Test the interface
    display_llama_fine_tuning_interface()
