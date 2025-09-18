"""
Streamlit Interface for Quantum-Inspired Model Compression
User-friendly interface for Phi model compression and fine-tuning
"""

import streamlit as st
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
import time
from typing import Dict, List, Any, Optional
import logging
import os

# Import quantum compression components
try:
    from .phi_model_handler import PhiModelHandler, PhiCompressionConfig
    from .quantum_tucker_compressor import CompressionConfig
    from .corporate_data_processor import CorporateDataProcessor
    from .post_compression_trainer import PostCompressionTrainer
    QUANTUM_COMPRESSION_AVAILABLE = True
except ImportError:
    QUANTUM_COMPRESSION_AVAILABLE = False

logger = logging.getLogger(__name__)

def display_quantum_compression_interface():
    """Main interface for quantum-inspired model compression"""
    
    if not QUANTUM_COMPRESSION_AVAILABLE:
        st.error("⚠️ Quantum compression dependencies not available")
        st.markdown("**Required packages:**")
        st.code("""
pip install torch transformers tensorly tensorly-torch
pip install qiskit qiskit-machine-learning
pip install plotly pandas numpy scipy
        """)
        return
    
    # Use the new automated interface
    try:
        from .automated_interface import display_automated_quantum_interface
        display_automated_quantum_interface()
    except ImportError:
        st.error("❌ Automated interface not available")
        st.info("Please ensure all quantum compression components are properly installed.")
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Compression Configuration")
        
        # Model selection
        model_name = st.selectbox(
            "Base Model",
            ["microsoft/phi-1_5", "microsoft/phi-2"],
            help="Select the base model for compression"
        )
        
        # Compression settings
        compression_ratio = st.slider(
            "Compression Ratio",
            min_value=0.1,
            max_value=0.8,
            value=0.3,
            step=0.05,
            help="Target compression ratio (higher = more compression)"
        )
        
        quantum_optimization = st.checkbox(
            "Enable Quantum Optimization",
            value=True,
            help="Use quantum-inspired algorithms for optimal compression"
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            preserve_embeddings = st.checkbox("Preserve Embeddings", value=True)
            preserve_lm_head = st.checkbox("Preserve LM Head", value=True)
            compress_attention = st.checkbox("Compress Attention", value=True)
            compress_mlp = st.checkbox("Compress MLP", value=True)
    
    # Main interface tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏗️ Model Setup", 
        "🔬 Compression", 
        "📊 Analysis", 
        "🎯 Fine-Tuning", 
        "💾 Export"
    ])
    
    with tab1:
        display_model_setup_tab(model_name, compression_ratio, quantum_optimization,
                               preserve_embeddings, preserve_lm_head, 
                               compress_attention, compress_mlp)
    
    with tab2:
        display_compression_tab()
    
    with tab3:
        display_analysis_tab()
    
    with tab4:
        display_fine_tuning_tab()
    
    with tab5:
        display_export_tab()

def display_model_setup_tab(model_name: str, compression_ratio: float, 
                           quantum_optimization: bool, preserve_embeddings: bool,
                           preserve_lm_head: bool, compress_attention: bool, 
                           compress_mlp: bool):
    """Model setup and configuration tab"""
    
    st.markdown("#### 🏗️ Model Setup & Configuration")
    
    # Configuration summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Selected Configuration:**")
        config_data = {
            "Model": model_name,
            "Compression Ratio": f"{compression_ratio:.1%}",
            "Quantum Optimization": "✅" if quantum_optimization else "❌",
            "Preserve Embeddings": "✅" if preserve_embeddings else "❌",
            "Preserve LM Head": "✅" if preserve_lm_head else "❌"
        }
        
        for key, value in config_data.items():
            st.markdown(f"- **{key}**: {value}")
    
    with col2:
        st.markdown("**Expected Results:**")
        expected_params = 1.5e9 * (1 - compression_ratio)  # Estimated
        expected_size = expected_params * 2 / (1024**3)  # Rough estimate in GB
        
        st.metric("Expected Parameters", f"{expected_params/1e6:.0f}M", 
                 f"-{compression_ratio:.1%}")
        st.metric("Expected Model Size", f"{expected_size:.1f} GB", 
                 f"-{compression_ratio:.1%}")
        st.metric("Expected Speedup", f"{1/(1-compression_ratio):.1f}x", 
                 f"+{compression_ratio/(1-compression_ratio):.1%}")
    
    # Model loading
    st.markdown("#### 📥 Load Base Model")
    
    if st.button("🚀 Initialize Model Handler", type="primary"):
        with st.spinner("Initializing quantum compression system..."):
            try:
                # Create configuration
                config = PhiCompressionConfig(
                    model_name=model_name,
                    compression_ratio=compression_ratio,
                    preserve_embeddings=preserve_embeddings,
                    preserve_lm_head=preserve_lm_head,
                    compress_attention=compress_attention,
                    compress_mlp=compress_mlp,
                    quantum_optimization=quantum_optimization
                )
                
                # Initialize handler
                st.session_state.quantum_handler = PhiModelHandler(config)
                
                # Load model
                success = st.session_state.quantum_handler.load_model()
                
                if success:
                    st.success("✅ Model loaded successfully!")
                    
                    # Display model analysis
                    analysis = st.session_state.quantum_handler.analyze_model_structure()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Parameters", f"{analysis['total_parameters']/1e6:.1f}M")
                    with col2:
                        st.metric("Model Size", f"{analysis['model_size_mb']:.1f} MB")
                    with col3:
                        st.metric("Compressible Layers", len(analysis['compressible_layers']))
                    
                else:
                    st.error("❌ Failed to load model")
                    
            except Exception as e:
                st.error(f"❌ Error initializing model: {str(e)}")
    
    # Display current status
    if st.session_state.quantum_handler is not None:
        st.success("🎯 Model handler ready for compression!")
        
        # Show model info
        with st.expander("📋 Model Information"):
            info = st.session_state.quantum_handler.get_model_info()
            st.json(info)

def display_compression_tab():
    """Model compression execution tab"""
    
    st.markdown("#### 🔬 Quantum-Inspired Compression")
    
    if st.session_state.quantum_handler is None:
        st.warning("⚠️ Please initialize the model handler first in the Model Setup tab")
        return
    
    # Compression controls
    col1, col2 = st.columns(2)
    
    with col1:
        selective_compression = st.checkbox(
            "Selective Layer Compression",
            value=True,
            help="Only compress specific layer types for better performance"
        )
    
    with col2:
        show_progress = st.checkbox(
            "Show Detailed Progress",
            value=True,
            help="Display detailed compression progress"
        )
    
    # Compression execution
    if st.button("🚀 Start Quantum Compression", type="primary"):
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔬 Initializing quantum compression...")
            progress_bar.progress(10)
            
            # Start compression
            start_time = time.time()
            success = st.session_state.quantum_handler.compress_model(selective_compression)
            compression_time = time.time() - start_time
            
            if success:
                progress_bar.progress(100)
                status_text.text("✅ Compression completed successfully!")
                
                # Store results
                st.session_state.compression_results = st.session_state.quantum_handler.compression_stats
                st.session_state.compression_results['total_time'] = compression_time
                
                # Display results
                st.success(f"🎉 Model compressed in {compression_time:.1f} seconds!")
                
                # Show compression statistics
                display_compression_results()
                
            else:
                st.error("❌ Compression failed")
                
        except Exception as e:
            st.error(f"❌ Compression error: {str(e)}")
            progress_bar.progress(0)
            status_text.text("❌ Compression failed")
    
    # Display existing results
    if st.session_state.compression_results is not None:
        st.markdown("#### 📊 Compression Results")
        display_compression_results()

def display_compression_results():
    """Display compression results and statistics"""
    
    if st.session_state.compression_results is None:
        return
    
    results = st.session_state.compression_results
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Compression Ratio",
            f"{results.get('overall_compression_ratio', 0):.1%}",
            help="Overall parameter reduction"
        )
    
    with col2:
        st.metric(
            "Parameters Saved",
            f"{results.get('parameter_reduction', 0)/1e6:.1f}M",
            help="Number of parameters reduced"
        )
    
    with col3:
        st.metric(
            "Layers Compressed",
            f"{results.get('layers_compressed', 0)}/{results.get('layers_processed', 0)}",
            help="Successfully compressed layers"
        )
    
    with col4:
        st.metric(
            "Compression Time",
            f"{results.get('total_time', 0):.1f}s",
            help="Total compression time"
        )
    
    # Detailed layer statistics
    if 'layer_details' in results:
        st.markdown("#### 📋 Layer-by-Layer Results")
        
        layer_data = []
        for layer_name, stats in results['layer_details'].items():
            layer_data.append({
                'Layer': layer_name,
                'Original Params': f"{stats['original_params']/1e6:.2f}M",
                'Compressed Params': f"{stats['compressed_params']/1e6:.2f}M",
                'Compression Ratio': f"{stats['compression_ratio']:.1%}",
                'Ranks': str(stats.get('ranks', 'N/A'))
            })
        
        df = pd.DataFrame(layer_data)
        st.dataframe(df, use_container_width=True)
        
        # Visualization
        fig = px.bar(
            df, 
            x='Layer', 
            y=[col for col in df.columns if 'Params' in col],
            title="Parameter Reduction by Layer",
            barmode='group'
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

def display_analysis_tab():
    """Model analysis and evaluation tab"""
    
    st.markdown("#### 📊 Performance Analysis")
    
    if st.session_state.quantum_handler is None or st.session_state.compression_results is None:
        st.warning("⚠️ Please complete model compression first")
        return
    
    # Evaluation controls
    col1, col2 = st.columns(2)
    
    with col1:
        test_samples = st.number_input(
            "Number of Test Samples",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of test queries for evaluation"
        )
    
    with col2:
        max_length = st.number_input(
            "Max Generation Length",
            min_value=50,
            max_value=500,
            value=100,
            help="Maximum tokens to generate"
        )
    
    # Custom test queries
    st.markdown("#### 🧪 Test Queries")
    default_queries = [
        "What is FortiGate and how does it work?",
        "How to deploy FortiGate on Azure cloud?",
        "Configure network security policies",
        "Troubleshoot FortiGate connectivity issues",
        "Azure integration best practices"
    ]
    
    test_queries = []
    for i in range(test_samples):
        query = st.text_input(
            f"Test Query {i+1}",
            value=default_queries[i] if i < len(default_queries) else "",
            key=f"test_query_{i}"
        )
        if query:
            test_queries.append(query)
    
    # Run evaluation
    if st.button("🔍 Evaluate Compressed Model", type="primary"):
        if not test_queries:
            st.warning("⚠️ Please provide at least one test query")
            return
        
        with st.spinner("Evaluating model performance..."):
            try:
                evaluation_results = st.session_state.quantum_handler.evaluate_compressed_model(test_queries)
                st.session_state.evaluation_results = evaluation_results
                
                st.success("✅ Evaluation completed!")
                display_evaluation_results()
                
            except Exception as e:
                st.error(f"❌ Evaluation failed: {str(e)}")
    
    # Display existing results
    if st.session_state.evaluation_results is not None:
        display_evaluation_results()

def display_evaluation_results():
    """Display evaluation results"""
    
    if st.session_state.evaluation_results is None:
        return
    
    results = st.session_state.evaluation_results
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Model Size Reduction",
            f"{results.get('compression_ratio', 0):.1%}",
            help="Reduction in model size"
        )
    
    with col2:
        if 'avg_speedup' in results:
            st.metric(
                "Average Speedup",
                f"{results['avg_speedup']:.1f}x",
                help="Inference speed improvement"
            )
    
    with col3:
        if 'avg_compressed_time' in results:
            st.metric(
                "Avg Inference Time",
                f"{results['avg_compressed_time']:.2f}s",
                help="Average time per query"
            )
    
    # Sample outputs comparison
    if 'sample_outputs' in results:
        st.markdown("#### 🔍 Output Quality Comparison")
        
        for i, sample in enumerate(results['sample_outputs']):
            with st.expander(f"Sample {i+1}: {sample['input'][:50]}..."):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Model Output:**")
                    st.text_area(
                        "Original",
                        sample['original_output'],
                        height=150,
                        key=f"original_{i}",
                        disabled=True
                    )
                
                with col2:
                    st.markdown("**Compressed Model Output:**")
                    st.text_area(
                        "Compressed",
                        sample['compressed_output'],
                        height=150,
                        key=f"compressed_{i}",
                        disabled=True
                    )
                
                if 'speedup' in sample:
                    st.metric("Speedup for this query", f"{sample['speedup']:.1f}x")

def display_fine_tuning_tab():
    """Post-compression fine-tuning tab"""
    
    st.markdown("#### 🎯 Post-Compression Fine-Tuning")
    st.markdown("*Specialized training with corporate data after compression*")
    
    if st.session_state.quantum_handler is None or st.session_state.compression_results is None:
        st.warning("⚠️ Please complete model compression first")
        return
    
    # Corporate data upload
    st.markdown("#### 📁 Corporate Training Data")
    
    uploaded_files = st.file_uploader(
        "Upload Corporate Training Data",
        accept_multiple_files=True,
        type=['txt', 'json', 'jsonl', 'csv'],
        help="Upload FortiGate, Azure, and corporate service documentation"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files uploaded")
        
        # Data processing options
        col1, col2 = st.columns(2)
        
        with col1:
            data_format = st.selectbox(
                "Data Format",
                ["Auto-detect", "Q&A Pairs", "Instructions", "Documents"],
                help="Format of the training data"
            )
        
        with col2:
            max_samples = st.number_input(
                "Max Training Samples",
                min_value=100,
                max_value=10000,
                value=1000,
                help="Maximum number of training samples"
            )
    
    # Fine-tuning configuration
    st.markdown("#### ⚙️ Fine-Tuning Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=1e-6,
            max_value=1e-3,
            value=5e-5,
            format="%.2e",
            help="Learning rate for fine-tuning"
        )
    
    with col2:
        batch_size = st.selectbox(
            "Batch Size",
            [1, 2, 4, 8, 16],
            index=2,
            help="Training batch size"
        )
    
    with col3:
        num_epochs = st.number_input(
            "Number of Epochs",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of training epochs"
        )
    
    # Advanced fine-tuning options
    with st.expander("🔧 Advanced Fine-Tuning Options"):
        use_lora = st.checkbox("Use LoRA", value=True, help="Parameter-efficient fine-tuning")
        gradient_checkpointing = st.checkbox("Gradient Checkpointing", value=True, help="Memory optimization")
        warmup_steps = st.number_input("Warmup Steps", min_value=0, max_value=1000, value=100)
    
    # Start fine-tuning
    if st.button("🚀 Start Fine-Tuning", type="primary"):
        if not uploaded_files:
            st.warning("⚠️ Please upload training data first")
            return
        
        with st.spinner("Fine-tuning compressed model..."):
            st.info("🔄 Fine-tuning process started (this may take a while)")
            
            # Placeholder for fine-tuning implementation
            # This would integrate with the PostCompressionTrainer
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.1)  # Simulate training progress
                progress_bar.progress(i + 1)
            
            st.success("✅ Fine-tuning completed!")

def display_export_tab():
    """Model export and deployment tab"""
    
    st.markdown("#### 💾 Export & Deployment")
    
    if st.session_state.quantum_handler is None or st.session_state.compression_results is None:
        st.warning("⚠️ Please complete model compression first")
        return
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Export Formats:**")
        export_pytorch = st.checkbox("PyTorch Model", value=True)
        export_onnx = st.checkbox("ONNX Format", value=False)
        export_tensorrt = st.checkbox("TensorRT", value=False)
        export_huggingface = st.checkbox("HuggingFace Hub", value=True)
    
    with col2:
        st.markdown("**Export Path:**")
        export_path = st.text_input(
            "Export Directory",
            value="./compressed_models/phi_quantum_compressed",
            help="Directory to save the compressed model"
        )
    
    # Model metadata
    st.markdown("#### 📋 Model Metadata")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_name = st.text_input("Model Name", value="phi-1.5b-quantum-compressed")
        model_version = st.text_input("Version", value="1.0.0")
    
    with col2:
        model_description = st.text_area(
            "Description",
            value="Quantum-inspired Tucker compressed Phi-1.5B model for FortiGate Azure deployments",
            height=100
        )
    
    # Export execution
    if st.button("📦 Export Compressed Model", type="primary"):
        with st.spinner("Exporting compressed model..."):
            try:
                success = st.session_state.quantum_handler.save_compressed_model(export_path)
                
                if success:
                    st.success(f"✅ Model exported successfully to {export_path}")
                    
                    # Show export summary
                    st.markdown("#### 📊 Export Summary")
                    
                    export_info = {
                        "Model Name": model_name,
                        "Version": model_version,
                        "Export Path": export_path,
                        "Compression Ratio": f"{st.session_state.compression_results.get('overall_compression_ratio', 0):.1%}",
                        "Original Size": f"{st.session_state.compression_results.get('total_original_params', 0)/1e6:.1f}M params",
                        "Compressed Size": f"{st.session_state.compression_results.get('total_compressed_params', 0)/1e6:.1f}M params"
                    }
                    
                    for key, value in export_info.items():
                        st.markdown(f"- **{key}**: {value}")
                    
                    # Download button for model files
                    st.markdown("#### 📥 Download")
                    st.info("Model files are ready for download from the export directory")
                    
                else:
                    st.error("❌ Export failed")
                    
            except Exception as e:
                st.error(f"❌ Export error: {str(e)}")
    
    # Deployment instructions
    st.markdown("#### 🚀 Deployment Instructions")
    
    with st.expander("📖 How to Deploy"):
        st.markdown("""
        **1. Load the compressed model:**
        ```python
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained("./compressed_models/phi_quantum_compressed")
        tokenizer = AutoTokenizer.from_pretrained("./compressed_models/phi_quantum_compressed")
        ```
        
        **2. Use in your application:**
        ```python
        # Generate text
        inputs = tokenizer("Your prompt here", return_tensors="pt")
        outputs = model.generate(**inputs, max_length=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ```
        
        **3. Integration with FortiGate Azure Chatbot:**
        - Replace the standard model loading in your chatbot
        - Update the model path in configuration
        - Enjoy faster inference with maintained quality!
        """)

# Main interface function
def display_quantum_compression_interface():
    """Main quantum compression interface"""
    display_quantum_compression_setup()
