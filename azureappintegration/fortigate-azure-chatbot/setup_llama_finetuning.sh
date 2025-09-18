#!/bin/bash

# Llama 7B Fine-Tuning Setup Script
# This script sets up the environment for local Llama fine-tuning

echo "🦙 Setting up Llama 7B Fine-Tuning Environment"
echo "=============================================="

# Check if running in virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: Not in a virtual environment"
    echo "   Consider activating your virtual environment first:"
    echo "   source fortinetvmazure/bin/activate"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "🐍 Python version: $python_version"

if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 0 ]]; then
    echo "❌ Python 3.8+ required. Current version: $python_version"
    exit 1
fi

# Check CUDA availability
echo "🔍 Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    cuda_available=true
else
    echo "⚠️  No NVIDIA GPU detected. Training will be very slow on CPU."
    cuda_available=false
fi

# Install PyTorch with CUDA support
echo "📦 Installing PyTorch with CUDA support..."
if [[ "$cuda_available" == true ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install transformers and related libraries
echo "📦 Installing Transformers and ML libraries..."
pip install transformers>=4.35.0
pip install tokenizers>=0.14.0
pip install accelerate>=0.24.0
pip install datasets>=2.14.0

# Install PEFT for parameter efficient fine-tuning
echo "📦 Installing PEFT (Parameter Efficient Fine-Tuning)..."
pip install peft>=0.6.0

# Install bitsandbytes for quantization
echo "📦 Installing bitsandbytes for quantization..."
if [[ "$cuda_available" == true ]]; then
    pip install bitsandbytes>=0.41.0
else
    echo "⚠️  Skipping bitsandbytes (requires CUDA)"
fi

# Install HuggingFace Hub
echo "📦 Installing HuggingFace Hub..."
pip install huggingface_hub>=0.17.0
pip install safetensors>=0.4.0

# Install additional dependencies
echo "📦 Installing additional dependencies..."
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install scipy>=1.10.0
pip install scikit-learn>=1.3.0

# Optional: Install monitoring tools
echo "📦 Installing monitoring tools..."
pip install wandb>=0.15.0
pip install tensorboard>=2.13.0

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models/llama_fine_tuned
mkdir -p training_data
mkdir -p logs

# Check HuggingFace token
echo "🔑 Checking HuggingFace token..."
if [[ -z "$HUGGINGFACE_TOKEN" ]]; then
    echo "⚠️  HUGGINGFACE_TOKEN not set"
    echo "   To access Llama models, you need a HuggingFace token:"
    echo "   1. Go to https://huggingface.co/settings/tokens"
    echo "   2. Create a new token with 'Read' permissions"
    echo "   3. Set the token: export HUGGINGFACE_TOKEN='your-token'"
    echo "   4. Accept Llama 2 license at: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf"
else
    echo "✅ HUGGINGFACE_TOKEN is set"
fi

# Test installation
echo "🧪 Testing installation..."
python3 -c "
import torch
import transformers
import peft
import datasets
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ PEFT: {peft.__version__}')
print(f'✅ Datasets: {datasets.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo ""
echo "🎉 Llama fine-tuning setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Set your HuggingFace token (if not already set):"
echo "   export HUGGINGFACE_TOKEN='your-token'"
echo ""
echo "2. Accept Llama 2 license:"
echo "   https://huggingface.co/meta-llama/Llama-2-7b-chat-hf"
echo ""
echo "3. Start the Streamlit app:"
echo "   streamlit run src/app.py"
echo ""
echo "4. Go to the 'Fine-Tuned Model' tab and select 'Llama 7B Local Fine-Tuning'"
echo ""
echo "💡 System Requirements Met:"
if [[ "$cuda_available" == true ]]; then
    echo "✅ GPU acceleration available"
else
    echo "⚠️  CPU-only mode (training will be slow)"
fi
echo "✅ All dependencies installed"
echo "✅ Directory structure created"
