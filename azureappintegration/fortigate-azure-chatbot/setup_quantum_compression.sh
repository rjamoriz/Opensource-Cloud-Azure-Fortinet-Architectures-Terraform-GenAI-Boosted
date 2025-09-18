#!/bin/bash

# Quantum-Inspired Model Compression Setup Script
# FortiGate Azure Chatbot - Advanced AI Optimization

echo "🔬 Setting up Quantum-Inspired Model Compression System..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in a virtual environment
check_virtual_env() {
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        print_success "Virtual environment detected: $VIRTUAL_ENV"
    else
        print_warning "No virtual environment detected. Consider using one for isolation."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Check system requirements
check_system_requirements() {
    print_status "Checking system requirements..."
    
    # Check Python version
    python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
    if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 1 ]]; then
        print_success "Python $python_version detected"
    else
        print_error "Python 3.8+ required, found $python_version"
        exit 1
    fi
    
    # Check available memory
    if command -v free &> /dev/null; then
        available_mem=$(free -g | awk '/^Mem:/{print $7}')
        if [[ $available_mem -lt 8 ]]; then
            print_warning "Less than 8GB RAM available. Quantum compression may be slow."
        else
            print_success "Sufficient memory available: ${available_mem}GB"
        fi
    fi
    
    # Check for CUDA
    if command -v nvidia-smi &> /dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        print_success "CUDA GPU detected: $gpu_info"
    else
        print_warning "No CUDA GPU detected. Using CPU for computations."
    fi
}

# Install core dependencies
install_core_dependencies() {
    print_status "Installing core dependencies..."
    
    # Upgrade pip first
    python3 -m pip install --upgrade pip setuptools wheel
    
    # Install PyTorch with CUDA support if available
    if command -v nvidia-smi &> /dev/null; then
        print_status "Installing PyTorch with CUDA support..."
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        print_status "Installing PyTorch CPU version..."
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    if [[ $? -eq 0 ]]; then
        print_success "PyTorch installed successfully"
    else
        print_error "Failed to install PyTorch"
        exit 1
    fi
}

# Install quantum computing libraries
install_quantum_libraries() {
    print_status "Installing quantum computing libraries..."
    
    # Install Qiskit
    python3 -m pip install qiskit qiskit-machine-learning qiskit-algorithms qiskit-optimization
    
    if [[ $? -eq 0 ]]; then
        print_success "Qiskit libraries installed successfully"
    else
        print_warning "Some quantum libraries may have failed to install"
    fi
}

# Install tensor decomposition libraries
install_tensor_libraries() {
    print_status "Installing tensor decomposition libraries..."
    
    python3 -m pip install tensorly tensorly-torch
    
    if [[ $? -eq 0 ]]; then
        print_success "TensorLy libraries installed successfully"
    else
        print_error "Failed to install tensor decomposition libraries"
        exit 1
    fi
}

# Install remaining dependencies
install_remaining_dependencies() {
    print_status "Installing remaining dependencies from requirements..."
    
    # Check if requirements file exists
    if [[ -f "requirements_quantum.txt" ]]; then
        python3 -m pip install -r requirements_quantum.txt
        
        if [[ $? -eq 0 ]]; then
            print_success "All dependencies installed successfully"
        else
            print_warning "Some dependencies may have failed to install"
        fi
    else
        print_error "requirements_quantum.txt not found"
        exit 1
    fi
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Test core imports
    python3 -c "
import torch
import transformers
import tensorly
import qiskit
print('✅ Core libraries imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'TensorLy version: {tensorly.__version__}')
print(f'Qiskit version: {qiskit.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
"
    
    if [[ $? -eq 0 ]]; then
        print_success "Installation verification passed"
    else
        print_error "Installation verification failed"
        exit 1
    fi
}

# Create test script
create_test_script() {
    print_status "Creating test script..."
    
    cat > test_quantum_compression.py << 'EOF'
#!/usr/bin/env python3
"""
Test script for Quantum-Inspired Model Compression
"""

import torch
import numpy as np
from src.quantum_compression import QUANTUM_COMPRESSION_AVAILABLE

def test_basic_functionality():
    """Test basic quantum compression functionality"""
    print("🔬 Testing Quantum Compression System...")
    
    if not QUANTUM_COMPRESSION_AVAILABLE:
        print("❌ Quantum compression not available")
        return False
    
    try:
        from src.quantum_compression.quantum_tucker_compressor import QuantumTuckerCompressor, CompressionConfig
        from src.quantum_compression.phi_model_handler import PhiModelHandler, PhiCompressionConfig
        
        print("✅ Core modules imported successfully")
        
        # Test compression config
        config = CompressionConfig(compression_ratio=0.3)
        compressor = QuantumTuckerCompressor(config)
        print("✅ Quantum compressor initialized")
        
        # Test Phi handler config
        phi_config = PhiCompressionConfig(compression_ratio=0.3)
        print("✅ Phi model handler config created")
        
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    exit(0 if success else 1)
EOF
    
    chmod +x test_quantum_compression.py
    print_success "Test script created: test_quantum_compression.py"
}

# Main installation process
main() {
    echo "🔬 Quantum-Inspired Model Compression Setup"
    echo "=========================================="
    echo
    
    check_virtual_env
    check_system_requirements
    
    echo
    print_status "Starting installation process..."
    
    install_core_dependencies
    install_quantum_libraries
    install_tensor_libraries
    install_remaining_dependencies
    
    echo
    verify_installation
    create_test_script
    
    echo
    echo "🎉 Quantum Compression Setup Complete!"
    echo "======================================"
    echo
    print_success "Installation completed successfully!"
    echo
    echo "Next steps:"
    echo "1. Run the test script: python3 test_quantum_compression.py"
    echo "2. Start the Streamlit app and navigate to the Quantum Compression tab"
    echo "3. Begin compressing your Phi-1.5B model with quantum-inspired techniques"
    echo
    echo "For more information, see: QUANTUM_COMPRESSION_PLAN.md"
    echo
    print_status "Happy quantum computing! 🚀"
}

# Run main function
main "$@"
