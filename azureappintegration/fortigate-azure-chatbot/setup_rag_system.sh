#!/bin/bash

# RAG System Setup Script
# FortiGate Azure Chatbot - Retrieval-Augmented Generation Setup

set -e  # Exit on any error

echo "🧠 Setting up RAG System for FortiGate Azure Chatbot..."
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

# Check if Python is installed
print_status "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_success "Python $PYTHON_VERSION found"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed. Please install pip."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv_rag" ]; then
    print_status "Creating virtual environment for RAG system..."
    python3 -m venv venv_rag
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv_rag/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install core RAG dependencies
print_status "Installing core RAG dependencies..."
pip install -r requirements_rag.txt

# Install system dependencies based on OS
print_status "Detecting operating system..."
OS="$(uname -s)"
case "${OS}" in
    Linux*)     
        print_status "Linux detected - installing system dependencies..."
        if command -v apt-get &> /dev/null; then
            # Ubuntu/Debian
            sudo apt-get update
            sudo apt-get install -y build-essential python3-dev
            sudo apt-get install -y libxml2-dev libxslt1-dev
            sudo apt-get install -y libjpeg-dev zlib1g-dev
        elif command -v yum &> /dev/null; then
            # CentOS/RHEL
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y python3-devel
            sudo yum install -y libxml2-devel libxslt-devel
            sudo yum install -y libjpeg-devel zlib-devel
        fi
        ;;
    Darwin*)    
        print_status "macOS detected - installing system dependencies..."
        if command -v brew &> /dev/null; then
            brew install libxml2 libxslt
            brew install jpeg zlib
        else
            print_warning "Homebrew not found. Please install Homebrew for better compatibility."
        fi
        ;;
    *)          
        print_warning "Unknown OS: ${OS}. Manual system dependency installation may be required."
        ;;
esac

# Download spaCy language model
print_status "Downloading spaCy English language model..."
python -m spacy download en_core_web_sm || print_warning "Failed to download spaCy model. You can install it later with: python -m spacy download en_core_web_sm"

# Setup Neo4j (Docker-based)
print_status "Setting up Neo4j database..."
if command -v docker &> /dev/null; then
    print_status "Docker found. Setting up Neo4j container..."
    
    # Check if Neo4j container already exists
    if [ "$(docker ps -aq -f name=neo4j-rag)" ]; then
        print_status "Neo4j container already exists. Starting it..."
        docker start neo4j-rag
    else
        print_status "Creating new Neo4j container..."
        docker run -d \
            --name neo4j-rag \
            -p 7474:7474 -p 7687:7687 \
            -e NEO4J_AUTH=neo4j/ragpassword \
            -e NEO4J_PLUGINS='["apoc"]' \
            -v neo4j_data:/data \
            -v neo4j_logs:/logs \
            neo4j:5.0
    fi
    
    print_success "Neo4j container started"
    print_status "Neo4j Browser: http://localhost:7474"
    print_status "Username: neo4j, Password: ragpassword"
    
else
    print_warning "Docker not found. Please install Neo4j manually:"
    echo "  1. Download Neo4j Desktop from https://neo4j.com/download/"
    echo "  2. Or use Docker: docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5.0"
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p data/vector_store
mkdir -p data/documents
mkdir -p data/exports
mkdir -p logs
mkdir -p temp

# Create environment file template
print_status "Creating environment configuration..."
cat > .env.rag << EOF
# RAG System Environment Configuration
# Copy this to .env and fill in your values

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=ragpassword

# RAG Configuration
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200
RAG_SIMILARITY_THRESHOLD=0.8
RAG_MAX_DOCS=5
RAG_VECTOR_STORE_PATH=./data/vector_store

# Optional: DataStax Configuration
# DATASTAX_CLIENT_ID=your_client_id
# DATASTAX_CLIENT_SECRET=your_client_secret
# DATASTAX_KEYSPACE=your_keyspace

# Optional: Pinecone Configuration
# PINECONE_API_KEY=your_pinecone_api_key
# PINECONE_ENVIRONMENT=your_environment

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/rag_system.log
EOF

# Test installation
print_status "Testing RAG system installation..."
python3 -c "
try:
    import langchain
    import openai
    import chromadb
    import neo4j
    print('✅ Core RAG dependencies imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Create sample data
print_status "Creating sample Azure-FortiGate documentation..."
cat > data/documents/azure_fortigate_sample.md << 'EOF'
# Azure FortiGate Integration Guide

## Overview
FortiGate virtual appliances can be deployed on Microsoft Azure to provide advanced security capabilities for cloud workloads.

## Deployment Options

### Single FortiGate VM
- Suitable for small to medium deployments
- Provides basic firewall and VPN capabilities
- Easy to configure and manage

### FortiGate High Availability (HA)
- Active-Passive configuration
- Automatic failover capabilities
- Requires two FortiGate VMs
- Uses Azure Load Balancer for traffic distribution

## Network Configuration

### Virtual Network Setup
1. Create Azure Virtual Network (VNet)
2. Configure subnets for management, external, and internal traffic
3. Set up Network Security Groups (NSGs)
4. Configure User Defined Routes (UDRs)

### FortiGate Network Interfaces
- Management interface: For administrative access
- External interface: Connects to internet-facing subnet
- Internal interface: Connects to protected subnet

## Security Features

### Firewall Policies
- Application control
- Intrusion Prevention System (IPS)
- Web filtering
- Antivirus scanning

### VPN Capabilities
- Site-to-site VPN
- SSL VPN for remote access
- IPSec tunnels

## Integration with Azure Services

### Azure Active Directory
- Single Sign-On (SSO) integration
- Multi-factor authentication
- User and group synchronization

### Azure Monitor
- Log forwarding to Azure Log Analytics
- Custom dashboards and alerts
- Performance monitoring

### Azure Security Center
- Security recommendations
- Threat detection
- Compliance monitoring

## Best Practices

### Performance Optimization
- Choose appropriate VM sizes
- Configure accelerated networking
- Optimize routing tables

### Security Hardening
- Regular firmware updates
- Strong authentication policies
- Network segmentation
- Regular security audits

## Troubleshooting

### Common Issues
1. **Connectivity Problems**
   - Check NSG rules
   - Verify UDR configuration
   - Test network connectivity

2. **Performance Issues**
   - Monitor CPU and memory usage
   - Check network throughput
   - Review firewall policies

3. **HA Failover Issues**
   - Verify HA configuration
   - Check Azure Load Balancer settings
   - Test failover scenarios

### Diagnostic Tools
- FortiGate CLI commands
- Azure Network Watcher
- Azure Monitor logs
- Packet capture analysis
EOF

print_success "Sample documentation created"

# Create test script
print_status "Creating RAG system test script..."
cat > test_rag_system.py << 'EOF'
#!/usr/bin/env python3
"""
Test script for RAG system
"""

import os
import sys
sys.path.append('src')

def test_rag_system():
    try:
        from rag.rag_system import FortiGateAzureRAGSystem, RAGConfig
        
        print("🧪 Testing RAG system initialization...")
        config = RAGConfig()
        rag_system = FortiGateAzureRAGSystem(config)
        
        print("✅ RAG system initialized successfully")
        
        # Test document ingestion
        print("📄 Testing document ingestion...")
        sample_docs = ['data/documents/azure_fortigate_sample.md']
        success = rag_system.ingest_documents(sample_docs)
        
        if success:
            print("✅ Document ingestion successful")
        else:
            print("⚠️ Document ingestion had issues")
        
        # Test query
        print("🔍 Testing query processing...")
        result = rag_system.query("How do I configure FortiGate HA on Azure?")
        
        print(f"Query: {result['question']}")
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Response time: {result['response_time']:.2f}s")
        
        # Get system stats
        stats = rag_system.get_system_stats()
        print(f"📊 System stats: {stats}")
        
        rag_system.close()
        print("✅ RAG system test completed successfully")
        
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_rag_system()
    sys.exit(0 if success else 1)
EOF

chmod +x test_rag_system.py

print_success "RAG system setup completed successfully!"
echo ""
echo "🎉 Setup Summary:"
echo "=================="
echo "✅ Virtual environment created: venv_rag"
echo "✅ RAG dependencies installed"
echo "✅ Neo4j database container started (if Docker available)"
echo "✅ Sample documentation created"
echo "✅ Environment template created"
echo ""
echo "📋 Next Steps:"
echo "=============="
echo "1. Copy .env.rag to .env and configure your API keys:"
echo "   cp .env.rag .env"
echo ""
echo "2. Set your OpenAI API key in .env:"
echo "   OPENAI_API_KEY=your_actual_api_key"
echo ""
echo "3. Test the RAG system:"
echo "   source venv_rag/bin/activate"
echo "   python test_rag_system.py"
echo ""
echo "4. Access Neo4j Browser at http://localhost:7474"
echo "   Username: neo4j, Password: ragpassword"
echo ""
echo "5. Run the Streamlit app with RAG enabled:"
echo "   streamlit run src/app.py"
echo ""
echo "🔧 Troubleshooting:"
echo "=================="
echo "- If Neo4j connection fails, ensure Docker is running"
echo "- For permission issues, check file permissions"
echo "- For import errors, verify virtual environment activation"
echo ""
print_success "RAG system is ready to use! 🚀"
