# FortiGate-VM Azure Deployment GenAI Integration

This project is a comprehensive Streamlit application that integrates advanced AI capabilities to provide intelligent assistance for deploying FortiGate-VM on Azure using Terraform. The application features a multimodal LLM chatbot with RAG (Retrieval-Augmented Generation) capabilities, enhanced voice chat, and multi-cloud deployment support.

## 🚀 Key Features

### 🤖 Advanced AI Integration
- **LangChain RAG System**: Intelligent document retrieval and knowledge management
- **Enhanced Voice Chat**: Real-time voice interaction with multiple AI providers (OpenAI, Cartesia)
- **Multi-Model Support**: GPT-4, GPT-4o, and fine-tuned models
- **Streaming Responses**: Real-time conversation flow

### ☁️ Multi-Cloud Support
- **Azure Integration**: Complete Azure deployment workflows
- **GCP Support**: Google Cloud Platform integration
- **AWS Compatibility**: Amazon Web Services deployment options
- **Terraform Automation**: Infrastructure as Code deployment

### 📊 Comprehensive Interface
- **7-Tab Application**: Organized feature access
- **RAG Knowledge System**: 5-tab knowledge management interface
- **Analytics Dashboard**: Performance and usage insights
- **Dark/Light Theme**: Customizable UI experience

## 📁 Project Structure

```
fortigate-azure-chatbot/
├── src/
│   ├── app.py                      # Main Streamlit application
│   ├── agents/                     # LangChain agent implementations
│   ├── analytics/                  # Analytics and monitoring
│   ├── chatbot/                    # Core chatbot functionality
│   ├── cloud_mcp/                  # Multi-cloud provider interfaces
│   ├── fine_tuning/                # Model fine-tuning capabilities
│   ├── multi_cloud_rag/            # Multi-cloud RAG implementation
│   ├── quantum_compression/         # Advanced compression algorithms
│   ├── rag/                        # RAG system components
│   │   ├── langchain_agent.py      # LangChain agent configuration
│   │   ├── rag_interface.py        # RAG user interface
│   │   ├── embedding_manager.py    # Vector embeddings management
│   │   └── config.py               # RAG configuration
│   ├── utils/                      # Utility functions
│   │   ├── enhanced_voice_chat.py  # Voice interaction system
│   │   ├── enhanced_voice_processor.py # Voice processing engine
│   │   └── azure_terraform.py      # Azure/Terraform utilities
│   └── types/                      # Type definitions
├── training_data/                  # Training datasets
├── tests/                          # Test suite
├── requirements/                   # Dependency files
├── setup_scripts/                  # Installation scripts
└── docs/                          # Documentation
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Git

### Quick Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rjamoriz/Opensource-Cloud-Azure-Fortinet-Architectures-Terraform-GenAI-Boosted.git
   cd Opensource-Cloud-Azure-Fortinet-Architectures-Terraform-GenAI-Boosted/fortigate-azure-chatbot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure secrets:**
   ```bash
   cp src/.streamlit/secrets.toml.example src/.streamlit/secrets.toml
   # Edit secrets.toml with your API keys
   ```

4. **Run the application:**
   ```bash
   streamlit run src/app.py
   ```

### Advanced Setup Options

#### Enhanced Voice Chat Setup
```bash
pip install -r requirements_voice.txt
chmod +x setup_voice_chat.sh
./setup_voice_chat.sh
```

#### RAG System Setup
```bash
pip install -r requirements_rag.txt
chmod +x setup_rag_system.sh
./setup_rag_system.sh
```

#### Multi-Cloud RAG Setup
```bash
pip install -r requirements_multi_cloud_rag.txt
chmod +x setup_multi_cloud_rag.sh
./setup_multi_cloud_rag.sh
```

## 🔧 Configuration

### API Keys Configuration
Edit `src/.streamlit/secrets.toml`:

```toml
[openai]
api_key = "your-openai-api-key"

[cartesia]
api_key = "your-cartesia-api-key"

[azure]
speech_key = "your-azure-speech-key"
speech_region = "your-azure-region"

[datastax]
api_endpoint = "your-datastax-endpoint"
api_key = "your-datastax-api-key"
keyspace = "fortigate_kb"
```

### Theme Configuration
Customize appearance in `src/.streamlit/config.toml`:

```toml
[theme]
base = "dark"
primaryColor = "#FF6B6B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
```

## 🚀 Usage

### Basic Chatbot Interaction
1. Launch the application
2. Navigate to the "💬 FortiGate Chat" tab
3. Ask questions about FortiGate deployment
4. Receive AI-powered responses with deployment guidance

### RAG Knowledge System
1. Go to the "📚 RAG Knowledge" tab
2. Upload documentation and configuration files
3. Use intelligent search across your knowledge base
4. Get contextual answers from your documents

### Enhanced Voice Chat
1. Access the "🎤 Enhanced Voice Chat" tab
2. Configure voice settings and AI provider
3. Start voice conversations
4. Receive spoken responses with real-time processing

### Multi-Cloud Deployment
1. Select your target cloud provider tab
2. Follow guided deployment workflows
3. Generate Terraform configurations
4. Deploy with integrated automation

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Run specific test modules:
```bash
python -m pytest tests/test_rag_system.py -v
python -m pytest tests/test_voice_chat.py -v
```

## 📈 Features Overview

### Core Capabilities
- ✅ **Intelligent Chatbot**: AI-powered FortiGate deployment assistance
- ✅ **RAG Integration**: Document-aware knowledge retrieval
- ✅ **Voice Interaction**: Real-time voice chat capabilities
- ✅ **Multi-Cloud Support**: Azure, GCP, AWS deployment options
- ✅ **Terraform Integration**: Infrastructure as Code automation
- ✅ **Analytics Dashboard**: Usage and performance monitoring
- ✅ **Dark Mode Support**: Customizable UI themes

### Advanced Features
- 🔬 **Quantum Compression**: Advanced data compression algorithms
- 🎯 **Model Fine-Tuning**: Custom model training capabilities
- 📊 **Visualization Tools**: Interactive charts and diagrams
- 🔗 **LangChain Integration**: Advanced agent-based conversations
- 🎙️ **Multi-Provider Voice**: OpenAI TTS, Cartesia AI support
- 📱 **Responsive Design**: Mobile-friendly interface

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- 📧 Email: support@fortigate-ai.com
- 💬 Discord: [FortiGate AI Community](https://discord.gg/fortigate-ai)
- 📖 Documentation: [docs.fortigate-ai.com](https://docs.fortigate-ai.com)

## 🙏 Acknowledgments

- Fortinet for FortiGate-VM technology
- OpenAI for GPT models and APIs
- LangChain for RAG framework
- Streamlit for the web application framework
- The open-source community for various tools and libraries

---

**Built with ❤️ for the FortiGate and Cloud Security Community**