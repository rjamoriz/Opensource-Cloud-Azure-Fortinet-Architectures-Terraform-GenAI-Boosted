# Multi-Cloud RAG System

## Overview

The Multi-Cloud RAG System is an advanced Retrieval-Augmented Generation (RAG) system designed specifically for multi-cloud VM architecture expertise. It combines comprehensive knowledge bases, vector search capabilities, and real-time cloud API integration to provide intelligent recommendations for Azure, GCP, and multi-cloud VM deployments.

## 🚀 Key Features

### Comprehensive Multi-Cloud Knowledge Base
- **Azure VM Expertise**: VM sizes, networking, security, best practices
- **GCP Compute Engine**: Machine types, pricing, optimization strategies  
- **Multi-Cloud Architecture**: Design patterns, cost optimization, governance
- **Metadata Tagging**: Cloud provider, topic, region, complexity classification
- **Structured Content**: JSON/YAML/Markdown export capabilities

### Advanced Vector Search
- **Multiple Vector Stores**: Pinecone, Azure AI Search, Weaviate support
- **Hybrid Search**: Semantic + keyword search capabilities
- **Intelligent Filtering**: By cloud, topic, region, complexity, use case
- **High-Performance Retrieval**: Optimized for large-scale knowledge bases

### Real-Time Cloud API Integration
- **Azure Resource Management**: VM specifications, pricing, deployment
- **GCP Compute Engine**: Machine types, cost estimation, recommendations
- **Live Pricing Data**: Real-time cost calculations and comparisons
- **Configuration Validation**: Automated deployment configuration checks

### Intelligent VM Recommendations
- **Requirement Analysis**: Extract technical needs from natural language
- **Multi-Provider Comparison**: Side-by-side Azure vs GCP recommendations
- **Cost Optimization**: Budget-aware suggestions with pricing breakdowns
- **Confidence Scoring**: AI-powered recommendation quality assessment

### Production-Ready Architecture
- **Modular Design**: Pluggable components for easy extension
- **Async Operations**: High-performance async/await implementation
- **Error Handling**: Robust error recovery and logging
- **Monitoring**: Built-in system statistics and health checks

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Interface                      │
├─────────────────────────────────────────────────────────────┤
│                Multi-Cloud RAG System                      │
├─────────────┬─────────────────┬─────────────────┬───────────┤
│ Vector      │ Cloud APIs      │ Knowledge Base  │ LLM       │
│ Stores      │                 │ Manager         │ Chain     │
│             │                 │                 │           │
│ • Pinecone  │ • Azure API     │ • Document      │ • OpenAI  │
│ • Azure AI  │ • GCP API       │   Storage       │ • Embeddings│
│ • Weaviate  │ • AWS API       │ • Metadata      │ • Retrieval│
│             │   (Future)      │   Management    │   Chain   │
└─────────────┴─────────────────┴─────────────────┴───────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.9+
- OpenAI API key
- Vector database credentials (Pinecone recommended)
- Cloud provider credentials (optional for live recommendations)

### Basic Installation
```bash
# Clone the repository
git clone <repository-url>
cd fortigate-azure-chatbot

# Install core dependencies
pip install -r requirements_multi_cloud_rag.txt

# Set environment variables
export OPENAI_API_KEY="your-openai-api-key"
export PINECONE_API_KEY="your-pinecone-api-key"
```

### Advanced Installation with Cloud APIs
```bash
# Install cloud SDK dependencies (optional)
pip install azure-mgmt-compute azure-mgmt-network azure-identity
pip install google-cloud-compute google-auth

# Configure cloud credentials
export AZURE_SUBSCRIPTION_ID="your-azure-subscription-id"
export AZURE_CLIENT_ID="your-azure-client-id"
export AZURE_CLIENT_SECRET="your-azure-client-secret"
export AZURE_TENANT_ID="your-azure-tenant-id"

export GCP_PROJECT_ID="your-gcp-project-id"
export GCP_CREDENTIALS_PATH="path-to-service-account.json"
```

## 🚀 Quick Start

### 1. Run the Streamlit Interface
```bash
cd src
streamlit run multi_cloud_rag_interface.py
```

### 2. Initialize the System
1. Open the Streamlit interface in your browser
2. Configure API credentials in the sidebar
3. Click "🚀 Initialize System"
4. Wait for the knowledge base to be seeded with initial content

### 3. Start Asking Questions
Examples of queries you can ask:

```
"What Azure VM size should I use for a web application with moderate traffic?"

"Compare Azure and GCP machine types for a high-memory database workload"

"Design a multi-cloud architecture for a global e-commerce platform"

"What are the best practices for Azure virtual network design?"

"How do I optimize costs for compute resources across Azure and GCP?"
```

## 💡 Example Usage

### Programmatic Usage
```python
import asyncio
from src.multi_cloud_rag import MultiCloudRAGSystem, RAGConfig

# Configure the system
config = RAGConfig(
    vector_store_type="pinecone",
    vector_store_config={
        "api_key": "your-pinecone-api-key",
        "environment": "us-west1-gcp-free"
    },
    cloud_configs={
        "azure": {
            "subscription_id": "your-azure-subscription",
            "client_id": "your-client-id",
            "client_secret": "your-client-secret",
            "tenant_id": "your-tenant-id"
        }
    }
)

# Initialize and use the system
async def main():
    rag_system = MultiCloudRAGSystem(config)
    await rag_system.initialize()
    
    response = await rag_system.process_query(
        "What's the best Azure VM for a web server?",
        include_vm_recommendations=True,
        output_format="json"
    )
    
    print(response.answer)
    for rec in response.vm_recommendations:
        print(f"- {rec.cloud_provider}: {rec.vm_specification.name}")

asyncio.run(main())
```

### Adding Custom Knowledge
```python
from src.multi_cloud_rag import DocumentMetadata, CloudProvider, DocumentType

# Add custom documentation
metadata = DocumentMetadata(
    cloud=CloudProvider.AZURE,
    topic=DocumentType.BEST_PRACTICES,
    region="global",
    complexity="advanced",
    use_case="enterprise-security"
)

doc_id = await rag_system.add_knowledge(
    title="Azure Security Best Practices",
    content="Your detailed security documentation...",
    metadata=metadata,
    tags=["security", "compliance", "enterprise"]
)
```

## 📊 System Capabilities

### Query Processing
- **Natural Language Understanding**: Extracts technical requirements from conversational queries
- **Context-Aware Responses**: Maintains conversation history for follow-up questions
- **Multi-Format Output**: Markdown, JSON, YAML response formats
- **Source Attribution**: Links to relevant knowledge base documents

### VM Recommendations
- **Intelligent Matching**: Matches workload requirements to optimal VM types
- **Cost Analysis**: Detailed pricing breakdown with monthly/yearly estimates
- **Performance Optimization**: CPU, memory, storage, and network considerations
- **Compliance Awareness**: Considers regulatory requirements in recommendations

### Knowledge Management
- **Automated Seeding**: Pre-populated with comprehensive multi-cloud content
- **Dynamic Updates**: Add new documentation and best practices
- **Version Control**: Track document changes and updates
- **Export Capabilities**: Backup and share knowledge in multiple formats

## 🔧 Configuration

### Vector Store Options
```python
# Pinecone (Recommended)
vector_config = {
    "api_key": "your-pinecone-key",
    "environment": "us-west1-gcp-free",
    "index_name": "multi-cloud-vm-rag"
}

# Azure AI Search
vector_config = {
    "search_endpoint": "https://your-search.search.windows.net",
    "api_key": "your-search-key",
    "index_name": "multi-cloud-vm-rag"
}
```

### LLM Configuration
```python
llm_config = {
    "model": "gpt-4",  # or "gpt-3.5-turbo"
    "temperature": 0.1,
    "max_tokens": 2000
}
```

### Cloud Provider Configuration
```python
cloud_configs = {
    "azure": {
        "subscription_id": "...",
        "client_id": "...",
        "client_secret": "...",
        "tenant_id": "..."
    },
    "gcp": {
        "project_id": "...",
        "credentials_path": "service-account.json"
    }
}
```

## 📈 Performance Optimization

### Vector Store Performance
- **Batch Operations**: Efficient bulk document indexing
- **Optimized Queries**: Smart filtering to reduce search space
- **Caching**: Frequently accessed embeddings cached locally
- **Parallel Processing**: Concurrent vector operations

### Cloud API Optimization
- **Connection Pooling**: Reuse HTTP connections for cloud APIs
- **Rate Limiting**: Respect cloud provider API limits
- **Caching**: Cache pricing and configuration data
- **Async Operations**: Non-blocking cloud API calls

### Memory Management
- **Lazy Loading**: Load components only when needed
- **Streaming**: Process large documents in chunks
- **Garbage Collection**: Efficient cleanup of temporary objects

## 🔐 Security Considerations

### API Key Management
- Environment variables for sensitive credentials
- Secure session storage in Streamlit interface
- No hardcoded credentials in source code

### Data Privacy
- Local knowledge base storage
- Configurable data retention policies
- Audit logging for system access

### Network Security
- HTTPS for all external API calls
- Certificate validation for cloud connections
- Secure token handling

## 🧪 Testing

### Unit Tests
```bash
pytest tests/unit/ -v
```

### Integration Tests
```bash
pytest tests/integration/ -v
```

### Load Testing
```bash
pytest tests/performance/ -v
```

## 📚 Documentation

### API Documentation
- Comprehensive docstrings for all classes and methods
- Type hints for better IDE support
- Usage examples in docstrings

### Architecture Documentation
- System design diagrams
- Component interaction flows
- Performance characteristics

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements_dev.txt

# Setup pre-commit hooks
pre-commit install

# Run code formatting
black src/
flake8 src/
```

### Adding New Cloud Providers
1. Implement `BaseCloudAPI` interface
2. Add provider to `CloudAPIFactory`
3. Create configuration templates
4. Add comprehensive tests

### Extending Vector Stores
1. Implement `BaseVectorStore` interface
2. Add store to `VectorStoreFactory`
3. Handle store-specific configurations
4. Ensure async compatibility

## 🚀 Roadmap

### Phase 1: Foundation ✅
- Multi-cloud knowledge base
- Vector search implementation
- Azure and GCP API integration
- Streamlit interface

### Phase 2: Enhancement 🔄
- AWS support
- Advanced filtering and search
- Performance optimizations
- Enhanced monitoring

### Phase 3: Enterprise 📋
- Multi-tenant support
- Advanced security features
- Custom model fine-tuning
- Enterprise integrations

## 🆘 Troubleshooting

### Common Issues

**Vector Store Connection Failed**
- Verify API credentials
- Check network connectivity
- Ensure proper environment configuration

**Cloud API Authentication Errors**
- Validate cloud provider credentials
- Check subscription/project permissions
- Verify service account configurations

**Memory Issues with Large Knowledge Base**
- Reduce batch sizes for document processing
- Implement document chunking
- Use streaming for large files

**Slow Query Performance**
- Optimize vector store index
- Implement query result caching
- Reduce search result limits

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT and embedding models
- Pinecone for vector database services
- LangChain for RAG framework
- Azure and GCP for cloud platform APIs
- Streamlit for the user interface framework

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation
- Review existing discussions

---

**Multi-Cloud RAG System** - Empowering intelligent multi-cloud VM architecture decisions through advanced AI and comprehensive knowledge management.
