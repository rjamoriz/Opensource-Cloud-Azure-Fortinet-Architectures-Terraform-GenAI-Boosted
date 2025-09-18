# 🎯 FortiGate Azure Fine-Tuning Guide

This guide walks you through creating a specialized AI model with deep FortiGate and Azure deployment expertise.

## 🎯 Overview

Fine-tuning creates a custom AI model trained specifically on FortiGate Azure deployment scenarios, making your chatbot significantly more knowledgeable and accurate for domain-specific questions.

### 🚀 Benefits of Fine-Tuning

- **🎯 Specialized Knowledge**: Deep understanding of FortiGate-specific configurations
- **🔧 Terraform Expertise**: Detailed knowledge of infrastructure as code patterns
- **🛡️ Security Best Practices**: Built-in security recommendations
- **🚨 Troubleshooting**: Context-aware problem-solving capabilities
- **📊 Cost Optimization**: Azure resource optimization suggestions

## 📋 Prerequisites

### 1. OpenAI API Access
```bash
# Set your OpenAI API key
export OPENAI_API_KEY='your-openai-api-key-here'
```

### 2. API Credits
- **Estimated Cost**: $8-20 USD for typical FortiGate dataset
- **Training Time**: 10-30 minutes depending on dataset size
- **Base Model**: GPT-3.5-turbo (recommended for fine-tuning)

### 3. Python Dependencies
```bash
pip install openai>=1.0.0
```

## 🛠️ Fine-Tuning Process

### Step 1: Generate Training Data
```bash
cd src/fine_tuning
python data_preparation.py
```

This creates a comprehensive dataset with:
- **Azure Deployment Scenarios**: Single VM, HA clusters, cross-zone setups
- **Terraform Knowledge**: Variables, resources, best practices
- **Troubleshooting Guides**: Common issues and solutions
- **Integration Patterns**: Application Gateway, Load Balancer, VWAN

### Step 2: Run Fine-Tuning
```bash
# Complete process (data + training)
python run_fine_tuning.py

# Or step by step:
python run_fine_tuning.py --data-only    # Generate data only
python run_fine_tuning.py --skip-data    # Use existing data
```

### Step 3: Monitor Progress
The script will show real-time progress:
```
🚀 Starting fine-tuning process...
📁 File uploaded successfully: file-abc123
✅ Fine-tuning job created: ftjob-xyz789
⏳ Job in progress... Status: running
🎉 Fine-tuning completed! Model: ft:gpt-3.5-turbo:your-org:fortigate-azure
```

## 📊 Training Data Structure

### Example Training Conversation
```json
{
  "messages": [
    {
      "role": "system", 
      "content": "You are an expert FortiGate Azure deployment assistant..."
    },
    {
      "role": "user", 
      "content": "How do I deploy FortiGate HA in Azure?"
    },
    {
      "role": "assistant", 
      "content": "To deploy FortiGate HA in Azure: 1. Create resource group..."
    }
  ]
}
```

### Training Categories

#### 🏗️ **Deployment Scenarios** (40% of dataset)
- Single FortiGate VM deployment
- High Availability configurations
- Cross-zone redundancy
- Multi-port setups
- Azure service integrations

#### 🔧 **Terraform Expertise** (30% of dataset)
- Variable definitions and validation
- Resource configurations
- Module structures
- State management
- Best practices

#### 🚨 **Troubleshooting** (20% of dataset)
- Common deployment issues
- Network connectivity problems
- Performance optimization
- Security configuration errors
- Azure-specific challenges

#### 🔗 **Integration Patterns** (10% of dataset)
- Application Gateway integration
- Load Balancer configurations
- VPN Gateway setups
- ExpressRoute connections
- Private Link implementations

## 🎮 Using the Fine-Tuned Model

### In the Streamlit App
1. **Navigate to "Fine-Tuned Model" tab**
2. **Select "Fine-Tuned FortiGate Expert"**
3. **Ask specialized questions**
4. **Compare with base GPT-4 responses**

### Programmatic Usage
```python
from fine_tuning.model_integration import FineTunedModelManager

manager = FineTunedModelManager()
response = manager.query_fine_tuned_model(
    "How do I configure FortiGate HA with floating IP in Azure?"
)
print(response)
```

## 📈 Performance Comparison

### Base GPT-4 vs Fine-Tuned Model

| Aspect | Base GPT-4 | Fine-Tuned Model |
|--------|------------|------------------|
| **General Knowledge** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **FortiGate Specifics** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Azure Integration** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Terraform Details** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Troubleshooting** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Response Speed** | Fast | Fast |
| **Cost per Query** | Higher | Lower |

## 🔄 Iterative Improvement

### Adding New Training Data
```python
from fine_tuning.data_preparation import FortiGateDataPreparator

preparator = FortiGateDataPreparator()
preparator.add_conversation(
    user_input="Your new question",
    assistant_response="Detailed expert answer"
)
preparator.save_training_data("updated_dataset.jsonl")
```

### Re-training Process
1. **Collect feedback** from users
2. **Identify knowledge gaps** in responses
3. **Add new training examples** addressing gaps
4. **Re-run fine-tuning** with updated dataset
5. **A/B test** new model against previous version

## 🛡️ Security & Best Practices

### API Key Management
```bash
# Use environment variables
export OPENAI_API_KEY='sk-...'

# Or use .env file (not committed to git)
echo "OPENAI_API_KEY=sk-..." > .env
```

### Data Privacy
- **No sensitive data** in training examples
- **Generic configurations** only
- **Public knowledge** sources
- **Compliance** with OpenAI usage policies

### Cost Management
```python
# Monitor fine-tuning costs
def estimate_cost(num_examples, avg_tokens_per_example):
    training_cost = (num_examples * avg_tokens_per_example) * 0.008 / 1000
    return f"Estimated cost: ${training_cost:.2f}"
```

## 🚨 Troubleshooting

### Common Issues

#### ❌ "API key not found"
```bash
export OPENAI_API_KEY='your-key-here'
# Verify with:
echo $OPENAI_API_KEY
```

#### ❌ "Training data validation failed"
- Check JSONL format
- Ensure all messages have role and content
- Verify minimum 10 examples

#### ❌ "Fine-tuning job failed"
- Check training data quality
- Verify sufficient API credits
- Review OpenAI status page

#### ❌ "Model not available"
- Wait for training completion
- Check model ID in logs
- Verify API permissions

### Getting Help
- **Check logs**: `fine_tuning.log`
- **OpenAI Documentation**: [Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- **Community Support**: OpenAI Community Forum

## 📊 Monitoring & Analytics

### Model Performance Metrics
```python
# Compare model responses
manager = FineTunedModelManager()
comparison = manager.compare_models("Your test question")

# Analyze response quality
# - Accuracy of technical details
# - Completeness of answers
# - Relevance to FortiGate/Azure context
```

### Usage Analytics
- **Response time** comparison
- **User satisfaction** ratings
- **Query complexity** handling
- **Cost per interaction**

## 🎯 Next Steps

1. **✅ Complete fine-tuning** following this guide
2. **🧪 Test extensively** with real FortiGate scenarios
3. **📊 Collect user feedback** on response quality
4. **🔄 Iterate and improve** based on usage patterns
5. **📈 Scale deployment** for production use

## 📚 Additional Resources

- [OpenAI Fine-tuning Documentation](https://platform.openai.com/docs/guides/fine-tuning)
- [FortiGate Azure Deployment Guide](https://docs.fortinet.com/document/fortigate-public-cloud/7.4.0/azure-administration-guide)
- [Terraform Azure Provider](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs)
- [Azure Architecture Center](https://docs.microsoft.com/en-us/azure/architecture/)

---

**🎉 Ready to create your specialized FortiGate AI expert!**

Run the fine-tuning process and transform your chatbot into a domain-specific expert with deep FortiGate and Azure knowledge.
