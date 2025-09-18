# 🦙 Llama 7B Fine-Tuning for FortiGate Azure Chatbot

This guide covers the complete setup and usage of local Llama 7B fine-tuning capabilities integrated into the FortiGate Azure Chatbot.

## 🎯 Overview

The Llama fine-tuning feature allows you to:
- Load and fine-tune Meta Llama 7B models locally
- Upload custom FortiGate/Azure training data through the web interface
- Use advanced techniques like LoRA and QLoRA for efficient training
- Generate specialized responses with your fine-tuned model
- Maintain complete privacy and control over your data

## 🔧 Prerequisites

### System Requirements

**Minimum Requirements:**
- Python 3.8+
- 16 GB RAM
- 50 GB free disk space
- CUDA-compatible GPU (recommended)

**Recommended Requirements:**
- Python 3.9+
- 32 GB RAM
- 100 GB free disk space
- NVIDIA GPU with 12+ GB VRAM
- CUDA 11.8 or later

### Dependencies

The following libraries are required:
- `torch>=2.0.0` (with CUDA support)
- `transformers>=4.35.0`
- `peft>=0.6.0` (Parameter Efficient Fine-Tuning)
- `bitsandbytes>=0.41.0` (for quantization)
- `datasets>=2.14.0`
- `accelerate>=0.24.0`
- `huggingface_hub>=0.17.0`

## 🚀 Quick Setup

### Option 1: Automated Setup (Recommended)

Run the automated setup script:

```bash
# Navigate to the project directory
cd /path/to/fortigate-azure-chatbot

# Run the setup script
./setup_llama_finetuning.sh
```

### Option 2: Manual Setup

1. **Install PyTorch with CUDA:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

2. **Install fine-tuning dependencies:**
```bash
pip install -r requirements_llama.txt
```

3. **Set up HuggingFace token:**
```bash
export HUGGINGFACE_TOKEN='your-huggingface-token'
```

4. **Accept Llama 2 license:**
Visit: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

## 🔑 HuggingFace Setup

1. **Create HuggingFace Account:**
   - Go to https://huggingface.co/join
   - Create a free account

2. **Generate Access Token:**
   - Visit https://huggingface.co/settings/tokens
   - Create a new token with "Read" permissions
   - Copy the token

3. **Accept Llama License:**
   - Visit https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
   - Click "Agree and access repository"
   - This is required to download Llama models

4. **Set Environment Variable:**
```bash
export HUGGINGFACE_TOKEN='hf_your_token_here'
```

## 📊 Using the Interface

### 1. Start the Application

```bash
streamlit run src/app.py
```

### 2. Navigate to Fine-Tuning

1. Open the app in your browser
2. Go to the "Fine-Tuned Model" tab
3. Select "🦙 Llama 7B Local Fine-Tuning"

### 3. System Check

The first tab shows:
- System requirements status
- GPU availability
- Dependency installation status
- HuggingFace token verification

### 4. Load Base Model

1. Go to the "🦙 Load Model" tab
2. Select a base model (default: `meta-llama/Llama-2-7b-chat-hf`)
3. Click "🚀 Load Base Model"
4. Wait for the model to download and load (5-10 minutes)

### 5. Upload Training Data

1. Go to the "📁 Upload Data" tab
2. Upload your training files (JSON, JSONL, TXT, CSV)
3. Click "🔄 Process Training Data"

**Supported Data Formats:**

**JSON (OpenAI Chat Format):**
```json
{
  "messages": [
    {"role": "system", "content": "You are a FortiGate expert..."},
    {"role": "user", "content": "How to configure FortiGate HA?"},
    {"role": "assistant", "content": "To configure FortiGate HA..."}
  ]
}
```

**JSONL (Alpaca Format):**
```json
{"instruction": "Explain FortiGate HA", "input": "", "output": "FortiGate HA configuration..."}
{"instruction": "Configure Azure VNET", "input": "Basic setup", "output": "Steps to configure..."}
```

### 6. Configure Fine-Tuning

1. Go to the "🔥 Fine-Tune" tab
2. Adjust training parameters:
   - **Epochs:** Number of training iterations (1-10)
   - **Learning Rate:** How fast the model learns (1e-5 to 5e-4)
   - **Batch Size:** Training batch size (1-8)
   - **LoRA Rank:** Parameter efficiency setting (8-64)

3. Review training estimates
4. Click "🚀 Start Fine-Tuning"

### 7. Test Your Model

1. Go to the "🧪 Test Model" tab
2. Enter FortiGate/Azure questions
3. Adjust response parameters
4. Generate and review responses

## ⚙️ Advanced Configuration

### LoRA Parameters

- **LoRA Rank (r):** Controls the rank of adaptation matrices
  - Lower values (8-16): Faster training, less capacity
  - Higher values (32-64): More capacity, slower training

- **LoRA Alpha:** Scaling parameter for LoRA updates
  - Typically 2x the rank value
  - Higher values: Stronger adaptation

### Training Parameters

- **Learning Rate:** 
  - Start with 2e-4 for most cases
  - Lower (1e-5) for stable training
  - Higher (5e-4) for faster convergence

- **Batch Size:**
  - Limited by GPU memory
  - Use gradient accumulation for effective larger batches

### Memory Optimization

The system uses several techniques to reduce memory usage:
- **4-bit Quantization:** Reduces model size by 75%
- **LoRA:** Only trains small adapter layers
- **Gradient Checkpointing:** Trades compute for memory

## 🔍 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```
Solution: Reduce batch size or use gradient accumulation
```

**2. Model Download Fails**
```
Solution: Check HuggingFace token and internet connection
```

**3. Slow Training**
```
Solution: Ensure GPU is available and CUDA is properly installed
```

**4. Import Errors**
```
Solution: Run setup script or install dependencies manually
```

### Performance Tips

1. **Use GPU:** Essential for reasonable training times
2. **Optimize Batch Size:** Find the largest batch size that fits in memory
3. **Monitor GPU Usage:** Use `nvidia-smi` to check utilization
4. **Quality Data:** Better training data leads to better results

## 📈 Expected Performance

### Training Times (Approximate)

| Dataset Size | GPU | Time per Epoch |
|-------------|-----|----------------|
| 100 examples | RTX 3080 | 2-3 minutes |
| 500 examples | RTX 3080 | 8-12 minutes |
| 1000 examples | RTX 3080 | 15-25 minutes |
| 100 examples | RTX 4090 | 1-2 minutes |
| 500 examples | RTX 4090 | 4-6 minutes |

### Memory Usage

| Model Component | Memory Usage |
|----------------|--------------|
| Base Model (4-bit) | ~4 GB |
| LoRA Adapters | ~100 MB |
| Training Overhead | ~2-4 GB |
| **Total** | **~6-8 GB** |

## 🎯 Best Practices

### Data Preparation

1. **Quality over Quantity:** 100 high-quality examples > 1000 poor ones
2. **Consistent Format:** Use the same format throughout your dataset
3. **Balanced Content:** Include various types of FortiGate/Azure scenarios
4. **Clear Instructions:** Make sure questions and answers are clear

### Training Strategy

1. **Start Small:** Begin with 3 epochs and adjust based on results
2. **Monitor Overfitting:** Watch for decreasing validation performance
3. **Save Checkpoints:** The system automatically saves your fine-tuned model
4. **Test Regularly:** Use the test interface to evaluate progress

### Production Deployment

1. **Model Versioning:** Keep track of different fine-tuned versions
2. **Performance Testing:** Compare with base model responses
3. **Backup Models:** Save successful fine-tuned models
4. **Documentation:** Record training parameters and data used

## 🔒 Security and Privacy

### Data Privacy
- All training happens locally on your machine
- No data is sent to external services during fine-tuning
- Training data is processed in memory only

### Model Security
- Fine-tuned models are saved locally
- No automatic uploading to cloud services
- You maintain full control over your models

## 📚 Additional Resources

### Documentation
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Llama 2 Paper](https://arxiv.org/abs/2307.09288)

### Community
- [HuggingFace Forums](https://discuss.huggingface.co/)
- [PyTorch Forums](https://discuss.pytorch.org/)

## 🆘 Support

If you encounter issues:

1. **Check System Status:** Use the system check tab in the interface
2. **Review Logs:** Check console output for detailed error messages
3. **Verify Setup:** Ensure all dependencies are properly installed
4. **GPU Issues:** Verify CUDA installation and GPU availability

## 🔄 Updates and Maintenance

### Keeping Dependencies Updated

```bash
# Update core libraries
pip install --upgrade torch transformers peft bitsandbytes

# Check for compatibility
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Model Management

- Fine-tuned models are saved in `models/llama_fine_tuned/`
- Training logs are stored in `logs/`
- You can manually backup these directories

---

## 🎉 Conclusion

The Llama 7B fine-tuning integration provides a powerful way to create specialized FortiGate Azure models while maintaining complete control over your data and training process. With the intuitive Streamlit interface, you can easily load models, upload training data, and fine-tune your own specialized AI assistant.

For questions or issues, refer to the troubleshooting section or check the system status in the application interface.
