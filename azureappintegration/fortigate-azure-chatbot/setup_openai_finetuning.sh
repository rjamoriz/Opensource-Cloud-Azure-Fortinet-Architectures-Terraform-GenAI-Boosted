#!/bin/bash

# Setup script for OpenAI Fine-Tuning Dependencies
# FortiGate Azure Chatbot - OpenAI Fine-Tuning Enhancement

echo "🤖 Setting up OpenAI Fine-Tuning Dependencies..."
echo "=================================================="

# Check if virtual environment is active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Consider activating one:"
    echo "   conda activate fortinetvmazure"
    echo "   # or"
    echo "   source venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install OpenAI fine-tuning dependencies
echo "📦 Installing OpenAI fine-tuning dependencies..."
pip install -r requirements_openai.txt

# Verify installations
echo ""
echo "🔍 Verifying installations..."

python -c "
import sys
try:
    import openai
    print(f'✅ OpenAI installed successfully (version: {openai.__version__})')
except ImportError:
    print('❌ OpenAI installation failed')
    sys.exit(1)

try:
    import pandas
    print(f'✅ pandas installed successfully (version: {pandas.__version__})')
except ImportError:
    print('❌ pandas installation failed')
    sys.exit(1)

try:
    import tiktoken
    print('✅ tiktoken installed successfully')
except ImportError:
    print('⚠️  tiktoken installation failed (optional for token counting)')

try:
    import jsonlines
    print('✅ jsonlines installed successfully')
except ImportError:
    print('⚠️  jsonlines installation failed (optional for JSONL processing)')

print('')
print('🎉 OpenAI fine-tuning setup complete!')
print('')
print('🔑 Next Steps:')
print('1. Set your OpenAI API key:')
print('   export OPENAI_API_KEY=\"your-api-key-here\"')
print('')
print('2. Or create a .env file with:')
print('   OPENAI_API_KEY=your-api-key-here')
print('')
print('📊 Available Features:')
print('  • GPT-3.5-Turbo and GPT-4 fine-tuning')
print('  • Training data validation and processing')
print('  • Fine-tuning job management and monitoring')
print('  • Model deployment and testing')
print('  • Performance metrics and evaluation')
print('')
print('🚀 You can now use the OpenAI Fine-Tuning interface!')
"

echo ""
echo "🎯 Quick Start:"
echo "1. Run your Streamlit app: streamlit run src/app.py"
echo "2. Navigate to the Fine-Tuning tab"
echo "3. Select 'OpenAI GPT Fine-Tuning'"
echo "4. Upload your training data and start fine-tuning"
echo ""
echo "📚 For more information, see the fine-tuning documentation."
