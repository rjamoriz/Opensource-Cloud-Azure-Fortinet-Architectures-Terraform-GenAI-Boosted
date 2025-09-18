#!/bin/bash

# Setup script for Apache ECharts Visualization Dependencies
# FortiGate Azure Chatbot - Fine-Tuning Visualization Enhancement

echo "🎨 Setting up Apache ECharts Visualization Dependencies..."
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

# Install visualization dependencies
echo "📦 Installing visualization dependencies..."
pip install -r requirements_visualization.txt

# Verify installations
echo ""
echo "🔍 Verifying installations..."

python -c "
import sys
try:
    import streamlit_echarts
    print('✅ streamlit-echarts installed successfully')
except ImportError:
    print('❌ streamlit-echarts installation failed')
    sys.exit(1)

try:
    import psutil
    print('✅ psutil installed successfully')
except ImportError:
    print('❌ psutil installation failed')
    sys.exit(1)

try:
    import GPUtil
    print('✅ GPUtil installed successfully')
except ImportError:
    print('⚠️  GPUtil installation failed (optional for NVIDIA GPU monitoring)')

print('')
print('🎉 Visualization setup complete!')
print('')
print('📊 Available Features:')
print('  • Real-time training progress charts')
print('  • Performance metrics visualization')
print('  • System resource monitoring')
print('  • Model comparison dashboards')
print('  • Interactive Apache ECharts integration')
print('')
print('🚀 You can now use the Performance Dashboard tabs in both:')
print('  • OpenAI Fine-Tuning interface')
print('  • Llama 7B Fine-Tuning interface')
"

echo ""
echo "🎯 Next Steps:"
echo "1. Run your Streamlit app: streamlit run src/app.py"
echo "2. Navigate to the Fine-Tuning tab"
echo "3. Select either OpenAI or Llama fine-tuning"
echo "4. Check out the 'Performance Dashboard' tab"
echo ""
echo "📚 For more information, see the visualization documentation."
