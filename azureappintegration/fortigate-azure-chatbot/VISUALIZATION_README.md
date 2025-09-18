# 📊 Apache ECharts Visualization Integration

## Overview

The FortiGate Azure Chatbot now includes advanced Apache ECharts visualizations for comprehensive fine-tuning monitoring and performance analysis. This enhancement provides real-time insights into both OpenAI and Llama 7B fine-tuning processes.

## 🎯 Features

### Real-Time Monitoring
- **Training Progress**: Live loss curves, learning rate tracking, and epoch progression
- **Performance Metrics**: Accuracy, precision, recall, F1-score, and perplexity visualization
- **System Resources**: CPU, memory, and GPU usage monitoring
- **Model Comparisons**: Side-by-side performance analysis of different models

### Interactive Dashboards
- **Training Timeline**: Milestone tracking and training phase visualization
- **Loss Landscape**: 3D visualization of training loss progression
- **Resource Utilization**: Real-time system monitoring with alerts
- **Performance Trends**: Historical performance tracking and analysis

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Run the automated setup script
./setup_visualization.sh

# Or install manually
pip install -r requirements_visualization.txt
```

### 2. Launch the Application
```bash
streamlit run src/app.py
```

### 3. Access Visualization Features
1. Navigate to the **Fine-Tuning** tab
2. Choose either **OpenAI GPT Fine-Tuning** or **Llama 7B Local Fine-Tuning**
3. Click on the **📊 Performance Dashboard** tab
4. Start monitoring and view real-time charts

## 📈 Available Charts

### 1. Training Progress Chart
- **Purpose**: Monitor training loss and validation loss over epochs
- **Features**: 
  - Dual-axis visualization
  - Learning rate overlay
  - Interactive zoom and pan
  - Real-time updates during training

### 2. Performance Metrics Chart
- **Purpose**: Track model performance indicators
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Perplexity
- **Features**: 
  - Multi-metric radar chart
  - Historical comparison
  - Threshold indicators

### 3. System Resource Monitor
- **Purpose**: Monitor system resources during training
- **Metrics**: CPU usage, Memory usage, GPU utilization
- **Features**: 
  - Real-time updates
  - Resource alerts
  - Historical trends

### 4. Model Comparison Dashboard
- **Purpose**: Compare different model versions and configurations
- **Features**: 
  - Side-by-side performance comparison
  - Training time analysis
  - Parameter efficiency metrics

### 5. Loss Landscape Visualization
- **Purpose**: 3D visualization of loss function behavior
- **Features**: 
  - Interactive 3D surface plots
  - Gradient visualization
  - Convergence analysis

### 6. Training Timeline
- **Purpose**: Track training milestones and phases
- **Features**: 
  - Gantt-style timeline
  - Milestone markers
  - Phase duration analysis

## 🔧 Configuration

### Dashboard Controls
- **🚀 Start Monitoring**: Begin real-time system resource monitoring
- **⏹️ Stop Monitoring**: Stop system monitoring to save resources
- **🔄 Auto Refresh**: Enable automatic chart updates (2-second intervals)
- **📈 Add Sample Metrics**: Add demonstration data for testing
- **🔄 Clear Metrics**: Reset all visualization data

### Customization Options
- Chart refresh intervals
- Metric thresholds and alerts
- Color schemes and themes
- Data retention periods

## 🛠️ Technical Implementation

### Architecture
```
visualization_charts.py
├── FineTuningVisualizer (Main Class)
├── Chart Components
│   ├── Training Progress Charts
│   ├── Performance Metrics Charts
│   ├── System Resource Monitors
│   ├── Model Comparison Charts
│   ├── Loss Landscape Visualization
│   └── Training Timeline Charts
└── System Monitoring
    ├── Background Thread Monitoring
    ├── Resource Data Collection
    └── Real-time Updates
```

### Dependencies
- **streamlit-echarts**: Apache ECharts integration for Streamlit
- **psutil**: System resource monitoring
- **GPUtil**: GPU monitoring (optional)
- **numpy/pandas**: Data processing and manipulation

### Integration Points
- **OpenAI Fine-Tuning**: Integrated into the OpenAI fine-tuning workflow
- **Llama Fine-Tuning**: Embedded in the Llama 7B fine-tuning interface
- **Real-time Updates**: Automatic metric collection during training processes

## 📊 Usage Examples

### OpenAI Fine-Tuning Visualization
1. Select "🤖 OpenAI GPT Fine-Tuning"
2. Navigate to "📊 Performance Dashboard" tab
3. Click "📈 Add Sample Metrics" to see demo data
4. Monitor real-time updates during actual fine-tuning

### Llama Fine-Tuning Visualization
1. Select "🦙 Llama 7B Local Fine-Tuning"
2. Go to "📊 Performance Dashboard" tab
3. Click "🚀 Start Monitoring" for system resource tracking
4. Begin fine-tuning to see live progress updates

### System Resource Monitoring
```python
# Automatic integration during fine-tuning
visualizer.start_system_monitoring()  # Starts background monitoring
# Training metrics are automatically collected
visualizer.stop_system_monitoring()   # Stops monitoring
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Charts Not Displaying
- **Cause**: Missing streamlit-echarts dependency
- **Solution**: Run `pip install streamlit-echarts`

#### 2. System Monitoring Not Working
- **Cause**: Missing psutil dependency
- **Solution**: Run `pip install psutil`

#### 3. GPU Monitoring Unavailable
- **Cause**: Missing GPUtil or no NVIDIA GPU
- **Solution**: Install GPUtil or use CPU-only monitoring

#### 4. Performance Issues
- **Cause**: High refresh rate or too much historical data
- **Solution**: Reduce refresh frequency or clear metrics

### Debug Mode
Enable debug logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🎨 Customization

### Adding Custom Charts
```python
def create_custom_chart(data):
    option = {
        "title": {"text": "Custom Chart"},
        "xAxis": {"type": "category", "data": data["categories"]},
        "yAxis": {"type": "value"},
        "series": [{"data": data["values"], "type": "line"}]
    }
    return option
```

### Custom Metrics Integration
```python
# Add custom metrics to visualizer
visualizer.add_training_metric({
    'epoch': epoch,
    'custom_metric': value,
    'timestamp': datetime.now().strftime('%H:%M:%S')
})
```

## 📚 Resources

### Apache ECharts Documentation
- [Official ECharts Documentation](https://echarts.apache.org/en/index.html)
- [Streamlit-ECharts GitHub](https://github.com/andfanilo/streamlit-echarts)

### Performance Monitoring
- [psutil Documentation](https://psutil.readthedocs.io/)
- [GPUtil Documentation](https://github.com/anderskm/gputil)

## 🤝 Contributing

To contribute to the visualization features:

1. Fork the repository
2. Create a feature branch
3. Add new chart types or enhance existing ones
4. Test with both OpenAI and Llama fine-tuning workflows
5. Submit a pull request

## 📝 License

This visualization enhancement is part of the FortiGate Azure Chatbot project and follows the same licensing terms.

---

**🎉 Enjoy your enhanced fine-tuning experience with beautiful, interactive visualizations!**
