"""
Advanced Fine-Tuning Visualization Module with Apache ECharts
Provides comprehensive monitoring and observability for model fine-tuning
"""

import streamlit as st
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
from streamlit_echarts import st_echarts
import psutil
import threading
import queue

class FineTuningVisualizer:
    """Advanced visualization system for fine-tuning monitoring"""
    
    def __init__(self):
        self.training_metrics = []
        self.system_metrics = []
        self.performance_history = []
        self.comparison_data = []
        self.is_monitoring = False
        self.metrics_queue = queue.Queue()
        
    def create_training_progress_chart(self, metrics_data: List[Dict]) -> Dict:
        """Create real-time training progress chart"""
        if not metrics_data:
            return self._empty_chart("Training Progress")
            
        epochs = [m.get('epoch', 0) for m in metrics_data]
        train_loss = [m.get('train_loss', 0) for m in metrics_data]
        val_loss = [m.get('val_loss', 0) for m in metrics_data]
        learning_rate = [m.get('learning_rate', 0) for m in metrics_data]
        
        option = {
            "title": {
                "text": "Training Progress",
                "left": "center",
                "textStyle": {"color": "#1f77b4", "fontSize": 18}
            },
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {"type": "cross"}
            },
            "legend": {
                "data": ["Training Loss", "Validation Loss", "Learning Rate"],
                "top": "10%"
            },
            "grid": {"left": "10%", "right": "10%", "bottom": "15%", "top": "25%"},
            "xAxis": {
                "type": "category",
                "data": epochs,
                "name": "Epoch",
                "nameLocation": "middle",
                "nameGap": 30
            },
            "yAxis": [
                {
                    "type": "value",
                    "name": "Loss",
                    "position": "left",
                    "axisLabel": {"formatter": "{value}"}
                },
                {
                    "type": "value",
                    "name": "Learning Rate",
                    "position": "right",
                    "axisLabel": {"formatter": "{value}"}
                }
            ],
            "series": [
                {
                    "name": "Training Loss",
                    "type": "line",
                    "data": train_loss,
                    "smooth": True,
                    "lineStyle": {"color": "#ff7f0e", "width": 3},
                    "symbol": "circle",
                    "symbolSize": 6
                },
                {
                    "name": "Validation Loss",
                    "type": "line",
                    "data": val_loss,
                    "smooth": True,
                    "lineStyle": {"color": "#2ca02c", "width": 3},
                    "symbol": "diamond",
                    "symbolSize": 6
                },
                {
                    "name": "Learning Rate",
                    "type": "line",
                    "yAxisIndex": 1,
                    "data": learning_rate,
                    "smooth": True,
                    "lineStyle": {"color": "#d62728", "width": 2, "type": "dashed"},
                    "symbol": "triangle",
                    "symbolSize": 4
                }
            ],
            "animation": True,
            "animationDuration": 1000
        }
        return option
    
    def create_performance_metrics_chart(self, performance_data: List[Dict]) -> Dict:
        """Create performance metrics visualization"""
        if not performance_data:
            return self._empty_chart("Performance Metrics")
            
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'perplexity']
        data = []
        
        for metric in metrics:
            values = [p.get(metric, 0) for p in performance_data]
            if any(v > 0 for v in values):
                data.append({
                    "name": metric.replace('_', ' ').title(),
                    "type": "bar",
                    "data": values,
                    "itemStyle": {"color": self._get_metric_color(metric)}
                })
        
        option = {
            "title": {
                "text": "Performance Metrics",
                "left": "center",
                "textStyle": {"color": "#1f77b4", "fontSize": 18}
            },
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {"type": "shadow"}
            },
            "legend": {
                "data": [d["name"] for d in data],
                "top": "10%"
            },
            "grid": {"left": "10%", "right": "10%", "bottom": "15%", "top": "25%"},
            "xAxis": {
                "type": "category",
                "data": [f"Epoch {i+1}" for i in range(len(performance_data))],
                "axisLabel": {"rotate": 45}
            },
            "yAxis": {
                "type": "value",
                "name": "Score",
                "min": 0,
                "max": 1
            },
            "series": data,
            "animation": True
        }
        return option
    
    def create_system_resource_chart(self, system_data: List[Dict]) -> Dict:
        """Create system resource monitoring chart"""
        if not system_data:
            return self._empty_chart("System Resources")
            
        timestamps = [s.get('timestamp', '') for s in system_data]
        cpu_usage = [s.get('cpu_percent', 0) for s in system_data]
        memory_usage = [s.get('memory_percent', 0) for s in system_data]
        gpu_usage = [s.get('gpu_percent', 0) for s in system_data]
        gpu_memory = [s.get('gpu_memory_percent', 0) for s in system_data]
        
        option = {
            "title": {
                "text": "System Resource Usage",
                "left": "center",
                "textStyle": {"color": "#1f77b4", "fontSize": 18}
            },
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {"type": "cross"}
            },
            "legend": {
                "data": ["CPU %", "Memory %", "GPU %", "GPU Memory %"],
                "top": "10%"
            },
            "grid": {"left": "10%", "right": "10%", "bottom": "15%", "top": "25%"},
            "xAxis": {
                "type": "category",
                "data": timestamps,
                "axisLabel": {"rotate": 45}
            },
            "yAxis": {
                "type": "value",
                "name": "Usage %",
                "min": 0,
                "max": 100
            },
            "series": [
                {
                    "name": "CPU %",
                    "type": "line",
                    "data": cpu_usage,
                    "smooth": True,
                    "lineStyle": {"color": "#1f77b4"},
                    "areaStyle": {"opacity": 0.3}
                },
                {
                    "name": "Memory %",
                    "type": "line",
                    "data": memory_usage,
                    "smooth": True,
                    "lineStyle": {"color": "#ff7f0e"},
                    "areaStyle": {"opacity": 0.3}
                },
                {
                    "name": "GPU %",
                    "type": "line",
                    "data": gpu_usage,
                    "smooth": True,
                    "lineStyle": {"color": "#2ca02c"},
                    "areaStyle": {"opacity": 0.3}
                },
                {
                    "name": "GPU Memory %",
                    "type": "line",
                    "data": gpu_memory,
                    "smooth": True,
                    "lineStyle": {"color": "#d62728"},
                    "areaStyle": {"opacity": 0.3}
                }
            ],
            "animation": True
        }
        return option
    
    def create_model_comparison_chart(self, comparison_data: List[Dict]) -> Dict:
        """Create model performance comparison chart"""
        if not comparison_data:
            return self._empty_chart("Model Comparison")
            
        models = [c.get('model_name', 'Unknown') for c in comparison_data]
        metrics = ['accuracy', 'f1_score', 'training_time', 'inference_speed']
        
        series_data = []
        for metric in metrics:
            values = [c.get(metric, 0) for c in comparison_data]
            if any(v > 0 for v in values):
                series_data.append({
                    "name": metric.replace('_', ' ').title(),
                    "type": "radar",
                    "data": [{
                        "value": values,
                        "name": "Performance"
                    }]
                })
        
        option = {
            "title": {
                "text": "Model Performance Comparison",
                "left": "center",
                "textStyle": {"color": "#1f77b4", "fontSize": 18}
            },
            "tooltip": {"trigger": "item"},
            "legend": {
                "data": models,
                "top": "10%"
            },
            "radar": {
                "indicator": [
                    {"name": "Accuracy", "max": 1},
                    {"name": "F1 Score", "max": 1},
                    {"name": "Training Time", "max": 100},
                    {"name": "Inference Speed", "max": 100}
                ],
                "center": ["50%", "60%"],
                "radius": "60%"
            },
            "series": [{
                "type": "radar",
                "data": [
                    {
                        "value": [c.get('accuracy', 0), c.get('f1_score', 0), 
                                c.get('training_time', 0), c.get('inference_speed', 0)],
                        "name": c.get('model_name', 'Unknown'),
                        "itemStyle": {"color": self._get_model_color(i)}
                    }
                    for i, c in enumerate(comparison_data)
                ]
            }]
        }
        return option
    
    def create_loss_landscape_chart(self, loss_data: List[Dict]) -> Dict:
        """Create 3D loss landscape visualization"""
        if not loss_data:
            return self._empty_chart("Loss Landscape")
            
        # Prepare 3D surface data
        surface_data = []
        for i, data in enumerate(loss_data):
            x = data.get('param1', i)
            y = data.get('param2', i)
            z = data.get('loss', 0)
            surface_data.append([x, y, z])
        
        option = {
            "title": {
                "text": "Loss Landscape",
                "left": "center",
                "textStyle": {"color": "#1f77b4", "fontSize": 18}
            },
            "tooltip": {},
            "visualMap": {
                "max": max([d[2] for d in surface_data]) if surface_data else 1,
                "inRange": {"color": ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#ffffcc", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"]}
            },
            "xAxis3D": {"type": "value"},
            "yAxis3D": {"type": "value"},
            "zAxis3D": {"type": "value"},
            "grid3D": {
                "viewControl": {"projection": "perspective"}
            },
            "series": [{
                "type": "surface",
                "data": surface_data,
                "shading": "color"
            }]
        }
        return option
    
    def create_training_timeline_chart(self, timeline_data: List[Dict]) -> Dict:
        """Create training timeline and milestones chart"""
        if not timeline_data:
            return self._empty_chart("Training Timeline")
            
        events = []
        for event in timeline_data:
            events.append({
                "name": event.get('event', 'Unknown'),
                "value": [event.get('timestamp', ''), event.get('value', 0)],
                "itemStyle": {"color": self._get_event_color(event.get('type', 'info'))}
            })
        
        option = {
            "title": {
                "text": "Training Timeline & Milestones",
                "left": "center",
                "textStyle": {"color": "#1f77b4", "fontSize": 18}
            },
            "tooltip": {
                "trigger": "axis",
                "formatter": "{b}: {c}"
            },
            "xAxis": {
                "type": "time",
                "name": "Time"
            },
            "yAxis": {
                "type": "value",
                "name": "Progress"
            },
            "series": [{
                "type": "scatter",
                "data": events,
                "symbolSize": 10,
                "emphasis": {"itemStyle": {"borderColor": "#333", "borderWidth": 2}}
            }]
        }
        return option
    
    def start_system_monitoring(self):
        """Start system resource monitoring in background"""
        self.is_monitoring = True
        
        def monitor():
            while self.is_monitoring:
                try:
                    # Get system metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    
                    # Try to get GPU metrics (if available)
                    gpu_percent = 0
                    gpu_memory_percent = 0
                    try:
                        import GPUtil
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu_percent = gpus[0].load * 100
                            gpu_memory_percent = gpus[0].memoryUtil * 100
                    except ImportError:
                        pass
                    
                    metric = {
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'gpu_percent': gpu_percent,
                        'gpu_memory_percent': gpu_memory_percent
                    }
                    
                    self.metrics_queue.put(metric)
                    time.sleep(2)  # Update every 2 seconds
                    
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    time.sleep(5)
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def stop_system_monitoring(self):
        """Stop system resource monitoring"""
        self.is_monitoring = False
    
    def get_latest_system_metrics(self, max_points: int = 50) -> List[Dict]:
        """Get latest system metrics from queue"""
        metrics = []
        while not self.metrics_queue.empty() and len(metrics) < max_points:
            try:
                metrics.append(self.metrics_queue.get_nowait())
            except queue.Empty:
                break
        
        # Keep only recent metrics
        self.system_metrics.extend(metrics)
        if len(self.system_metrics) > max_points:
            self.system_metrics = self.system_metrics[-max_points:]
        
        return self.system_metrics
    
    def add_training_metric(self, metric: Dict):
        """Add training metric to history"""
        self.training_metrics.append(metric)
    
    def add_performance_metric(self, metric: Dict):
        """Add performance metric to history"""
        self.performance_history.append(metric)
    
    def add_comparison_data(self, data: Dict):
        """Add model comparison data"""
        self.comparison_data.append(data)
    
    def _empty_chart(self, title: str) -> Dict:
        """Create empty chart placeholder"""
        return {
            "title": {
                "text": f"{title} - No Data Available",
                "left": "center",
                "textStyle": {"color": "#999", "fontSize": 16}
            },
            "graphic": {
                "type": "text",
                "left": "center",
                "top": "middle",
                "style": {
                    "text": "Start training to see visualizations",
                    "fontSize": 14,
                    "fill": "#999"
                }
            }
        }
    
    def _get_metric_color(self, metric: str) -> str:
        """Get color for specific metric"""
        colors = {
            'accuracy': '#2ca02c',
            'precision': '#ff7f0e',
            'recall': '#d62728',
            'f1_score': '#9467bd',
            'perplexity': '#8c564b'
        }
        return colors.get(metric, '#1f77b4')
    
    def _get_model_color(self, index: int) -> str:
        """Get color for model comparison"""
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        return colors[index % len(colors)]
    
    def _get_event_color(self, event_type: str) -> str:
        """Get color for timeline events"""
        colors = {
            'start': '#2ca02c',
            'checkpoint': '#ff7f0e',
            'milestone': '#1f77b4',
            'error': '#d62728',
            'complete': '#9467bd'
        }
        return colors.get(event_type, '#1f77b4')

def display_visualization_dashboard(visualizer: FineTuningVisualizer):
    """Display comprehensive visualization dashboard"""
    st.markdown("## 📊 Fine-Tuning Performance Dashboard")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Training Progress", 
        "🎯 Performance Metrics", 
        "💻 System Resources", 
        "🔄 Model Comparison",
        "🗓️ Timeline"
    ])
    
    with tab1:
        st.markdown("### Training Progress & Loss Curves")
        col1, col2 = st.columns(2)
        
        with col1:
            # Training progress chart
            training_chart = visualizer.create_training_progress_chart(visualizer.training_metrics)
            st_echarts(options=training_chart, height="400px", key="training_progress")
        
        with col2:
            # Loss landscape (if available)
            loss_chart = visualizer.create_loss_landscape_chart([])
            st_echarts(options=loss_chart, height="400px", key="loss_landscape")
    
    with tab2:
        st.markdown("### Performance Metrics & Evaluation")
        performance_chart = visualizer.create_performance_metrics_chart(visualizer.performance_history)
        st_echarts(options=performance_chart, height="500px", key="performance_metrics")
        
        # Performance summary table
        if visualizer.performance_history:
            st.markdown("#### Latest Performance Summary")
            latest = visualizer.performance_history[-1] if visualizer.performance_history else {}
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{latest.get('accuracy', 0):.3f}")
            with col2:
                st.metric("F1 Score", f"{latest.get('f1_score', 0):.3f}")
            with col3:
                st.metric("Precision", f"{latest.get('precision', 0):.3f}")
            with col4:
                st.metric("Recall", f"{latest.get('recall', 0):.3f}")
    
    with tab3:
        st.markdown("### System Resource Monitoring")
        
        # Real-time system metrics
        system_metrics = visualizer.get_latest_system_metrics()
        system_chart = visualizer.create_system_resource_chart(system_metrics)
        st_echarts(options=system_chart, height="400px", key="system_resources")
        
        # Current system status
        if system_metrics:
            latest_system = system_metrics[-1]
            st.markdown("#### Current System Status")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                cpu_val = latest_system.get('cpu_percent', 0)
                st.metric("CPU Usage", f"{cpu_val:.1f}%", 
                         delta=f"{cpu_val-50:.1f}%" if cpu_val > 50 else None)
            with col2:
                mem_val = latest_system.get('memory_percent', 0)
                st.metric("Memory Usage", f"{mem_val:.1f}%",
                         delta=f"{mem_val-70:.1f}%" if mem_val > 70 else None)
            with col3:
                gpu_val = latest_system.get('gpu_percent', 0)
                st.metric("GPU Usage", f"{gpu_val:.1f}%")
            with col4:
                gpu_mem_val = latest_system.get('gpu_memory_percent', 0)
                st.metric("GPU Memory", f"{gpu_mem_val:.1f}%")
    
    with tab4:
        st.markdown("### Model Performance Comparison")
        comparison_chart = visualizer.create_model_comparison_chart(visualizer.comparison_data)
        st_echarts(options=comparison_chart, height="500px", key="model_comparison")
        
        # Comparison table
        if visualizer.comparison_data:
            st.markdown("#### Detailed Comparison")
            df = pd.DataFrame(visualizer.comparison_data)
            st.dataframe(df, use_container_width=True)
    
    with tab5:
        st.markdown("### Training Timeline & Milestones")
        timeline_chart = visualizer.create_training_timeline_chart([])
        st_echarts(options=timeline_chart, height="400px", key="training_timeline")
        
        # Training log
        st.markdown("#### Training Log")
        if visualizer.training_metrics:
            for i, metric in enumerate(reversed(visualizer.training_metrics[-10:])):
                with st.expander(f"Epoch {metric.get('epoch', i)} - {metric.get('timestamp', 'Unknown')}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Training Loss:** {metric.get('train_loss', 'N/A')}")
                        st.write(f"**Learning Rate:** {metric.get('learning_rate', 'N/A')}")
                    with col2:
                        st.write(f"**Validation Loss:** {metric.get('val_loss', 'N/A')}")
                        st.write(f"**Duration:** {metric.get('duration', 'N/A')}")

# Global visualizer instance
_global_visualizer = None

def get_visualizer() -> FineTuningVisualizer:
    """Get global visualizer instance"""
    global _global_visualizer
    if _global_visualizer is None:
        _global_visualizer = FineTuningVisualizer()
    return _global_visualizer
