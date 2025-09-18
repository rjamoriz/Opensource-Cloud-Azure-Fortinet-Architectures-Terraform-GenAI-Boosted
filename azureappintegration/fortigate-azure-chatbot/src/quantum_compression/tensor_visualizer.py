"""
3D Tensor Visualization for Tucker Decomposition
Real-time visualization of compression and gradient descent processes
"""

import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TensorVisualizer:
    """3D visualization for tensor compression and optimization"""
    
    def __init__(self):
        self.compression_history = []
        self.gradient_history = []
        self.tensor_shapes = []
        self.decomposition_progress = {}
        
    def visualize_compression_progress(self, 
                                     original_tensor: torch.Tensor,
                                     compressed_tensor: torch.Tensor,
                                     compression_ratio: float,
                                     layer_name: str) -> go.Figure:
        """Create real-time compression visualization"""
        
        # Calculate compression metrics
        original_size = original_tensor.numel()
        compressed_size = compressed_tensor.numel()
        actual_ratio = 1 - (compressed_size / original_size)
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f'Original Tensor Shape: {original_tensor.shape}',
                f'Compressed Tensor Shape: {compressed_tensor.shape}',
                'Compression Progress',
                'Value Distribution Comparison'
            ],
            specs=[[{"type": "surface"}, {"type": "surface"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # 3D visualization of original tensor (sample slice)
        if len(original_tensor.shape) >= 2:
            original_slice = original_tensor[:min(50, original_tensor.shape[0]), 
                                           :min(50, original_tensor.shape[1])].detach().cpu().numpy()
            
            fig.add_trace(
                go.Surface(
                    z=original_slice,
                    colorscale='Viridis',
                    name='Original',
                    showscale=False
                ),
                row=1, col=1
            )
        
        # 3D visualization of compressed tensor (sample slice)
        if len(compressed_tensor.shape) >= 2:
            compressed_slice = compressed_tensor[:min(50, compressed_tensor.shape[0]), 
                                               :min(50, compressed_tensor.shape[1])].detach().cpu().numpy()
            
            fig.add_trace(
                go.Surface(
                    z=compressed_slice,
                    colorscale='Plasma',
                    name='Compressed',
                    showscale=False
                ),
                row=1, col=2
            )
        
        # Compression progress over layers
        self.compression_history.append({
            'layer': layer_name,
            'target_ratio': compression_ratio,
            'actual_ratio': actual_ratio,
            'original_size': original_size,
            'compressed_size': compressed_size
        })
        
        if len(self.compression_history) > 1:
            layers = [h['layer'] for h in self.compression_history]
            ratios = [h['actual_ratio'] for h in self.compression_history]
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(layers))),
                    y=ratios,
                    mode='lines+markers',
                    name='Compression Ratio',
                    line=dict(color='blue', width=3)
                ),
                row=2, col=1
            )
        
        # Value distribution comparison
        original_values = original_tensor.flatten().detach().cpu().numpy()
        compressed_values = compressed_tensor.flatten().detach().cpu().numpy()
        
        fig.add_trace(
            go.Histogram(
                x=original_values[:10000],  # Sample for performance
                name='Original Values',
                opacity=0.7,
                nbinsx=50,
                marker_color='blue'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=compressed_values[:10000],  # Sample for performance
                name='Compressed Values',
                opacity=0.7,
                nbinsx=50,
                marker_color='red'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Tucker Decomposition: {layer_name}',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def visualize_tucker_decomposition_3d(self, 
                                        core_tensor: torch.Tensor,
                                        factors: List[torch.Tensor],
                                        iteration: int = 0) -> go.Figure:
        """Visualize Tucker decomposition components in 3D"""
        
        # Create subplot for core and factors
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Tucker Core Tensor',
                'Factor Matrix 1',
                'Factor Matrix 2',
                'Reconstruction Error'
            ],
            specs=[[{"type": "surface"}, {"type": "heatmap"}],
                   [{"type": "heatmap"}, {"type": "scatter3d"}]]
        )
        
        # Visualize core tensor (3D surface)
        if len(core_tensor.shape) >= 3:
            # Take a slice of the core tensor
            core_slice = core_tensor[0, :, :].detach().cpu().numpy()
        else:
            core_slice = core_tensor.detach().cpu().numpy()
        
        fig.add_trace(
            go.Surface(
                z=core_slice,
                colorscale='RdBu',
                name='Core Tensor'
            ),
            row=1, col=1
        )
        
        # Visualize factor matrices
        if len(factors) >= 2:
            factor1 = factors[0].detach().cpu().numpy()
            factor2 = factors[1].detach().cpu().numpy()
            
            fig.add_trace(
                go.Heatmap(
                    z=factor1,
                    colorscale='Blues',
                    name='Factor 1'
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Heatmap(
                    z=factor2,
                    colorscale='Reds',
                    name='Factor 2'
                ),
                row=2, col=1
            )
        
        # 3D scatter plot of core tensor values
        if len(core_tensor.shape) >= 3:
            core_np = core_tensor.detach().cpu().numpy()
            # Sample points for visualization
            indices = np.random.choice(core_np.size, min(1000, core_np.size), replace=False)
            coords = np.unravel_index(indices, core_np.shape)
            values = core_np.flatten()[indices]
            
            fig.add_trace(
                go.Scatter3d(
                    x=coords[0],
                    y=coords[1],
                    z=coords[2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=values,
                        colorscale='Viridis',
                        showscale=True
                    ),
                    name='Core Values'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f'Tucker Decomposition Components (Iteration {iteration})',
            height=800
        )
        
        return fig
    
    def visualize_gradient_descent(self, 
                                 gradients: List[torch.Tensor],
                                 losses: List[float],
                                 iteration: int) -> go.Figure:
        """Visualize gradient descent process for tensor optimization"""
        
        # Store gradient information
        self.gradient_history.append({
            'iteration': iteration,
            'loss': losses[-1] if losses else 0,
            'gradient_norm': torch.norm(gradients[0]).item() if gradients else 0
        })
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Loss Convergence',
                'Gradient Magnitude',
                'Gradient Flow 3D',
                'Optimization Landscape'
            ],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter3d"}, {"type": "surface"}]]
        )
        
        # Loss convergence plot
        if len(losses) > 1:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(losses))),
                    y=losses,
                    mode='lines+markers',
                    name='Loss',
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )
        
        # Gradient magnitude over iterations
        if len(self.gradient_history) > 1:
            iterations = [h['iteration'] for h in self.gradient_history]
            grad_norms = [h['gradient_norm'] for h in self.gradient_history]
            
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=grad_norms,
                    mode='lines+markers',
                    name='Gradient Norm',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=2
            )
        
        # 3D gradient flow visualization
        if gradients and len(gradients[0].shape) >= 2:
            grad_tensor = gradients[0].detach().cpu().numpy()
            
            # Sample gradient vectors for 3D visualization
            if len(grad_tensor.shape) >= 2:
                h, w = grad_tensor.shape[:2]
                step = max(1, min(h, w) // 10)
                
                x_coords = []
                y_coords = []
                z_coords = []
                u_vals = []
                v_vals = []
                w_vals = []
                
                for i in range(0, h, step):
                    for j in range(0, w, step):
                        x_coords.append(i)
                        y_coords.append(j)
                        z_coords.append(0)
                        u_vals.append(grad_tensor[i, j] if len(grad_tensor.shape) == 2 else grad_tensor[i, j, 0])
                        v_vals.append(0)
                        w_vals.append(grad_tensor[i, j] if len(grad_tensor.shape) == 2 else 
                                    (grad_tensor[i, j, 1] if grad_tensor.shape[2] > 1 else 0))
                
                fig.add_trace(
                    go.Scatter3d(
                        x=x_coords,
                        y=y_coords,
                        z=z_coords,
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=u_vals,
                            colorscale='RdYlBu',
                            showscale=True
                        ),
                        name='Gradient Field'
                    ),
                    row=2, col=1
                )
        
        # Optimization landscape (loss surface approximation)
        if len(self.gradient_history) >= 4:
            # Create a simple 2D loss landscape visualization
            x_range = np.linspace(-2, 2, 20)
            y_range = np.linspace(-2, 2, 20)
            X, Y = np.meshgrid(x_range, y_range)
            
            # Simulate loss landscape (quadratic with noise)
            Z = X**2 + Y**2 + 0.1 * np.random.randn(*X.shape)
            
            fig.add_trace(
                go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    colorscale='Viridis',
                    opacity=0.8,
                    name='Loss Landscape'
                ),
                row=2, col=2
            )
            
            # Add optimization path
            if len(self.gradient_history) > 1:
                path_x = np.linspace(-1, 1, len(self.gradient_history))
                path_y = np.linspace(-1, 1, len(self.gradient_history))
                path_z = [h['loss'] for h in self.gradient_history]
                
                fig.add_trace(
                    go.Scatter3d(
                        x=path_x,
                        y=path_y,
                        z=path_z,
                        mode='lines+markers',
                        line=dict(color='red', width=5),
                        marker=dict(size=4, color='red'),
                        name='Optimization Path'
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title=f'Gradient Descent Visualization (Iteration {iteration})',
            height=800
        )
        
        return fig
    
    def create_real_time_progress_bar(self, 
                                    current_layer: int,
                                    total_layers: int,
                                    layer_name: str,
                                    compression_ratio: float) -> None:
        """Create animated progress visualization"""
        
        progress = current_layer / total_layers
        
        # Create progress bar with custom styling
        progress_html = f"""
        <div style="background-color: #f0f0f0; border-radius: 10px; padding: 10px; margin: 10px 0;">
            <h4>🗜️ Compressing Layer: {layer_name}</h4>
            <div style="background-color: #e0e0e0; border-radius: 5px; height: 20px; position: relative;">
                <div style="background: linear-gradient(90deg, #4CAF50, #2196F3); 
                           width: {progress*100}%; height: 100%; border-radius: 5px; 
                           transition: width 0.3s ease;"></div>
                <div style="position: absolute; top: 0; left: 50%; transform: translateX(-50%); 
                           line-height: 20px; color: white; font-weight: bold;">
                    {progress*100:.1f}%
                </div>
            </div>
            <p>Layer {current_layer}/{total_layers} | Target Compression: {compression_ratio:.1%}</p>
        </div>
        """
        
        st.markdown(progress_html, unsafe_allow_html=True)
    
    def create_compression_metrics_dashboard(self, stats: Dict) -> go.Figure:
        """Create comprehensive metrics dashboard"""
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Model Size Reduction',
                'Layer Compression Efficiency',
                'Memory Usage',
                'Compression Speed',
                'Quality Metrics',
                'Hardware Utilization'
            ],
            specs=[[{"type": "indicator"}, {"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "radar"}, {"type": "pie"}]]
        )
        
        # Model size reduction gauge
        size_reduction = stats.get('size_reduction', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=size_reduction * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Size Reduction %"},
                delta={'reference': 30},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 50], 'color': "gray"},
                        {'range': [50, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )
        
        # Layer compression efficiency
        if 'layer_stats' in stats:
            layer_names = list(stats['layer_stats'].keys())
            compression_ratios = [stats['layer_stats'][name].get('compression_ratio', 0) 
                                for name in layer_names]
            
            fig.add_trace(
                go.Bar(
                    x=layer_names,
                    y=compression_ratios,
                    name='Layer Compression',
                    marker_color='lightblue'
                ),
                row=1, col=2
            )
        
        # Memory usage over time
        if hasattr(self, 'memory_history'):
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(self.memory_history))),
                    y=self.memory_history,
                    mode='lines+markers',
                    name='Memory Usage (MB)',
                    line=dict(color='orange')
                ),
                row=1, col=3
            )
        
        # Compression speed
        compression_times = stats.get('layer_times', [])
        if compression_times:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(compression_times))),
                    y=compression_times,
                    mode='lines+markers',
                    name='Time per Layer (s)',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
        
        # Quality metrics radar chart
        quality_metrics = {
            'Accuracy Retention': stats.get('accuracy_retention', 0.95) * 100,
            'Speed Improvement': stats.get('speed_improvement', 1.5) * 20,
            'Memory Efficiency': stats.get('memory_efficiency', 0.7) * 100,
            'Compression Ratio': size_reduction * 100,
            'Stability': stats.get('stability_score', 0.9) * 100
        }
        
        fig.add_trace(
            go.Scatterpolar(
                r=list(quality_metrics.values()),
                theta=list(quality_metrics.keys()),
                fill='toself',
                name='Quality Metrics'
            ),
            row=2, col=2
        )
        
        # Hardware utilization pie chart
        hw_usage = {
            'GPU Memory': stats.get('gpu_memory_used', 60),
            'CPU Usage': stats.get('cpu_usage', 30),
            'Available': 100 - stats.get('gpu_memory_used', 60) - stats.get('cpu_usage', 30)
        }
        
        fig.add_trace(
            go.Pie(
                labels=list(hw_usage.keys()),
                values=list(hw_usage.values()),
                name="Hardware Usage"
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title='Tucker Compression Analytics Dashboard',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def animate_tensor_decomposition(self, 
                                   original_shape: Tuple,
                                   target_ranks: List[int],
                                   steps: int = 10) -> go.Figure:
        """Create animated visualization of tensor decomposition process"""
        
        frames = []
        
        for step in range(steps):
            # Simulate decomposition progress
            progress = step / (steps - 1)
            
            # Create frame data
            frame_data = []
            
            # Simulate core tensor evolution
            current_ranks = [int(rank * progress + 1) for rank in target_ranks]
            
            # Create visualization data for this step
            x = np.arange(original_shape[0])
            y = np.arange(original_shape[1])
            X, Y = np.meshgrid(x, y)
            
            # Simulate tensor values changing during decomposition
            Z = np.sin(X * progress + Y * progress) * np.exp(-progress)
            
            frame_data.append(
                go.Surface(
                    x=X, y=Y, z=Z,
                    colorscale='Viridis',
                    name=f'Step {step}'
                )
            )
            
            frames.append(go.Frame(data=frame_data, name=f'frame_{step}'))
        
        # Create initial figure
        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            title='Animated Tucker Decomposition Process',
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Tensor Values'
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 500, 'redraw': True},
                            'fromcurrent': True
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }]
        )
        
        return fig
    
    def clear_history(self):
        """Clear visualization history"""
        self.compression_history = []
        self.gradient_history = []
        self.tensor_shapes = []
        self.decomposition_progress = {}
