"""
Pytest configuration and shared fixtures for Tucker Decomposition tests
"""

import pytest
import torch
import numpy as np
import sys
import os
from unittest.mock import Mock, patch

# Add src to path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture(scope="session")
def test_device():
    """Determine test device (CPU for CI/CD compatibility)"""
    return "cpu"

@pytest.fixture(scope="session") 
def seed_random():
    """Set random seeds for reproducible tests"""
    torch.manual_seed(42)
    np.random.seed(42)

@pytest.fixture
def sample_phi_model_structure():
    """Mock Phi model structure for testing"""
    return {
        'config': {
            'vocab_size': 51200,
            'hidden_size': 2048,
            'intermediate_size': 8192,
            'num_hidden_layers': 24,
            'num_attention_heads': 32,
            'max_position_embeddings': 2048
        },
        'layers': [
            'model.embed_tokens.weight',
            'model.layers.0.self_attn.q_proj.weight',
            'model.layers.0.self_attn.k_proj.weight', 
            'model.layers.0.self_attn.v_proj.weight',
            'model.layers.0.self_attn.dense.weight',
            'model.layers.0.mlp.gate_proj.weight',
            'model.layers.0.mlp.up_proj.weight',
            'model.layers.0.mlp.down_proj.weight'
        ]
    }

@pytest.fixture
def mock_model_weights():
    """Generate realistic mock model weights"""
    torch.manual_seed(42)
    return {
        'model.embed_tokens.weight': torch.randn(51200, 2048) * 0.02,
        'model.layers.0.self_attn.q_proj.weight': torch.randn(2048, 2048) * 0.02,
        'model.layers.0.self_attn.k_proj.weight': torch.randn(2048, 2048) * 0.02,
        'model.layers.0.self_attn.v_proj.weight': torch.randn(2048, 2048) * 0.02,
        'model.layers.0.self_attn.dense.weight': torch.randn(2048, 2048) * 0.02,
        'model.layers.0.mlp.gate_proj.weight': torch.randn(8192, 2048) * 0.02,
        'model.layers.0.mlp.up_proj.weight': torch.randn(8192, 2048) * 0.02,
        'model.layers.0.mlp.down_proj.weight': torch.randn(2048, 8192) * 0.02,
    }

@pytest.fixture
def compression_test_configs():
    """Standard compression configurations for testing"""
    return {
        'light': {'compression_ratio': 0.8, 'ranks_multiplier': 0.9},
        'medium': {'compression_ratio': 0.5, 'ranks_multiplier': 0.7},
        'aggressive': {'compression_ratio': 0.2, 'ranks_multiplier': 0.4}
    }

@pytest.fixture
def mock_streamlit_session():
    """Mock Streamlit session state"""
    with patch('streamlit.session_state', {}) as mock_session:
        yield mock_session

@pytest.fixture
def mock_streamlit_ui():
    """Mock Streamlit UI components"""
    with patch('streamlit.progress') as mock_progress, \
         patch('streamlit.success') as mock_success, \
         patch('streamlit.error') as mock_error, \
         patch('streamlit.info') as mock_info, \
         patch('streamlit.warning') as mock_warning, \
         patch('streamlit.columns') as mock_columns:
        
        mock_columns.return_value = [Mock(), Mock(), Mock()]
        
        yield {
            'progress': mock_progress,
            'success': mock_success,
            'error': mock_error,
            'info': mock_info,
            'warning': mock_warning,
            'columns': mock_columns
        }

@pytest.fixture
def performance_thresholds():
    """Performance thresholds for benchmarking"""
    return {
        'compression_time_max': 60.0,  # seconds
        'memory_overhead_max': 2.0,    # 2x original memory max
        'reconstruction_error_max': 0.5, # 50% max error
        'min_compression_ratio': 0.1,   # 10% minimum compression
        'max_compression_ratio': 0.9    # 90% maximum compression
    }

@pytest.fixture
def quantum_optimization_configs():
    """Quantum optimization test configurations"""
    return {
        'annealing': {
            'iterations': [1, 5, 10],
            'temperature_schedules': [
                None,  # Default
                [1.0, 0.5, 0.1],  # Custom
                [0.1] * 5  # Constant low temperature
            ]
        },
        'superposition': {
            'num_states': [2, 4, 8, 16]
        },
        'tunneling': {
            'thresholds': [1e-6, 1e-4, 1e-2]
        }
    }

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as performance benchmarks"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Mark GPU tests
        if "gpu" in item.nodeid.lower() or "cuda" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Mark slow tests
        if "benchmark" in item.nodeid.lower() or "performance" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.benchmark)
        
        # Mark integration tests
        if "integration" in item.nodeid.lower() or "complete" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
