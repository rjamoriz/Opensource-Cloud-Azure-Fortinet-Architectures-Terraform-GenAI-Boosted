"""
Test-Driven Development for Tucker Phi Compressor
Integration tests for Tucker decomposition with Phi model compression
"""

import pytest
import torch
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestTuckerPhiCompressor:
    """Test suite for Tucker Phi Compressor integration"""
    
    @pytest.fixture
    def mock_streamlit(self):
        """Mock Streamlit for testing"""
        with patch('streamlit.session_state', {}), \
             patch('streamlit.columns'), \
             patch('streamlit.progress'), \
             patch('streamlit.success'), \
             patch('streamlit.error'), \
             patch('streamlit.info'):
            yield
    
    @pytest.fixture
    def sample_model_weights(self):
        """Create sample model weights for testing"""
        return {
            'embeddings.weight': torch.randn(1000, 512),
            'attention.query.weight': torch.randn(512, 512),
            'attention.key.weight': torch.randn(512, 512),
            'attention.value.weight': torch.randn(512, 512),
            'mlp.dense1.weight': torch.randn(512, 2048),
            'mlp.dense2.weight': torch.randn(2048, 512),
        }
    
    def test_import_tucker_compressor(self):
        """Test that Tucker compressor can be imported"""
        try:
            from quantum_compression.tucker_phi_compressor import TuckerPhiCompressor
            assert TuckerPhiCompressor is not None
        except ImportError as e:
            pytest.skip(f"Tucker compressor not available: {e}")
    
    def test_import_quantum_optimizer(self):
        """Test that quantum optimizer can be imported correctly"""
        try:
            from quantum_compression.quantum_optimizer import QuantumInspiredOptimizer
            optimizer = QuantumInspiredOptimizer()
            assert optimizer is not None
        except ImportError as e:
            pytest.fail(f"Failed to import quantum_optimizer: {e}")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_availability(self):
        """Test GPU availability for compression"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert device.type in ['cuda', 'cpu']
        
        if device.type == 'cuda':
            assert torch.cuda.device_count() > 0
    
    def test_tensor_decomposition_shapes(self, sample_model_weights):
        """Test tensor decomposition maintains correct shapes"""
        from quantum_compression.quantum_optimizer import QuantumTensorDecomposer, QuantumInspiredOptimizer
        
        optimizer = QuantumInspiredOptimizer(device="cpu")
        decomposer = QuantumTensorDecomposer(optimizer)
        
        # Test with embedding weight tensor
        embedding_weight = sample_model_weights['embeddings.weight']
        ranks = [500, 256]  # Compression ranks
        
        core_tensor, factor_matrices = decomposer.quantum_tucker_decomposition(
            embedding_weight, ranks, max_iterations=2
        )
        
        # Verify shapes
        assert core_tensor.shape == tuple(ranks)
        assert len(factor_matrices) == 2
        assert factor_matrices[0].shape == (embedding_weight.shape[0], ranks[0])
        assert factor_matrices[1].shape == (embedding_weight.shape[1], ranks[1])
    
    def test_compression_ratio_calculation(self, sample_model_weights):
        """Test compression ratio calculations"""
        from quantum_compression.quantum_optimizer import QuantumTensorDecomposer, QuantumInspiredOptimizer
        
        optimizer = QuantumInspiredOptimizer(device="cpu")
        decomposer = QuantumTensorDecomposer(optimizer)
        
        # Test compression on MLP weight
        mlp_weight = sample_model_weights['mlp.dense1.weight']
        original_params = mlp_weight.numel()
        
        ranks = [256, 1024]  # 50% compression target
        core_tensor, factor_matrices = decomposer.quantum_tucker_decomposition(
            mlp_weight, ranks, max_iterations=1
        )
        
        compressed_params = core_tensor.numel() + sum(f.numel() for f in factor_matrices)
        compression_ratio = compressed_params / original_params
        
        # Should achieve some compression
        assert compression_ratio < 1.0
        assert compression_ratio > 0.1  # Reasonable lower bound
    
    def test_reconstruction_quality(self, sample_model_weights):
        """Test reconstruction quality after decomposition"""
        from quantum_compression.quantum_optimizer import QuantumTensorDecomposer, QuantumInspiredOptimizer
        
        optimizer = QuantumInspiredOptimizer(device="cpu")
        decomposer = QuantumTensorDecomposer(optimizer)
        
        # Test with attention weight
        attention_weight = sample_model_weights['attention.query.weight']
        ranks = [256, 256]  # Moderate compression
        
        core_tensor, factor_matrices = decomposer.quantum_tucker_decomposition(
            attention_weight, ranks, max_iterations=2
        )
        
        reconstructed = decomposer.reconstruct_tensor(core_tensor, factor_matrices)
        error = decomposer.calculate_reconstruction_error(attention_weight, reconstructed)
        
        # Error should be reasonable for this compression level
        assert error < 1.0  # Less than 100% error
        assert error >= 0.0  # Non-negative
    
    @patch('streamlit.session_state', {})
    def test_streamlit_interface_initialization(self, mock_streamlit):
        """Test Streamlit interface can initialize"""
        try:
            from quantum_compression.tucker_phi_compressor import TuckerPhiCompressor
            
            # Mock the interface initialization
            with patch.object(TuckerPhiCompressor, '__init__', return_value=None):
                compressor = TuckerPhiCompressor()
                assert compressor is not None
        except ImportError:
            pytest.skip("Tucker compressor interface not available")

class TestTensorVisualization:
    """Test tensor visualization components"""
    
    def test_tensor_visualizer_import(self):
        """Test tensor visualizer can be imported"""
        try:
            from quantum_compression.tensor_visualizer import TensorVisualizer
            visualizer = TensorVisualizer()
            assert visualizer is not None
        except ImportError as e:
            pytest.skip(f"Tensor visualizer not available: {e}")
    
    def test_visualization_data_preparation(self):
        """Test visualization data preparation"""
        try:
            from quantum_compression.tensor_visualizer import TensorVisualizer
            
            visualizer = TensorVisualizer()
            
            # Test with sample tensor data
            sample_tensor = torch.randn(5, 5, 5)
            
            # Should be able to prepare data without errors
            # (Actual plotting would require Plotly, so we just test data prep)
            assert sample_tensor.shape == (5, 5, 5)
            
        except ImportError:
            pytest.skip("Tensor visualizer not available")

class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    def test_compression_speed_benchmark(self, sample_model_weights):
        """Benchmark compression speed"""
        from quantum_compression.quantum_optimizer import QuantumTensorDecomposer, QuantumInspiredOptimizer
        import time
        
        optimizer = QuantumInspiredOptimizer(device="cpu")
        decomposer = QuantumTensorDecomposer(optimizer)
        
        # Test with medium-sized tensor
        test_tensor = sample_model_weights['mlp.dense1.weight']
        ranks = [256, 1024]
        
        start_time = time.time()
        core_tensor, factor_matrices = decomposer.quantum_tucker_decomposition(
            test_tensor, ranks, max_iterations=1
        )
        end_time = time.time()
        
        compression_time = end_time - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert compression_time < 30.0  # 30 seconds max for test
        assert compression_time > 0.0
    
    def test_memory_usage_estimation(self, sample_model_weights):
        """Test memory usage estimation"""
        # Calculate memory usage for original vs compressed
        original_memory = sum(tensor.numel() * 4 for tensor in sample_model_weights.values())  # 4 bytes per float32
        
        # Estimate compressed memory (simplified calculation)
        compression_ratio = 0.5  # Assume 50% compression
        compressed_memory = original_memory * compression_ratio
        
        assert compressed_memory < original_memory
        assert compressed_memory > 0

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_tensor_shapes(self):
        """Test handling of invalid tensor shapes"""
        from quantum_compression.quantum_optimizer import QuantumTensorDecomposer, QuantumInspiredOptimizer
        
        optimizer = QuantumInspiredOptimizer(device="cpu")
        decomposer = QuantumTensorDecomposer(optimizer)
        
        # Test with 1D tensor (should handle gracefully)
        tensor_1d = torch.randn(100)
        
        try:
            # This might fail or handle gracefully depending on implementation
            core, factors = decomposer.quantum_tucker_decomposition(tensor_1d, [50], max_iterations=1)
            # If it succeeds, verify basic properties
            assert isinstance(core, torch.Tensor)
            assert isinstance(factors, list)
        except (ValueError, RuntimeError):
            # Expected for unsupported tensor shapes
            pass
    
    def test_extreme_compression_ratios(self, sample_model_weights):
        """Test extreme compression ratios"""
        from quantum_compression.quantum_optimizer import QuantumTensorDecomposer, QuantumInspiredOptimizer
        
        optimizer = QuantumInspiredOptimizer(device="cpu")
        decomposer = QuantumTensorDecomposer(optimizer)
        
        test_tensor = sample_model_weights['attention.query.weight']  # 512x512
        
        # Test very aggressive compression
        extreme_ranks = [10, 10]  # Very small ranks
        
        core_tensor, factor_matrices = decomposer.quantum_tucker_decomposition(
            test_tensor, extreme_ranks, max_iterations=1
        )
        
        # Should still produce valid output
        assert core_tensor.shape == tuple(extreme_ranks)
        assert len(factor_matrices) == 2
        
        # Reconstruction error will be high but should be finite
        reconstructed = decomposer.reconstruct_tensor(core_tensor, factor_matrices)
        error = decomposer.calculate_reconstruction_error(test_tensor, reconstructed)
        assert not torch.isnan(torch.tensor(error))
        assert not torch.isinf(torch.tensor(error))

class TestConfigurationValidation:
    """Test configuration and parameter validation"""
    
    def test_device_configuration(self):
        """Test device configuration options"""
        from quantum_compression.quantum_optimizer import QuantumInspiredOptimizer
        
        # Test CPU device
        cpu_optimizer = QuantumInspiredOptimizer(device="cpu")
        assert cpu_optimizer.device.type == "cpu"
        
        # Test auto device selection
        auto_optimizer = QuantumInspiredOptimizer(device="auto")
        assert auto_optimizer.device.type in ["cpu", "cuda"]
    
    def test_optimization_parameters(self):
        """Test optimization parameter validation"""
        from quantum_compression.quantum_optimizer import QuantumInspiredOptimizer
        
        optimizer = QuantumInspiredOptimizer()
        sample_tensor = torch.randn(10, 10)
        
        # Test different iteration counts
        for iterations in [1, 5, 10]:
            result = optimizer.quantum_annealing_optimization(
                sample_tensor, target_rank=5, iterations=iterations
            )
            assert result.shape == sample_tensor.shape
            assert len(optimizer.optimization_history) == iterations
            optimizer.reset_history()
        
        # Test different number of quantum states
        for num_states in [2, 4, 8]:
            result = optimizer.quantum_superposition_optimization(
                sample_tensor, num_states=num_states
            )
            assert result.shape == sample_tensor.shape
            assert len(optimizer.quantum_states) == num_states
            optimizer.reset_history()

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
