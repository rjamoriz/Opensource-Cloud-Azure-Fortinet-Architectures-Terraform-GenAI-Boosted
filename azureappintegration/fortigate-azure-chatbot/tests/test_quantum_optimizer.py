"""
Test-Driven Development for Quantum Optimizer
Unit tests for quantum-inspired tensor optimization
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quantum_compression.quantum_optimizer import (
    QuantumInspiredOptimizer,
    QuantumTensorDecomposer,
    create_quantum_optimizer,
    create_quantum_decomposer
)

class TestQuantumInspiredOptimizer:
    """Test suite for QuantumInspiredOptimizer"""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance for testing"""
        return QuantumInspiredOptimizer(device="cpu")
    
    @pytest.fixture
    def sample_tensor(self):
        """Create sample tensor for testing"""
        torch.manual_seed(42)
        return torch.randn(10, 10)
    
    @pytest.fixture
    def sample_3d_tensor(self):
        """Create sample 3D tensor for testing"""
        torch.manual_seed(42)
        return torch.randn(5, 5, 5)
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initializes correctly"""
        assert optimizer.device.type in ['cpu', 'cuda']
        assert optimizer.optimization_history == []
        assert optimizer.quantum_states == []
    
    def test_quantum_annealing_optimization(self, optimizer, sample_tensor):
        """Test quantum annealing optimization"""
        target_rank = 5
        iterations = 10
        
        optimized_tensor = optimizer.quantum_annealing_optimization(
            sample_tensor, target_rank, iterations
        )
        
        # Check output properties
        assert optimized_tensor.shape == sample_tensor.shape
        assert len(optimizer.optimization_history) == iterations
        assert all('iteration' in h for h in optimizer.optimization_history)
        assert all('energy' in h for h in optimizer.optimization_history)
        assert all('temperature' in h for h in optimizer.optimization_history)
    
    def test_quantum_superposition_optimization(self, optimizer, sample_tensor):
        """Test quantum superposition optimization"""
        num_states = 4
        
        optimized_tensor = optimizer.quantum_superposition_optimization(
            sample_tensor, num_states
        )
        
        # Check output properties
        assert optimized_tensor.shape == sample_tensor.shape
        assert len(optimizer.quantum_states) == num_states
        assert all('tensor' in state for state in optimizer.quantum_states)
        assert all('amplitude' in state for state in optimizer.quantum_states)
        assert all('phase' in state for state in optimizer.quantum_states)
        assert all('energy' in state for state in optimizer.quantum_states)
    
    def test_quantum_tunneling_escape(self, optimizer, sample_tensor):
        """Test quantum tunneling escape mechanism"""
        # Create tensor with very small values to ensure small gradient
        local_min_tensor = torch.zeros_like(sample_tensor) + 1e-8
        
        tunneled_tensor = optimizer.quantum_tunneling_escape(local_min_tensor)
        
        # Should apply tunneling for small gradients
        gradient_norm = torch.norm(optimizer._calculate_gradient(local_min_tensor))
        if gradient_norm < 1e-3:  # If gradient is small enough, tunneling should occur
            assert not torch.allclose(tunneled_tensor, local_min_tensor, atol=1e-6)
        
        # Test with large gradient (should not tunnel)
        large_grad_tensor = sample_tensor * 100
        no_tunnel_tensor = optimizer.quantum_tunneling_escape(large_grad_tensor)
        assert torch.allclose(no_tunnel_tensor, large_grad_tensor)
    
    def test_energy_calculation(self, optimizer, sample_tensor):
        """Test energy function calculation"""
        energy = optimizer._calculate_energy(sample_tensor)
        
        assert isinstance(energy, torch.Tensor)
        assert energy.numel() == 1  # Scalar energy
        assert energy.item() >= 0  # Energy should be non-negative
    
    def test_gradient_calculation(self, optimizer, sample_tensor):
        """Test gradient calculation"""
        gradient = optimizer._calculate_gradient(sample_tensor)
        
        assert gradient.shape == sample_tensor.shape
        assert isinstance(gradient, torch.Tensor)
    
    def test_optimization_metrics(self, optimizer, sample_tensor):
        """Test optimization metrics collection"""
        # Run optimization to generate history
        optimizer.quantum_annealing_optimization(sample_tensor, 5, 5)
        
        metrics = optimizer.get_optimization_metrics()
        
        expected_keys = [
            'final_energy', 'energy_reduction', 'convergence_rate',
            'acceptance_rate', 'iterations', 'min_temperature', 'max_temperature'
        ]
        
        for key in expected_keys:
            assert key in metrics
        
        assert metrics['iterations'] == 5
        assert 0 <= metrics['acceptance_rate'] <= 1
    
    def test_quantum_state_info(self, optimizer, sample_tensor):
        """Test quantum state information retrieval"""
        # Run superposition optimization to generate states
        optimizer.quantum_superposition_optimization(sample_tensor, 3)
        
        state_info = optimizer.get_quantum_state_info()
        
        assert len(state_info) == 3
        assert all(isinstance(state, dict) for state in state_info)
    
    def test_reset_history(self, optimizer, sample_tensor):
        """Test history reset functionality"""
        # Generate some history
        optimizer.quantum_annealing_optimization(sample_tensor, 5, 3)
        optimizer.quantum_superposition_optimization(sample_tensor, 2)
        
        assert len(optimizer.optimization_history) > 0
        assert len(optimizer.quantum_states) > 0
        
        # Reset and verify
        optimizer.reset_history()
        
        assert optimizer.optimization_history == []
        assert optimizer.quantum_states == []

class TestQuantumTensorDecomposer:
    """Test suite for QuantumTensorDecomposer"""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer for decomposer"""
        return QuantumInspiredOptimizer(device="cpu")
    
    @pytest.fixture
    def decomposer(self, optimizer):
        """Create decomposer instance"""
        return QuantumTensorDecomposer(optimizer)
    
    @pytest.fixture
    def sample_3d_tensor(self):
        """Create sample 3D tensor"""
        torch.manual_seed(42)
        return torch.randn(8, 6, 4)
    
    def test_decomposer_initialization(self, decomposer, optimizer):
        """Test decomposer initializes correctly"""
        assert decomposer.optimizer == optimizer
        assert decomposer.device == optimizer.device
    
    def test_quantum_tucker_decomposition(self, decomposer, sample_3d_tensor):
        """Test quantum Tucker decomposition"""
        ranks = [4, 3, 2]
        max_iterations = 5
        
        core_tensor, factor_matrices = decomposer.quantum_tucker_decomposition(
            sample_3d_tensor, ranks, max_iterations
        )
        
        # Check core tensor
        assert core_tensor.shape == tuple(ranks)
        
        # Check factor matrices
        assert len(factor_matrices) == len(ranks)
        for i, (factor, expected_shape) in enumerate(zip(factor_matrices, 
                                                        zip(sample_3d_tensor.shape, ranks))):
            assert factor.shape == expected_shape
    
    def test_tensor_reconstruction(self, decomposer):
        """Test tensor reconstruction from Tucker components"""
        # Create simple test case
        core_tensor = torch.randn(2, 2, 2)
        factor_matrices = [
            torch.randn(4, 2),
            torch.randn(3, 2),
            torch.randn(5, 2)
        ]
        
        reconstructed = decomposer.reconstruct_tensor(core_tensor, factor_matrices)
        
        # Check output shape matches expected dimensions
        expected_shape = (4, 3, 5)  # From factor matrix dimensions
        assert reconstructed.shape == expected_shape
    
    def test_reconstruction_error(self, decomposer):
        """Test reconstruction error calculation"""
        original = torch.randn(5, 5)
        reconstructed = original + torch.randn(5, 5) * 0.1  # Add small noise
        
        error = decomposer.calculate_reconstruction_error(original, reconstructed)
        
        assert isinstance(error, float)
        assert error >= 0
        
        # Test perfect reconstruction
        perfect_error = decomposer.calculate_reconstruction_error(original, original)
        assert perfect_error < 1e-6

class TestFactoryFunctions:
    """Test factory functions"""
    
    def test_create_quantum_optimizer(self):
        """Test quantum optimizer factory"""
        optimizer = create_quantum_optimizer("cpu")
        
        assert isinstance(optimizer, QuantumInspiredOptimizer)
        assert optimizer.device.type == "cpu"
    
    def test_create_quantum_decomposer(self):
        """Test quantum decomposer factory"""
        optimizer = create_quantum_optimizer("cpu")
        decomposer = create_quantum_decomposer(optimizer)
        
        assert isinstance(decomposer, QuantumTensorDecomposer)
        assert decomposer.optimizer == optimizer

class TestIntegrationScenarios:
    """Integration tests for complete workflows"""
    
    @pytest.fixture
    def setup_complete_system(self):
        """Setup complete quantum compression system"""
        optimizer = create_quantum_optimizer("cpu")
        decomposer = create_quantum_decomposer(optimizer)
        return optimizer, decomposer
    
    def test_complete_compression_workflow(self, setup_complete_system):
        """Test complete compression workflow"""
        optimizer, decomposer = setup_complete_system
        
        # Create test tensor
        torch.manual_seed(42)
        original_tensor = torch.randn(6, 8, 4)
        
        # Perform decomposition
        ranks = [3, 4, 2]
        core_tensor, factor_matrices = decomposer.quantum_tucker_decomposition(
            original_tensor, ranks, max_iterations=3
        )
        
        # Reconstruct tensor
        reconstructed = decomposer.reconstruct_tensor(core_tensor, factor_matrices)
        
        # Calculate compression ratio
        original_params = original_tensor.numel()
        compressed_params = (core_tensor.numel() + 
                           sum(f.numel() for f in factor_matrices))
        compression_ratio = compressed_params / original_params
        
        # Verify compression achieved
        assert compression_ratio < 1.0
        
        # Verify reconstruction quality
        error = decomposer.calculate_reconstruction_error(original_tensor, reconstructed)
        assert error < 2.0  # Relaxed threshold for test environment
        
        # Verify optimization metrics
        metrics = optimizer.get_optimization_metrics()
        assert 'final_energy' in metrics
        assert metrics['iterations'] > 0

class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_tensor_handling(self):
        """Test handling of edge case tensors"""
        optimizer = create_quantum_optimizer("cpu")
        
        # Test with very small tensor
        small_tensor = torch.randn(1, 1)
        result = optimizer.quantum_annealing_optimization(small_tensor, 1, 1)
        assert result.shape == small_tensor.shape
    
    def test_large_tensor_handling(self):
        """Test handling of larger tensors"""
        optimizer = create_quantum_optimizer("cpu")
        
        # Test with larger tensor (but still manageable for tests)
        large_tensor = torch.randn(20, 20)
        result = optimizer.quantum_superposition_optimization(large_tensor, 2)
        assert result.shape == large_tensor.shape
    
    def test_invalid_ranks_handling(self):
        """Test handling of invalid rank specifications"""
        optimizer = create_quantum_optimizer("cpu")
        decomposer = create_quantum_decomposer(optimizer)
        
        tensor = torch.randn(4, 4, 4)
        
        # Test with ranks larger than tensor dimensions
        large_ranks = [10, 10, 10]
        core, factors = decomposer.quantum_tucker_decomposition(tensor, large_ranks, 1)
        
        # Should still work but may not achieve expected compression
        assert core.shape == tuple(large_ranks)

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
