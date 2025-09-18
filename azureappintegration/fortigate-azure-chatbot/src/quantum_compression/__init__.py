"""
Quantum-Inspired Model Compression System
FortiGate Azure Chatbot - Advanced AI Optimization

This module implements quantum-inspired Tucker decomposition for model compression
with specialized fine-tuning capabilities for corporate services.
"""

from .quantum_tucker_compressor import QuantumTuckerCompressor
from .phi_model_handler import PhiModelHandler
from .quantum_optimizer import QuantumInspiredOptimizer
from .corporate_data_processor import CorporateDataProcessor
from .post_compression_trainer import PostCompressionTrainer
from .performance_evaluator import PerformanceEvaluator

__version__ = "1.0.0"
__author__ = "FortiGate Azure Chatbot Team"

# Check for required dependencies
try:
    import torch
    import tensorly
    import transformers
    QUANTUM_COMPRESSION_AVAILABLE = True
except ImportError:
    QUANTUM_COMPRESSION_AVAILABLE = False

__all__ = [
    "QuantumTuckerCompressor",
    "PhiModelHandler", 
    "QuantumInspiredOptimizer",
    "CorporateDataProcessor",
    "PostCompressionTrainer",
    "PerformanceEvaluator",
    "QUANTUM_COMPRESSION_AVAILABLE"
]
