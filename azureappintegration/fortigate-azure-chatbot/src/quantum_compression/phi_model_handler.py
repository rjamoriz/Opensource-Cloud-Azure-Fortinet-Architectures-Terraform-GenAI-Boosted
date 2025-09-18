"""
Microsoft Phi-1.5B Model Handler
Specialized handling for Phi model compression and fine-tuning
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
    TrainingArguments,
    Trainer
)
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
from dataclasses import dataclass
import json
import time

from .quantum_tucker_compressor import QuantumTuckerCompressor, CompressionConfig

logger = logging.getLogger(__name__)

@dataclass
class PhiCompressionConfig:
    """Configuration for Phi model compression"""
    model_name: str = "microsoft/phi-1_5"
    compression_ratio: float = 0.3
    preserve_embeddings: bool = True
    preserve_lm_head: bool = True
    compress_attention: bool = True
    compress_mlp: bool = True
    quantum_optimization: bool = True
    output_dir: str = "./compressed_models/phi_compressed"
    
class PhiModelHandler:
    """Handler for Microsoft Phi-1.5B model compression and management"""
    
    def __init__(self, config: PhiCompressionConfig = None):
        self.config = config or PhiCompressionConfig()
        self.model = None
        self.tokenizer = None
        self.compressed_model = None
        self.compression_stats = {}
        
        # Initialize compressor
        compression_config = CompressionConfig(
            compression_ratio=self.config.compression_ratio,
            quantum_optimization=self.config.quantum_optimization
        )
        self.compressor = QuantumTuckerCompressor(compression_config)
        
    def load_model(self) -> bool:
        """Load the Phi-1.5B model and tokenizer"""
        try:
            logger.info(f"Loading Phi model: {self.config.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            logger.info("Phi model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Phi model: {e}")
            return False
    
    def analyze_model_structure(self) -> Dict[str, Any]:
        """Analyze the Phi model structure for compression planning"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        analysis = {
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024),
            'layers': {},
            'compressible_layers': []
        }
        
        # Analyze each layer
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layer_params = sum(p.numel() for p in module.parameters())
                analysis['layers'][name] = {
                    'type': type(module).__name__,
                    'parameters': layer_params,
                    'shape': list(module.weight.shape) if hasattr(module, 'weight') else None,
                    'compressible': True
                }
                analysis['compressible_layers'].append(name)
            elif hasattr(module, 'weight') and module.weight.dim() >= 3:
                layer_params = sum(p.numel() for p in module.parameters())
                analysis['layers'][name] = {
                    'type': type(module).__name__,
                    'parameters': layer_params,
                    'shape': list(module.weight.shape),
                    'compressible': True
                }
                analysis['compressible_layers'].append(name)
        
        logger.info(f"Model analysis complete: {len(analysis['compressible_layers'])} compressible layers found")
        return analysis
    
    def compress_model(self, selective_compression: bool = True) -> bool:
        """
        Compress the Phi model using quantum-inspired Tucker decomposition
        
        Args:
            selective_compression: If True, only compress specific layer types
            
        Returns:
            Success status
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info("Starting Phi model compression...")
        start_time = time.time()
        
        try:
            # Create a copy of the model for compression
            self.compressed_model = type(self.model)(self.model.config)
            self.compressed_model.load_state_dict(self.model.state_dict())
            
            # Compress layers selectively
            compressed_layers = 0
            total_layers = 0
            
            for name, module in self.compressed_model.named_modules():
                total_layers += 1
                
                # Skip certain layers if configured
                if not self._should_compress_layer(name, module):
                    continue
                
                # Compress the layer
                try:
                    compressed_layer = self.compressor.compress_layer(module, name)
                    
                    # Replace the layer in the model
                    self._replace_layer_in_model(name, compressed_layer)
                    compressed_layers += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to compress layer {name}: {e}")
                    continue
            
            # Get compression statistics
            self.compression_stats = self.compressor.get_compression_summary()
            self.compression_stats['layers_processed'] = total_layers
            self.compression_stats['layers_compressed'] = compressed_layers
            self.compression_stats['compression_time'] = time.time() - start_time
            
            logger.info(f"Compression complete: {compressed_layers}/{total_layers} layers compressed")
            logger.info(f"Overall compression ratio: {self.compression_stats.get('overall_compression_ratio', 0):.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Model compression failed: {e}")
            return False
    
    def _should_compress_layer(self, layer_name: str, module: nn.Module) -> bool:
        """Determine if a layer should be compressed based on configuration"""
        
        # Skip embeddings if configured
        if not self.config.preserve_embeddings and 'embed' in layer_name.lower():
            return False
        
        # Skip language model head if configured
        if not self.config.preserve_lm_head and ('lm_head' in layer_name.lower() or 'output' in layer_name.lower()):
            return False
        
        # Skip attention layers if configured
        if not self.config.compress_attention and ('attn' in layer_name.lower() or 'attention' in layer_name.lower()):
            return False
        
        # Skip MLP layers if configured
        if not self.config.compress_mlp and ('mlp' in layer_name.lower() or 'feed_forward' in layer_name.lower()):
            return False
        
        # Check if layer is compressible
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            return True
        elif hasattr(module, 'weight') and module.weight.dim() >= 3:
            return True
        
        return False
    
    def _replace_layer_in_model(self, layer_name: str, new_layer: nn.Module):
        """Replace a layer in the model with the compressed version"""
        # Split the layer name to navigate the model hierarchy
        parts = layer_name.split('.')
        current = self.compressed_model
        
        # Navigate to the parent of the target layer
        for part in parts[:-1]:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                # Handle indexed access (e.g., layers.0)
                if part.isdigit():
                    current = current[int(part)]
                else:
                    raise AttributeError(f"Cannot find {part} in model structure")
        
        # Replace the final layer
        final_part = parts[-1]
        if hasattr(current, final_part):
            setattr(current, final_part, new_layer)
        elif final_part.isdigit():
            current[int(final_part)] = new_layer
        else:
            raise AttributeError(f"Cannot replace layer {layer_name}")
    
    def evaluate_compressed_model(self, test_texts: List[str] = None) -> Dict[str, Any]:
        """Evaluate the compressed model performance"""
        if self.compressed_model is None:
            raise ValueError("No compressed model available. Run compress_model() first.")
        
        if test_texts is None:
            test_texts = [
                "What is FortiGate?",
                "How to deploy FortiGate on Azure?",
                "Configure Azure network security",
                "Troubleshoot FortiGate connectivity issues"
            ]
        
        evaluation_results = {
            'original_model_size': self._get_model_size(self.model),
            'compressed_model_size': self._get_model_size(self.compressed_model),
            'compression_ratio': 0,
            'inference_times': {'original': [], 'compressed': []},
            'sample_outputs': []
        }
        
        # Calculate compression ratio
        if evaluation_results['original_model_size'] > 0:
            evaluation_results['compression_ratio'] = 1 - (
                evaluation_results['compressed_model_size'] / evaluation_results['original_model_size']
            )
        
        # Test inference performance and quality
        for text in test_texts:
            try:
                # Test original model
                original_time, original_output = self._generate_text(self.model, text)
                evaluation_results['inference_times']['original'].append(original_time)
                
                # Test compressed model
                compressed_time, compressed_output = self._generate_text(self.compressed_model, text)
                evaluation_results['inference_times']['compressed'].append(compressed_time)
                
                evaluation_results['sample_outputs'].append({
                    'input': text,
                    'original_output': original_output,
                    'compressed_output': compressed_output,
                    'speedup': original_time / compressed_time if compressed_time > 0 else 0
                })
                
            except Exception as e:
                logger.warning(f"Evaluation failed for text '{text}': {e}")
        
        # Calculate average performance metrics
        if evaluation_results['inference_times']['original']:
            evaluation_results['avg_original_time'] = sum(evaluation_results['inference_times']['original']) / len(evaluation_results['inference_times']['original'])
        if evaluation_results['inference_times']['compressed']:
            evaluation_results['avg_compressed_time'] = sum(evaluation_results['inference_times']['compressed']) / len(evaluation_results['inference_times']['compressed'])
        
        if 'avg_original_time' in evaluation_results and 'avg_compressed_time' in evaluation_results:
            evaluation_results['avg_speedup'] = evaluation_results['avg_original_time'] / evaluation_results['avg_compressed_time']
        
        return evaluation_results
    
    def _get_model_size(self, model: nn.Module) -> int:
        """Get model size in bytes"""
        return sum(p.numel() * p.element_size() for p in model.parameters())
    
    def _generate_text(self, model: nn.Module, input_text: str, max_length: int = 100) -> Tuple[float, str]:
        """Generate text and measure inference time"""
        model.eval()
        
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate text with timing
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        inference_time = time.time() - start_time
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return inference_time, generated_text
    
    def save_compressed_model(self, save_path: str = None) -> bool:
        """Save the compressed model"""
        if self.compressed_model is None:
            raise ValueError("No compressed model to save. Run compress_model() first.")
        
        save_path = save_path or self.config.output_dir
        os.makedirs(save_path, exist_ok=True)
        
        try:
            # Save model
            self.compressed_model.save_pretrained(save_path)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(save_path)
            
            # Save compression statistics
            stats_path = os.path.join(save_path, "compression_stats.json")
            with open(stats_path, 'w') as f:
                json.dump(self.compression_stats, f, indent=2)
            
            # Save configuration
            config_path = os.path.join(save_path, "compression_config.json")
            with open(config_path, 'w') as f:
                json.dump(self.config.__dict__, f, indent=2)
            
            logger.info(f"Compressed model saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save compressed model: {e}")
            return False
    
    def load_compressed_model(self, model_path: str) -> bool:
        """Load a previously compressed model"""
        try:
            self.compressed_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load compression statistics if available
            stats_path = os.path.join(model_path, "compression_stats.json")
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    self.compression_stats = json.load(f)
            
            logger.info(f"Compressed model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load compressed model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        info = {
            'original_model_loaded': self.model is not None,
            'compressed_model_available': self.compressed_model is not None,
            'compression_stats': self.compression_stats,
            'config': self.config.__dict__
        }
        
        if self.model is not None:
            info['original_model_params'] = sum(p.numel() for p in self.model.parameters())
            info['original_model_size_mb'] = self._get_model_size(self.model) / (1024 * 1024)
        
        if self.compressed_model is not None:
            info['compressed_model_params'] = sum(p.numel() for p in self.compressed_model.parameters())
            info['compressed_model_size_mb'] = self._get_model_size(self.compressed_model) / (1024 * 1024)
        
        return info
