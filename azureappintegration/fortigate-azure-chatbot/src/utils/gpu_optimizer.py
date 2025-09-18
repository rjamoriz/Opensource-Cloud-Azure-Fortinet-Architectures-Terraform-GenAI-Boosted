"""
GPU Optimization Utilities for Windows ASUS ProArt
Optimized for NVIDIA GPU acceleration
"""

import os
import torch
import psutil
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class GPUOptimizer:
    """GPU optimization and monitoring for Windows systems"""
    
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.cuda_available else 0
        self.current_device = None
        self.gpu_memory_total = 0
        self.gpu_memory_allocated = 0
        
        if self.cuda_available:
            self.current_device = torch.cuda.current_device()
            self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
            self._optimize_cuda_settings()
    
    def _optimize_cuda_settings(self):
        """Apply CUDA optimizations for Windows"""
        try:
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Set environment variables for optimal performance
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
            os.environ['TORCH_USE_CUDA_DSA'] = '1'
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            
            # Memory management
            torch.cuda.empty_cache()
            
            logger.info("CUDA optimizations applied successfully")
            
        except Exception as e:
            logger.warning(f"Could not apply all CUDA optimizations: {e}")
    
    def get_optimal_device(self) -> str:
        """Get the optimal device for computation"""
        if self.cuda_available:
            return f"cuda:{self.current_device}"
        else:
            return "cpu"
    
    def get_optimal_dtype(self) -> torch.dtype:
        """Get optimal data type based on available hardware"""
        if self.cuda_available:
            # Use float16 for GPU to save memory and increase speed
            return torch.float16
        else:
            return torch.float32
    
    def get_optimal_batch_size(self, model_size_gb: float = 2.6) -> int:
        """Calculate optimal batch size based on GPU memory"""
        if not self.cuda_available:
            return 1  # Conservative for CPU
        
        # Estimate available memory (leave 2GB buffer)
        available_memory_gb = (self.gpu_memory_total - 2 * 1024**3) / (1024**3)
        
        # Rough estimation: batch_size = available_memory / (model_size * 4)
        optimal_batch = max(1, int(available_memory_gb / (model_size_gb * 4)))
        
        # Cap at reasonable limits
        return min(optimal_batch, 16)
    
    def monitor_gpu_usage(self) -> Dict[str, Any]:
        """Monitor current GPU usage"""
        if not self.cuda_available:
            return {"gpu_available": False}
        
        try:
            memory_allocated = torch.cuda.memory_allocated(0)
            memory_reserved = torch.cuda.memory_reserved(0)
            memory_total = self.gpu_memory_total
            
            return {
                "gpu_available": True,
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated_gb": memory_allocated / (1024**3),
                "memory_reserved_gb": memory_reserved / (1024**3),
                "memory_total_gb": memory_total / (1024**3),
                "memory_utilization": (memory_allocated / memory_total) * 100,
                "temperature": self._get_gpu_temperature()
            }
        except Exception as e:
            logger.error(f"Error monitoring GPU: {e}")
            return {"gpu_available": True, "error": str(e)}
    
    def _get_gpu_temperature(self) -> Optional[float]:
        """Get GPU temperature (Windows-specific)"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].temperature
        except ImportError:
            pass
        return None
    
    def optimize_model_loading(self, model_name: str) -> Dict[str, Any]:
        """Get optimized parameters for model loading"""
        device = self.get_optimal_device()
        dtype = self.get_optimal_dtype()
        
        config = {
            "torch_dtype": dtype,
            "device_map": "auto" if self.cuda_available else None,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True
        }
        
        # Add GPU-specific optimizations
        if self.cuda_available:
            config.update({
                "use_cache": True,
                "attn_implementation": "flash_attention_2" if self._supports_flash_attention() else None
            })
        
        return config
    
    def _supports_flash_attention(self) -> bool:
        """Check if GPU supports Flash Attention"""
        if not self.cuda_available:
            return False
        
        # Check GPU compute capability
        try:
            major, minor = torch.cuda.get_device_capability(0)
            # Flash Attention requires compute capability >= 8.0 (Ampere+)
            return major >= 8
        except:
            return False
    
    def clear_gpu_cache(self):
        """Clear GPU cache to free memory"""
        if self.cuda_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            "cuda_available": self.cuda_available,
            "gpu_count": self.device_count,
            "system_ram_gb": psutil.virtual_memory().total / (1024**3),
            "cpu_count": psutil.cpu_count(),
            "platform": "Windows"
        }
        
        if self.cuda_available:
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "pytorch_version": torch.__version__,
                "gpu_memory_gb": self.gpu_memory_total / (1024**3),
                "compute_capability": torch.cuda.get_device_capability(0)
            })
        
        return info

# Global GPU optimizer instance
gpu_optimizer = GPUOptimizer()

def get_gpu_status() -> Dict[str, Any]:
    """Get current GPU status for UI display"""
    return gpu_optimizer.monitor_gpu_usage()

def optimize_for_inference(model, tokenizer):
    """Optimize model and tokenizer for inference"""
    if gpu_optimizer.cuda_available:
        model = model.to(gpu_optimizer.get_optimal_device())
        model.eval()
        
        # Enable inference optimizations
        with torch.no_grad():
            model = torch.jit.optimize_for_inference(model)
    
    return model, tokenizer

def get_recommended_settings() -> Dict[str, Any]:
    """Get recommended settings for the current hardware"""
    return {
        "device": gpu_optimizer.get_optimal_device(),
        "dtype": gpu_optimizer.get_optimal_dtype(),
        "batch_size": gpu_optimizer.get_optimal_batch_size(),
        "use_gpu": gpu_optimizer.cuda_available,
        "system_info": gpu_optimizer.get_system_info()
    }
