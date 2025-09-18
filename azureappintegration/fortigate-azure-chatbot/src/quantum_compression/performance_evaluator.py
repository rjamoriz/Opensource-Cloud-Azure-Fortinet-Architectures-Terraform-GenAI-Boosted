"""
Performance Evaluator for Tucker Decomposition
Comprehensive evaluation metrics for compressed models
"""

import torch
import torch.nn as nn
import time
import psutil
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class PerformanceEvaluator:
    """Evaluate performance of compressed models"""
    
    def __init__(self, device: str = "auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.evaluation_history = []
        
    def evaluate_compression_quality(self, 
                                   original_model: nn.Module,
                                   compressed_model: nn.Module,
                                   test_inputs: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate compression quality by comparing model outputs
        
        Args:
            original_model: Original uncompressed model
            compressed_model: Compressed model
            test_inputs: Test input tensors
            
        Returns:
            Quality metrics
        """
        original_model.eval()
        compressed_model.eval()
        
        with torch.no_grad():
            # Get outputs from both models
            original_outputs = original_model(test_inputs.to(self.device))
            compressed_outputs = compressed_model(test_inputs.to(self.device))
            
            # Extract logits if outputs are complex objects
            if hasattr(original_outputs, 'logits'):
                original_logits = original_outputs.logits
                compressed_logits = compressed_outputs.logits
            else:
                original_logits = original_outputs
                compressed_logits = compressed_outputs
            
            # Calculate various distance metrics
            mse_loss = nn.MSELoss()(compressed_logits, original_logits)
            mae_loss = nn.L1Loss()(compressed_logits, original_logits)
            
            # Cosine similarity
            cos_sim = nn.CosineSimilarity(dim=-1)(
                original_logits.flatten(), 
                compressed_logits.flatten()
            ).mean()
            
            # Relative error
            relative_error = torch.norm(compressed_logits - original_logits) / torch.norm(original_logits)
            
            # Top-k accuracy preservation (for classification tasks)
            top1_preserved = self._calculate_top_k_preservation(original_logits, compressed_logits, k=1)
            top5_preserved = self._calculate_top_k_preservation(original_logits, compressed_logits, k=5)
            
        return {
            'mse_loss': mse_loss.item(),
            'mae_loss': mae_loss.item(),
            'cosine_similarity': cos_sim.item(),
            'relative_error': relative_error.item(),
            'top1_preservation': top1_preserved,
            'top5_preservation': top5_preserved
        }
    
    def evaluate_inference_speed(self, 
                                model: nn.Module,
                                test_inputs: torch.Tensor,
                                num_runs: int = 100,
                                warmup_runs: int = 10) -> Dict[str, float]:
        """
        Evaluate model inference speed
        
        Args:
            model: Model to evaluate
            test_inputs: Test input tensors
            num_runs: Number of inference runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Speed metrics
        """
        model.eval()
        test_inputs = test_inputs.to(self.device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(test_inputs)
        
        # Synchronize GPU if available
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Timed runs
        inference_times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(test_inputs)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                inference_times.append(end_time - start_time)
        
        # Calculate statistics
        mean_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        
        # Throughput (samples per second)
        batch_size = test_inputs.shape[0]
        throughput = batch_size / mean_time
        
        return {
            'mean_inference_time': mean_time,
            'std_inference_time': std_time,
            'min_inference_time': min_time,
            'max_inference_time': max_time,
            'throughput_samples_per_sec': throughput,
            'batch_size': batch_size
        }
    
    def evaluate_memory_usage(self, model: nn.Module) -> Dict[str, float]:
        """
        Evaluate model memory usage
        
        Args:
            model: Model to evaluate
            
        Returns:
            Memory metrics
        """
        # Model parameters memory
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate parameter memory (assuming float32)
        param_memory_mb = total_params * 4 / (1024 * 1024)
        
        # System memory usage
        process = psutil.Process()
        system_memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # GPU memory usage if available
        gpu_memory_mb = 0
        if self.device.type == 'cuda' and torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_memory_mb': param_memory_mb,
            'system_memory_mb': system_memory_mb,
            'gpu_memory_mb': gpu_memory_mb
        }
    
    def evaluate_compression_ratio(self, 
                                 original_model: nn.Module,
                                 compressed_model: nn.Module) -> Dict[str, float]:
        """
        Calculate compression ratios
        
        Args:
            original_model: Original model
            compressed_model: Compressed model
            
        Returns:
            Compression metrics
        """
        # Parameter counts
        original_params = sum(p.numel() for p in original_model.parameters())
        compressed_params = sum(p.numel() for p in compressed_model.parameters())
        
        # Memory estimates
        original_memory = original_params * 4 / (1024 * 1024)  # MB
        compressed_memory = compressed_params * 4 / (1024 * 1024)  # MB
        
        # Ratios
        param_compression_ratio = compressed_params / original_params
        memory_compression_ratio = compressed_memory / original_memory
        
        # Savings
        param_reduction = 1 - param_compression_ratio
        memory_savings_mb = original_memory - compressed_memory
        
        return {
            'original_parameters': original_params,
            'compressed_parameters': compressed_params,
            'parameter_compression_ratio': param_compression_ratio,
            'parameter_reduction_percent': param_reduction * 100,
            'original_memory_mb': original_memory,
            'compressed_memory_mb': compressed_memory,
            'memory_compression_ratio': memory_compression_ratio,
            'memory_savings_mb': memory_savings_mb
        }
    
    def comprehensive_evaluation(self,
                               original_model: nn.Module,
                               compressed_model: nn.Module,
                               test_inputs: torch.Tensor,
                               num_speed_runs: int = 50) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of compressed model
        
        Args:
            original_model: Original model
            compressed_model: Compressed model
            test_inputs: Test inputs for evaluation
            num_speed_runs: Number of runs for speed evaluation
            
        Returns:
            Complete evaluation results
        """
        evaluation_start = time.time()
        
        # Quality evaluation
        quality_metrics = self.evaluate_compression_quality(
            original_model, compressed_model, test_inputs
        )
        
        # Speed evaluation
        original_speed = self.evaluate_inference_speed(
            original_model, test_inputs, num_speed_runs
        )
        compressed_speed = self.evaluate_inference_speed(
            compressed_model, test_inputs, num_speed_runs
        )
        
        # Memory evaluation
        original_memory = self.evaluate_memory_usage(original_model)
        compressed_memory = self.evaluate_memory_usage(compressed_model)
        
        # Compression ratios
        compression_ratios = self.evaluate_compression_ratio(
            original_model, compressed_model
        )
        
        # Speed improvement
        speed_improvement = (original_speed['mean_inference_time'] / 
                           compressed_speed['mean_inference_time'])
        
        # Overall score (weighted combination of metrics)
        overall_score = self._calculate_overall_score(
            quality_metrics, compression_ratios, speed_improvement
        )
        
        evaluation_time = time.time() - evaluation_start
        
        results = {
            'quality_metrics': quality_metrics,
            'original_speed': original_speed,
            'compressed_speed': compressed_speed,
            'original_memory': original_memory,
            'compressed_memory': compressed_memory,
            'compression_ratios': compression_ratios,
            'speed_improvement': speed_improvement,
            'overall_score': overall_score,
            'evaluation_time': evaluation_time,
            'timestamp': time.time()
        }
        
        self.evaluation_history.append(results)
        
        return results
    
    def _calculate_top_k_preservation(self, 
                                    original_logits: torch.Tensor,
                                    compressed_logits: torch.Tensor,
                                    k: int = 1) -> float:
        """Calculate top-k prediction preservation"""
        _, original_topk = torch.topk(original_logits, k, dim=-1)
        _, compressed_topk = torch.topk(compressed_logits, k, dim=-1)
        
        # Calculate overlap
        matches = 0
        total = original_topk.shape[0]
        
        for i in range(total):
            orig_set = set(original_topk[i].cpu().numpy())
            comp_set = set(compressed_topk[i].cpu().numpy())
            matches += len(orig_set.intersection(comp_set)) / k
        
        return matches / total
    
    def _calculate_overall_score(self,
                               quality_metrics: Dict[str, float],
                               compression_ratios: Dict[str, float],
                               speed_improvement: float) -> float:
        """Calculate overall compression score"""
        # Weights for different aspects
        quality_weight = 0.4
        compression_weight = 0.3
        speed_weight = 0.3
        
        # Quality score (higher cosine similarity is better)
        quality_score = quality_metrics.get('cosine_similarity', 0.0)
        
        # Compression score (lower ratio is better compression)
        compression_score = 1.0 - compression_ratios.get('parameter_compression_ratio', 1.0)
        
        # Speed score (higher improvement is better)
        speed_score = min(speed_improvement / 2.0, 1.0)  # Cap at 2x improvement
        
        overall_score = (quality_weight * quality_score +
                        compression_weight * compression_score +
                        speed_weight * speed_score)
        
        return overall_score
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable evaluation report"""
        report = []
        report.append("=" * 60)
        report.append("TUCKER DECOMPOSITION EVALUATION REPORT")
        report.append("=" * 60)
        
        # Compression Summary
        comp_ratios = results['compression_ratios']
        report.append(f"\n📊 COMPRESSION SUMMARY:")
        report.append(f"  Parameter Reduction: {comp_ratios['parameter_reduction_percent']:.1f}%")
        report.append(f"  Memory Savings: {comp_ratios['memory_savings_mb']:.1f} MB")
        report.append(f"  Compression Ratio: {comp_ratios['parameter_compression_ratio']:.3f}")
        
        # Quality Metrics
        quality = results['quality_metrics']
        report.append(f"\n🎯 QUALITY METRICS:")
        report.append(f"  Cosine Similarity: {quality['cosine_similarity']:.4f}")
        report.append(f"  Relative Error: {quality['relative_error']:.4f}")
        report.append(f"  Top-1 Preservation: {quality['top1_preservation']:.3f}")
        report.append(f"  Top-5 Preservation: {quality['top5_preservation']:.3f}")
        
        # Speed Improvement
        report.append(f"\n⚡ SPEED IMPROVEMENT:")
        report.append(f"  Speed Improvement: {results['speed_improvement']:.2f}x")
        orig_time = results['original_speed']['mean_inference_time']
        comp_time = results['compressed_speed']['mean_inference_time']
        report.append(f"  Original Time: {orig_time*1000:.2f} ms")
        report.append(f"  Compressed Time: {comp_time*1000:.2f} ms")
        
        # Overall Score
        report.append(f"\n🏆 OVERALL SCORE: {results['overall_score']:.3f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], save_path: str):
        """Save evaluation results"""
        import json
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert tensors to lists for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {save_path}")
    
    def _make_serializable(self, obj):
        """Convert torch tensors and numpy arrays to lists"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

def create_performance_evaluator(device: str = "auto") -> PerformanceEvaluator:
    """Factory function to create performance evaluator"""
    return PerformanceEvaluator(device=device)
