"""
Post-Compression Trainer for Tucker Decomposition
Fine-tuning compressed models with corporate data
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class PostCompressionTrainer:
    """Fine-tune compressed models with corporate data"""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = "auto",
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.model.to(self.device)
        self.optimizer = None
        self.scheduler = None
        self.training_history = []
        
    def setup_optimizer(self, optimizer_type: str = "adamw") -> torch.optim.Optimizer:
        """Setup optimizer for training"""
        if optimizer_type.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_type.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_type.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        return self.optimizer
    
    def setup_scheduler(self, scheduler_type: str = "cosine", num_training_steps: int = 1000):
        """Setup learning rate scheduler"""
        if self.optimizer is None:
            raise ValueError("Optimizer must be setup before scheduler")
        
        if scheduler_type.lower() == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_training_steps
            )
        elif scheduler_type.lower() == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_training_steps
            )
        elif scheduler_type.lower() == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=num_training_steps // 3, gamma=0.5
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")
        
        return self.scheduler
    
    def train_epoch(self, 
                   dataloader: DataLoader,
                   loss_fn: Callable,
                   progress_callback: Optional[Callable] = None) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get model outputs
            outputs = self.model(**batch)
            
            # Calculate loss
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                # Custom loss calculation
                loss = loss_fn(outputs, batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += loss.item()
            
            # Progress callback
            if progress_callback:
                progress_callback(batch_idx + 1, num_batches, loss.item())
        
        avg_loss = total_loss / num_batches
        current_lr = self.optimizer.param_groups[0]['lr']
        
        return {
            'loss': avg_loss,
            'learning_rate': current_lr
        }
    
    def validate(self, 
                dataloader: DataLoader,
                loss_fn: Callable) -> Dict[str, float]:
        """Validate model performance"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Calculate loss
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                else:
                    loss = loss_fn(outputs, batch)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        
        return {
            'val_loss': avg_loss
        }
    
    def train(self,
              train_dataloader: DataLoader,
              val_dataloader: Optional[DataLoader] = None,
              num_epochs: int = 3,
              loss_fn: Optional[Callable] = None,
              progress_callback: Optional[Callable] = None,
              validation_callback: Optional[Callable] = None) -> List[Dict[str, float]]:
        """
        Complete training loop
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            num_epochs: Number of training epochs
            loss_fn: Loss function (uses model's loss if None)
            progress_callback: Callback for training progress
            validation_callback: Callback for validation results
            
        Returns:
            Training history
        """
        if self.optimizer is None:
            self.setup_optimizer()
        
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        self.training_history = []
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(
                train_dataloader, 
                loss_fn, 
                progress_callback
            )
            
            # Validation
            val_metrics = {}
            if val_dataloader is not None:
                val_metrics = self.validate(val_dataloader, loss_fn)
            
            # Combine metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_time': time.time() - epoch_start_time,
                **train_metrics,
                **val_metrics
            }
            
            self.training_history.append(epoch_metrics)
            
            # Validation callback
            if validation_callback:
                validation_callback(epoch + 1, epoch_metrics)
            
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - "
                       f"Loss: {train_metrics['loss']:.4f}")
            
            if val_metrics:
                logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
        
        return self.training_history
    
    def save_model(self, save_path: str):
        """Save trained model"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_history': self.training_history,
            'config': {
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'device': str(self.device)
            }
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load trained model"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        logger.info(f"Model loaded from {load_path}")
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training performance metrics"""
        if not self.training_history:
            return {}
        
        train_losses = [h['loss'] for h in self.training_history]
        val_losses = [h.get('val_loss') for h in self.training_history if 'val_loss' in h]
        
        metrics = {
            'num_epochs': len(self.training_history),
            'final_train_loss': train_losses[-1],
            'best_train_loss': min(train_losses),
            'train_loss_improvement': train_losses[0] - train_losses[-1] if len(train_losses) > 1 else 0,
            'total_training_time': sum(h.get('train_time', 0) for h in self.training_history)
        }
        
        if val_losses:
            metrics.update({
                'final_val_loss': val_losses[-1],
                'best_val_loss': min(val_losses),
                'val_loss_improvement': val_losses[0] - val_losses[-1] if len(val_losses) > 1 else 0
            })
        
        return metrics
    
    def calculate_model_size(self) -> Dict[str, int]:
        """Calculate model size metrics"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Estimate memory usage (assuming float32)
        memory_mb = total_params * 4 / (1024 * 1024)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'memory_mb': memory_mb
        }

def create_post_compression_trainer(model: nn.Module, 
                                  device: str = "auto",
                                  learning_rate: float = 1e-4,
                                  weight_decay: float = 0.01) -> PostCompressionTrainer:
    """Factory function to create post-compression trainer"""
    return PostCompressionTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
