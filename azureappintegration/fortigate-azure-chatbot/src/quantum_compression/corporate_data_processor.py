"""
Corporate Data Processor for Tucker Decomposition
Handles corporate services data for post-compression fine-tuning
"""

import torch
import pandas as pd
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class CorporateDataProcessor:
    """Process corporate data for model fine-tuning after compression"""
    
    def __init__(self, tokenizer=None, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processed_data = []
        self.data_statistics = {}
        
    def load_data(self, file_path: Union[str, Path], file_format: str = "auto") -> List[Dict[str, Any]]:
        """
        Load corporate data from various file formats
        
        Args:
            file_path: Path to data file
            file_format: Format of the file (json, jsonl, csv, txt, auto)
            
        Returns:
            List of data samples
        """
        file_path = Path(file_path)
        
        if file_format == "auto":
            file_format = file_path.suffix.lower().lstrip('.')
        
        try:
            if file_format in ['json']:
                return self._load_json(file_path)
            elif file_format in ['jsonl', 'ndjson']:
                return self._load_jsonl(file_path)
            elif file_format in ['csv']:
                return self._load_csv(file_path)
            elif file_format in ['txt']:
                return self._load_txt(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
                
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def _load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSON format data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError("JSON data must be a list or dict")
    
    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSONL format data"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def _load_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load CSV format data"""
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    
    def _load_txt(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load plain text data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Convert lines to dict format
        data = []
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                data.append({
                    'text': line,
                    'id': i
                })
        return data
    
    def preprocess_data(self, data: List[Dict[str, Any]], 
                       text_field: str = "text",
                       label_field: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Preprocess corporate data for training
        
        Args:
            data: Raw data samples
            text_field: Field containing text data
            label_field: Field containing labels (optional)
            
        Returns:
            Preprocessed data samples
        """
        processed = []
        
        for sample in data:
            if text_field not in sample:
                logger.warning(f"Text field '{text_field}' not found in sample")
                continue
            
            text = sample[text_field]
            if not isinstance(text, str):
                text = str(text)
            
            # Basic text cleaning
            text = self._clean_text(text)
            
            processed_sample = {
                'text': text,
                'original_length': len(text)
            }
            
            # Add label if available
            if label_field and label_field in sample:
                processed_sample['label'] = sample[label_field]
            
            # Add any additional metadata
            for key, value in sample.items():
                if key not in [text_field, label_field]:
                    processed_sample[f'meta_{key}'] = value
            
            processed.append(processed_sample)
        
        self.processed_data = processed
        self._calculate_statistics()
        
        return processed
    
    def _clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove very short texts
        if len(text.strip()) < 10:
            return ""
        
        return text.strip()
    
    def tokenize_data(self, data: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, torch.Tensor]]:
        """
        Tokenize preprocessed data
        
        Args:
            data: Data to tokenize (uses self.processed_data if None)
            
        Returns:
            Tokenized data samples
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not provided")
        
        if data is None:
            data = self.processed_data
        
        tokenized = []
        
        for sample in data:
            text = sample['text']
            if not text:
                continue
            
            # Tokenize with truncation and padding
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            tokenized_sample = {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'original_text': text
            }
            
            # Add labels if available
            if 'label' in sample:
                tokenized_sample['labels'] = torch.tensor(sample['label'])
            
            tokenized.append(tokenized_sample)
        
        return tokenized
    
    def create_training_dataset(self, data: List[Dict[str, torch.Tensor]]) -> torch.utils.data.Dataset:
        """Create PyTorch dataset for training"""
        return CorporateDataset(data)
    
    def _calculate_statistics(self):
        """Calculate data statistics"""
        if not self.processed_data:
            return
        
        texts = [sample['text'] for sample in self.processed_data if sample['text']]
        lengths = [len(text) for text in texts]
        
        self.data_statistics = {
            'total_samples': len(self.processed_data),
            'valid_samples': len(texts),
            'avg_length': sum(lengths) / len(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'empty_samples': len(self.processed_data) - len(texts)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get data processing statistics"""
        return self.data_statistics
    
    def validate_data_quality(self, min_samples: int = 10, min_avg_length: int = 50) -> Dict[str, Any]:
        """
        Validate data quality for training
        
        Args:
            min_samples: Minimum number of valid samples required
            min_avg_length: Minimum average text length required
            
        Returns:
            Validation results
        """
        stats = self.get_statistics()
        
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check minimum samples
        if stats.get('valid_samples', 0) < min_samples:
            validation_results['is_valid'] = False
            validation_results['issues'].append(
                f"Insufficient samples: {stats.get('valid_samples', 0)} < {min_samples}"
            )
        
        # Check average length
        if stats.get('avg_length', 0) < min_avg_length:
            validation_results['warnings'].append(
                f"Low average text length: {stats.get('avg_length', 0)} < {min_avg_length}"
            )
        
        # Check for empty samples
        if stats.get('empty_samples', 0) > 0:
            validation_results['warnings'].append(
                f"Found {stats.get('empty_samples', 0)} empty samples"
            )
        
        return validation_results

class CorporateDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for corporate data"""
    
    def __init__(self, data: List[Dict[str, torch.Tensor]]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def create_corporate_processor(tokenizer=None, max_length: int = 512) -> CorporateDataProcessor:
    """Factory function to create corporate data processor"""
    return CorporateDataProcessor(tokenizer=tokenizer, max_length=max_length)
