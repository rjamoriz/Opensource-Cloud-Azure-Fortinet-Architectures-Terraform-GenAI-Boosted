"""
Llama 7B Fine-Tuning Module with LoRA and QLoRA
Advanced fine-tuning using transformers, bitsandbytes, and PEFT
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# Core ML libraries
try:
    import torch
    import pandas as pd
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        BitsAndBytesConfig
    )
    from peft import (
        get_peft_model, 
        TaskType, 
        LoraConfig, 
        PeftModel, 
        prepare_model_for_kbit_training
    )
    from datasets import Dataset
    import bitsandbytes as bnb
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)
    
    # Create dummy classes to prevent import errors
    class torch:
        class cuda:
            @staticmethod
            def is_available():
                return False
            @staticmethod
            def device_count():
                return 0
            @staticmethod
            def get_device_name(i):
                return "No GPU"
        @staticmethod
        def __version__():
            return "Not installed"
    
    class LoraConfig:
        def __init__(self, *args, **kwargs):
            pass
    
    class PeftModel:
        pass
    
    class Dataset:
        def __init__(self, *args, **kwargs):
            pass
        @staticmethod
        def from_dict(data):
            return Dataset()
    
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return None
    
    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return None
    
    class TrainingArguments:
        def __init__(self, *args, **kwargs):
            pass
    
    class Trainer:
        def __init__(self, *args, **kwargs):
            pass
        def train(self):
            pass
        def save_model(self, path):
            pass
    
    class BitsAndBytesConfig:
        def __init__(self, *args, **kwargs):
            pass
    
    def get_peft_model(*args, **kwargs):
        return None
    
    def prepare_model_for_kbit_training(*args, **kwargs):
        return None
    
    class TaskType:
        CAUSAL_LM = "CAUSAL_LM"
    
    class pd:
        @staticmethod
        def read_csv(*args, **kwargs):
            return None
        @staticmethod
        def DataFrame(*args, **kwargs):
            return None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaFineTuner:
    """Advanced Llama 7B fine-tuning with QLoRA and efficient training"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.training_data = []
        self.fine_tuned_model = None
        self.output_dir = Path("models/llama_fine_tuned")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not DEPENDENCIES_AVAILABLE:
            logger.error(f"Required dependencies not available: {IMPORT_ERROR}")
            return
        
        # Check for GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        if self.device == "cpu":
            logger.warning("⚠️ GPU not available. Fine-tuning will be very slow on CPU.")
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check system dependencies and GPU availability"""
        status = {
            "dependencies_available": DEPENDENCIES_AVAILABLE,
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_memory": [],
            "torch_version": "Not installed",
            "huggingface_token": bool(os.getenv("HUGGINGFACE_TOKEN")),
            "system_ram_gb": 16,
            "disk_space_gb": 20
        }
        
        if not DEPENDENCIES_AVAILABLE:
            return status
        
        if torch and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                status["gpu_memory"].append(f"GPU {i}: {gpu_memory:.1f} GB")
        
        return status
    
    def load_base_model(self) -> bool:
        """Load Llama 7B base model with quantization"""
        if not DEPENDENCIES_AVAILABLE:
            logger.error("Dependencies not available for model loading")
            return False
        
        try:
            logger.info(f"🦙 Loading Llama model: {self.model_name}")
            
            # Configure quantization for efficient training
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            
            # Prepare model for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
            
            logger.info("✅ Llama model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            return False
    
    def setup_lora_config(self) -> LoraConfig:
        """Configure LoRA for parameter-efficient fine-tuning"""
        lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,  # Alpha parameter
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        return lora_config
    
    def process_uploaded_data(self, uploaded_files: List[Any]) -> bool:
        """Process uploaded training data files"""
        self.training_data = []
        
        for uploaded_file in uploaded_files:
            try:
                # Read file content
                if uploaded_file.name.endswith('.json'):
                    content = json.loads(uploaded_file.read().decode())
                    if isinstance(content, list):
                        self.training_data.extend(content)
                    else:
                        self.training_data.append(content)
                        
                elif uploaded_file.name.endswith('.jsonl'):
                    lines = uploaded_file.read().decode().strip().split('\n')
                    for line in lines:
                        if line.strip():
                            self.training_data.append(json.loads(line))
                            
                elif uploaded_file.name.endswith('.txt'):
                    content = uploaded_file.read().decode()
                    # Convert plain text to training format
                    self.training_data.append({
                        "instruction": "Answer the following FortiGate Azure question:",
                        "input": content[:500],  # Truncate for example
                        "output": "This requires expert FortiGate knowledge to answer properly."
                    })
                    
                elif uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    for _, row in df.iterrows():
                        self.training_data.append(row.to_dict())
                        
            except Exception as e:
                logger.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                return False
        
        logger.info(f"📊 Processed {len(self.training_data)} training examples")
        return True
    
    def format_training_data(self) -> Dataset:
        """Format data for Llama training"""
        formatted_data = []
        
        for item in self.training_data:
            # Handle different data formats
            if "messages" in item:
                # OpenAI format
                text = self.format_chat_messages(item["messages"])
            elif "instruction" in item:
                # Alpaca format
                instruction = item.get("instruction", "")
                input_text = item.get("input", "")
                output = item.get("output", "")
                
                if input_text:
                    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            else:
                # Generic format
                text = str(item)
            
            formatted_data.append({"text": text})
        
        return Dataset.from_list(formatted_data)
    
    def format_chat_messages(self, messages: List[Dict]) -> str:
        """Format chat messages for Llama training"""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted += f"<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n\n"
            elif role == "user":
                formatted += f"{content} [/INST] "
            elif role == "assistant":
                formatted += f"{content} </s><s>[INST] "
        
        return formatted.rstrip("<s>[INST] ")
    
    def tokenize_data(self, dataset: Dataset) -> Dataset:
        """Tokenize the training dataset"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=2048,
                return_overflowing_tokens=False,
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        return tokenized_dataset
    
    def start_fine_tuning(self, 
                         epochs: int = 3,
                         learning_rate: float = 2e-4,
                         batch_size: int = 4) -> bool:
        """Start the fine-tuning process"""
        if not self.model or not self.tokenizer:
            logger.error("❌ Model not loaded. Please load the base model first.")
            return False
        
        if not self.training_data:
            logger.error("❌ No training data available. Please upload training data first.")
            return False
        
        try:
            logger.info("🚀 Starting Llama fine-tuning process...")
            
            # Setup LoRA
            lora_config = self.setup_lora_config()
            self.model = get_peft_model(self.model, lora_config)
            
            # Prepare training data
            dataset = self.format_training_data()
            tokenized_dataset = self.tokenize_data(dataset)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,
                optim="paged_adamw_32bit",
                save_steps=100,
                logging_steps=10,
                learning_rate=learning_rate,
                weight_decay=0.001,
                fp16=False,
                bf16=True,
                max_grad_norm=0.3,
                max_steps=-1,
                warmup_ratio=0.03,
                group_by_length=True,
                lr_scheduler_type="constant",
                report_to=None,
                save_total_limit=3,
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
            )
            
            # Start training
            logger.info("🔥 Training started...")
            trainer.train()
            
            # Save the fine-tuned model
            trainer.save_model()
            self.tokenizer.save_pretrained(str(self.output_dir))
            
            # Save model info
            model_info = {
                "model_name": self.model_name,
                "fine_tuned_path": str(self.output_dir),
                "training_examples": len(self.training_data),
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "created_at": datetime.now().isoformat(),
                "lora_config": {
                    "r": lora_config.r,
                    "alpha": lora_config.lora_alpha,
                    "dropout": lora_config.lora_dropout
                }
            }
            
            with open(self.output_dir / "model_info.json", "w") as f:
                json.dump(model_info, f, indent=2)
            
            logger.info("✅ Fine-tuning completed successfully!")
            logger.info(f"📁 Model saved to: {self.output_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Fine-tuning failed: {str(e)}")
            return False
    
    def load_fine_tuned_model(self, model_path: str) -> bool:
        """Load a previously fine-tuned model"""
        try:
            from peft import PeftModel
            
            # Load base model
            if not self.load_base_model():
                return False
            
            # Load LoRA weights
            self.fine_tuned_model = PeftModel.from_pretrained(
                self.model,
                model_path,
                torch_dtype=torch.bfloat16
            )
            
            logger.info(f"✅ Fine-tuned model loaded from: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load fine-tuned model: {str(e)}")
            return False
    
    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        """Generate response using fine-tuned model"""
        if not self.fine_tuned_model:
            return "Fine-tuned model not loaded"
        
        try:
            # Format prompt for Llama
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.fine_tuned_model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        info = {
            "base_model": self.model_name,
            "model_loaded": self.model is not None,
            "fine_tuned_available": self.fine_tuned_model is not None,
            "training_examples": len(self.training_data),
            "device": self.device,
            "dependencies_available": DEPENDENCIES_AVAILABLE
        }
        
        # Add model file info if available
        if (self.output_dir / "model_info.json").exists():
            with open(self.output_dir / "model_info.json", "r") as f:
                saved_info = json.load(f)
                info.update(saved_info)
        
        return info

# Utility functions for Streamlit integration
def get_system_requirements() -> Dict[str, Any]:
    """Get system requirements for Llama fine-tuning"""
    return {
        "minimum_requirements": {
            "gpu_memory": "12 GB VRAM (RTX 3060 12GB or better)",
            "system_ram": "16 GB RAM",
            "disk_space": "20 GB free space",
            "python_version": "3.8+",
            "cuda_version": "11.8+"
        },
        "recommended_requirements": {
            "gpu_memory": "24 GB VRAM (RTX 4090 or A100)",
            "system_ram": "32 GB RAM",
            "disk_space": "50 GB free space",
            "python_version": "3.10+",
            "cuda_version": "12.1+"
        },
        "dependencies": [
            "torch>=2.0.0",
            "transformers>=4.35.0",
            "peft>=0.6.0",
            "bitsandbytes>=0.41.0",
            "datasets>=2.14.0",
            "accelerate>=0.24.0"
        ]
    }

if __name__ == "__main__":
    # Test the fine-tuner
    fine_tuner = LlamaFineTuner()
    status = fine_tuner.check_dependencies()
    print("System Status:", json.dumps(status, indent=2))
