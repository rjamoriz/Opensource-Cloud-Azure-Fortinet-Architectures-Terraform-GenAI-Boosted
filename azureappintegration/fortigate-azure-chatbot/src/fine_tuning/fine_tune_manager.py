"""
FortiGate Azure Fine-Tuning Manager
Handles OpenAI fine-tuning process for specialized FortiGate Azure model
"""

import os
import time
import json
from typing import Optional, Dict, Any
from pathlib import Path
import logging
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FortiGateFineTuneManager:
    """Manage OpenAI fine-tuning for FortiGate Azure expertise"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = "gpt-3.5-turbo"  # Base model for fine-tuning
        self.fine_tuned_model_id = None
        
    def upload_training_file(self, file_path: str) -> str:
        """Upload training data file to OpenAI"""
        logger.info(f"Uploading training file: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                response = self.client.files.create(
                    file=f,
                    purpose='fine-tune'
                )
            
            file_id = response.id
            logger.info(f"✅ File uploaded successfully: {file_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"❌ Failed to upload file: {str(e)}")
            raise
    
    def create_fine_tuning_job(self, training_file_id: str, model_suffix: str = "fortigate-azure") -> str:
        """Create fine-tuning job"""
        logger.info(f"Creating fine-tuning job with file: {training_file_id}")
        
        try:
            response = self.client.fine_tuning.jobs.create(
                training_file=training_file_id,
                model=self.model_name,
                suffix=model_suffix,
                hyperparameters={
                    "n_epochs": 3,  # Number of training epochs
                    "batch_size": 1,  # Batch size for training
                    "learning_rate_multiplier": 0.1  # Learning rate multiplier
                }
            )
            
            job_id = response.id
            logger.info(f"✅ Fine-tuning job created: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create fine-tuning job: {str(e)}")
            raise
    
    def monitor_fine_tuning_job(self, job_id: str) -> Dict[str, Any]:
        """Monitor fine-tuning job progress"""
        logger.info(f"Monitoring fine-tuning job: {job_id}")
        
        while True:
            try:
                job = self.client.fine_tuning.jobs.retrieve(job_id)
                status = job.status
                
                logger.info(f"Job status: {status}")
                
                if status == "succeeded":
                    self.fine_tuned_model_id = job.fine_tuned_model
                    logger.info(f"🎉 Fine-tuning completed! Model: {self.fine_tuned_model_id}")
                    return {
                        "status": "completed",
                        "model_id": self.fine_tuned_model_id,
                        "job": job
                    }
                elif status == "failed":
                    logger.error(f"❌ Fine-tuning failed: {job.error}")
                    return {
                        "status": "failed",
                        "error": job.error,
                        "job": job
                    }
                elif status in ["validating_files", "queued", "running"]:
                    logger.info(f"⏳ Job in progress... Status: {status}")
                    time.sleep(60)  # Wait 1 minute before checking again
                else:
                    logger.warning(f"⚠️ Unknown status: {status}")
                    time.sleep(30)
                    
            except Exception as e:
                logger.error(f"Error monitoring job: {str(e)}")
                time.sleep(30)
    
    def test_fine_tuned_model(self, test_prompts: list) -> Dict[str, str]:
        """Test the fine-tuned model with sample prompts"""
        if not self.fine_tuned_model_id:
            raise ValueError("No fine-tuned model available")
        
        logger.info(f"Testing fine-tuned model: {self.fine_tuned_model_id}")
        results = {}
        
        for prompt in test_prompts:
            try:
                response = self.client.chat.completions.create(
                    model=self.fine_tuned_model_id,
                    messages=[
                        {"role": "system", "content": "You are an expert FortiGate Azure deployment assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
                
                results[prompt] = response.choices[0].message.content.strip()
                logger.info(f"✅ Test prompt completed: {prompt[:50]}...")
                
            except Exception as e:
                logger.error(f"❌ Test failed for prompt: {prompt[:50]}... Error: {str(e)}")
                results[prompt] = f"Error: {str(e)}"
        
        return results
    
    def save_model_info(self, output_dir: str = "models"):
        """Save fine-tuned model information"""
        if not self.fine_tuned_model_id:
            logger.warning("No fine-tuned model to save")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        model_info = {
            "model_id": self.fine_tuned_model_id,
            "base_model": self.model_name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "FortiGate Azure deployment specialist model",
            "use_case": "Azure FortiGate deployment assistance and troubleshooting"
        }
        
        info_file = output_path / "fortigate_model_info.json"
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"📄 Model info saved: {info_file}")
        return str(info_file)
    
    def complete_fine_tuning_process(self, training_file_path: str) -> str:
        """Complete end-to-end fine-tuning process"""
        logger.info("🚀 Starting complete fine-tuning process...")
        
        try:
            # Step 1: Upload training file
            file_id = self.upload_training_file(training_file_path)
            
            # Step 2: Create fine-tuning job
            job_id = self.create_fine_tuning_job(file_id)
            
            # Step 3: Monitor job progress
            result = self.monitor_fine_tuning_job(job_id)
            
            if result["status"] == "completed":
                # Step 4: Save model information
                self.save_model_info()
                
                # Step 5: Test the model
                test_prompts = [
                    "How do I deploy FortiGate HA in Azure?",
                    "What are the network requirements for FortiGate in Azure?",
                    "How do I troubleshoot FortiGate connectivity issues?",
                    "What Terraform variables do I need for FortiGate deployment?"
                ]
                
                logger.info("🧪 Testing fine-tuned model...")
                test_results = self.test_fine_tuned_model(test_prompts)
                
                # Save test results
                test_file = Path("models") / "test_results.json"
                with open(test_file, 'w') as f:
                    json.dump(test_results, f, indent=2)
                
                logger.info(f"✅ Fine-tuning process completed successfully!")
                logger.info(f"🎯 Model ID: {self.fine_tuned_model_id}")
                logger.info(f"📊 Test results saved: {test_file}")
                
                return self.fine_tuned_model_id
            else:
                logger.error("❌ Fine-tuning process failed")
                return None
                
        except Exception as e:
            logger.error(f"❌ Fine-tuning process error: {str(e)}")
            raise

def main():
    """Main function to run fine-tuning process"""
    print("🎯 FortiGate Azure Fine-Tuning Manager")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        return
    
    try:
        # Initialize manager
        manager = FortiGateFineTuneManager(api_key)
        
        # Check for training data
        training_file = "training_data/fortigate_training_data.jsonl"
        if not os.path.exists(training_file):
            print(f"❌ Training file not found: {training_file}")
            print("Please run data_preparation.py first to generate training data.")
            return
        
        print(f"📁 Using training file: {training_file}")
        
        # Start fine-tuning process
        model_id = manager.complete_fine_tuning_process(training_file)
        
        if model_id:
            print("\n🎉 SUCCESS!")
            print(f"Fine-tuned model ready: {model_id}")
            print("\nNext steps:")
            print("1. Update your chatbot to use the fine-tuned model")
            print("2. Test the model with real FortiGate questions")
            print("3. Monitor performance and iterate if needed")
        else:
            print("\n❌ Fine-tuning failed. Check logs for details.")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
