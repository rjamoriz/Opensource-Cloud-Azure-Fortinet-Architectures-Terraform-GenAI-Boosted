#!/usr/bin/env python3
"""
Complete Fine-Tuning Execution Script
Orchestrates the entire fine-tuning process for FortiGate Azure model
"""

import os
import sys
import argparse
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from fine_tuning.data_preparation import FortiGateDataPreparator
from fine_tuning.fine_tune_manager import FortiGateFineTuneManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fine_tuning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_prerequisites():
    """Check if all prerequisites are met"""
    logger.info("🔍 Checking prerequisites...")
    
    # Check OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("❌ OPENAI_API_KEY environment variable not set")
        logger.info("Please set your OpenAI API key:")
        logger.info("export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    logger.info("✅ OpenAI API key found")
    
    # Check if we have sufficient API credits (this is informational)
    logger.info("ℹ️ Note: Fine-tuning requires OpenAI API credits")
    logger.info("ℹ️ Estimated cost: $8-20 for a typical FortiGate dataset")
    
    return True

def prepare_training_data():
    """Prepare training data for fine-tuning"""
    logger.info("📚 Preparing training data...")
    
    try:
        preparator = FortiGateDataPreparator()
        training_file = preparator.generate_full_dataset()
        
        if training_file:
            logger.info(f"✅ Training data prepared: {training_file}")
            logger.info(f"📊 Total examples: {len(preparator.training_data)}")
            return training_file
        else:
            logger.error("❌ Failed to prepare training data")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error preparing training data: {str(e)}")
        return None

def run_fine_tuning(training_file: str):
    """Run the fine-tuning process"""
    logger.info("🚀 Starting fine-tuning process...")
    
    try:
        manager = FortiGateFineTuneManager()
        model_id = manager.complete_fine_tuning_process(training_file)
        
        if model_id:
            logger.info("🎉 Fine-tuning completed successfully!")
            logger.info(f"🎯 Model ID: {model_id}")
            return model_id
        else:
            logger.error("❌ Fine-tuning failed")
            return None
            
    except Exception as e:
        logger.error(f"❌ Fine-tuning error: {str(e)}")
        return None

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="FortiGate Azure Fine-Tuning Pipeline")
    parser.add_argument("--data-only", action="store_true", help="Only prepare training data")
    parser.add_argument("--skip-data", action="store_true", help="Skip data preparation")
    parser.add_argument("--training-file", help="Use existing training file")
    
    args = parser.parse_args()
    
    print("🎯 FortiGate Azure Fine-Tuning Pipeline")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    training_file = None
    
    # Prepare training data
    if not args.skip_data:
        training_file = prepare_training_data()
        if not training_file:
            logger.error("❌ Cannot proceed without training data")
            sys.exit(1)
    elif args.training_file:
        training_file = args.training_file
        if not os.path.exists(training_file):
            logger.error(f"❌ Training file not found: {training_file}")
            sys.exit(1)
    else:
        # Look for default training file
        default_file = "training_data/fortigate_training_data.jsonl"
        if os.path.exists(default_file):
            training_file = default_file
            logger.info(f"📁 Using existing training file: {training_file}")
        else:
            logger.error("❌ No training file found. Run without --skip-data or specify --training-file")
            sys.exit(1)
    
    # Stop here if only preparing data
    if args.data_only:
        logger.info("✅ Data preparation completed. Use --skip-data to run fine-tuning.")
        return
    
    # Run fine-tuning
    model_id = run_fine_tuning(training_file)
    
    if model_id:
        print("\n🎉 SUCCESS!")
        print(f"Fine-tuned model ready: {model_id}")
        print("\nNext steps:")
        print("1. Restart your Streamlit app to use the new model")
        print("2. Test the fine-tuned model in the chat interface")
        print("3. Compare responses with the base GPT-4 model")
        print("4. Monitor performance and iterate if needed")
        
        # Create integration instructions
        print("\n📋 Integration Instructions:")
        print("The fine-tuned model is automatically available in your chatbot.")
        print("Look for the 'Fine-Tuned FortiGate Expert' option in the model selection.")
    else:
        print("\n❌ Fine-tuning failed. Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
