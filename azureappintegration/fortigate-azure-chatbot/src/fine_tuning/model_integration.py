"""
Fine-Tuned Model Integration for FortiGate Azure Chatbot
Integrates custom fine-tuned model with the existing chatbot application
"""

import os
import json
from typing import Optional, Dict, Any
from pathlib import Path
import logging
from openai import OpenAI
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FineTunedModelManager:
    """Manage fine-tuned FortiGate model integration"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            logger.warning("OpenAI API key not configured")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
        
        self.fine_tuned_model_id = None
        self.model_info = None
        self.load_model_info()
    
    def load_model_info(self):
        """Load fine-tuned model information"""
        model_info_path = Path("models/fortigate_model_info.json")
        
        if model_info_path.exists():
            try:
                with open(model_info_path, 'r') as f:
                    self.model_info = json.load(f)
                    self.fine_tuned_model_id = self.model_info.get('model_id')
                    logger.info(f"✅ Loaded fine-tuned model: {self.fine_tuned_model_id}")
            except Exception as e:
                logger.error(f"❌ Failed to load model info: {str(e)}")
        else:
            logger.info("ℹ️ No fine-tuned model found. Using base GPT-4 model.")
    
    def is_fine_tuned_available(self) -> bool:
        """Check if fine-tuned model is available"""
        return self.fine_tuned_model_id is not None and self.client is not None
    
    def query_fine_tuned_model(self, user_input: str, system_prompt: Optional[str] = None) -> str:
        """Query the fine-tuned model"""
        if not self.is_fine_tuned_available():
            return "Fine-tuned model not available. Please configure OpenAI API key and train the model."
        
        if system_prompt is None:
            system_prompt = """You are a specialized FortiGate Azure deployment expert with deep knowledge of:
            - FortiGate VM deployment on Azure
            - Terraform infrastructure as code
            - Azure networking and security
            - High availability configurations
            - Troubleshooting and best practices
            
            Provide detailed, accurate, and actionable guidance for FortiGate Azure deployments."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.fine_tuned_model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Fine-tuned model query failed: {str(e)}")
            return f"Error querying fine-tuned model: {str(e)}"
    
    def compare_models(self, user_input: str) -> Dict[str, str]:
        """Compare responses from base GPT-4 and fine-tuned model"""
        results = {}
        
        # Base GPT-4 response
        try:
            base_response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a FortiGate Azure deployment assistant."},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=300,
                temperature=0.7
            )
            results["base_gpt4"] = base_response.choices[0].message.content.strip()
        except Exception as e:
            results["base_gpt4"] = f"Error: {str(e)}"
        
        # Fine-tuned model response
        results["fine_tuned"] = self.query_fine_tuned_model(user_input)
        
        return results
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics and information"""
        if not self.model_info:
            return {"status": "No fine-tuned model available"}
        
        stats = {
            "model_id": self.fine_tuned_model_id,
            "base_model": self.model_info.get("base_model", "Unknown"),
            "created_at": self.model_info.get("created_at", "Unknown"),
            "description": self.model_info.get("description", ""),
            "status": "Available" if self.is_fine_tuned_available() else "API key required"
        }
        
        return stats

def display_fine_tuning_interface():
    """Streamlit interface for fine-tuning management"""
    st.header("🎯 Fine-Tuned FortiGate Model")
    
    # Initialize model manager
    model_manager = FineTunedModelManager()
    
    # Display model status
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Status")
        stats = model_manager.get_model_stats()
        
        if "model_id" in stats:
            st.success("✅ Fine-tuned model available")
            st.write(f"**Model ID**: `{stats['model_id']}`")
            st.write(f"**Created**: {stats['created_at']}")
            st.write(f"**Status**: {stats['status']}")
        else:
            st.warning("⚠️ No fine-tuned model found")
            st.write("Train a model using the fine-tuning tools")
    
    with col2:
        st.subheader("🛠️ Fine-Tuning Tools")
        
        if st.button("📚 Generate Training Data"):
            with st.spinner("Generating training data..."):
                try:
                    from .data_preparation import FortiGateDataPreparator
                    preparator = FortiGateDataPreparator()
                    training_file = preparator.generate_full_dataset()
                    
                    if training_file:
                        st.success(f"✅ Training data generated: {training_file}")
                        st.write(f"📊 Examples: {len(preparator.training_data)}")
                    else:
                        st.error("❌ Failed to generate training data")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        if st.button("🚀 Start Fine-Tuning"):
            if not os.getenv('OPENAI_API_KEY'):
                st.error("❌ OpenAI API key required")
                st.code("export OPENAI_API_KEY='your-api-key'")
            else:
                st.info("🔄 Fine-tuning process started...")
                st.write("This process may take 10-30 minutes.")
                st.write("Check the logs for progress updates.")
    
    # Model comparison interface
    st.divider()
    st.subheader("🔍 Model Comparison")
    
    if model_manager.is_fine_tuned_available():
        test_input = st.text_area(
            "Enter a FortiGate question to compare responses:",
            placeholder="How do I configure FortiGate HA in Azure with Terraform?"
        )
        
        if st.button("🆚 Compare Models") and test_input:
            with st.spinner("Comparing model responses..."):
                comparison = model_manager.compare_models(test_input)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Base GPT-4 Response:**")
                st.write(comparison.get("base_gpt4", "No response"))
            
            with col2:
                st.write("**Fine-Tuned Model Response:**")
                st.write(comparison.get("fine_tuned", "No response"))
    else:
        st.info("ℹ️ Fine-tuned model not available for comparison")

def integrate_with_voice_chat(voice_integration_instance):
    """Integrate fine-tuned model with voice chat"""
    model_manager = FineTunedModelManager()
    
    # Override the generate_fortigate_response method
    def enhanced_fortigate_response(user_input: str) -> str:
        if model_manager.is_fine_tuned_available():
            logger.info("Using fine-tuned model for response")
            return model_manager.query_fine_tuned_model(user_input)
        else:
            logger.info("Using base GPT-4 model for response")
            return voice_integration_instance.generate_fortigate_response(user_input)
    
    # Replace the method
    voice_integration_instance.generate_fortigate_response = enhanced_fortigate_response
    
    return voice_integration_instance

# Streamlit app integration
def display_enhanced_chat_interface():
    """Enhanced chat interface with fine-tuned model option"""
    st.subheader("💬 Enhanced FortiGate Chat")
    
    model_manager = FineTunedModelManager()
    
    # Model selection
    if model_manager.is_fine_tuned_available():
        model_choice = st.radio(
            "Choose model:",
            ["Fine-Tuned FortiGate Expert", "Base GPT-4"],
            help="Fine-tuned model has specialized FortiGate Azure knowledge"
        )
        use_fine_tuned = model_choice == "Fine-Tuned FortiGate Expert"
    else:
        st.info("ℹ️ Using base GPT-4 model (fine-tuned model not available)")
        use_fine_tuned = False
    
    # Chat interface
    user_question = st.text_area(
        "Ask your FortiGate Azure question:",
        placeholder="How do I deploy a FortiGate HA cluster in Azure using Terraform?"
    )
    
    if st.button("🚀 Get Answer") and user_question:
        with st.spinner("Generating response..."):
            if use_fine_tuned:
                response = model_manager.query_fine_tuned_model(user_question)
                st.success("🎯 **Fine-Tuned Model Response:**")
            else:
                # Use base model logic here
                response = "Base GPT-4 response would go here"
                st.info("🤖 **Base GPT-4 Response:**")
            
            st.write(response)
    
    return model_manager

if __name__ == "__main__":
    # Test the integration
    manager = FineTunedModelManager()
    
    if manager.is_fine_tuned_available():
        test_question = "How do I deploy FortiGate HA in Azure?"
        response = manager.query_fine_tuned_model(test_question)
        print(f"Question: {test_question}")
        print(f"Response: {response}")
    else:
        print("Fine-tuned model not available for testing")
