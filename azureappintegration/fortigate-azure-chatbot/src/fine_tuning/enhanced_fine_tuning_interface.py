#!/usr/bin/env python3
"""
Enhanced Fine-Tuning Interface with Automatic Launch and Progress Tracking
FortiGate Azure Chatbot - Advanced Fine-Tuning Management
"""

import streamlit as st
import os
import json
import time
import threading
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Optional, Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available")

class EnhancedFineTuningManager:
    """Enhanced fine-tuning manager with real-time progress tracking"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = None
        self.current_job = None
        self.training_progress = {}
        
        if self.api_key and OPENAI_AVAILABLE:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
    
    def validate_training_data(self, data: List[Dict]) -> Dict[str, Any]:
        """Validate training data format and quality"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {
                "total_examples": len(data),
                "avg_prompt_length": 0,
                "avg_completion_length": 0,
                "unique_prompts": 0
            }
        }
        
        if len(data) < 10:
            validation_results["errors"].append("Minimum 10 training examples required")
            validation_results["valid"] = False
        
        prompt_lengths = []
        completion_lengths = []
        prompts = set()
        
        for i, example in enumerate(data):
            if not isinstance(example, dict):
                validation_results["errors"].append(f"Example {i+1}: Must be a dictionary")
                continue
                
            if "messages" not in example:
                validation_results["errors"].append(f"Example {i+1}: Missing 'messages' field")
                continue
                
            messages = example["messages"]
            if not isinstance(messages, list) or len(messages) < 2:
                validation_results["errors"].append(f"Example {i+1}: 'messages' must be a list with at least 2 messages")
                continue
            
            # Extract prompt and completion for stats
            user_msg = next((msg for msg in messages if msg.get("role") == "user"), None)
            assistant_msg = next((msg for msg in messages if msg.get("role") == "assistant"), None)
            
            if user_msg and assistant_msg:
                prompt_lengths.append(len(user_msg.get("content", "")))
                completion_lengths.append(len(assistant_msg.get("content", "")))
                prompts.add(user_msg.get("content", ""))
        
        if prompt_lengths:
            validation_results["stats"]["avg_prompt_length"] = sum(prompt_lengths) / len(prompt_lengths)
            validation_results["stats"]["avg_completion_length"] = sum(completion_lengths) / len(completion_lengths)
            validation_results["stats"]["unique_prompts"] = len(prompts)
        
        if len(validation_results["errors"]) > 0:
            validation_results["valid"] = False
            
        return validation_results
    
    def upload_training_file(self, file_path: str) -> Optional[str]:
        """Upload training file to OpenAI"""
        if not self.client:
            return None
            
        try:
            with open(file_path, 'rb') as f:
                response = self.client.files.create(
                    file=f,
                    purpose='fine-tune'
                )
            return response.id
        except Exception as e:
            logger.error(f"Failed to upload training file: {e}")
            return None
    
    def start_fine_tuning(self, file_id: str, model: str = "gpt-3.5-turbo", 
                         suffix: Optional[str] = None) -> Optional[str]:
        """Start fine-tuning job"""
        if not self.client:
            return None
            
        try:
            job = self.client.fine_tuning.jobs.create(
                training_file=file_id,
                model=model,
                suffix=suffix or f"fortigate-{int(time.time())}"
            )
            self.current_job = job.id
            return job.id
        except Exception as e:
            logger.error(f"Failed to start fine-tuning: {e}")
            return None
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get fine-tuning job status"""
        if not self.client:
            return None
            
        try:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            return {
                "id": job.id,
                "status": job.status,
                "created_at": datetime.fromtimestamp(job.created_at).strftime("%Y-%m-%d %H:%M:%S"),
                "finished_at": datetime.fromtimestamp(job.finished_at).strftime("%Y-%m-%d %H:%M:%S") if job.finished_at else None,
                "fine_tuned_model": job.fine_tuned_model,
                "training_file": job.training_file,
                "validation_file": job.validation_file,
                "hyperparameters": job.hyperparameters,
                "trained_tokens": job.trained_tokens,
                "error": job.error
            }
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return None
    
    def list_fine_tuned_models(self) -> List[Dict]:
        """List available fine-tuned models"""
        if not self.client:
            return []
            
        try:
            models = self.client.models.list()
            fine_tuned = [
                {
                    "id": model.id,
                    "created": datetime.fromtimestamp(model.created).strftime("%Y-%m-%d %H:%M:%S"),
                    "owned_by": model.owned_by
                }
                for model in models.data 
                if model.id.startswith("ft:")
            ]
            return fine_tuned
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

def display_training_data_uploader():
    """Display training data upload interface"""
    st.subheader("📁 Training Data Upload")
    
    # File upload options
    upload_method = st.radio(
        "Choose upload method:",
        ["📄 Upload JSONL File", "✏️ Manual Entry", "🤖 Generate Sample Data"],
        horizontal=True
    )
    
    training_data = []
    
    if upload_method == "📄 Upload JSONL File":
        uploaded_file = st.file_uploader(
            "Upload training data (JSONL format)",
            type=['jsonl', 'json'],
            help="Each line should be a JSON object with 'messages' field containing conversation data"
        )
        
        if uploaded_file:
            try:
                content = uploaded_file.read().decode('utf-8')
                lines = content.strip().split('\n')
                
                for line in lines:
                    if line.strip():
                        training_data.append(json.loads(line))
                
                st.success(f"✅ Loaded {len(training_data)} training examples")
                
                # Show preview
                if training_data:
                    st.write("**Data Preview:**")
                    st.json(training_data[0])
                    
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    elif upload_method == "✏️ Manual Entry":
        st.write("**Add Training Examples:**")
        
        # Initialize session state for manual entries
        if 'manual_training_data' not in st.session_state:
            st.session_state.manual_training_data = []
        
        with st.form("add_training_example"):
            col1, col2 = st.columns(2)
            
            with col1:
                system_prompt = st.text_area(
                    "System Prompt (optional):",
                    placeholder="You are a FortiGate expert assistant...",
                    height=100
                )
                
                user_prompt = st.text_area(
                    "User Question:",
                    placeholder="How do I configure FortiGate HA in Azure?",
                    height=100
                )
            
            with col2:
                assistant_response = st.text_area(
                    "Assistant Response:",
                    placeholder="To configure FortiGate HA in Azure...",
                    height=200
                )
            
            if st.form_submit_button("➕ Add Example"):
                if user_prompt and assistant_response:
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": user_prompt})
                    messages.append({"role": "assistant", "content": assistant_response})
                    
                    example = {"messages": messages}
                    st.session_state.manual_training_data.append(example)
                    st.success("✅ Example added!")
                    st.rerun()
        
        # Display current manual entries
        if st.session_state.manual_training_data:
            st.write(f"**Current Examples: {len(st.session_state.manual_training_data)}**")
            
            for i, example in enumerate(st.session_state.manual_training_data):
                with st.expander(f"Example {i+1}"):
                    st.json(example)
                    if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                        st.session_state.manual_training_data.pop(i)
                        st.rerun()
            
            training_data = st.session_state.manual_training_data
    
    elif upload_method == "🤖 Generate Sample Data":
        st.write("**Generate FortiGate Training Data:**")
        
        categories = st.multiselect(
            "Select categories:",
            ["Azure Deployment", "HA Configuration", "Security Policies", "VPN Setup", "Troubleshooting"],
            default=["Azure Deployment", "HA Configuration"]
        )
        
        num_examples = st.slider("Number of examples per category:", 5, 20, 10)
        
        if st.button("🎲 Generate Sample Data"):
            with st.spinner("Generating training data..."):
                training_data = generate_sample_training_data(categories, num_examples)
                st.success(f"✅ Generated {len(training_data)} training examples")
                
                # Store in session state
                st.session_state.generated_training_data = training_data
        
        if 'generated_training_data' in st.session_state:
            training_data = st.session_state.generated_training_data
            st.write(f"**Generated Examples: {len(training_data)}**")
            
            # Show preview
            if training_data:
                with st.expander("Preview Generated Data"):
                    st.json(training_data[0])
    
    return training_data

def generate_sample_training_data(categories: List[str], num_examples: int) -> List[Dict]:
    """Generate sample training data for FortiGate scenarios"""
    
    sample_data_templates = {
        "Azure Deployment": [
            {
                "user": "How do I deploy FortiGate in Azure using Terraform?",
                "assistant": "To deploy FortiGate in Azure using Terraform:\n\n1. **Set up Terraform configuration**:\n   - Configure Azure provider\n   - Define resource group\n   - Set up virtual network and subnets\n\n2. **Deploy FortiGate VM**:\n   - Use FortiGate marketplace image\n   - Configure network interfaces\n   - Set up public IP and NSG rules\n\n3. **Configure storage and diagnostics**:\n   - Attach managed disks\n   - Enable boot diagnostics\n\n4. **Apply Terraform configuration**:\n   ```bash\n   terraform init\n   terraform plan\n   terraform apply\n   ```\n\nThis creates a fully functional FortiGate instance in Azure."
            },
            {
                "user": "What are the network requirements for FortiGate in Azure?",
                "assistant": "FortiGate network requirements in Azure:\n\n**Minimum Requirements:**\n- 2 network interfaces (WAN/LAN)\n- Separate subnets for each interface\n- Public IP for WAN interface\n- Network Security Groups with appropriate rules\n\n**Recommended Setup:**\n- 4 network interfaces for full functionality\n- DMZ and HA sync subnets\n- User-defined routes for traffic steering\n- Azure Load Balancer for HA scenarios\n\n**IP Addressing:**\n- Static private IPs recommended\n- Reserve IP ranges for VPN clients\n- Plan for future expansion\n\n**Security:**\n- NSG rules aligned with FortiGate policies\n- Disable IP forwarding on Azure level\n- Enable accelerated networking for performance"
            }
        ],
        "HA Configuration": [
            {
                "user": "How do I configure FortiGate HA in Azure?",
                "assistant": "FortiGate HA configuration in Azure:\n\n**1. Prerequisites:**\n- Two identical FortiGate VMs\n- Dedicated HA sync subnet\n- Azure Load Balancer configured\n- Shared storage for configuration sync\n\n**2. HA Configuration Steps:**\n```\nconfig system ha\n    set group-name \"azure-ha\"\n    set mode a-p\n    set hbdev \"port3\" 50\n    set session-pickup enable\n    set session-pickup-connectionless enable\n    set ha-mgmt-status enable\n    set ha-mgmt-interface \"port4\"\nend\n```\n\n**3. Azure-specific settings:**\n- Configure Azure SDN connector\n- Set up route table updates\n- Enable API access for failover\n\n**4. Testing:**\n- Verify HA status\n- Test failover scenarios\n- Monitor synchronization"
            }
        ],
        "Security Policies": [
            {
                "user": "How do I create security policies for Azure workloads?",
                "assistant": "Creating FortiGate security policies for Azure:\n\n**1. Define Address Objects:**\n```\nconfig firewall address\n    edit \"Azure-Subnet-Web\"\n        set subnet 10.0.1.0 255.255.255.0\n    next\n    edit \"Azure-Subnet-DB\"\n        set subnet 10.0.2.0 255.255.255.0\n    next\nend\n```\n\n**2. Create Service Objects:**\n```\nconfig firewall service custom\n    edit \"HTTPS-8443\"\n        set tcp-portrange 8443\n    next\nend\n```\n\n**3. Security Policies:**\n```\nconfig firewall policy\n    edit 1\n        set name \"Web-to-DB-Allow\"\n        set srcintf \"port2\"\n        set dstintf \"port3\"\n        set srcaddr \"Azure-Subnet-Web\"\n        set dstaddr \"Azure-Subnet-DB\"\n        set action accept\n        set service \"MYSQL\"\n        set logtraffic all\n    next\nend\n```\n\n**Best Practices:**\n- Use least privilege principle\n- Enable logging for compliance\n- Regular policy reviews\n- Application control integration"
            }
        ],
        "VPN Setup": [
            {
                "user": "How do I configure site-to-site VPN between Azure and on-premises?",
                "assistant": "FortiGate site-to-site VPN configuration:\n\n**1. Phase 1 (IKE) Configuration:**\n```\nconfig vpn ipsec phase1-interface\n    edit \"Azure-to-OnPrem\"\n        set interface \"port1\"\n        set peertype any\n        set net-device disable\n        set proposal aes256-sha256\n        set dhgrp 14\n        set remote-gw 203.0.113.1\n        set psksecret \"your-preshared-key\"\n    next\nend\n```\n\n**2. Phase 2 (IPSec) Configuration:**\n```\nconfig vpn ipsec phase2-interface\n    edit \"Azure-to-OnPrem\"\n        set phase1name \"Azure-to-OnPrem\"\n        set proposal aes256-sha256\n        set dhgrp 14\n        set src-subnet 10.0.0.0 255.255.0.0\n        set dst-subnet 192.168.0.0 255.255.0.0\n    next\nend\n```\n\n**3. Routing Configuration:**\n```\nconfig router static\n    edit 1\n        set dst 192.168.0.0 255.255.0.0\n        set device \"Azure-to-OnPrem\"\n    next\nend\n```\n\n**4. Firewall Policies:**\n- Allow traffic between subnets\n- Enable NAT if required\n- Configure appropriate logging"
            }
        ],
        "Troubleshooting": [
            {
                "user": "How do I troubleshoot FortiGate connectivity issues in Azure?",
                "assistant": "FortiGate Azure troubleshooting steps:\n\n**1. Basic Connectivity:**\n```bash\n# Check interface status\nget system interface physical\n\n# Verify routing table\nget router info routing-table all\n\n# Test connectivity\nexecute ping 8.8.8.8\n```\n\n**2. Azure-specific Checks:**\n- Verify NSG rules allow traffic\n- Check User Defined Routes\n- Confirm public IP associations\n- Validate subnet configurations\n\n**3. FortiGate Diagnostics:**\n```bash\n# Check system resources\nget system performance status\n\n# Monitor real-time traffic\ndiagnose sniffer packet any 'host 10.0.1.100' 4\n\n# Review logs\nexecute log filter category traffic\nexecute log display\n```\n\n**4. Common Issues:**\n- Asymmetric routing\n- MTU size problems\n- License limitations\n- Azure service limits\n\n**5. Performance Optimization:**\n- Enable accelerated networking\n- Optimize FortiGate settings\n- Monitor Azure metrics\n- Regular firmware updates"
            }
        ]
    }
    
    training_data = []
    
    for category in categories:
        if category in sample_data_templates:
            templates = sample_data_templates[category]
            for i in range(min(num_examples, len(templates))):
                template = templates[i % len(templates)]
                
                # Add variation to avoid exact duplicates
                variation_suffix = f" (Scenario {i+1})" if i > 0 else ""
                
                messages = [
                    {"role": "system", "content": "You are a FortiGate expert assistant specializing in Azure deployments and network security."},
                    {"role": "user", "content": template["user"] + variation_suffix},
                    {"role": "assistant", "content": template["assistant"]}
                ]
                
                training_data.append({"messages": messages})
    
    return training_data

def display_fine_tuning_progress(manager: EnhancedFineTuningManager, job_id: str):
    """Display real-time fine-tuning progress"""
    st.subheader("🚀 Fine-Tuning Progress")
    
    # Create placeholders for dynamic updates
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    details_placeholder = st.empty()
    
    # Progress tracking loop
    max_checks = 180  # 30 minutes with 10-second intervals
    check_count = 0
    
    while check_count < max_checks:
        job_status = manager.get_job_status(job_id)
        
        if job_status:
            status = job_status["status"]
            
            with status_placeholder.container():
                if status == "validating_files":
                    st.info("🔍 Validating training files...")
                elif status == "queued":
                    st.info("⏳ Job queued for processing...")
                elif status == "running":
                    st.success("🏃‍♂️ Fine-tuning in progress...")
                elif status == "succeeded":
                    st.success("🎉 Fine-tuning completed successfully!")
                    break
                elif status == "failed":
                    st.error("❌ Fine-tuning failed")
                    if job_status.get("error"):
                        st.error(f"Error: {job_status['error']}")
                    break
                elif status == "cancelled":
                    st.warning("⚠️ Fine-tuning was cancelled")
                    break
            
            with details_placeholder.container():
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Status", status.replace("_", " ").title())
                
                with col2:
                    if job_status.get("trained_tokens"):
                        st.metric("Trained Tokens", f"{job_status['trained_tokens']:,}")
                
                with col3:
                    st.metric("Job ID", job_id[:8] + "...")
                
                # Show detailed information
                with st.expander("📋 Job Details"):
                    st.json(job_status)
            
            if status in ["succeeded", "failed", "cancelled"]:
                break
        
        time.sleep(10)  # Wait 10 seconds before next check
        check_count += 1
    
    return job_status

def display_enhanced_fine_tuning_interface():
    """Main enhanced fine-tuning interface"""
    st.header("🎯 Enhanced Fine-Tuning Interface")
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        st.error("❌ OpenAI API key required")
        st.code("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    if not OPENAI_AVAILABLE:
        st.error("❌ OpenAI library not installed")
        st.code("pip install openai")
        return
    
    # Initialize manager
    manager = EnhancedFineTuningManager()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Data Upload", 
        "🚀 Start Fine-Tuning", 
        "📊 Progress Tracking", 
        "🎯 Model Management"
    ])
    
    with tab1:
        training_data = display_training_data_uploader()
        
        if training_data:
            st.divider()
            st.subheader("🔍 Data Validation")
            
            validation = manager.validate_training_data(training_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if validation["valid"]:
                    st.success("✅ Training data is valid!")
                else:
                    st.error("❌ Training data has issues:")
                    for error in validation["errors"]:
                        st.error(f"• {error}")
                
                for warning in validation["warnings"]:
                    st.warning(f"⚠️ {warning}")
            
            with col2:
                st.write("**Data Statistics:**")
                stats = validation["stats"]
                st.metric("Total Examples", stats["total_examples"])
                st.metric("Unique Prompts", stats["unique_prompts"])
                st.metric("Avg Prompt Length", f"{stats['avg_prompt_length']:.0f} chars")
                st.metric("Avg Response Length", f"{stats['avg_completion_length']:.0f} chars")
            
            # Save training data for next tab
            if validation["valid"]:
                st.session_state.validated_training_data = training_data
    
    with tab2:
        st.subheader("🚀 Launch Fine-Tuning")
        
        if 'validated_training_data' not in st.session_state:
            st.info("📁 Please upload and validate training data first")
            return
        
        training_data = st.session_state.validated_training_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_choice = st.selectbox(
                "Base Model:",
                ["gpt-3.5-turbo", "gpt-4o-mini"],
                help="Choose the base model for fine-tuning"
            )
            
            model_suffix = st.text_input(
                "Model Suffix:",
                value=f"fortigate-{datetime.now().strftime('%Y%m%d')}",
                help="Custom suffix for your fine-tuned model"
            )
        
        with col2:
            st.write("**Training Summary:**")
            st.metric("Examples", len(training_data))
            st.metric("Base Model", model_choice)
            st.metric("Estimated Cost", "$8-20")
            st.metric("Estimated Time", "10-30 min")
        
        if st.button("🚀 Start Fine-Tuning Process", type="primary"):
            with st.spinner("Preparing fine-tuning job..."):
                # Save training data to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                training_file_path = f"training_data_{timestamp}.jsonl"
                
                with open(training_file_path, 'w') as f:
                    for example in training_data:
                        f.write(json.dumps(example) + '\n')
                
                # Upload file
                file_id = manager.upload_training_file(training_file_path)
                
                if file_id:
                    st.success(f"✅ Training file uploaded: {file_id}")
                    
                    # Start fine-tuning
                    job_id = manager.start_fine_tuning(file_id, model_choice, model_suffix)
                    
                    if job_id:
                        st.success(f"🚀 Fine-tuning job started: {job_id}")
                        st.session_state.current_job_id = job_id
                        
                        # Switch to progress tracking tab
                        st.info("🔄 Switching to progress tracking...")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("❌ Failed to start fine-tuning job")
                else:
                    st.error("❌ Failed to upload training file")
                
                # Clean up temporary file
                if os.path.exists(training_file_path):
                    os.remove(training_file_path)
    
    with tab3:
        st.subheader("📊 Progress Tracking")
        
        # Check for active job
        if 'current_job_id' in st.session_state:
            job_id = st.session_state.current_job_id
            st.info(f"Tracking job: {job_id}")
            
            if st.button("🔄 Start Real-Time Tracking"):
                job_status = display_fine_tuning_progress(manager, job_id)
                
                if job_status and job_status["status"] == "succeeded":
                    st.balloons()
                    st.success(f"🎉 Model ready: {job_status['fine_tuned_model']}")
        else:
            st.info("No active fine-tuning job found")
            
            # Manual job ID input
            manual_job_id = st.text_input("Enter Job ID to track:")
            if manual_job_id and st.button("Track Job"):
                st.session_state.current_job_id = manual_job_id
                st.rerun()
    
    with tab4:
        st.subheader("🎯 Model Management")
        
        # List fine-tuned models
        models = manager.list_fine_tuned_models()
        
        if models:
            st.write(f"**Available Fine-Tuned Models: {len(models)}**")
            
            for model in models:
                with st.expander(f"📋 {model['id']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Created:** {model['created']}")
                        st.write(f"**Owner:** {model['owned_by']}")
                    
                    with col2:
                        if st.button(f"🧪 Test Model", key=f"test_{model['id']}"):
                            st.session_state.test_model_id = model['id']
                        
                        if st.button(f"🗑️ Delete Model", key=f"delete_{model['id']}"):
                            # Add confirmation dialog
                            st.warning("Model deletion not implemented in demo")
            
            # Model testing interface
            if 'test_model_id' in st.session_state:
                st.divider()
                st.subheader("🧪 Model Testing")
                
                test_prompt = st.text_area(
                    "Test your fine-tuned model:",
                    placeholder="How do I configure FortiGate HA in Azure?"
                )
                
                if st.button("🧪 Test") and test_prompt:
                    with st.spinner("Testing model..."):
                        try:
                            response = manager.client.chat.completions.create(
                                model=st.session_state.test_model_id,
                                messages=[
                                    {"role": "system", "content": "You are a FortiGate expert assistant."},
                                    {"role": "user", "content": test_prompt}
                                ],
                                max_tokens=500
                            )
                            
                            st.write("**Model Response:**")
                            st.write(response.choices[0].message.content)
                            
                        except Exception as e:
                            st.error(f"Error testing model: {str(e)}")
        else:
            st.info("No fine-tuned models found")
            st.write("Complete the fine-tuning process to see your models here.")

if __name__ == "__main__":
    display_enhanced_fine_tuning_interface()
