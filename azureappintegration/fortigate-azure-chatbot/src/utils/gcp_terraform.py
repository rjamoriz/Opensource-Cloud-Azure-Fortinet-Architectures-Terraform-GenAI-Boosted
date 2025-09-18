"""
Google Cloud Platform Terraform Integration
Manages FortiGate deployments on Google Cloud Platform
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import subprocess
import streamlit as st

try:
    from google.cloud import compute_v1
    from google.cloud import resource_manager_v3
    from google.oauth2 import service_account
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

# Import GCP authentication component
try:
    from .gcp_auth_component import (
        ensure_gcp_configured,
        get_gcp_project_id,
        get_gcp_credentials,
        display_gcp_quick_setup
    )
    GCP_AUTH_COMPONENT_AVAILABLE = True
except ImportError:
    GCP_AUTH_COMPONENT_AVAILABLE = False

logger = logging.getLogger(__name__)

class GCPTerraformManager:
    """Manages FortiGate Terraform deployments on Google Cloud Platform"""
    
    def __init__(self, project_id: str, credentials_path: Optional[str] = None):
        """
        Initialize GCP Terraform Manager
        
        Args:
            project_id: GCP Project ID
            credentials_path: Path to service account credentials JSON file
        """
        self.project_id = project_id
        self.credentials_path = credentials_path
        self.terraform_base_path = self._get_terraform_base_path()
        
        if not GCP_AVAILABLE:
            logger.warning("Google Cloud SDK not available. Install requirements_gcp.txt")
            return
            
        self._initialize_gcp_clients()
    
    def _get_terraform_base_path(self) -> Path:
        """Get the base path for GCP Terraform templates"""
        # Assuming the terraform templates are in the parent directory structure
        current_dir = Path(__file__).parent
        terraform_path = current_dir.parent.parent.parent / "fortigate-terraform-deploy" / "gcp"
        return terraform_path
    
    def _initialize_gcp_clients(self):
        """Initialize GCP service clients"""
        try:
            if self.credentials_path:
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                self.compute_client = compute_v1.InstancesClient(credentials=credentials)
                self.resource_manager_client = resource_manager_v3.ProjectsClient(credentials=credentials)
            else:
                # Use default credentials (from gcloud auth or service account)
                self.compute_client = compute_v1.InstancesClient()
                self.resource_manager_client = resource_manager_v3.ProjectsClient()
                
            logger.info("GCP clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GCP clients: {e}")
            self.compute_client = None
            self.resource_manager_client = None
    
    def list_available_templates(self) -> Dict[str, List[str]]:
        """List available FortiGate templates for GCP"""
        templates = {}
        
        if not self.terraform_base_path.exists():
            logger.warning(f"Terraform templates path not found: {self.terraform_base_path}")
            return templates
        
        try:
            for version_dir in self.terraform_base_path.iterdir():
                if version_dir.is_dir():
                    version = version_dir.name
                    template_types = []
                    
                    for template_dir in version_dir.iterdir():
                        if template_dir.is_dir():
                            template_types.append(template_dir.name)
                    
                    templates[version] = sorted(template_types)
            
            logger.info(f"Found {len(templates)} FortiGate template versions for GCP")
            
        except Exception as e:
            logger.error(f"Error listing templates: {e}")
        
        return templates
    
    def get_template_path(self, version: str, template_type: str) -> Optional[Path]:
        """Get the full path to a specific template"""
        template_path = self.terraform_base_path / version / template_type
        
        if template_path.exists():
            return template_path
        else:
            logger.error(f"Template not found: {template_path}")
            return None
    
    def validate_template(self, version: str, template_type: str) -> Tuple[bool, str]:
        """Validate a Terraform template"""
        template_path = self.get_template_path(version, template_type)
        
        if not template_path:
            return False, "Template path not found"
        
        try:
            # Check if required Terraform files exist
            main_tf = template_path / "main.tf"
            variables_tf = template_path / "variables.tf"
            
            if not main_tf.exists():
                return False, "main.tf file not found"
            
            if not variables_tf.exists():
                return False, "variables.tf file not found"
            
            # Run terraform validate
            result = subprocess.run(
                ["terraform", "validate"],
                cwd=template_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return True, "Template validation successful"
            else:
                return False, f"Terraform validation failed: {result.stderr}"
                
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def initialize_terraform(self, version: str, template_type: str) -> Tuple[bool, str]:
        """Initialize Terraform in the specified template directory"""
        template_path = self.get_template_path(version, template_type)
        
        if not template_path:
            return False, "Template path not found"
        
        try:
            # Run terraform init
            result = subprocess.run(
                ["terraform", "init"],
                cwd=template_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Terraform initialized successfully for {version}/{template_type}")
                return True, "Terraform initialization successful"
            else:
                logger.error(f"Terraform init failed: {result.stderr}")
                return False, f"Terraform initialization failed: {result.stderr}"
                
        except Exception as e:
            logger.error(f"Error during terraform init: {e}")
            return False, f"Initialization error: {str(e)}"
    
    def plan_deployment(self, version: str, template_type: str, variables: Dict[str, str]) -> Tuple[bool, str]:
        """Create a Terraform plan for the deployment"""
        template_path = self.get_template_path(version, template_type)
        
        if not template_path:
            return False, "Template path not found"
        
        try:
            # Create terraform.tfvars file
            tfvars_content = []
            for key, value in variables.items():
                if isinstance(value, str):
                    tfvars_content.append(f'{key} = "{value}"')
                else:
                    tfvars_content.append(f'{key} = {value}')
            
            tfvars_path = template_path / "terraform.tfvars"
            with open(tfvars_path, 'w') as f:
                f.write('\n'.join(tfvars_content))
            
            # Run terraform plan
            result = subprocess.run(
                ["terraform", "plan", "-out=tfplan"],
                cwd=template_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Terraform plan created successfully for {version}/{template_type}")
                return True, result.stdout
            else:
                logger.error(f"Terraform plan failed: {result.stderr}")
                return False, f"Terraform plan failed: {result.stderr}"
                
        except Exception as e:
            logger.error(f"Error during terraform plan: {e}")
            return False, f"Planning error: {str(e)}"
    
    def apply_deployment(self, version: str, template_type: str) -> Tuple[bool, str]:
        """Apply the Terraform deployment"""
        template_path = self.get_template_path(version, template_type)
        
        if not template_path:
            return False, "Template path not found"
        
        try:
            # Check if plan exists
            plan_path = template_path / "tfplan"
            if not plan_path.exists():
                return False, "Terraform plan not found. Please run planning first."
            
            # Run terraform apply
            result = subprocess.run(
                ["terraform", "apply", "-auto-approve", "tfplan"],
                cwd=template_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Terraform apply successful for {version}/{template_type}")
                return True, result.stdout
            else:
                logger.error(f"Terraform apply failed: {result.stderr}")
                return False, f"Terraform apply failed: {result.stderr}"
                
        except Exception as e:
            logger.error(f"Error during terraform apply: {e}")
            return False, f"Apply error: {str(e)}"
    
    def destroy_deployment(self, version: str, template_type: str) -> Tuple[bool, str]:
        """Destroy the Terraform deployment"""
        template_path = self.get_template_path(version, template_type)
        
        if not template_path:
            return False, "Template path not found"
        
        try:
            # Run terraform destroy
            result = subprocess.run(
                ["terraform", "destroy", "-auto-approve"],
                cwd=template_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Terraform destroy successful for {version}/{template_type}")
                return True, result.stdout
            else:
                logger.error(f"Terraform destroy failed: {result.stderr}")
                return False, f"Terraform destroy failed: {result.stderr}"
                
        except Exception as e:
            logger.error(f"Error during terraform destroy: {e}")
            return False, f"Destroy error: {str(e)}"
    
    def get_deployment_status(self, version: str, template_type: str) -> Dict[str, any]:
        """Get the current status of a deployment"""
        template_path = self.get_template_path(version, template_type)
        
        if not template_path:
            return {"status": "error", "message": "Template path not found"}
        
        try:
            # Check if terraform state exists
            state_path = template_path / "terraform.tfstate"
            
            if not state_path.exists():
                return {"status": "not_deployed", "message": "No deployment found"}
            
            # Run terraform show to get current state
            result = subprocess.run(
                ["terraform", "show", "-json"],
                cwd=template_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                state_data = json.loads(result.stdout)
                
                resources = []
                if "values" in state_data and "root_module" in state_data["values"]:
                    root_module = state_data["values"]["root_module"]
                    if "resources" in root_module:
                        resources = root_module["resources"]
                
                return {
                    "status": "deployed",
                    "resources": len(resources),
                    "details": state_data
                }
            else:
                return {"status": "error", "message": f"Failed to get state: {result.stderr}"}
                
        except Exception as e:
            return {"status": "error", "message": f"Status check error: {str(e)}"}
    
    def list_instances(self, zone: str = None) -> List[Dict[str, str]]:
        """List FortiGate instances in the project"""
        if not self.compute_client:
            return []
        
        instances = []
        
        try:
            if zone:
                zones = [zone]
            else:
                # List all zones in the project (simplified - you might want to get this dynamically)
                zones = ["us-central1-a", "us-central1-b", "us-central1-c"]
            
            for zone_name in zones:
                request = compute_v1.ListInstancesRequest(
                    project=self.project_id,
                    zone=zone_name
                )
                
                page_result = self.compute_client.list(request=request)
                
                for instance in page_result:
                    # Filter for FortiGate instances (assuming they have 'fortigate' in the name)
                    if 'fortigate' in instance.name.lower():
                        instances.append({
                            "name": instance.name,
                            "zone": zone_name,
                            "status": instance.status,
                            "machine_type": instance.machine_type.split('/')[-1],
                            "internal_ip": instance.network_interfaces[0].network_i_p if instance.network_interfaces else "",
                            "external_ip": instance.network_interfaces[0].access_configs[0].nat_i_p if instance.network_interfaces and instance.network_interfaces[0].access_configs else ""
                        })
                        
        except Exception as e:
            logger.error(f"Error listing instances: {e}")
        
        return instances

def display_gcp_terraform_interface():
    """Streamlit interface for GCP Terraform operations"""
    st.subheader("🌐 Google Cloud Platform Deployment")
    
    if not GCP_AVAILABLE:
        st.error("Google Cloud SDK not installed. Please install requirements_gcp.txt")
        return
    
    # Check and ensure GCP authentication
    if GCP_AUTH_COMPONENT_AVAILABLE:
        if not ensure_gcp_configured():
            st.warning("Please configure Google Cloud Platform authentication above.")
            return
        
        project_id = get_gcp_project_id()
        credentials_path = get_gcp_credentials()
        
        # Show current configuration
        st.success(f"✅ Using GCP Project: **{project_id}**")
        
    else:
        # Fallback to manual project input
        st.warning("GCP Authentication component not available. Using manual configuration.")
        
        # Project configuration
        col1, col2 = st.columns(2)
        
        with col1:
            project_id = st.text_input("GCP Project ID", help="Your Google Cloud Project ID")
        
        with col2:
            credentials_file = st.file_uploader(
                "Service Account Credentials (Optional)",
                type="json",
                help="Upload service account JSON file, or leave empty to use default credentials"
            )
        
        if not project_id:
            st.warning("Please enter your GCP Project ID to continue")
            return
        
        credentials_path = None
        if credentials_file:
            # Save uploaded file temporarily
            credentials_path = f"/tmp/{credentials_file.name}"
            with open(credentials_path, "wb") as f:
                f.write(credentials_file.getbuffer())
    
    # Initialize manager
    try:
        gcp_manager = GCPTerraformManager(project_id, credentials_path)
        
        # Template selection
        st.subheader("Select FortiGate Template")
        
        templates = gcp_manager.list_available_templates()
        
        if not templates:
            st.error("No FortiGate templates found. Please check the terraform templates directory.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            version = st.selectbox("FortiGate Version", options=list(templates.keys()))
        
        with col2:
            template_type = st.selectbox("Deployment Type", options=templates.get(version, []))
        
        if version and template_type:
            # Validate template
            is_valid, validation_message = gcp_manager.validate_template(version, template_type)
            
            if is_valid:
                st.success(f"✅ {validation_message}")
            else:
                st.error(f"❌ {validation_message}")
                return
            
            # Deployment configuration
            st.subheader("Deployment Configuration")
            
            # Basic configuration
            col1, col2 = st.columns(2)
            
            with col1:
                deployment_name = st.text_input("Deployment Name", value=f"fortigate-{template_type}")
                region = st.selectbox("Region", ["us-central1", "us-west1", "us-east1", "europe-west1", "asia-southeast1"])
                zone = st.selectbox("Zone", [f"{region}-a", f"{region}-b", f"{region}-c"])
            
            with col2:
                machine_type = st.selectbox("Machine Type", ["e2-standard-4", "n1-standard-4", "n2-standard-4"])
                disk_size = st.number_input("Boot Disk Size (GB)", min_value=10, max_value=1000, value=30)
            
            # Network configuration
            st.subheader("Network Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                vpc_name = st.text_input("VPC Name", value="fortigate-vpc")
                subnet_name = st.text_input("Subnet Name", value="fortigate-subnet")
            
            with col2:
                subnet_cidr = st.text_input("Subnet CIDR", value="10.0.1.0/24")
                
            # Create variables dictionary
            variables = {
                "project_id": project_id,
                "region": region,
                "zone": zone,
                "deployment_name": deployment_name,
                "machine_type": machine_type,
                "boot_disk_size": disk_size,
                "vpc_name": vpc_name,
                "subnet_name": subnet_name,
                "subnet_cidr": subnet_cidr
            }
            
            # Deployment actions
            st.subheader("Deployment Actions")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🔧 Initialize"):
                    with st.spinner("Initializing Terraform..."):
                        success, message = gcp_manager.initialize_terraform(version, template_type)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
            
            with col2:
                if st.button("📋 Plan"):
                    with st.spinner("Creating deployment plan..."):
                        success, message = gcp_manager.plan_deployment(version, template_type, variables)
                        if success:
                            st.success("Plan created successfully!")
                            with st.expander("View Plan Details"):
                                st.code(message, language="text")
                        else:
                            st.error(message)
            
            with col3:
                if st.button("🚀 Deploy"):
                    with st.spinner("Deploying FortiGate..."):
                        success, message = gcp_manager.apply_deployment(version, template_type)
                        if success:
                            st.success("Deployment successful!")
                            with st.expander("View Deployment Details"):
                                st.code(message, language="text")
                        else:
                            st.error(message)
            
            with col4:
                if st.button("🗑️ Destroy"):
                    if st.checkbox("I understand this will destroy the deployment"):
                        with st.spinner("Destroying deployment..."):
                            success, message = gcp_manager.destroy_deployment(version, template_type)
                            if success:
                                st.success("Deployment destroyed successfully!")
                            else:
                                st.error(message)
            
            # Deployment status
            st.subheader("Deployment Status")
            
            status = gcp_manager.get_deployment_status(version, template_type)
            
            if status["status"] == "deployed":
                st.success(f"✅ Deployment active with {status['resources']} resources")
            elif status["status"] == "not_deployed":
                st.info("ℹ️ No active deployment found")
            else:
                st.error(f"❌ {status['message']}")
            
            # Instance list
            if status["status"] == "deployed":
                st.subheader("FortiGate Instances")
                
                instances = gcp_manager.list_instances()
                
                if instances:
                    for instance in instances:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.write(f"**{instance['name']}**")
                        with col2:
                            st.write(f"Status: {instance['status']}")
                        with col3:
                            st.write(f"Internal IP: {instance['internal_ip']}")
                        with col4:
                            st.write(f"External IP: {instance['external_ip']}")
                else:
                    st.info("No FortiGate instances found")
        
    except Exception as e:
        st.error(f"Error initializing GCP manager: {str(e)}")
        logger.error(f"GCP manager initialization error: {e}")

if __name__ == "__main__":
    # For testing
    display_gcp_terraform_interface()
