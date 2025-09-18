"""
FortiGate Manager for Multi-Cloud Deployment Management
"""

import streamlit as st
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FortiGateManager:
    """FortiGate deployment and configuration manager"""
    
    def __init__(self):
        """Initialize FortiGate manager"""
        self.deployments = {}
        self.configurations = {}
        
    def create_deployment(self, config: Dict[str, Any]) -> str:
        """
        Create a new FortiGate deployment
        
        Args:
            config: Deployment configuration
            
        Returns:
            str: Deployment ID
        """
        deployment_id = f"fg-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        self.deployments[deployment_id] = {
            "id": deployment_id,
            "name": config.get("name", f"FortiGate-{deployment_id}"),
            "cloud_provider": config.get("cloud_provider", "azure"),
            "region": config.get("region", "eastus"),
            "instance_type": config.get("instance_type", "Standard_D4s_v3"),
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "config": config
        }
        
        logger.info(f"Created FortiGate deployment: {deployment_id}")
        return deployment_id
    
    def get_deployment(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment by ID"""
        return self.deployments.get(deployment_id)
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all deployments"""
        return list(self.deployments.values())
    
    def update_deployment_status(self, deployment_id: str, status: str):
        """Update deployment status"""
        if deployment_id in self.deployments:
            self.deployments[deployment_id]["status"] = status
            self.deployments[deployment_id]["updated_at"] = datetime.now().isoformat()
    
    def delete_deployment(self, deployment_id: str) -> bool:
        """Delete a deployment"""
        if deployment_id in self.deployments:
            del self.deployments[deployment_id]
            logger.info(f"Deleted FortiGate deployment: {deployment_id}")
            return True
        return False
    
    def get_deployment_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get available deployment templates"""
        return {
            "single_vm": {
                "name": "Single FortiGate VM",
                "description": "Basic single FortiGate VM deployment",
                "resources": ["VM", "Network Security Group", "Public IP"],
                "use_cases": ["Small office", "Branch office", "Testing"]
            },
            "ha_active_passive": {
                "name": "HA Active-Passive",
                "description": "High availability with active-passive configuration",
                "resources": ["2x VMs", "Load Balancer", "Shared Storage"],
                "use_cases": ["Production", "Critical workloads", "High availability"]
            },
            "ha_active_active": {
                "name": "HA Active-Active",
                "description": "High availability with active-active configuration",
                "resources": ["2x VMs", "Load Balancer", "Session sync"],
                "use_cases": ["High throughput", "Load distribution", "Scalability"]
            },
            "hub_spoke": {
                "name": "Hub and Spoke",
                "description": "Centralized security with hub and spoke topology",
                "resources": ["Hub VM", "Spoke connections", "Route tables"],
                "use_cases": ["Multi-site", "Centralized security", "Branch connectivity"]
            }
        }
    
    def generate_terraform_config(self, deployment_id: str) -> str:
        """Generate Terraform configuration for deployment"""
        deployment = self.get_deployment(deployment_id)
        if not deployment:
            return ""
        
        config = deployment["config"]
        cloud_provider = config.get("cloud_provider", "azure")
        
        if cloud_provider == "azure":
            return self._generate_azure_terraform(config)
        elif cloud_provider == "gcp":
            return self._generate_gcp_terraform(config)
        else:
            return ""
    
    def _generate_azure_terraform(self, config: Dict[str, Any]) -> str:
        """Generate Azure Terraform configuration"""
        return f"""
# FortiGate Azure Deployment
terraform {{
  required_providers {{
    azurerm = {{
      source  = "hashicorp/azurerm"
      version = "~>3.0"
    }}
  }}
}}

provider "azurerm" {{
  features {{}}
}}

resource "azurerm_resource_group" "fortigate" {{
  name     = "{config.get('resource_group', 'rg-fortigate')}"
  location = "{config.get('region', 'East US')}"
}}

resource "azurerm_virtual_network" "fortigate" {{
  name                = "{config.get('vnet_name', 'vnet-fortigate')}"
  address_space       = ["{config.get('address_space', '10.0.0.0/16')}"]
  location            = azurerm_resource_group.fortigate.location
  resource_group_name = azurerm_resource_group.fortigate.name
}}

resource "azurerm_subnet" "external" {{
  name                 = "external"
  resource_group_name  = azurerm_resource_group.fortigate.name
  virtual_network_name = azurerm_virtual_network.fortigate.name
  address_prefixes     = ["{config.get('external_subnet', '10.0.1.0/24')}"]
}}

resource "azurerm_subnet" "internal" {{
  name                 = "internal"
  resource_group_name  = azurerm_resource_group.fortigate.name
  virtual_network_name = azurerm_virtual_network.fortigate.name
  address_prefixes     = ["{config.get('internal_subnet', '10.0.2.0/24')}"]
}}

resource "azurerm_public_ip" "fortigate" {{
  name                = "pip-fortigate"
  location            = azurerm_resource_group.fortigate.location
  resource_group_name = azurerm_resource_group.fortigate.name
  allocation_method   = "Static"
  sku                 = "Standard"
}}

resource "azurerm_network_security_group" "fortigate" {{
  name                = "nsg-fortigate"
  location            = azurerm_resource_group.fortigate.location
  resource_group_name = azurerm_resource_group.fortigate.name

  security_rule {{
    name                       = "HTTPS"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }}

  security_rule {{
    name                       = "SSH"
    priority                   = 1002
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }}
}}

resource "azurerm_network_interface" "external" {{
  name                = "nic-fortigate-external"
  location            = azurerm_resource_group.fortigate.location
  resource_group_name = azurerm_resource_group.fortigate.name

  ip_configuration {{
    name                          = "external"
    subnet_id                     = azurerm_subnet.external.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.fortigate.id
  }}
}}

resource "azurerm_network_interface" "internal" {{
  name                = "nic-fortigate-internal"
  location            = azurerm_resource_group.fortigate.location
  resource_group_name = azurerm_resource_group.fortigate.name

  ip_configuration {{
    name                          = "internal"
    subnet_id                     = azurerm_subnet.internal.id
    private_ip_address_allocation = "Dynamic"
  }}
}}

resource "azurerm_virtual_machine" "fortigate" {{
  name                = "{config.get('vm_name', 'vm-fortigate')}"
  location            = azurerm_resource_group.fortigate.location
  resource_group_name = azurerm_resource_group.fortigate.name
  network_interface_ids = [
    azurerm_network_interface.external.id,
    azurerm_network_interface.internal.id,
  ]
  primary_network_interface_id = azurerm_network_interface.external.id
  vm_size                      = "{config.get('instance_type', 'Standard_D4s_v3')}"

  storage_image_reference {{
    publisher = "fortinet"
    offer     = "fortinet_fortigate-vm_v5"
    sku       = "fortinet_fg-vm"
    version   = "latest"
  }}

  storage_os_disk {{
    name              = "osdisk-fortigate"
    caching           = "ReadWrite"
    create_option     = "FromImage"
    managed_disk_type = "Premium_LRS"
  }}

  os_profile {{
    computer_name  = "fortigate"
    admin_username = "{config.get('admin_username', 'azureuser')}"
    admin_password = "{config.get('admin_password', 'P@ssw0rd123!')}"
  }}

  os_profile_linux_config {{
    disable_password_authentication = false
  }}

  plan {{
    name      = "fortinet_fg-vm"
    publisher = "fortinet"
    product   = "fortinet_fortigate-vm_v5"
  }}
}}

output "public_ip" {{
  value = azurerm_public_ip.fortigate.ip_address
}}

output "admin_url" {{
  value = "https://${{azurerm_public_ip.fortigate.ip_address}}"
}}
"""
    
    def _generate_gcp_terraform(self, config: Dict[str, Any]) -> str:
        """Generate GCP Terraform configuration"""
        return f"""
# FortiGate GCP Deployment
terraform {{
  required_providers {{
    google = {{
      source  = "hashicorp/google"
      version = "~> 4.0"
    }}
  }}
}}

provider "google" {{
  project = "{config.get('project_id', 'your-project-id')}"
  region  = "{config.get('region', 'us-central1')}"
}}

resource "google_compute_network" "fortigate_vpc" {{
  name                    = "{config.get('vpc_name', 'fortigate-vpc')}"
  auto_create_subnetworks = false
}}

resource "google_compute_subnetwork" "external" {{
  name          = "external-subnet"
  ip_cidr_range = "{config.get('external_subnet', '10.0.1.0/24')}"
  region        = "{config.get('region', 'us-central1')}"
  network       = google_compute_network.fortigate_vpc.id
}}

resource "google_compute_subnetwork" "internal" {{
  name          = "internal-subnet"
  ip_cidr_range = "{config.get('internal_subnet', '10.0.2.0/24')}"
  region        = "{config.get('region', 'us-central1')}"
  network       = google_compute_network.fortigate_vpc.id
}}

resource "google_compute_firewall" "fortigate_admin" {{
  name    = "allow-fortigate-admin"
  network = google_compute_network.fortigate_vpc.name

  allow {{
    protocol = "tcp"
    ports    = ["443", "22"]
  }}

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["fortigate"]
}}

resource "google_compute_instance" "fortigate" {{
  name         = "{config.get('instance_name', 'fortigate-vm')}"
  machine_type = "{config.get('machine_type', 'n1-standard-4')}"
  zone         = "{config.get('zone', 'us-central1-a')}"

  tags = ["fortigate"]

  boot_disk {{
    initialize_params {{
      image = "projects/fortigcp-project-001/global/images/fortinet-fgtondemand-724-20230301-001-w-license"
      size  = 10
      type  = "pd-standard"
    }}
  }}

  network_interface {{
    subnetwork = google_compute_subnetwork.external.id
    access_config {{
      // Ephemeral public IP
    }}
  }}

  network_interface {{
    subnetwork = google_compute_subnetwork.internal.id
  }}

  metadata = {{
    user-data = "{config.get('user_data', '')}"
  }}

  service_account {{
    scopes = ["cloud-platform"]
  }}
}}

output "public_ip" {{
  value = google_compute_instance.fortigate.network_interface[0].access_config[0].nat_ip
}}

output "admin_url" {{
  value = "https://${{google_compute_instance.fortigate.network_interface[0].access_config[0].nat_ip}}"
}}
"""
    
    def get_deployment_cost_estimate(self, deployment_id: str) -> Dict[str, Any]:
        """Get cost estimate for deployment"""
        deployment = self.get_deployment(deployment_id)
        if not deployment:
            return {"error": "Deployment not found"}
        
        config = deployment["config"]
        cloud_provider = config.get("cloud_provider", "azure")
        instance_type = config.get("instance_type", "Standard_D4s_v3")
        
        # Mock pricing data
        if cloud_provider == "azure":
            pricing = {
                "Standard_D2s_v3": {"hourly": 0.096, "monthly": 70.08},
                "Standard_D4s_v3": {"hourly": 0.192, "monthly": 140.16},
                "Standard_D8s_v3": {"hourly": 0.384, "monthly": 280.32}
            }
        else:  # GCP
            pricing = {
                "n1-standard-2": {"hourly": 0.095, "monthly": 69.35},
                "n1-standard-4": {"hourly": 0.190, "monthly": 138.70},
                "n1-standard-8": {"hourly": 0.380, "monthly": 277.40}
            }
        
        cost_info = pricing.get(instance_type, {"hourly": 0.192, "monthly": 140.16})
        
        return {
            "deployment_id": deployment_id,
            "cloud_provider": cloud_provider,
            "instance_type": instance_type,
            "hourly_cost": cost_info["hourly"],
            "monthly_estimate": cost_info["monthly"],
            "currency": "USD",
            "includes": ["Compute", "Storage", "Network"]
        }
    
    def render_dashboard(self):
        """Render FortiGate manager dashboard"""
        st.subheader("🛡️ FortiGate Deployment Manager")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🚀 Deploy", "⚙️ Manage", "📋 Templates"])
        
        with tab1:
            self._render_overview()
        
        with tab2:
            self._render_deployment_form()
        
        with tab3:
            self._render_management()
        
        with tab4:
            self._render_templates()
    
    def _render_overview(self):
        """Render overview tab"""
        deployments = self.list_deployments()
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Deployments", len(deployments))
        
        with col2:
            active_count = len([d for d in deployments if d["status"] == "running"])
            st.metric("Active", active_count)
        
        with col3:
            pending_count = len([d for d in deployments if d["status"] == "pending"])
            st.metric("Pending", pending_count)
        
        with col4:
            total_cost = sum([140.16 for d in deployments if d["status"] == "running"])
            st.metric("Monthly Cost", f"${total_cost:.2f}")
        
        # Recent deployments
        if deployments:
            st.subheader("Recent Deployments")
            st.dataframe(deployments, use_container_width=True)
        else:
            st.info("No deployments found. Create your first deployment in the Deploy tab.")
    
    def _render_deployment_form(self):
        """Render deployment form"""
        st.subheader("🚀 Create New Deployment")
        
        with st.form("fortigate_deployment"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Deployment Name", value="FortiGate-Production")
                cloud_provider = st.selectbox("Cloud Provider", ["azure", "gcp"])
                template = st.selectbox("Template", list(self.get_deployment_templates().keys()))
            
            with col2:
                region = st.selectbox("Region", 
                    ["eastus", "westus2", "centralus"] if cloud_provider == "azure" 
                    else ["us-central1", "us-east1", "us-west1"])
                
                instance_type = st.selectbox("Instance Type",
                    ["Standard_D2s_v3", "Standard_D4s_v3", "Standard_D8s_v3"] if cloud_provider == "azure"
                    else ["n1-standard-2", "n1-standard-4", "n1-standard-8"])
            
            # Advanced options
            with st.expander("Advanced Configuration"):
                admin_username = st.text_input("Admin Username", value="azureuser")
                admin_password = st.text_input("Admin Password", type="password", value="P@ssw0rd123!")
                
                col3, col4 = st.columns(2)
                with col3:
                    external_subnet = st.text_input("External Subnet", value="10.0.1.0/24")
                with col4:
                    internal_subnet = st.text_input("Internal Subnet", value="10.0.2.0/24")
            
            submitted = st.form_submit_button("Create Deployment", type="primary")
            
            if submitted:
                config = {
                    "name": name,
                    "cloud_provider": cloud_provider,
                    "region": region,
                    "instance_type": instance_type,
                    "template": template,
                    "admin_username": admin_username,
                    "admin_password": admin_password,
                    "external_subnet": external_subnet,
                    "internal_subnet": internal_subnet
                }
                
                deployment_id = self.create_deployment(config)
                st.success(f"✅ Deployment created: {deployment_id}")
                
                # Show cost estimate
                cost_estimate = self.get_deployment_cost_estimate(deployment_id)
                st.info(f"💰 Estimated monthly cost: ${cost_estimate['monthly_estimate']:.2f}")
                
                st.rerun()
    
    def _render_management(self):
        """Render management tab"""
        st.subheader("⚙️ Manage Deployments")
        
        deployments = self.list_deployments()
        if not deployments:
            st.info("No deployments to manage.")
            return
        
        for deployment in deployments:
            with st.expander(f"{deployment['name']} ({deployment['id']})"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**Status:** {deployment['status']}")
                    st.write(f"**Cloud:** {deployment['cloud_provider']}")
                    st.write(f"**Region:** {deployment['region']}")
                    st.write(f"**Created:** {deployment['created_at']}")
                
                with col2:
                    if st.button("View Config", key=f"config_{deployment['id']}"):
                        terraform_config = self.generate_terraform_config(deployment['id'])
                        st.code(terraform_config, language="hcl")
                
                with col3:
                    if st.button("Delete", key=f"delete_{deployment['id']}", type="secondary"):
                        if self.delete_deployment(deployment['id']):
                            st.success("Deployment deleted")
                            st.rerun()
    
    def _render_templates(self):
        """Render templates tab"""
        st.subheader("📋 Deployment Templates")
        
        templates = self.get_deployment_templates()
        
        for template_id, template_info in templates.items():
            with st.expander(f"{template_info['name']}"):
                st.write(f"**Description:** {template_info['description']}")
                st.write(f"**Resources:** {', '.join(template_info['resources'])}")
                st.write(f"**Use Cases:** {', '.join(template_info['use_cases'])}")
