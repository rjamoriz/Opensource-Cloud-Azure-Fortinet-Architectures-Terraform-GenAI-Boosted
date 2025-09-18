"""
FortiGate Azure Fine-Tuning Data Preparation Module
Prepares training data for OpenAI fine-tuning with Azure and FortiGate expertise
"""

import json
import os
from typing import List, Dict, Any
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FortiGateDataPreparator:
    """Prepare training data for FortiGate Azure fine-tuning"""
    
    def __init__(self, output_dir: str = "training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.training_data = []
        
    def add_conversation(self, user_input: str, assistant_response: str, system_prompt: str = None):
        """Add a conversation pair to training data"""
        if system_prompt is None:
            system_prompt = "You are an expert FortiGate Azure deployment assistant with deep knowledge of Terraform, Azure networking, and FortiGate security appliances."
        
        conversation = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_response}
            ]
        }
        self.training_data.append(conversation)
        
    def load_fortigate_knowledge_base(self):
        """Load FortiGate and Azure specific knowledge"""
        
        # Azure FortiGate deployment scenarios
        azure_scenarios = [
            {
                "user": "How do I deploy a single FortiGate VM in Azure?",
                "assistant": """To deploy a single FortiGate VM in Azure:

1. **Resource Group**: Create a dedicated resource group
2. **Virtual Network**: Set up VNet with subnets:
   - External subnet (public-facing)
   - Internal subnet (protected resources)
   - Management subnet (FortiGate management)

3. **Network Security Groups**: Configure NSGs for each subnet
4. **Public IP**: Assign static public IP for external interface
5. **FortiGate VM**: Deploy from Azure Marketplace
6. **Storage Account**: For boot diagnostics and logging

**Terraform Example**:
```hcl
resource "azurerm_virtual_machine" "fortigate" {
  name                = "fortigate-vm"
  location            = var.location
  resource_group_name = var.resource_group_name
  vm_size            = "Standard_F2s"
  
  storage_image_reference {
    publisher = "fortinet"
    offer     = "fortinet_fortigate-vm_v5"
    sku       = "fortinet_fg-vm"
    version   = "latest"
  }
}
```

**Key Considerations**:
- Choose appropriate VM size based on throughput requirements
- Configure proper routing for traffic flow
- Set up monitoring and logging
- Implement backup and disaster recovery"""
            },
            {
                "user": "What's the difference between HA active-passive and active-active?",
                "assistant": """FortiGate HA deployment modes in Azure:

**Active-Passive HA**:
- One FortiGate processes traffic (active)
- Second FortiGate stands by (passive)
- Automatic failover when active fails
- Shared virtual IP addresses
- Session synchronization between units
- Best for: Stateful applications, session persistence

**Active-Active HA**:
- Both FortiGates process traffic simultaneously
- Load balancing across both units
- Higher throughput and utilization
- Independent processing of connections
- Best for: High-throughput environments, stateless applications

**Azure Implementation**:
- Use Azure Load Balancer for traffic distribution
- Configure health probes for failover detection
- Set up route tables for proper traffic flow
- Implement Azure availability sets/zones for redundancy

**Terraform Configuration**:
```hcl
resource "azurerm_lb" "fortigate_lb" {
  name                = "fortigate-lb"
  location            = var.location
  resource_group_name = var.resource_group_name
  sku                = "Standard"
  
  frontend_ip_configuration {
    name                 = "external"
    public_ip_address_id = azurerm_public_ip.lb_pip.id
  }
}
```"""
            },
            {
                "user": "How do I configure FortiGate with Azure Application Gateway?",
                "assistant": """Integrating FortiGate with Azure Application Gateway:

**Architecture Overview**:
- Application Gateway: Layer 7 load balancer (frontend)
- FortiGate: Security inspection and filtering
- Backend servers: Protected applications

**Configuration Steps**:

1. **Application Gateway Setup**:
   - Deploy in dedicated subnet
   - Configure backend pool pointing to FortiGate internal IPs
   - Set up health probes for FortiGate availability

2. **FortiGate Configuration**:
   - Configure virtual IPs for backend services
   - Set up firewall policies for traffic inspection
   - Enable SSL inspection if required
   - Configure load balancing to backend servers

3. **Network Flow**:
   ```
   Internet → App Gateway → FortiGate → Backend Servers
   ```

**Terraform Implementation**:
```hcl
resource "azurerm_application_gateway" "main" {
  name                = "fortigate-appgw"
  resource_group_name = var.resource_group_name
  location            = var.location

  sku {
    name     = "WAF_v2"
    tier     = "WAF_v2"
    capacity = 2
  }

  backend_address_pool {
    name = "fortigate-backend"
    ip_addresses = [
      azurerm_network_interface.fortigate_internal.private_ip_address
    ]
  }
}
```

**Benefits**:
- SSL termination at Application Gateway
- Web Application Firewall (WAF) protection
- Advanced security inspection by FortiGate
- High availability and scalability"""
            }
        ]
        
        # Add Azure scenarios to training data
        for scenario in azure_scenarios:
            self.add_conversation(scenario["user"], scenario["assistant"])
            
        logger.info(f"Added {len(azure_scenarios)} Azure FortiGate scenarios")
        
    def load_terraform_knowledge(self):
        """Load Terraform-specific FortiGate knowledge"""
        
        terraform_scenarios = [
            {
                "user": "What are the essential Terraform variables for FortiGate deployment?",
                "assistant": """Essential Terraform variables for FortiGate Azure deployment:

**Core Variables**:
```hcl
variable "resource_group_name" {
  description = "Name of the Azure resource group"
  type        = string
}

variable "location" {
  description = "Azure region for deployment"
  type        = string
  default     = "East US"
}

variable "fortigate_vm_size" {
  description = "Size of the FortiGate VM"
  type        = string
  default     = "Standard_F2s_v2"
  validation {
    condition = contains([
      "Standard_F2s_v2", "Standard_F4s_v2", "Standard_F8s_v2"
    ], var.fortigate_vm_size)
    error_message = "VM size must be a supported FortiGate size."
  }
}

variable "admin_username" {
  description = "Administrator username for FortiGate"
  type        = string
  default     = "azureuser"
}

variable "admin_password" {
  description = "Administrator password for FortiGate"
  type        = string
  sensitive   = true
}
```

**Network Variables**:
```hcl
variable "vnet_address_space" {
  description = "Address space for the virtual network"
  type        = list(string)
  default     = ["10.0.0.0/16"]
}

variable "external_subnet_prefix" {
  description = "Address prefix for external subnet"
  type        = string
  default     = "10.0.1.0/24"
}

variable "internal_subnet_prefix" {
  description = "Address prefix for internal subnet"
  type        = string
  default     = "10.0.2.0/24"
}
```

**Security Variables**:
```hcl
variable "allowed_source_ips" {
  description = "List of allowed source IP addresses"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "enable_accelerated_networking" {
  description = "Enable accelerated networking"
  type        = bool
  default     = true
}
```"""
            }
        ]
        
        for scenario in terraform_scenarios:
            self.add_conversation(scenario["user"], scenario["assistant"])
            
        logger.info(f"Added {len(terraform_scenarios)} Terraform scenarios")
        
    def load_troubleshooting_knowledge(self):
        """Load troubleshooting and common issues"""
        
        troubleshooting_scenarios = [
            {
                "user": "FortiGate VM is not accessible after deployment in Azure",
                "assistant": """Troubleshooting FortiGate VM accessibility issues:

**Common Causes & Solutions**:

1. **Network Security Group (NSG) Rules**:
   - Check if management ports are open (443, 22)
   - Verify source IP restrictions
   - Ensure proper priority ordering

2. **Public IP Configuration**:
   - Confirm static public IP is assigned
   - Check if IP is associated with correct NIC
   - Verify DNS name resolution

3. **FortiGate Boot Process**:
   - Check boot diagnostics in Azure portal
   - Verify VM size meets minimum requirements
   - Ensure proper storage account configuration

4. **Firewall Policies**:
   - Default FortiGate blocks all traffic
   - Create explicit allow policies for management
   - Check interface assignments

**Diagnostic Commands**:
```bash
# Check VM status
az vm show --resource-group myRG --name fortigate-vm --show-details

# Verify NSG rules
az network nsg rule list --resource-group myRG --nsg-name fortigate-nsg

# Check public IP
az network public-ip show --resource-group myRG --name fortigate-pip
```

**Quick Fix Steps**:
1. Verify NSG allows HTTPS (443) from your IP
2. Check public IP is static and assigned
3. Try accessing via serial console if available
4. Review Azure activity logs for deployment errors"""
            }
        ]
        
        for scenario in troubleshooting_scenarios:
            self.add_conversation(scenario["user"], scenario["assistant"])
            
        logger.info(f"Added {len(troubleshooting_scenarios)} troubleshooting scenarios")
    
    def save_training_data(self, filename: str = "fortigate_training_data.jsonl"):
        """Save training data to JSONL format for OpenAI fine-tuning"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for conversation in self.training_data:
                f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(self.training_data)} training examples to {output_path}")
        return str(output_path)
    
    def validate_training_data(self):
        """Validate training data format for OpenAI fine-tuning"""
        errors = []
        
        for i, conversation in enumerate(self.training_data):
            # Check required structure
            if "messages" not in conversation:
                errors.append(f"Conversation {i}: Missing 'messages' key")
                continue
                
            messages = conversation["messages"]
            if len(messages) < 2:
                errors.append(f"Conversation {i}: Need at least 2 messages")
                continue
            
            # Check message format
            for j, message in enumerate(messages):
                if "role" not in message or "content" not in message:
                    errors.append(f"Conversation {i}, Message {j}: Missing role or content")
                
                if message["role"] not in ["system", "user", "assistant"]:
                    errors.append(f"Conversation {i}, Message {j}: Invalid role")
        
        if errors:
            logger.error(f"Validation errors found: {len(errors)}")
            for error in errors[:10]:  # Show first 10 errors
                logger.error(error)
            return False
        else:
            logger.info("Training data validation passed!")
            return True
    
    def generate_full_dataset(self):
        """Generate complete training dataset"""
        logger.info("Generating FortiGate Azure fine-tuning dataset...")
        
        self.load_fortigate_knowledge_base()
        self.load_terraform_knowledge()
        self.load_troubleshooting_knowledge()
        
        # Validate data
        if self.validate_training_data():
            # Save training data
            training_file = self.save_training_data()
            logger.info(f"Training dataset ready: {training_file}")
            return training_file
        else:
            logger.error("Training data validation failed!")
            return None

if __name__ == "__main__":
    # Generate training dataset
    preparator = FortiGateDataPreparator()
    training_file = preparator.generate_full_dataset()
    
    if training_file:
        print(f"✅ Training data prepared: {training_file}")
        print(f"📊 Total examples: {len(preparator.training_data)}")
        print("🚀 Ready for OpenAI fine-tuning!")
    else:
        print("❌ Failed to prepare training data")
