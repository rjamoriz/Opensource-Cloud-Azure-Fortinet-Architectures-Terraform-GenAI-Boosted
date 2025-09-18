"""
Base Cloud API Interface
Abstract interface for cloud provider API integrations
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class VMSize(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"

class StorageType(Enum):
    STANDARD = "standard"
    PREMIUM = "premium"
    ULTRA = "ultra"

@dataclass
class VMSpecification:
    """VM specification data structure"""
    name: str
    cpu_cores: int
    memory_gb: float
    storage_gb: int
    storage_type: StorageType
    network_performance: str
    price_per_hour: float
    availability_zones: List[str]
    suitable_workloads: List[str]

@dataclass
class NetworkConfiguration:
    """Network configuration data structure"""
    vpc_id: str
    subnet_id: str
    security_group_ids: List[str]
    public_ip: bool
    availability_zone: str

@dataclass
class VMDeploymentRequest:
    """VM deployment request structure"""
    vm_spec: VMSpecification
    network_config: NetworkConfiguration
    os_image: str
    key_pair: str
    user_data: Optional[str] = None
    tags: Optional[Dict[str, str]] = None

@dataclass
class VMDeploymentResult:
    """VM deployment result structure"""
    vm_id: str
    status: str
    public_ip: Optional[str]
    private_ip: str
    deployment_time: str
    cost_estimate: Dict[str, float]

class BaseCloudAPI(ABC):
    """Abstract base class for cloud API integrations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.region = config.get('region', 'us-east-1')
        self.credentials = config.get('credentials', {})
        
    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with cloud provider"""
        pass
    
    @abstractmethod
    async def list_vm_sizes(self, region: Optional[str] = None) -> List[VMSpecification]:
        """List available VM sizes for the region"""
        pass
    
    @abstractmethod
    async def get_vm_recommendations(
        self, 
        workload_type: str,
        performance_requirements: Dict[str, Any],
        budget_constraints: Optional[Dict[str, float]] = None
    ) -> List[VMSpecification]:
        """Get VM recommendations based on requirements"""
        pass
    
    @abstractmethod
    async def get_pricing_info(
        self, 
        vm_spec: VMSpecification,
        region: Optional[str] = None
    ) -> Dict[str, float]:
        """Get detailed pricing information for VM specification"""
        pass
    
    @abstractmethod
    async def create_vm(self, request: VMDeploymentRequest) -> VMDeploymentResult:
        """Create VM instance"""
        pass
    
    @abstractmethod
    async def list_regions(self) -> List[Dict[str, str]]:
        """List available regions"""
        pass
    
    @abstractmethod
    async def list_availability_zones(self, region: str) -> List[str]:
        """List availability zones for region"""
        pass
    
    @abstractmethod
    async def get_network_options(self, region: str) -> Dict[str, Any]:
        """Get networking options for region"""
        pass
    
    @abstractmethod
    async def validate_configuration(self, request: VMDeploymentRequest) -> Dict[str, Any]:
        """Validate VM deployment configuration"""
        pass
    
    @abstractmethod
    async def estimate_costs(
        self, 
        vm_spec: VMSpecification,
        usage_hours: int = 730  # Default to monthly
    ) -> Dict[str, float]:
        """Estimate costs for VM usage"""
        pass

class CloudProviderFactory:
    """Factory for creating cloud provider API instances"""
    
    @staticmethod
    def get_provider_config_template(provider: str) -> Dict[str, Any]:
        """Get configuration template for cloud provider"""
        templates = {
            'azure': {
                'subscription_id': '',
                'client_id': '',
                'client_secret': '',
                'tenant_id': '',
                'region': 'eastus'
            },
            'gcp': {
                'project_id': '',
                'credentials_path': '',
                'region': 'us-central1'
            },
            'aws': {
                'access_key_id': '',
                'secret_access_key': '',
                'region': 'us-east-1'
            }
        }
        return templates.get(provider, {})
    
    @staticmethod
    def validate_provider_config(provider: str, config: Dict[str, Any]) -> bool:
        """Validate cloud provider configuration"""
        required_fields = {
            'azure': ['subscription_id', 'client_id', 'client_secret', 'tenant_id'],
            'gcp': ['project_id', 'credentials_path'],
            'aws': ['access_key_id', 'secret_access_key']
        }
        
        required = required_fields.get(provider, [])
        return all(field in config and config[field] for field in required)
