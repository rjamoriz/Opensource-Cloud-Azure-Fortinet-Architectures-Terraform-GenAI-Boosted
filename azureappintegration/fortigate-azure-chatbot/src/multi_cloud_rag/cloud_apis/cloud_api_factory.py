"""
Cloud API Factory
Creates appropriate cloud API instances based on provider
"""

from typing import Dict, Any, Optional
import logging
from .base_cloud_api import BaseCloudAPI, CloudProviderFactory
from .azure_api import AzureCloudAPI
from .gcp_api import GCPCloudAPI

logger = logging.getLogger(__name__)

class CloudAPIFactory:
    """Factory for creating cloud API instances"""
    
    SUPPORTED_PROVIDERS = {
        'azure': AzureCloudAPI,
        'gcp': GCPCloudAPI,
        # 'aws': AWSCloudAPI  # Can be added later
    }
    
    @classmethod
    def create_cloud_api(
        self, 
        provider: str, 
        config: Dict[str, Any]
    ) -> Optional[BaseCloudAPI]:
        """Create cloud API instance based on provider"""
        
        if provider not in self.SUPPORTED_PROVIDERS:
            logger.error(f"Unsupported cloud provider: {provider}")
            logger.info(f"Supported providers: {list(self.SUPPORTED_PROVIDERS.keys())}")
            return None
        
        # Validate configuration
        if not CloudProviderFactory.validate_provider_config(provider, config):
            logger.error(f"Invalid configuration for {provider}")
            return None
        
        try:
            api_class = self.SUPPORTED_PROVIDERS[provider]
            return api_class(config)
        except Exception as e:
            logger.error(f"Failed to create {provider} cloud API: {e}")
            return None
    
    @classmethod
    def get_default_config(self, provider: str) -> Dict[str, Any]:
        """Get default configuration template for provider"""
        return CloudProviderFactory.get_provider_config_template(provider)
    
    @classmethod
    def validate_config(self, provider: str, config: Dict[str, Any]) -> bool:
        """Validate configuration for provider"""
        return CloudProviderFactory.validate_provider_config(provider, config)
