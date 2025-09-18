"""
Cloud Credential Manager for Multi-Cloud MCP Server
Secure storage and management of Azure and GCP credentials
"""

import os
import json
import base64
from typing import Dict, Optional, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import streamlit as st
import logging

logger = logging.getLogger(__name__)

class CloudCredentialManager:
    """Secure credential management for cloud providers"""
    
    def __init__(self, master_key: Optional[str] = None):
        self.master_key = master_key or self._get_master_key()
        self.cipher_suite = self._initialize_cipher()
        self.credentials = {}
        
    def _get_master_key(self) -> str:
        """Get or generate master encryption key"""
        # Try to get from Streamlit secrets first
        if hasattr(st, 'secrets') and 'cloud_mcp' in st.secrets:
            return st.secrets.cloud_mcp.get('master_key', self._generate_master_key())
        
        # Fallback to environment variable
        return os.getenv('CLOUD_MCP_MASTER_KEY', self._generate_master_key())
    
    def _generate_master_key(self) -> str:
        """Generate a new master key"""
        key = Fernet.generate_key()
        return base64.urlsafe_b64encode(key).decode()
    
    def _initialize_cipher(self) -> Fernet:
        """Initialize encryption cipher"""
        try:
            key_bytes = base64.urlsafe_b64decode(self.master_key.encode())
            return Fernet(key_bytes)
        except Exception:
            # Generate new key if invalid
            new_key = self._generate_master_key()
            key_bytes = base64.urlsafe_b64decode(new_key.encode())
            return Fernet(key_bytes)
    
    def store_azure_credentials(self, 
                               tenant_id: str,
                               client_id: str, 
                               client_secret: str,
                               subscription_id: str) -> bool:
        """Store Azure Service Principal credentials"""
        try:
            azure_creds = {
                'tenant_id': tenant_id,
                'client_id': client_id,
                'client_secret': client_secret,
                'subscription_id': subscription_id,
                'provider': 'azure'
            }
            
            encrypted_creds = self._encrypt_credentials(azure_creds)
            self.credentials['azure'] = encrypted_creds
            
            logger.info("Azure credentials stored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store Azure credentials: {e}")
            return False
    
    def store_gcp_credentials(self,
                             project_id: str,
                             service_account_key: Dict[str, Any]) -> bool:
        """Store Google Cloud Service Account credentials"""
        try:
            gcp_creds = {
                'project_id': project_id,
                'service_account_key': service_account_key,
                'provider': 'gcp'
            }
            
            encrypted_creds = self._encrypt_credentials(gcp_creds)
            self.credentials['gcp'] = encrypted_creds
            
            logger.info("GCP credentials stored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store GCP credentials: {e}")
            return False
    
    def store_fortigate_credentials(self,
                                   api_key: str,
                                   base_url: str,
                                   username: Optional[str] = None,
                                   password: Optional[str] = None) -> bool:
        """Store FortiGate API credentials"""
        try:
            fortigate_creds = {
                'api_key': api_key,
                'base_url': base_url,
                'username': username,
                'password': password,
                'provider': 'fortigate'
            }
            
            encrypted_creds = self._encrypt_credentials(fortigate_creds)
            self.credentials['fortigate'] = encrypted_creds
            
            logger.info("FortiGate credentials stored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store FortiGate credentials: {e}")
            return False
    
    def get_azure_credentials(self) -> Optional[Dict[str, str]]:
        """Retrieve and decrypt Azure credentials"""
        try:
            if 'azure' not in self.credentials:
                return None
            
            encrypted_creds = self.credentials['azure']
            return self._decrypt_credentials(encrypted_creds)
            
        except Exception as e:
            logger.error(f"Failed to retrieve Azure credentials: {e}")
            return None
    
    def get_gcp_credentials(self) -> Optional[Dict[str, Any]]:
        """Retrieve and decrypt GCP credentials"""
        try:
            if 'gcp' not in self.credentials:
                return None
            
            encrypted_creds = self.credentials['gcp']
            return self._decrypt_credentials(encrypted_creds)
            
        except Exception as e:
            logger.error(f"Failed to retrieve GCP credentials: {e}")
            return None
    
    def get_fortigate_credentials(self) -> Optional[Dict[str, str]]:
        """Retrieve and decrypt FortiGate credentials"""
        try:
            if 'fortigate' not in self.credentials:
                return None
            
            encrypted_creds = self.credentials['fortigate']
            return self._decrypt_credentials(encrypted_creds)
            
        except Exception as e:
            logger.error(f"Failed to retrieve FortiGate credentials: {e}")
            return None
    
    def _encrypt_credentials(self, credentials: Dict[str, Any]) -> str:
        """Encrypt credentials dictionary"""
        try:
            creds_json = json.dumps(credentials)
            encrypted_data = self.cipher_suite.encrypt(creds_json.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def _decrypt_credentials(self, encrypted_credentials: str) -> Dict[str, Any]:
        """Decrypt credentials string"""
        try:
            encrypted_data = base64.urlsafe_b64decode(encrypted_credentials.encode())
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def validate_azure_credentials(self) -> bool:
        """Validate Azure credentials by testing connection"""
        try:
            creds = self.get_azure_credentials()
            if not creds:
                return False
            
            # Test Azure connection
            from azure.identity import ClientSecretCredential
            from azure.mgmt.resource import ResourceManagementClient
            
            credential = ClientSecretCredential(
                tenant_id=creds['tenant_id'],
                client_id=creds['client_id'],
                client_secret=creds['client_secret']
            )
            
            resource_client = ResourceManagementClient(
                credential, creds['subscription_id']
            )
            
            # Try to list resource groups (minimal permission test)
            list(resource_client.resource_groups.list())
            return True
            
        except Exception as e:
            logger.error(f"Azure credential validation failed: {e}")
            return False
    
    def validate_gcp_credentials(self) -> bool:
        """Validate GCP credentials by testing connection"""
        try:
            creds = self.get_gcp_credentials()
            if not creds:
                return False
            
            # Test GCP connection
            from google.oauth2 import service_account
            from google.cloud import compute_v1
            
            credentials = service_account.Credentials.from_service_account_info(
                creds['service_account_key']
            )
            
            compute_client = compute_v1.InstancesClient(credentials=credentials)
            
            # Try to list instances (minimal permission test)
            request = compute_v1.AggregatedListInstancesRequest(
                project=creds['project_id']
            )
            compute_client.aggregated_list(request=request)
            return True
            
        except Exception as e:
            logger.error(f"GCP credential validation failed: {e}")
            return False
    
    def validate_fortigate_credentials(self) -> bool:
        """Validate FortiGate credentials by testing API connection"""
        try:
            creds = self.get_fortigate_credentials()
            if not creds:
                return False
            
            import requests
            
            # Test FortiGate API connection
            headers = {
                'Authorization': f"Bearer {creds['api_key']}",
                'Content-Type': 'application/json'
            }
            
            response = requests.get(
                f"{creds['base_url']}/api/v2/cmdb/system/status",
                headers=headers,
                timeout=10,
                verify=False  # FortiGate often uses self-signed certs
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"FortiGate credential validation failed: {e}")
            return False
    
    def get_credential_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all stored credentials"""
        status = {}
        
        for provider in ['azure', 'gcp', 'fortigate']:
            status[provider] = {
                'stored': provider in self.credentials,
                'valid': False,
                'last_validated': None
            }
            
            if status[provider]['stored']:
                if provider == 'azure':
                    status[provider]['valid'] = self.validate_azure_credentials()
                elif provider == 'gcp':
                    status[provider]['valid'] = self.validate_gcp_credentials()
                elif provider == 'fortigate':
                    status[provider]['valid'] = self.validate_fortigate_credentials()
        
        return status
    
    def clear_credentials(self, provider: Optional[str] = None):
        """Clear stored credentials"""
        if provider:
            if provider in self.credentials:
                del self.credentials[provider]
                logger.info(f"Cleared {provider} credentials")
        else:
            self.credentials.clear()
            logger.info("Cleared all credentials")

def create_credential_manager(master_key: Optional[str] = None) -> CloudCredentialManager:
    """Factory function to create credential manager"""
    return CloudCredentialManager(master_key=master_key)
