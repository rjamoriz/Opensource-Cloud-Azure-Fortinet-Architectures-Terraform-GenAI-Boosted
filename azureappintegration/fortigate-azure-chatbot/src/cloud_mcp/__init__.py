"""
Multi-Cloud FortiGate MCP Server
Model Context Protocol server for Azure and Google Cloud FortiGate management
"""

from .credential_manager import CloudCredentialManager
from .azure_connector import AzureConnector
from .gcp_connector import GCPConnector
from .fortigate_manager import FortiGateManager
from .security_scanner import SecurityScanner
from .resource_monitor import ResourceMonitor
from .deployment_engine import DeploymentEngine
from .analytics_engine import AnalyticsEngine

__version__ = "1.0.0"
__author__ = "FortiGate Azure Chatbot Team"

# Check for required dependencies
try:
    import azure.identity
    import azure.mgmt.resource
    import azure.mgmt.compute
    import azure.mgmt.network
    import google.cloud.compute_v1
    import google.cloud.asset
    import google.auth
    import requests
    import cryptography
    CLOUD_MCP_AVAILABLE = True
except ImportError:
    CLOUD_MCP_AVAILABLE = False

__all__ = [
    "CloudCredentialManager",
    "AzureConnector", 
    "GCPConnector",
    "FortiGateManager",
    "SecurityScanner",
    "ResourceMonitor",
    "DeploymentEngine",
    "AnalyticsEngine",
    "CLOUD_MCP_AVAILABLE"
]
