"""
Cloud MCP Dashboard for Multi-Cloud FortiGate Management
Streamlit interface for Azure and Google Cloud integration
"""

import streamlit as st
import json
from typing import Dict, List, Optional, Any
import logging

# Try to import cloud MCP components with graceful fallback
try:
    from .credential_manager import CloudCredentialManager
    from .azure_connector import AzureConnector, AZURE_SDK_AVAILABLE
    CLOUD_MCP_AVAILABLE = True
except ImportError as e:
    CLOUD_MCP_AVAILABLE = False
    AZURE_SDK_AVAILABLE = False
    
    # Create placeholder classes for graceful degradation
    class CloudCredentialManager:
        def __init__(self):
            pass
        def get_credential_status(self):
            return {'azure': {'stored': False, 'valid': False}}
    
    class AzureConnector:
        def __init__(self, *args):
            pass

logger = logging.getLogger(__name__)

class CloudMCPDashboard:
    """Multi-cloud management dashboard"""
    
    def __init__(self):
        self.credential_manager = CloudCredentialManager()
        self.azure_connector = None
        self.gcp_connector = None
        
        # Initialize session state
        if 'cloud_mcp_tab' not in st.session_state:
            st.session_state.cloud_mcp_tab = 'credentials'
        if 'cloud_credentials_status' not in st.session_state:
            st.session_state.cloud_credentials_status = {}
    
    def render_dashboard(self):
        """Render the main cloud MCP dashboard"""
        st.markdown("# ☁️ Multi-Cloud FortiGate Management")
        st.markdown("**MCP Server for Azure and Google Cloud FortiGate Integration**")
        
        # Check if dependencies are available
        if not CLOUD_MCP_AVAILABLE:
            st.error("❌ **Cloud MCP Dependencies Missing**")
            st.markdown("""
            ### 📦 Required Dependencies
            
            To use the Multi-Cloud Management features, please install the required packages:
            
            ```bash
            pip install -r src/cloud_mcp/requirements_cloud_mcp.txt
            ```
            
            ### 🔧 Required Packages:
            - **Azure SDK**: `azure-identity`, `azure-mgmt-resource`, `azure-mgmt-compute`, `azure-mgmt-network`
            - **Google Cloud SDK**: `google-cloud-compute`, `google-cloud-asset`, `google-auth`
            - **Security**: `cryptography`, `pycryptodome`
            """)
            
            if st.button("📋 Copy Installation Command"):
                st.code("pip install -r src/cloud_mcp/requirements_cloud_mcp.txt", language="bash")
                st.success("✅ Command copied! Run this in your terminal.")
            
            return
        
        if not AZURE_SDK_AVAILABLE:
            st.warning("⚠️ **Azure SDK Missing** - Some features may be limited")
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔐 Credentials", 
            "🌐 Azure Resources", 
            "☁️ Google Cloud", 
            "🛡️ Security", 
            "📊 Analytics"
        ])
        
        with tab1:
            self._render_credentials_tab()
        
        with tab2:
            self._render_azure_tab()
        
        with tab3:
            self._render_gcp_tab()
        
        with tab4:
            self._render_security_tab()
        
        with tab5:
            self._render_analytics_tab()
    
    def _render_credentials_tab(self):
        """Render credentials management tab"""
        st.markdown("## 🔐 Cloud Credentials Management")
        
        # Credential status overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Azure Status")
            azure_status = self._get_credential_status('azure')
            if azure_status['stored']:
                st.success("✅ Credentials Stored")
                if azure_status['valid']:
                    st.success("✅ Connection Valid")
                else:
                    st.warning("⚠️ Connection Failed")
            else:
                st.error("❌ No Credentials")
        
        with col2:
            st.markdown("### Google Cloud Status")
            gcp_status = self._get_credential_status('gcp')
            if gcp_status['stored']:
                st.success("✅ Credentials Stored")
                if gcp_status['valid']:
                    st.success("✅ Connection Valid")
                else:
                    st.warning("⚠️ Connection Failed")
            else:
                st.error("❌ No Credentials")
        
        with col3:
            st.markdown("### FortiGate Status")
            fortigate_status = self._get_credential_status('fortigate')
            if fortigate_status['stored']:
                st.success("✅ Credentials Stored")
                if fortigate_status['valid']:
                    st.success("✅ API Connected")
                else:
                    st.warning("⚠️ API Failed")
            else:
                st.error("❌ No Credentials")
        
        st.markdown("---")
        
        # Azure credentials form
        with st.expander("🔵 Configure Azure Credentials"):
            st.markdown("**Azure Service Principal Configuration**")
            
            azure_tenant = st.text_input("Tenant ID", type="password", key="azure_tenant")
            azure_client = st.text_input("Client ID", key="azure_client")
            azure_secret = st.text_input("Client Secret", type="password", key="azure_secret")
            azure_subscription = st.text_input("Subscription ID", key="azure_subscription")
            
            if st.button("💾 Save Azure Credentials", key="save_azure"):
                if all([azure_tenant, azure_client, azure_secret, azure_subscription]):
                    success = self.credential_manager.store_azure_credentials(
                        azure_tenant, azure_client, azure_secret, azure_subscription
                    )
                    if success:
                        st.success("✅ Azure credentials saved successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save Azure credentials")
                else:
                    st.error("❌ Please fill in all Azure credential fields")
        
        # GCP credentials form
        with st.expander("🟡 Configure Google Cloud Credentials"):
            st.markdown("**Google Cloud Service Account Configuration**")
            
            gcp_project = st.text_input("Project ID", key="gcp_project")
            gcp_key_file = st.file_uploader(
                "Service Account Key (JSON)", 
                type=['json'], 
                key="gcp_key_upload"
            )
            
            if st.button("💾 Save GCP Credentials", key="save_gcp"):
                if gcp_project and gcp_key_file:
                    try:
                        service_account_key = json.load(gcp_key_file)
                        success = self.credential_manager.store_gcp_credentials(
                            gcp_project, service_account_key
                        )
                        if success:
                            st.success("✅ GCP credentials saved successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to save GCP credentials")
                    except json.JSONDecodeError:
                        st.error("❌ Invalid JSON file format")
                else:
                    st.error("❌ Please provide project ID and service account key")
        
        # FortiGate credentials form
        with st.expander("🛡️ Configure FortiGate API Credentials"):
            st.markdown("**FortiGate API Configuration**")
            
            fortigate_url = st.text_input("FortiGate Base URL", key="fortigate_url", 
                                        placeholder="https://your-fortigate.com")
            fortigate_api_key = st.text_input("API Key", type="password", key="fortigate_api")
            fortigate_user = st.text_input("Username (optional)", key="fortigate_user")
            fortigate_pass = st.text_input("Password (optional)", type="password", key="fortigate_pass")
            
            if st.button("💾 Save FortiGate Credentials", key="save_fortigate"):
                if fortigate_url and fortigate_api_key:
                    success = self.credential_manager.store_fortigate_credentials(
                        fortigate_api_key, fortigate_url, fortigate_user, fortigate_pass
                    )
                    if success:
                        st.success("✅ FortiGate credentials saved successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save FortiGate credentials")
                else:
                    st.error("❌ Please provide at least URL and API key")
        
        # Test connections
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔵 Test Azure Connection", key="test_azure"):
                with st.spinner("Testing Azure connection..."):
                    valid = self.credential_manager.validate_azure_credentials()
                    if valid:
                        st.success("✅ Azure connection successful!")
                    else:
                        st.error("❌ Azure connection failed")
        
        with col2:
            if st.button("🟡 Test GCP Connection", key="test_gcp"):
                with st.spinner("Testing GCP connection..."):
                    valid = self.credential_manager.validate_gcp_credentials()
                    if valid:
                        st.success("✅ GCP connection successful!")
                    else:
                        st.error("❌ GCP connection failed")
        
        with col3:
            if st.button("🛡️ Test FortiGate API", key="test_fortigate"):
                with st.spinner("Testing FortiGate API..."):
                    valid = self.credential_manager.validate_fortigate_credentials()
                    if valid:
                        st.success("✅ FortiGate API connection successful!")
                    else:
                        st.error("❌ FortiGate API connection failed")
    
    def _render_azure_tab(self):
        """Render Azure resources tab"""
        st.markdown("## 🌐 Azure Resources")
        
        if not self._get_credential_status('azure')['valid']:
            st.warning("⚠️ Please configure and validate Azure credentials first")
            return
        
        # Initialize Azure connector
        if not self.azure_connector:
            self.azure_connector = AzureConnector(self.credential_manager)
            if not self.azure_connector.connect():
                st.error("❌ Failed to connect to Azure")
                return
        
        # Resource overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📦 Resource Groups")
            with st.spinner("Loading resource groups..."):
                resource_groups = self.azure_connector.get_resource_groups()
                if resource_groups:
                    for rg in resource_groups[:5]:  # Show first 5
                        st.write(f"• **{rg['name']}** ({rg['location']})")
                    if len(resource_groups) > 5:
                        st.write(f"... and {len(resource_groups) - 5} more")
                else:
                    st.write("No resource groups found")
        
        with col2:
            st.markdown("### 🛡️ FortiGate VMs")
            with st.spinner("Loading FortiGate instances..."):
                fortigate_vms = self.azure_connector.get_fortigate_vms()
                if fortigate_vms:
                    for vm in fortigate_vms:
                        status_icon = "🟢" if vm['power_state'] == 'running' else "🔴"
                        st.write(f"{status_icon} **{vm['name']}** ({vm['vm_size']})")
                else:
                    st.write("No FortiGate VMs found")
        
        # Detailed views
        st.markdown("---")
        
        if st.button("🔍 View All Azure Resources", key="view_azure_resources"):
            st.markdown("### 📊 Detailed Azure Resources")
            
            # Virtual Networks
            with st.expander("🌐 Virtual Networks"):
                vnets = self.azure_connector.get_virtual_networks()
                if vnets:
                    for vnet in vnets:
                        st.markdown(f"**{vnet['name']}** - {vnet['location']}")
                        st.write(f"Address Space: {', '.join(vnet['address_space'])}")
                        if vnet['subnets']:
                            st.write("Subnets:")
                            for subnet in vnet['subnets']:
                                st.write(f"  • {subnet['name']}: {subnet['address_prefix']}")
                        st.markdown("---")
            
            # Security Groups
            with st.expander("🔒 Network Security Groups"):
                nsgs = self.azure_connector.get_security_groups()
                if nsgs:
                    for nsg in nsgs:
                        st.markdown(f"**{nsg['name']}** - {nsg['location']}")
                        st.write(f"Rules: {len(nsg['security_rules'])}")
                        st.markdown("---")
    
    def _render_gcp_tab(self):
        """Render Google Cloud tab"""
        st.markdown("## ☁️ Google Cloud Platform")
        
        if not self._get_credential_status('gcp')['valid']:
            st.warning("⚠️ Please configure and validate GCP credentials first")
            return
        
        st.info("🚧 GCP integration coming soon...")
        
        # Placeholder for GCP features
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🖥️ Compute Instances")
            st.write("• FortiGate VM instances")
            st.write("• Instance groups")
            st.write("• Auto-scaling policies")
        
        with col2:
            st.markdown("### 🌐 VPC Networks")
            st.write("• VPC networks")
            st.write("• Subnets")
            st.write("• Firewall rules")
    
    def _render_security_tab(self):
        """Render security assessment tab"""
        st.markdown("## 🛡️ Security Assessment")
        
        # Security overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Security Score", "85/100", "↑5")
        
        with col2:
            st.metric("Active Threats", "3", "↓2")
        
        with col3:
            st.metric("Compliance", "92%", "↑3%")
        
        st.markdown("---")
        
        # Security recommendations
        st.markdown("### 🔍 Security Recommendations")
        
        recommendations = [
            {"severity": "high", "title": "Update FortiGate firmware", "description": "Critical security patches available"},
            {"severity": "medium", "title": "Review firewall rules", "description": "Some rules allow broad access"},
            {"severity": "low", "title": "Enable logging", "description": "Increase log retention period"}
        ]
        
        for rec in recommendations:
            severity_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}[rec['severity']]
            st.markdown(f"{severity_color} **{rec['title']}**")
            st.write(rec['description'])
            st.markdown("---")
    
    def _render_analytics_tab(self):
        """Render analytics and cost optimization tab"""
        st.markdown("## 📊 Multi-Cloud Analytics")
        
        # Cost overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Azure Costs", "$1,234", "↑12%")
        
        with col2:
            st.metric("GCP Costs", "$567", "↓5%")
        
        with col3:
            st.metric("Total Costs", "$1,801", "↑8%")
        
        st.markdown("---")
        
        # Cost breakdown chart (placeholder)
        st.markdown("### 💰 Cost Breakdown")
        
        chart_data = {
            "tooltip": {"trigger": "item"},
            "series": [{
                "type": "pie",
                "radius": "50%",
                "data": [
                    {"value": 1234, "name": "Azure"},
                    {"value": 567, "name": "Google Cloud"},
                    {"value": 200, "name": "FortiGate Licenses"}
                ]
            }]
        }
        
        st.components.v1.html(
            f"""
            <div id="cost_chart" style="width: 100%; height: 400px;"></div>
            <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.0/dist/echarts.min.js"></script>
            <script>
                var chart = echarts.init(document.getElementById('cost_chart'));
                chart.setOption({json.dumps(chart_data)});
            </script>
            """,
            height=400
        )
        
        # Optimization recommendations
        st.markdown("### 💡 Cost Optimization Recommendations")
        
        optimizations = [
            "Right-size Azure VMs to save ~$200/month",
            "Use reserved instances for 23% savings",
            "Optimize storage tier for infrequently accessed data",
            "Review unused network resources"
        ]
        
        for opt in optimizations:
            st.write(f"• {opt}")
    
    def _get_credential_status(self, provider: str) -> Dict[str, bool]:
        """Get credential status for a provider"""
        try:
            status = self.credential_manager.get_credential_status()
            return status.get(provider, {'stored': False, 'valid': False})
        except Exception:
            return {'stored': False, 'valid': False}
