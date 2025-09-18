"""
Deployment Engine for FortiGate Multi-Cloud Deployments
"""

import streamlit as st
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeploymentEngine:
    """Deployment engine for FortiGate multi-cloud deployments"""
    
    def __init__(self):
        """Initialize deployment engine"""
        self.active_deployments = {}
        self.deployment_history = []
        
    def create_deployment(self, config: Dict[str, Any]) -> str:
        """
        Create a new deployment
        
        Args:
            config: Deployment configuration
            
        Returns:
            str: Deployment ID
        """
        deployment_id = f"deploy-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        deployment = {
            "id": deployment_id,
            "name": config.get("name", f"Deployment-{deployment_id}"),
            "cloud_provider": config.get("cloud_provider", "azure"),
            "template": config.get("template", "single_vm"),
            "region": config.get("region", "eastus"),
            "status": "initializing",
            "progress": 0,
            "created_at": datetime.now().isoformat(),
            "config": config,
            "steps": self._get_deployment_steps(config),
            "current_step": 0
        }
        
        self.active_deployments[deployment_id] = deployment
        logger.info(f"Created deployment: {deployment_id}")
        
        return deployment_id
    
    def _get_deployment_steps(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get deployment steps based on configuration"""
        cloud_provider = config.get("cloud_provider", "azure")
        template = config.get("template", "single_vm")
        
        if cloud_provider == "azure":
            return self._get_azure_steps(template)
        elif cloud_provider == "gcp":
            return self._get_gcp_steps(template)
        else:
            return []
    
    def _get_azure_steps(self, template: str) -> List[Dict[str, Any]]:
        """Get Azure deployment steps"""
        base_steps = [
            {"name": "Validate Configuration", "duration": 30, "status": "pending"},
            {"name": "Create Resource Group", "duration": 60, "status": "pending"},
            {"name": "Create Virtual Network", "duration": 120, "status": "pending"},
            {"name": "Create Subnets", "duration": 90, "status": "pending"},
            {"name": "Create Network Security Groups", "duration": 60, "status": "pending"},
            {"name": "Create Public IP", "duration": 45, "status": "pending"},
            {"name": "Create Network Interfaces", "duration": 75, "status": "pending"},
            {"name": "Deploy FortiGate VM", "duration": 300, "status": "pending"},
            {"name": "Configure Initial Settings", "duration": 180, "status": "pending"},
            {"name": "Verify Deployment", "duration": 120, "status": "pending"}
        ]
        
        if template == "ha_active_passive":
            base_steps.extend([
                {"name": "Deploy Secondary VM", "duration": 300, "status": "pending"},
                {"name": "Configure HA Cluster", "duration": 240, "status": "pending"},
                {"name": "Setup Load Balancer", "duration": 180, "status": "pending"}
            ])
        
        return base_steps
    
    def _get_gcp_steps(self, template: str) -> List[Dict[str, Any]]:
        """Get GCP deployment steps"""
        base_steps = [
            {"name": "Validate Configuration", "duration": 30, "status": "pending"},
            {"name": "Create VPC Network", "duration": 90, "status": "pending"},
            {"name": "Create Subnets", "duration": 75, "status": "pending"},
            {"name": "Create Firewall Rules", "duration": 60, "status": "pending"},
            {"name": "Create Instance Template", "duration": 120, "status": "pending"},
            {"name": "Deploy Compute Instance", "duration": 240, "status": "pending"},
            {"name": "Configure Startup Script", "duration": 150, "status": "pending"},
            {"name": "Verify Deployment", "duration": 90, "status": "pending"}
        ]
        
        if template == "ha_active_passive":
            base_steps.extend([
                {"name": "Deploy Secondary Instance", "duration": 240, "status": "pending"},
                {"name": "Configure HA Setup", "duration": 180, "status": "pending"},
                {"name": "Setup Load Balancing", "duration": 120, "status": "pending"}
            ])
        
        return base_steps
    
    def start_deployment(self, deployment_id: str) -> bool:
        """Start a deployment"""
        if deployment_id not in self.active_deployments:
            return False
        
        deployment = self.active_deployments[deployment_id]
        deployment["status"] = "running"
        deployment["started_at"] = datetime.now().isoformat()
        
        logger.info(f"Started deployment: {deployment_id}")
        return True
    
    def update_deployment_progress(self, deployment_id: str) -> Dict[str, Any]:
        """Update deployment progress"""
        if deployment_id not in self.active_deployments:
            return {"error": "Deployment not found"}
        
        deployment = self.active_deployments[deployment_id]
        
        if deployment["status"] != "running":
            return deployment
        
        # Simulate progress
        current_step = deployment["current_step"]
        steps = deployment["steps"]
        
        if current_step < len(steps):
            # Mark current step as completed
            steps[current_step]["status"] = "completed"
            deployment["current_step"] += 1
            
            # Calculate overall progress
            completed_steps = len([s for s in steps if s["status"] == "completed"])
            deployment["progress"] = int((completed_steps / len(steps)) * 100)
            
            # Check if deployment is complete
            if deployment["current_step"] >= len(steps):
                deployment["status"] = "completed"
                deployment["completed_at"] = datetime.now().isoformat()
                self.deployment_history.append(deployment.copy())
        
        return deployment
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment status"""
        return self.active_deployments.get(deployment_id)
    
    def list_active_deployments(self) -> List[Dict[str, Any]]:
        """List all active deployments"""
        return list(self.active_deployments.values())
    
    def list_deployment_history(self) -> List[Dict[str, Any]]:
        """List deployment history"""
        return self.deployment_history
    
    def cancel_deployment(self, deployment_id: str) -> bool:
        """Cancel a deployment"""
        if deployment_id not in self.active_deployments:
            return False
        
        deployment = self.active_deployments[deployment_id]
        deployment["status"] = "cancelled"
        deployment["cancelled_at"] = datetime.now().isoformat()
        
        logger.info(f"Cancelled deployment: {deployment_id}")
        return True
    
    def delete_deployment(self, deployment_id: str) -> bool:
        """Delete a deployment"""
        if deployment_id in self.active_deployments:
            del self.active_deployments[deployment_id]
            logger.info(f"Deleted deployment: {deployment_id}")
            return True
        return False
    
    def get_deployment_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get available deployment templates"""
        return {
            "single_vm": {
                "name": "Single FortiGate VM",
                "description": "Basic single FortiGate VM deployment",
                "estimated_time": "15-20 minutes",
                "resources": ["VM", "VNet", "NSG", "Public IP"],
                "use_cases": ["Small office", "Branch office", "Testing"]
            },
            "ha_active_passive": {
                "name": "HA Active-Passive",
                "description": "High availability with active-passive configuration",
                "estimated_time": "25-35 minutes",
                "resources": ["2x VMs", "Load Balancer", "Shared Storage"],
                "use_cases": ["Production", "Critical workloads", "High availability"]
            },
            "ha_active_active": {
                "name": "HA Active-Active",
                "description": "High availability with active-active configuration",
                "estimated_time": "30-40 minutes",
                "resources": ["2x VMs", "Load Balancer", "Session sync"],
                "use_cases": ["High throughput", "Load distribution", "Scalability"]
            },
            "hub_spoke": {
                "name": "Hub and Spoke",
                "description": "Centralized security with hub and spoke topology",
                "estimated_time": "45-60 minutes",
                "resources": ["Hub VM", "Spoke connections", "Route tables"],
                "use_cases": ["Multi-site", "Centralized security", "Branch connectivity"]
            }
        }
    
    def render_dashboard(self):
        """Render deployment engine dashboard"""
        st.subheader("🚀 Deployment Engine")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🚀 Deploy", "📋 Active", "📚 History"])
        
        with tab1:
            self._render_overview()
        
        with tab2:
            self._render_deploy()
        
        with tab3:
            self._render_active()
        
        with tab4:
            self._render_history()
    
    def _render_overview(self):
        """Render overview tab"""
        st.subheader("📊 Deployment Overview")
        
        active_deployments = self.list_active_deployments()
        history = self.list_deployment_history()
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Deployments", len(active_deployments))
        
        with col2:
            running_count = len([d for d in active_deployments if d["status"] == "running"])
            st.metric("Running", running_count)
        
        with col3:
            completed_count = len([d for d in history if d["status"] == "completed"])
            st.metric("Completed", completed_count)
        
        with col4:
            failed_count = len([d for d in history if d["status"] == "failed"])
            st.metric("Failed", failed_count)
        
        # Recent activity
        if active_deployments or history:
            st.subheader("Recent Activity")
            
            all_deployments = active_deployments + history
            recent = sorted(all_deployments, key=lambda x: x["created_at"], reverse=True)[:5]
            
            for deployment in recent:
                status_emoji = {
                    "initializing": "🔄",
                    "running": "⚡",
                    "completed": "✅",
                    "failed": "❌",
                    "cancelled": "⏹️"
                }
                
                with st.expander(f"{status_emoji.get(deployment['status'], '⚪')} {deployment['name']} - {deployment['status'].title()}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**ID:** {deployment['id']}")
                        st.write(f"**Cloud:** {deployment['cloud_provider'].title()}")
                        st.write(f"**Template:** {deployment['template']}")
                    
                    with col2:
                        st.write(f"**Region:** {deployment['region']}")
                        st.write(f"**Created:** {deployment['created_at']}")
                        if deployment["status"] == "running":
                            st.write(f"**Progress:** {deployment['progress']}%")
        else:
            st.info("No deployments found. Create your first deployment in the Deploy tab.")
    
    def _render_deploy(self):
        """Render deploy tab"""
        st.subheader("🚀 Create New Deployment")
        
        # Template selection
        templates = self.get_deployment_templates()
        
        st.write("**Select Deployment Template:**")
        
        selected_template = None
        for template_id, template_info in templates.items():
            if st.button(f"📋 {template_info['name']}", key=f"template_{template_id}", use_container_width=True):
                selected_template = template_id
        
        if selected_template:
            st.session_state.selected_template = selected_template
        
        if hasattr(st.session_state, 'selected_template'):
            template_id = st.session_state.selected_template
            template_info = templates[template_id]
            
            st.success(f"Selected: {template_info['name']}")
            st.write(f"**Description:** {template_info['description']}")
            st.write(f"**Estimated Time:** {template_info['estimated_time']}")
            st.write(f"**Resources:** {', '.join(template_info['resources'])}")
            
            # Configuration form
            with st.form("deployment_config"):
                st.write("**Deployment Configuration:**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    name = st.text_input("Deployment Name", value=f"FortiGate-{template_id}")
                    cloud_provider = st.selectbox("Cloud Provider", ["azure", "gcp"])
                    region = st.selectbox("Region", 
                        ["eastus", "westus2", "centralus"] if cloud_provider == "azure" 
                        else ["us-central1", "us-east1", "us-west1"])
                
                with col2:
                    instance_type = st.selectbox("Instance Type",
                        ["Standard_D2s_v3", "Standard_D4s_v3", "Standard_D8s_v3"] if cloud_provider == "azure"
                        else ["n1-standard-2", "n1-standard-4", "n1-standard-8"])
                    
                    admin_username = st.text_input("Admin Username", value="azureuser")
                    admin_password = st.text_input("Admin Password", type="password")
                
                # Advanced options
                with st.expander("Advanced Configuration"):
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        vnet_cidr = st.text_input("VNet CIDR", value="10.0.0.0/16")
                        external_subnet = st.text_input("External Subnet", value="10.0.1.0/24")
                    
                    with col4:
                        internal_subnet = st.text_input("Internal Subnet", value="10.0.2.0/24")
                        enable_monitoring = st.checkbox("Enable Monitoring", value=True)
                
                submitted = st.form_submit_button("Start Deployment", type="primary")
                
                if submitted:
                    if not admin_password:
                        st.error("Admin password is required")
                    else:
                        config = {
                            "name": name,
                            "template": template_id,
                            "cloud_provider": cloud_provider,
                            "region": region,
                            "instance_type": instance_type,
                            "admin_username": admin_username,
                            "admin_password": admin_password,
                            "vnet_cidr": vnet_cidr,
                            "external_subnet": external_subnet,
                            "internal_subnet": internal_subnet,
                            "enable_monitoring": enable_monitoring
                        }
                        
                        deployment_id = self.create_deployment(config)
                        self.start_deployment(deployment_id)
                        
                        st.success(f"✅ Deployment started: {deployment_id}")
                        st.info(f"⏱️ Estimated completion time: {template_info['estimated_time']}")
                        
                        # Clear selection
                        del st.session_state.selected_template
                        st.rerun()
    
    def _render_active(self):
        """Render active deployments tab"""
        st.subheader("📋 Active Deployments")
        
        active_deployments = self.list_active_deployments()
        
        if not active_deployments:
            st.info("No active deployments.")
            return
        
        for deployment in active_deployments:
            with st.expander(f"{deployment['name']} ({deployment['id']})"):
                # Update progress
                updated_deployment = self.update_deployment_progress(deployment['id'])
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**Status:** {updated_deployment['status'].title()}")
                    st.write(f"**Cloud:** {updated_deployment['cloud_provider'].title()}")
                    st.write(f"**Template:** {updated_deployment['template']}")
                    st.write(f"**Region:** {updated_deployment['region']}")
                
                with col2:
                    if updated_deployment["status"] == "running":
                        st.progress(updated_deployment["progress"] / 100)
                        st.write(f"Progress: {updated_deployment['progress']}%")
                    
                    st.write(f"**Created:** {updated_deployment['created_at']}")
                
                with col3:
                    if st.button("Cancel", key=f"cancel_{deployment['id']}", type="secondary"):
                        if self.cancel_deployment(deployment['id']):
                            st.success("Deployment cancelled")
                            st.rerun()
                    
                    if st.button("Delete", key=f"delete_{deployment['id']}", type="secondary"):
                        if self.delete_deployment(deployment['id']):
                            st.success("Deployment deleted")
                            st.rerun()
                
                # Show deployment steps
                if updated_deployment["steps"]:
                    st.write("**Deployment Steps:**")
                    
                    for i, step in enumerate(updated_deployment["steps"]):
                        status_emoji = {
                            "pending": "⏳",
                            "running": "⚡",
                            "completed": "✅",
                            "failed": "❌"
                        }
                        
                        current_step = updated_deployment["current_step"]
                        if i < current_step:
                            step_status = "completed"
                        elif i == current_step and updated_deployment["status"] == "running":
                            step_status = "running"
                        else:
                            step_status = "pending"
                        
                        st.write(f"{status_emoji.get(step_status, '⚪')} {step['name']} ({step['duration']}s)")
    
    def _render_history(self):
        """Render deployment history tab"""
        st.subheader("📚 Deployment History")
        
        history = self.list_deployment_history()
        
        if not history:
            st.info("No deployment history available.")
            return
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            status_filter = st.selectbox("Filter by Status", ["All", "Completed", "Failed", "Cancelled"])
        
        with col2:
            cloud_filter = st.selectbox("Filter by Cloud", ["All", "Azure", "GCP"])
        
        # Apply filters
        filtered_history = history
        
        if status_filter != "All":
            filtered_history = [d for d in filtered_history if d["status"].lower() == status_filter.lower()]
        
        if cloud_filter != "All":
            filtered_history = [d for d in filtered_history if d["cloud_provider"].lower() == cloud_filter.lower()]
        
        # Display history
        if filtered_history:
            st.dataframe([
                {
                    "Name": d["name"],
                    "ID": d["id"],
                    "Cloud": d["cloud_provider"].title(),
                    "Template": d["template"],
                    "Status": d["status"].title(),
                    "Created": d["created_at"],
                    "Completed": d.get("completed_at", "N/A")
                }
                for d in filtered_history
            ], use_container_width=True)
        else:
            st.info("No deployments match the selected filters.")
