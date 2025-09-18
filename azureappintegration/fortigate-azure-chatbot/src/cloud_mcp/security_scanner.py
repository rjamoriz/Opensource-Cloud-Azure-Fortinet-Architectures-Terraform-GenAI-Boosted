"""
Security Scanner for FortiGate Multi-Cloud Deployments
"""

import streamlit as st
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityScanner:
    """Security scanner for FortiGate deployments"""
    
    def __init__(self):
        """Initialize security scanner"""
        self.scan_results = {}
        self.security_policies = {}
        
    def scan_deployment(self, deployment_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scan a FortiGate deployment for security issues
        
        Args:
            deployment_id: Deployment identifier
            config: Deployment configuration
            
        Returns:
            dict: Scan results
        """
        scan_id = f"scan-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Perform security checks
        findings = []
        
        # Check network security
        network_findings = self._check_network_security(config)
        findings.extend(network_findings)
        
        # Check access controls
        access_findings = self._check_access_controls(config)
        findings.extend(access_findings)
        
        # Check encryption
        encryption_findings = self._check_encryption(config)
        findings.extend(encryption_findings)
        
        # Check compliance
        compliance_findings = self._check_compliance(config)
        findings.extend(compliance_findings)
        
        # Calculate security score
        security_score = self._calculate_security_score(findings)
        
        scan_result = {
            "scan_id": scan_id,
            "deployment_id": deployment_id,
            "timestamp": datetime.now().isoformat(),
            "security_score": security_score,
            "total_findings": len(findings),
            "critical_findings": len([f for f in findings if f["severity"] == "critical"]),
            "high_findings": len([f for f in findings if f["severity"] == "high"]),
            "medium_findings": len([f for f in findings if f["severity"] == "medium"]),
            "low_findings": len([f for f in findings if f["severity"] == "low"]),
            "findings": findings,
            "recommendations": self._generate_recommendations(findings)
        }
        
        self.scan_results[scan_id] = scan_result
        logger.info(f"Security scan completed: {scan_id}")
        
        return scan_result
    
    def _check_network_security(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check network security configuration"""
        findings = []
        
        # Check for open ports
        if config.get("allow_all_ports", False):
            findings.append({
                "category": "Network Security",
                "severity": "critical",
                "title": "All Ports Open",
                "description": "All ports are open to the internet",
                "recommendation": "Restrict access to only necessary ports"
            })
        
        # Check for weak firewall rules
        firewall_rules = config.get("firewall_rules", [])
        for rule in firewall_rules:
            if rule.get("source") == "0.0.0.0/0" and rule.get("protocol") == "any":
                findings.append({
                    "category": "Network Security",
                    "severity": "high",
                    "title": "Overly Permissive Firewall Rule",
                    "description": f"Rule '{rule.get('name')}' allows all traffic from anywhere",
                    "recommendation": "Restrict source IP ranges and protocols"
                })
        
        # Check subnet configuration
        external_subnet = config.get("external_subnet", "")
        if not external_subnet or external_subnet == "0.0.0.0/0":
            findings.append({
                "category": "Network Security",
                "severity": "medium",
                "title": "Subnet Configuration Issue",
                "description": "External subnet not properly configured",
                "recommendation": "Configure proper subnet ranges"
            })
        
        return findings
    
    def _check_access_controls(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check access control configuration"""
        findings = []
        
        # Check admin password strength
        admin_password = config.get("admin_password", "")
        if len(admin_password) < 12:
            findings.append({
                "category": "Access Control",
                "severity": "high",
                "title": "Weak Admin Password",
                "description": "Admin password is too short",
                "recommendation": "Use a password with at least 12 characters"
            })
        
        # Check for default credentials
        if admin_password in ["admin", "password", "123456", "P@ssw0rd"]:
            findings.append({
                "category": "Access Control",
                "severity": "critical",
                "title": "Default/Weak Credentials",
                "description": "Using default or commonly used passwords",
                "recommendation": "Change to a strong, unique password"
            })
        
        # Check SSH key configuration
        if not config.get("ssh_key") and config.get("ssh_enabled", True):
            findings.append({
                "category": "Access Control",
                "severity": "medium",
                "title": "SSH Key Not Configured",
                "description": "SSH access enabled without key-based authentication",
                "recommendation": "Configure SSH key-based authentication"
            })
        
        return findings
    
    def _check_encryption(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check encryption configuration"""
        findings = []
        
        # Check disk encryption
        if not config.get("disk_encryption_enabled", False):
            findings.append({
                "category": "Encryption",
                "severity": "high",
                "title": "Disk Encryption Disabled",
                "description": "VM disks are not encrypted",
                "recommendation": "Enable disk encryption for data at rest"
            })
        
        # Check TLS configuration
        tls_version = config.get("min_tls_version", "1.0")
        if float(tls_version) < 1.2:
            findings.append({
                "category": "Encryption",
                "severity": "medium",
                "title": "Outdated TLS Version",
                "description": f"Minimum TLS version is {tls_version}",
                "recommendation": "Set minimum TLS version to 1.2 or higher"
            })
        
        return findings
    
    def _check_compliance(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check compliance requirements"""
        findings = []
        
        # Check logging configuration
        if not config.get("logging_enabled", False):
            findings.append({
                "category": "Compliance",
                "severity": "medium",
                "title": "Logging Disabled",
                "description": "Security logging is not enabled",
                "recommendation": "Enable comprehensive security logging"
            })
        
        # Check backup configuration
        if not config.get("backup_enabled", False):
            findings.append({
                "category": "Compliance",
                "severity": "medium",
                "title": "Backup Not Configured",
                "description": "Configuration backup is not enabled",
                "recommendation": "Enable automated configuration backups"
            })
        
        # Check update policy
        if not config.get("auto_updates", False):
            findings.append({
                "category": "Compliance",
                "severity": "low",
                "title": "Auto-Updates Disabled",
                "description": "Automatic security updates are disabled",
                "recommendation": "Enable automatic security updates"
            })
        
        return findings
    
    def _calculate_security_score(self, findings: List[Dict[str, Any]]) -> int:
        """Calculate security score based on findings"""
        base_score = 100
        
        for finding in findings:
            severity = finding["severity"]
            if severity == "critical":
                base_score -= 25
            elif severity == "high":
                base_score -= 15
            elif severity == "medium":
                base_score -= 10
            elif severity == "low":
                base_score -= 5
        
        return max(0, base_score)
    
    def _generate_recommendations(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        # Priority recommendations based on critical findings
        critical_findings = [f for f in findings if f["severity"] == "critical"]
        if critical_findings:
            recommendations.append("🚨 Address critical security issues immediately")
        
        # Category-based recommendations
        categories = set(f["category"] for f in findings)
        
        if "Network Security" in categories:
            recommendations.append("🔒 Review and tighten firewall rules")
        
        if "Access Control" in categories:
            recommendations.append("🔑 Strengthen authentication mechanisms")
        
        if "Encryption" in categories:
            recommendations.append("🛡️ Enable encryption for data at rest and in transit")
        
        if "Compliance" in categories:
            recommendations.append("📋 Implement compliance monitoring and logging")
        
        return recommendations
    
    def get_scan_result(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Get scan result by ID"""
        return self.scan_results.get(scan_id)
    
    def list_scan_results(self) -> List[Dict[str, Any]]:
        """List all scan results"""
        return list(self.scan_results.values())
    
    def get_security_policies(self) -> Dict[str, Any]:
        """Get available security policies"""
        return {
            "baseline": {
                "name": "Security Baseline",
                "description": "Basic security requirements for FortiGate deployments",
                "requirements": [
                    "Strong admin passwords (12+ characters)",
                    "Restricted firewall rules",
                    "TLS 1.2 or higher",
                    "Disk encryption enabled",
                    "Security logging enabled"
                ]
            },
            "enterprise": {
                "name": "Enterprise Security",
                "description": "Enhanced security for enterprise deployments",
                "requirements": [
                    "All baseline requirements",
                    "Multi-factor authentication",
                    "Network segmentation",
                    "Regular security scans",
                    "Compliance monitoring"
                ]
            },
            "government": {
                "name": "Government/Compliance",
                "description": "High security standards for government and regulated industries",
                "requirements": [
                    "All enterprise requirements",
                    "FIPS 140-2 compliance",
                    "Advanced threat protection",
                    "Audit logging",
                    "Regular penetration testing"
                ]
            }
        }
    
    def render_dashboard(self):
        """Render security scanner dashboard"""
        st.subheader("🔍 Security Scanner")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Scan Results", "📋 Policies", "⚙️ Settings"])
        
        with tab1:
            self._render_overview()
        
        with tab2:
            self._render_scan_results()
        
        with tab3:
            self._render_policies()
        
        with tab4:
            self._render_settings()
    
    def _render_overview(self):
        """Render overview tab"""
        st.subheader("📊 Security Overview")
        
        scan_results = self.list_scan_results()
        
        if scan_results:
            # Latest scan metrics
            latest_scan = max(scan_results, key=lambda x: x["timestamp"])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                score_color = "green" if latest_scan["security_score"] >= 80 else "orange" if latest_scan["security_score"] >= 60 else "red"
                st.metric("Security Score", f"{latest_scan['security_score']}/100")
            
            with col2:
                st.metric("Critical Issues", latest_scan["critical_findings"], delta=-latest_scan["critical_findings"] if latest_scan["critical_findings"] > 0 else None)
            
            with col3:
                st.metric("High Issues", latest_scan["high_findings"])
            
            with col4:
                st.metric("Total Scans", len(scan_results))
            
            # Security score trend (mock data)
            st.subheader("Security Score Trend")
            chart_data = {
                "Date": ["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15"],
                "Score": [65, 72, 78, latest_scan["security_score"]]
            }
            st.line_chart(chart_data, x="Date", y="Score")
            
        else:
            st.info("No security scans available. Run your first scan to see security metrics.")
            
            if st.button("Run Sample Scan", type="primary"):
                # Run a sample scan
                sample_config = {
                    "admin_password": "P@ssw0rd123!",
                    "firewall_rules": [
                        {"name": "allow-all", "source": "0.0.0.0/0", "protocol": "tcp"}
                    ],
                    "disk_encryption_enabled": False,
                    "logging_enabled": True
                }
                
                result = self.scan_deployment("sample-deployment", sample_config)
                st.success(f"Sample scan completed! Security Score: {result['security_score']}/100")
                st.rerun()
    
    def _render_scan_results(self):
        """Render scan results tab"""
        st.subheader("🔍 Scan Results")
        
        scan_results = self.list_scan_results()
        
        if not scan_results:
            st.info("No scan results available.")
            return
        
        # Display scan results
        for result in sorted(scan_results, key=lambda x: x["timestamp"], reverse=True):
            with st.expander(f"Scan {result['scan_id']} - Score: {result['security_score']}/100"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Deployment:** {result['deployment_id']}")
                    st.write(f"**Timestamp:** {result['timestamp']}")
                    st.write(f"**Total Findings:** {result['total_findings']}")
                
                with col2:
                    # Severity breakdown
                    st.write("**Severity Breakdown:**")
                    st.write(f"🔴 Critical: {result['critical_findings']}")
                    st.write(f"🟠 High: {result['high_findings']}")
                    st.write(f"🟡 Medium: {result['medium_findings']}")
                    st.write(f"🟢 Low: {result['low_findings']}")
                
                # Findings details
                if result["findings"]:
                    st.write("**Findings:**")
                    for finding in result["findings"]:
                        severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
                        st.write(f"{severity_emoji[finding['severity']]} **{finding['title']}** ({finding['category']})")
                        st.write(f"   {finding['description']}")
                        st.write(f"   💡 {finding['recommendation']}")
                
                # Recommendations
                if result["recommendations"]:
                    st.write("**Recommendations:**")
                    for rec in result["recommendations"]:
                        st.write(f"• {rec}")
    
    def _render_policies(self):
        """Render policies tab"""
        st.subheader("📋 Security Policies")
        
        policies = self.get_security_policies()
        
        for policy_id, policy_info in policies.items():
            with st.expander(f"{policy_info['name']}"):
                st.write(f"**Description:** {policy_info['description']}")
                st.write("**Requirements:**")
                for req in policy_info["requirements"]:
                    st.write(f"• {req}")
    
    def _render_settings(self):
        """Render settings tab"""
        st.subheader("⚙️ Scanner Settings")
        
        with st.form("scanner_settings"):
            st.write("**Scan Configuration:**")
            
            scan_frequency = st.selectbox("Scan Frequency", ["Manual", "Daily", "Weekly", "Monthly"])
            
            st.write("**Notification Settings:**")
            notify_critical = st.checkbox("Notify on Critical Findings", value=True)
            notify_high = st.checkbox("Notify on High Findings", value=True)
            
            email_notifications = st.text_input("Email for Notifications")
            
            st.write("**Compliance Standards:**")
            enable_pci = st.checkbox("PCI DSS Compliance Checks")
            enable_hipaa = st.checkbox("HIPAA Compliance Checks")
            enable_sox = st.checkbox("SOX Compliance Checks")
            
            submitted = st.form_submit_button("Save Settings")
            
            if submitted:
                st.success("Settings saved successfully!")
