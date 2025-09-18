"""
Resource Monitor for FortiGate Multi-Cloud Deployments
"""

import streamlit as st
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResourceMonitor:
    """Resource monitor for FortiGate deployments"""
    
    def __init__(self):
        """Initialize resource monitor"""
        self.metrics_data = {}
        self.alerts = []
        
    def collect_metrics(self, deployment_id: str) -> Dict[str, Any]:
        """
        Collect resource metrics for a deployment
        
        Args:
            deployment_id: Deployment identifier
            
        Returns:
            dict: Resource metrics
        """
        timestamp = datetime.now()
        
        # Mock metrics data
        metrics = {
            "timestamp": timestamp.isoformat(),
            "deployment_id": deployment_id,
            "cpu": {
                "usage_percent": 45.2,
                "cores": 4,
                "load_average": [1.2, 1.5, 1.8]
            },
            "memory": {
                "usage_percent": 62.8,
                "total_gb": 16,
                "used_gb": 10.05,
                "available_gb": 5.95
            },
            "network": {
                "bytes_in": 1024000,
                "bytes_out": 2048000,
                "packets_in": 1500,
                "packets_out": 1800,
                "connections": 245
            },
            "disk": {
                "usage_percent": 35.4,
                "total_gb": 100,
                "used_gb": 35.4,
                "available_gb": 64.6,
                "iops": 150
            },
            "security": {
                "blocked_attacks": 12,
                "allowed_connections": 1200,
                "vpn_sessions": 8,
                "ssl_sessions": 45
            }
        }
        
        # Store metrics
        if deployment_id not in self.metrics_data:
            self.metrics_data[deployment_id] = []
        
        self.metrics_data[deployment_id].append(metrics)
        
        # Keep only last 100 data points
        if len(self.metrics_data[deployment_id]) > 100:
            self.metrics_data[deployment_id] = self.metrics_data[deployment_id][-100:]
        
        # Check for alerts
        self._check_alerts(deployment_id, metrics)
        
        return metrics
    
    def get_metrics_history(self, deployment_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics history for a deployment"""
        if deployment_id not in self.metrics_data:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            m for m in self.metrics_data[deployment_id]
            if datetime.fromisoformat(m["timestamp"]) > cutoff_time
        ]
    
    def _check_alerts(self, deployment_id: str, metrics: Dict[str, Any]):
        """Check for alert conditions"""
        alerts = []
        
        # CPU alerts
        if metrics["cpu"]["usage_percent"] > 80:
            alerts.append({
                "type": "cpu_high",
                "severity": "warning",
                "message": f"High CPU usage: {metrics['cpu']['usage_percent']:.1f}%",
                "deployment_id": deployment_id,
                "timestamp": metrics["timestamp"]
            })
        
        # Memory alerts
        if metrics["memory"]["usage_percent"] > 85:
            alerts.append({
                "type": "memory_high",
                "severity": "critical",
                "message": f"High memory usage: {metrics['memory']['usage_percent']:.1f}%",
                "deployment_id": deployment_id,
                "timestamp": metrics["timestamp"]
            })
        
        # Disk alerts
        if metrics["disk"]["usage_percent"] > 90:
            alerts.append({
                "type": "disk_full",
                "severity": "critical",
                "message": f"Disk space critical: {metrics['disk']['usage_percent']:.1f}%",
                "deployment_id": deployment_id,
                "timestamp": metrics["timestamp"]
            })
        
        # Security alerts
        if metrics["security"]["blocked_attacks"] > 50:
            alerts.append({
                "type": "security_high",
                "severity": "warning",
                "message": f"High attack volume: {metrics['security']['blocked_attacks']} blocked",
                "deployment_id": deployment_id,
                "timestamp": metrics["timestamp"]
            })
        
        self.alerts.extend(alerts)
        
        # Keep only last 1000 alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
    
    def get_active_alerts(self, deployment_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active alerts"""
        if deployment_id:
            return [a for a in self.alerts if a["deployment_id"] == deployment_id]
        return self.alerts
    
    def get_performance_summary(self, deployment_id: str) -> Dict[str, Any]:
        """Get performance summary for a deployment"""
        history = self.get_metrics_history(deployment_id, 24)
        
        if not history:
            return {"error": "No metrics data available"}
        
        # Calculate averages
        cpu_avg = sum(m["cpu"]["usage_percent"] for m in history) / len(history)
        memory_avg = sum(m["memory"]["usage_percent"] for m in history) / len(history)
        disk_avg = sum(m["disk"]["usage_percent"] for m in history) / len(history)
        
        # Calculate peaks
        cpu_peak = max(m["cpu"]["usage_percent"] for m in history)
        memory_peak = max(m["memory"]["usage_percent"] for m in history)
        disk_peak = max(m["disk"]["usage_percent"] for m in history)
        
        # Network totals
        total_bytes_in = sum(m["network"]["bytes_in"] for m in history)
        total_bytes_out = sum(m["network"]["bytes_out"] for m in history)
        
        # Security totals
        total_blocked = sum(m["security"]["blocked_attacks"] for m in history)
        total_connections = sum(m["security"]["allowed_connections"] for m in history)
        
        return {
            "deployment_id": deployment_id,
            "period_hours": 24,
            "data_points": len(history),
            "averages": {
                "cpu_percent": round(cpu_avg, 1),
                "memory_percent": round(memory_avg, 1),
                "disk_percent": round(disk_avg, 1)
            },
            "peaks": {
                "cpu_percent": round(cpu_peak, 1),
                "memory_percent": round(memory_peak, 1),
                "disk_percent": round(disk_peak, 1)
            },
            "network": {
                "total_bytes_in": total_bytes_in,
                "total_bytes_out": total_bytes_out,
                "total_gb_in": round(total_bytes_in / (1024**3), 2),
                "total_gb_out": round(total_bytes_out / (1024**3), 2)
            },
            "security": {
                "total_blocked_attacks": total_blocked,
                "total_allowed_connections": total_connections,
                "block_rate": round(total_blocked / (total_blocked + total_connections) * 100, 2) if (total_blocked + total_connections) > 0 else 0
            }
        }
    
    def render_dashboard(self):
        """Render resource monitor dashboard"""
        st.subheader("📊 Resource Monitor")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Real-time", "📊 Analytics", "🚨 Alerts", "⚙️ Settings"])
        
        with tab1:
            self._render_realtime()
        
        with tab2:
            self._render_analytics()
        
        with tab3:
            self._render_alerts()
        
        with tab4:
            self._render_settings()
    
    def _render_realtime(self):
        """Render real-time monitoring tab"""
        st.subheader("📈 Real-time Monitoring")
        
        # Deployment selector
        deployment_id = st.selectbox("Select Deployment", ["fortigate-prod-001", "fortigate-dev-002", "fortigate-test-003"])
        
        # Auto-refresh
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)
        
        if auto_refresh:
            st.rerun()
        
        # Collect current metrics
        current_metrics = self.collect_metrics(deployment_id)
        
        # Display current metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cpu_color = "red" if current_metrics["cpu"]["usage_percent"] > 80 else "orange" if current_metrics["cpu"]["usage_percent"] > 60 else "green"
            st.metric("CPU Usage", f"{current_metrics['cpu']['usage_percent']:.1f}%")
        
        with col2:
            memory_color = "red" if current_metrics["memory"]["usage_percent"] > 85 else "orange" if current_metrics["memory"]["usage_percent"] > 70 else "green"
            st.metric("Memory Usage", f"{current_metrics['memory']['usage_percent']:.1f}%")
        
        with col3:
            st.metric("Network In", f"{current_metrics['network']['bytes_in'] / 1024:.1f} KB/s")
        
        with col4:
            st.metric("Active Connections", current_metrics["network"]["connections"])
        
        # Resource usage charts
        history = self.get_metrics_history(deployment_id, 1)  # Last hour
        
        if len(history) > 1:
            # CPU chart
            st.subheader("CPU Usage Trend")
            cpu_data = {
                "Time": [datetime.fromisoformat(m["timestamp"]).strftime("%H:%M") for m in history],
                "CPU %": [m["cpu"]["usage_percent"] for m in history]
            }
            st.line_chart(cpu_data, x="Time", y="CPU %")
            
            # Memory chart
            st.subheader("Memory Usage Trend")
            memory_data = {
                "Time": [datetime.fromisoformat(m["timestamp"]).strftime("%H:%M") for m in history],
                "Memory %": [m["memory"]["usage_percent"] for m in history]
            }
            st.line_chart(memory_data, x="Time", y="Memory %")
        
        # Security metrics
        st.subheader("Security Metrics")
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric("Blocked Attacks", current_metrics["security"]["blocked_attacks"])
        
        with col6:
            st.metric("VPN Sessions", current_metrics["security"]["vpn_sessions"])
        
        with col7:
            st.metric("SSL Sessions", current_metrics["security"]["ssl_sessions"])
        
        with col8:
            st.metric("Allowed Connections", current_metrics["security"]["allowed_connections"])
    
    def _render_analytics(self):
        """Render analytics tab"""
        st.subheader("📊 Performance Analytics")
        
        deployment_id = st.selectbox("Select Deployment", ["fortigate-prod-001", "fortigate-dev-002", "fortigate-test-003"], key="analytics_deployment")
        
        # Time range selector
        time_range = st.selectbox("Time Range", ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last Week"])
        
        hours_map = {"Last Hour": 1, "Last 6 Hours": 6, "Last 24 Hours": 24, "Last Week": 168}
        hours = hours_map[time_range]
        
        # Get performance summary
        summary = self.get_performance_summary(deployment_id)
        
        if "error" not in summary:
            # Summary metrics
            st.subheader("Performance Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Average Usage:**")
                st.write(f"CPU: {summary['averages']['cpu_percent']}%")
                st.write(f"Memory: {summary['averages']['memory_percent']}%")
                st.write(f"Disk: {summary['averages']['disk_percent']}%")
            
            with col2:
                st.write("**Peak Usage:**")
                st.write(f"CPU: {summary['peaks']['cpu_percent']}%")
                st.write(f"Memory: {summary['peaks']['memory_percent']}%")
                st.write(f"Disk: {summary['peaks']['disk_percent']}%")
            
            with col3:
                st.write("**Network Traffic:**")
                st.write(f"Inbound: {summary['network']['total_gb_in']} GB")
                st.write(f"Outbound: {summary['network']['total_gb_out']} GB")
                st.write(f"Block Rate: {summary['security']['block_rate']}%")
            
            # Historical trends
            history = self.get_metrics_history(deployment_id, hours)
            
            if len(history) > 1:
                st.subheader("Resource Trends")
                
                # Combined resource chart
                chart_data = {
                    "Time": [datetime.fromisoformat(m["timestamp"]).strftime("%m-%d %H:%M") for m in history],
                    "CPU %": [m["cpu"]["usage_percent"] for m in history],
                    "Memory %": [m["memory"]["usage_percent"] for m in history],
                    "Disk %": [m["disk"]["usage_percent"] for m in history]
                }
                st.line_chart(chart_data, x="Time")
                
                # Network traffic chart
                st.subheader("Network Traffic")
                network_data = {
                    "Time": [datetime.fromisoformat(m["timestamp"]).strftime("%m-%d %H:%M") for m in history],
                    "Bytes In": [m["network"]["bytes_in"] for m in history],
                    "Bytes Out": [m["network"]["bytes_out"] for m in history]
                }
                st.line_chart(network_data, x="Time")
        else:
            st.info("No analytics data available. Start monitoring to see performance trends.")
    
    def _render_alerts(self):
        """Render alerts tab"""
        st.subheader("🚨 System Alerts")
        
        # Alert filters
        col1, col2 = st.columns(2)
        
        with col1:
            severity_filter = st.selectbox("Filter by Severity", ["All", "Critical", "Warning", "Info"])
        
        with col2:
            deployment_filter = st.selectbox("Filter by Deployment", ["All", "fortigate-prod-001", "fortigate-dev-002", "fortigate-test-003"])
        
        # Get alerts
        alerts = self.get_active_alerts()
        
        # Apply filters
        if severity_filter != "All":
            alerts = [a for a in alerts if a["severity"].lower() == severity_filter.lower()]
        
        if deployment_filter != "All":
            alerts = [a for a in alerts if a["deployment_id"] == deployment_filter]
        
        # Display alerts
        if alerts:
            st.write(f"**{len(alerts)} alerts found**")
            
            for alert in sorted(alerts, key=lambda x: x["timestamp"], reverse=True)[:50]:  # Show last 50
                severity_emoji = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
                
                with st.expander(f"{severity_emoji.get(alert['severity'], '⚪')} {alert['message']}"):
                    st.write(f"**Type:** {alert['type']}")
                    st.write(f"**Severity:** {alert['severity'].title()}")
                    st.write(f"**Deployment:** {alert['deployment_id']}")
                    st.write(f"**Time:** {alert['timestamp']}")
        else:
            st.success("✅ No alerts found")
    
    def _render_settings(self):
        """Render settings tab"""
        st.subheader("⚙️ Monitor Settings")
        
        with st.form("monitor_settings"):
            st.write("**Collection Settings:**")
            
            collection_interval = st.selectbox("Collection Interval", ["30 seconds", "1 minute", "5 minutes", "15 minutes"])
            retention_days = st.number_input("Data Retention (days)", min_value=1, max_value=365, value=30)
            
            st.write("**Alert Thresholds:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cpu_warning = st.number_input("CPU Warning (%)", min_value=1, max_value=100, value=80)
                cpu_critical = st.number_input("CPU Critical (%)", min_value=1, max_value=100, value=90)
                
                memory_warning = st.number_input("Memory Warning (%)", min_value=1, max_value=100, value=85)
                memory_critical = st.number_input("Memory Critical (%)", min_value=1, max_value=100, value=95)
            
            with col2:
                disk_warning = st.number_input("Disk Warning (%)", min_value=1, max_value=100, value=80)
                disk_critical = st.number_input("Disk Critical (%)", min_value=1, max_value=100, value=90)
                
                attack_threshold = st.number_input("Attack Alert Threshold", min_value=1, value=50)
            
            st.write("**Notification Settings:**")
            
            enable_email = st.checkbox("Email Notifications", value=True)
            email_address = st.text_input("Email Address")
            
            enable_webhook = st.checkbox("Webhook Notifications")
            webhook_url = st.text_input("Webhook URL")
            
            submitted = st.form_submit_button("Save Settings")
            
            if submitted:
                st.success("Monitor settings saved successfully!")
