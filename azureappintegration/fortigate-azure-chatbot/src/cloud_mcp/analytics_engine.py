"""
Analytics Engine for FortiGate Multi-Cloud Analytics and Insights
"""

import streamlit as st
import pandas as pd
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalyticsEngine:
    """Analytics engine for FortiGate multi-cloud analytics and insights"""
    
    def __init__(self):
        """Initialize analytics engine"""
        self.metrics_data = {}
        self.alerts = []
        self.reports = []
        
    def collect_metrics(self, source: str, metrics: Dict[str, Any]) -> bool:
        """
        Collect metrics from various sources
        
        Args:
            source: Source of metrics (e.g., 'azure', 'gcp', 'fortigate')
            metrics: Metrics data
            
        Returns:
            bool: Success status
        """
        timestamp = datetime.now().isoformat()
        
        if source not in self.metrics_data:
            self.metrics_data[source] = []
        
        metric_entry = {
            "timestamp": timestamp,
            "metrics": metrics
        }
        
        self.metrics_data[source].append(metric_entry)
        
        # Keep only last 1000 entries per source
        if len(self.metrics_data[source]) > 1000:
            self.metrics_data[source] = self.metrics_data[source][-1000:]
        
        logger.info(f"Collected metrics from {source}")
        return True
    
    def generate_insights(self, timeframe: str = "24h") -> Dict[str, Any]:
        """
        Generate insights from collected metrics
        
        Args:
            timeframe: Analysis timeframe
            
        Returns:
            Dict containing insights
        """
        insights = {
            "summary": {},
            "trends": {},
            "anomalies": [],
            "recommendations": []
        }
        
        # Calculate timeframe
        now = datetime.now()
        if timeframe == "1h":
            start_time = now - timedelta(hours=1)
        elif timeframe == "24h":
            start_time = now - timedelta(hours=24)
        elif timeframe == "7d":
            start_time = now - timedelta(days=7)
        else:
            start_time = now - timedelta(hours=24)
        
        # Analyze each source
        for source, data in self.metrics_data.items():
            # Filter data by timeframe
            filtered_data = [
                entry for entry in data
                if datetime.fromisoformat(entry["timestamp"]) >= start_time
            ]
            
            if not filtered_data:
                continue
            
            # Generate summary
            insights["summary"][source] = self._analyze_source_summary(filtered_data)
            
            # Generate trends
            insights["trends"][source] = self._analyze_source_trends(filtered_data)
            
            # Detect anomalies
            anomalies = self._detect_anomalies(source, filtered_data)
            insights["anomalies"].extend(anomalies)
        
        # Generate recommendations
        insights["recommendations"] = self._generate_recommendations(insights)
        
        return insights
    
    def _analyze_source_summary(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze summary statistics for a source"""
        if not data:
            return {}
        
        # Extract numeric metrics
        numeric_metrics = {}
        for entry in data:
            for key, value in entry["metrics"].items():
                if isinstance(value, (int, float)):
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(value)
        
        # Calculate statistics
        summary = {}
        for metric, values in numeric_metrics.items():
            if values:
                summary[metric] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        
        return summary
    
    def _analyze_source_trends(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends for a source"""
        if len(data) < 2:
            return {}
        
        trends = {}
        
        # Simple trend analysis (comparing first half vs second half)
        mid_point = len(data) // 2
        first_half = data[:mid_point]
        second_half = data[mid_point:]
        
        # Extract numeric metrics for both halves
        first_metrics = {}
        second_metrics = {}
        
        for entry in first_half:
            for key, value in entry["metrics"].items():
                if isinstance(value, (int, float)):
                    if key not in first_metrics:
                        first_metrics[key] = []
                    first_metrics[key].append(value)
        
        for entry in second_half:
            for key, value in entry["metrics"].items():
                if isinstance(value, (int, float)):
                    if key not in second_metrics:
                        second_metrics[key] = []
                    second_metrics[key].append(value)
        
        # Calculate trend direction
        for metric in first_metrics:
            if metric in second_metrics:
                first_avg = sum(first_metrics[metric]) / len(first_metrics[metric])
                second_avg = sum(second_metrics[metric]) / len(second_metrics[metric])
                
                if second_avg > first_avg * 1.1:
                    trend = "increasing"
                elif second_avg < first_avg * 0.9:
                    trend = "decreasing"
                else:
                    trend = "stable"
                
                trends[metric] = {
                    "direction": trend,
                    "change_percent": ((second_avg - first_avg) / first_avg) * 100 if first_avg > 0 else 0
                }
        
        return trends
    
    def _detect_anomalies(self, source: str, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalies in the data"""
        anomalies = []
        
        if len(data) < 10:  # Need sufficient data for anomaly detection
            return anomalies
        
        # Extract numeric metrics
        numeric_metrics = {}
        for entry in data:
            for key, value in entry["metrics"].items():
                if isinstance(value, (int, float)):
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append((entry["timestamp"], value))
        
        # Simple anomaly detection using standard deviation
        for metric, values in numeric_metrics.items():
            if len(values) < 10:
                continue
            
            # Calculate mean and std dev
            metric_values = [v[1] for v in values]
            mean_val = sum(metric_values) / len(metric_values)
            variance = sum((x - mean_val) ** 2 for x in metric_values) / len(metric_values)
            std_dev = variance ** 0.5
            
            # Detect outliers (values beyond 2 standard deviations)
            threshold = 2 * std_dev
            
            for timestamp, value in values[-10:]:  # Check last 10 values
                if abs(value - mean_val) > threshold:
                    anomalies.append({
                        "source": source,
                        "metric": metric,
                        "timestamp": timestamp,
                        "value": value,
                        "expected_range": [mean_val - threshold, mean_val + threshold],
                        "severity": "high" if abs(value - mean_val) > 3 * std_dev else "medium"
                    })
        
        return anomalies
    
    def _generate_recommendations(self, insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on insights"""
        recommendations = []
        
        # Check for high resource utilization
        for source, summary in insights["summary"].items():
            for metric, stats in summary.items():
                if "cpu" in metric.lower() and stats["avg"] > 80:
                    recommendations.append({
                        "type": "performance",
                        "priority": "high",
                        "title": f"High CPU utilization in {source}",
                        "description": f"Average CPU usage is {stats['avg']:.1f}%. Consider scaling up resources.",
                        "action": "Scale up instance or optimize workload"
                    })
                
                if "memory" in metric.lower() and stats["avg"] > 85:
                    recommendations.append({
                        "type": "performance",
                        "priority": "high",
                        "title": f"High memory utilization in {source}",
                        "description": f"Average memory usage is {stats['avg']:.1f}%. Risk of performance degradation.",
                        "action": "Increase memory allocation or optimize memory usage"
                    })
        
        # Check for increasing trends
        for source, trends in insights["trends"].items():
            for metric, trend_data in trends.items():
                if trend_data["direction"] == "increasing" and trend_data["change_percent"] > 50:
                    recommendations.append({
                        "type": "trend",
                        "priority": "medium",
                        "title": f"Rapidly increasing {metric} in {source}",
                        "description": f"{metric} has increased by {trend_data['change_percent']:.1f}% recently.",
                        "action": "Monitor closely and investigate root cause"
                    })
        
        # Check for anomalies
        high_severity_anomalies = [a for a in insights["anomalies"] if a["severity"] == "high"]
        if high_severity_anomalies:
            recommendations.append({
                "type": "anomaly",
                "priority": "critical",
                "title": f"Critical anomalies detected",
                "description": f"Found {len(high_severity_anomalies)} high-severity anomalies across systems.",
                "action": "Investigate anomalies immediately"
            })
        
        return recommendations
    
    def create_report(self, report_type: str, config: Dict[str, Any]) -> str:
        """
        Create analytics report
        
        Args:
            report_type: Type of report
            config: Report configuration
            
        Returns:
            str: Report ID
        """
        report_id = f"report-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        insights = self.generate_insights(config.get("timeframe", "24h"))
        
        report = {
            "id": report_id,
            "type": report_type,
            "created_at": datetime.now().isoformat(),
            "config": config,
            "insights": insights,
            "status": "completed"
        }
        
        self.reports.append(report)
        logger.info(f"Created report: {report_id}")
        
        return report_id
    
    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get report by ID"""
        for report in self.reports:
            if report["id"] == report_id:
                return report
        return None
    
    def list_reports(self) -> List[Dict[str, Any]]:
        """List all reports"""
        return self.reports
    
    def generate_sample_data(self):
        """Generate sample data for demonstration"""
        # Generate sample Azure metrics
        for i in range(50):
            timestamp = datetime.now() - timedelta(minutes=i * 30)
            azure_metrics = {
                "cpu_usage": random.uniform(20, 90),
                "memory_usage": random.uniform(30, 85),
                "network_in": random.uniform(100, 1000),
                "network_out": random.uniform(50, 800),
                "disk_usage": random.uniform(40, 70),
                "active_connections": random.randint(10, 500)
            }
            
            self.metrics_data.setdefault("azure", []).append({
                "timestamp": timestamp.isoformat(),
                "metrics": azure_metrics
            })
        
        # Generate sample GCP metrics
        for i in range(40):
            timestamp = datetime.now() - timedelta(minutes=i * 45)
            gcp_metrics = {
                "cpu_usage": random.uniform(15, 75),
                "memory_usage": random.uniform(25, 80),
                "network_throughput": random.uniform(200, 1200),
                "storage_usage": random.uniform(35, 65),
                "request_count": random.randint(50, 1000)
            }
            
            self.metrics_data.setdefault("gcp", []).append({
                "timestamp": timestamp.isoformat(),
                "metrics": gcp_metrics
            })
        
        # Generate sample FortiGate metrics
        for i in range(60):
            timestamp = datetime.now() - timedelta(minutes=i * 20)
            fortigate_metrics = {
                "cpu_usage": random.uniform(10, 60),
                "memory_usage": random.uniform(20, 70),
                "session_count": random.randint(100, 2000),
                "throughput_mbps": random.uniform(50, 500),
                "blocked_threats": random.randint(0, 50),
                "policy_hits": random.randint(100, 5000)
            }
            
            self.metrics_data.setdefault("fortigate", []).append({
                "timestamp": timestamp.isoformat(),
                "metrics": fortigate_metrics
            })
    
    def render_dashboard(self):
        """Render analytics dashboard"""
        st.subheader("📊 Analytics Engine")
        
        # Generate sample data if empty
        if not self.metrics_data:
            if st.button("Generate Sample Data", type="primary"):
                self.generate_sample_data()
                st.success("Sample data generated!")
                st.rerun()
            st.info("Click 'Generate Sample Data' to see analytics in action.")
            return
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "🔍 Insights", "⚠️ Anomalies", "📋 Reports", "⚙️ Settings"])
        
        with tab1:
            self._render_overview()
        
        with tab2:
            self._render_insights()
        
        with tab3:
            self._render_anomalies()
        
        with tab4:
            self._render_reports()
        
        with tab5:
            self._render_settings()
    
    def _render_overview(self):
        """Render overview tab"""
        st.subheader("📈 Analytics Overview")
        
        # Timeframe selector
        timeframe = st.selectbox("Select Timeframe", ["1h", "24h", "7d"], index=1)
        
        insights = self.generate_insights(timeframe)
        
        # Summary metrics
        if insights["summary"]:
            st.subheader("📊 Summary Metrics")
            
            for source, summary in insights["summary"].items():
                with st.expander(f"{source.title()} Metrics"):
                    if summary:
                        # Create columns for metrics
                        metrics_list = list(summary.items())
                        cols = st.columns(min(len(metrics_list), 4))
                        
                        for i, (metric, stats) in enumerate(metrics_list):
                            with cols[i % 4]:
                                st.metric(
                                    label=metric.replace("_", " ").title(),
                                    value=f"{stats['avg']:.1f}",
                                    delta=f"Min: {stats['min']:.1f}, Max: {stats['max']:.1f}"
                                )
                    else:
                        st.info("No metrics available for this source.")
        
        # Trends overview
        if insights["trends"]:
            st.subheader("📈 Trends Overview")
            
            trend_data = []
            for source, trends in insights["trends"].items():
                for metric, trend_info in trends.items():
                    trend_data.append({
                        "Source": source.title(),
                        "Metric": metric.replace("_", " ").title(),
                        "Direction": trend_info["direction"].title(),
                        "Change %": f"{trend_info['change_percent']:.1f}%"
                    })
            
            if trend_data:
                df = pd.DataFrame(trend_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No trend data available.")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_sources = len(self.metrics_data)
            st.metric("Data Sources", total_sources)
        
        with col2:
            total_metrics = sum(len(data) for data in self.metrics_data.values())
            st.metric("Total Metrics", total_metrics)
        
        with col3:
            anomaly_count = len(insights["anomalies"])
            st.metric("Anomalies", anomaly_count)
        
        with col4:
            recommendation_count = len(insights["recommendations"])
            st.metric("Recommendations", recommendation_count)
    
    def _render_insights(self):
        """Render insights tab"""
        st.subheader("🔍 Insights & Recommendations")
        
        timeframe = st.selectbox("Analysis Timeframe", ["1h", "24h", "7d"], index=1, key="insights_timeframe")
        
        insights = self.generate_insights(timeframe)
        
        # Recommendations
        if insights["recommendations"]:
            st.subheader("💡 Recommendations")
            
            # Group by priority
            priorities = {"critical": [], "high": [], "medium": [], "low": []}
            for rec in insights["recommendations"]:
                priority = rec.get("priority", "low")
                priorities[priority].append(rec)
            
            for priority, recs in priorities.items():
                if recs:
                    priority_emoji = {"critical": "🚨", "high": "⚠️", "medium": "📋", "low": "💡"}
                    st.write(f"**{priority_emoji.get(priority, '📋')} {priority.title()} Priority**")
                    
                    for rec in recs:
                        with st.expander(f"{rec['title']}"):
                            st.write(f"**Type:** {rec['type'].title()}")
                            st.write(f"**Description:** {rec['description']}")
                            st.write(f"**Recommended Action:** {rec['action']}")
        else:
            st.info("No recommendations available. System appears to be running optimally.")
        
        # Detailed trends
        if insights["trends"]:
            st.subheader("📈 Detailed Trends")
            
            for source, trends in insights["trends"].items():
                if trends:
                    with st.expander(f"{source.title()} Trends"):
                        for metric, trend_data in trends.items():
                            direction_emoji = {
                                "increasing": "📈",
                                "decreasing": "📉",
                                "stable": "➡️"
                            }
                            
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write(f"**{metric.replace('_', ' ').title()}**")
                                st.write(f"Direction: {direction_emoji.get(trend_data['direction'], '➡️')} {trend_data['direction'].title()}")
                            
                            with col2:
                                change = trend_data['change_percent']
                                color = "red" if change > 20 else "orange" if change > 10 else "green"
                                st.markdown(f"<span style='color: {color}'>{change:+.1f}%</span>", unsafe_allow_html=True)
    
    def _render_anomalies(self):
        """Render anomalies tab"""
        st.subheader("⚠️ Anomaly Detection")
        
        timeframe = st.selectbox("Detection Timeframe", ["1h", "24h", "7d"], index=1, key="anomaly_timeframe")
        
        insights = self.generate_insights(timeframe)
        anomalies = insights["anomalies"]
        
        if anomalies:
            # Filter options
            col1, col2 = st.columns(2)
            
            with col1:
                severity_filter = st.selectbox("Filter by Severity", ["All", "Critical", "High", "Medium", "Low"])
            
            with col2:
                source_filter = st.selectbox("Filter by Source", ["All"] + list(set(a["source"] for a in anomalies)))
            
            # Apply filters
            filtered_anomalies = anomalies
            
            if severity_filter != "All":
                filtered_anomalies = [a for a in filtered_anomalies if a["severity"].lower() == severity_filter.lower()]
            
            if source_filter != "All":
                filtered_anomalies = [a for a in filtered_anomalies if a["source"] == source_filter.lower()]
            
            # Display anomalies
            if filtered_anomalies:
                st.write(f"**Found {len(filtered_anomalies)} anomalies:**")
                
                for anomaly in filtered_anomalies:
                    severity_emoji = {"critical": "🚨", "high": "⚠️", "medium": "📋", "low": "💡"}
                    
                    with st.expander(f"{severity_emoji.get(anomaly['severity'], '📋')} {anomaly['source'].title()} - {anomaly['metric']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Metric:** {anomaly['metric'].replace('_', ' ').title()}")
                            st.write(f"**Source:** {anomaly['source'].title()}")
                            st.write(f"**Severity:** {anomaly['severity'].title()}")
                            st.write(f"**Timestamp:** {anomaly['timestamp']}")
                        
                        with col2:
                            st.write(f"**Actual Value:** {anomaly['value']:.2f}")
                            expected_min, expected_max = anomaly['expected_range']
                            st.write(f"**Expected Range:** {expected_min:.2f} - {expected_max:.2f}")
                            
                            deviation = abs(anomaly['value'] - ((expected_min + expected_max) / 2))
                            st.write(f"**Deviation:** {deviation:.2f}")
            else:
                st.info("No anomalies match the selected filters.")
        else:
            st.success("✅ No anomalies detected. All systems operating within normal parameters.")
    
    def _render_reports(self):
        """Render reports tab"""
        st.subheader("📋 Analytics Reports")
        
        # Create new report
        with st.expander("📝 Create New Report"):
            col1, col2 = st.columns(2)
            
            with col1:
                report_type = st.selectbox("Report Type", ["summary", "performance", "security", "trends"])
                timeframe = st.selectbox("Timeframe", ["1h", "24h", "7d", "30d"], index=1)
            
            with col2:
                include_recommendations = st.checkbox("Include Recommendations", value=True)
                include_anomalies = st.checkbox("Include Anomalies", value=True)
            
            if st.button("Generate Report", type="primary"):
                config = {
                    "timeframe": timeframe,
                    "include_recommendations": include_recommendations,
                    "include_anomalies": include_anomalies
                }
                
                report_id = self.create_report(report_type, config)
                st.success(f"✅ Report generated: {report_id}")
                st.rerun()
        
        # List existing reports
        reports = self.list_reports()
        
        if reports:
            st.subheader("📚 Generated Reports")
            
            for report in reversed(reports):  # Show newest first
                with st.expander(f"📄 {report['type'].title()} Report - {report['id']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Type:** {report['type'].title()}")
                        st.write(f"**Created:** {report['created_at']}")
                        st.write(f"**Status:** {report['status'].title()}")
                    
                    with col2:
                        st.write(f"**Timeframe:** {report['config']['timeframe']}")
                        st.write(f"**Recommendations:** {'Yes' if report['config'].get('include_recommendations') else 'No'}")
                        st.write(f"**Anomalies:** {'Yes' if report['config'].get('include_anomalies') else 'No'}")
                    
                    # Report summary
                    insights = report['insights']
                    
                    if insights['summary']:
                        st.write("**Summary:**")
                        summary_text = f"Analyzed {len(insights['summary'])} data sources. "
                        summary_text += f"Found {len(insights['anomalies'])} anomalies and "
                        summary_text += f"generated {len(insights['recommendations'])} recommendations."
                        st.write(summary_text)
                    
                    # Download button (placeholder)
                    if st.button(f"Download Report", key=f"download_{report['id']}"):
                        st.info("Report download functionality would be implemented here.")
        else:
            st.info("No reports generated yet. Create your first report above.")
    
    def _render_settings(self):
        """Render settings tab"""
        st.subheader("⚙️ Analytics Settings")
        
        # Data retention settings
        with st.expander("📊 Data Retention"):
            st.write("Configure how long to keep analytics data:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                metrics_retention = st.selectbox("Metrics Data", ["1 day", "7 days", "30 days", "90 days"], index=2)
                reports_retention = st.selectbox("Reports", ["30 days", "90 days", "1 year", "Forever"], index=1)
            
            with col2:
                anomaly_retention = st.selectbox("Anomaly Data", ["7 days", "30 days", "90 days"], index=1)
                max_metrics_per_source = st.number_input("Max Metrics per Source", min_value=100, max_value=10000, value=1000)
            
            if st.button("Save Retention Settings"):
                st.success("✅ Retention settings saved!")
        
        # Alert thresholds
        with st.expander("🚨 Alert Thresholds"):
            st.write("Configure thresholds for automatic alerts:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cpu_threshold = st.slider("CPU Usage Alert (%)", 0, 100, 80)
                memory_threshold = st.slider("Memory Usage Alert (%)", 0, 100, 85)
            
            with col2:
                anomaly_sensitivity = st.selectbox("Anomaly Detection Sensitivity", ["Low", "Medium", "High"], index=1)
                alert_frequency = st.selectbox("Alert Frequency", ["Immediate", "Every 5 min", "Every 15 min", "Hourly"], index=1)
            
            if st.button("Save Alert Settings"):
                st.success("✅ Alert settings saved!")
        
        # Data sources
        with st.expander("🔌 Data Sources"):
            st.write("Configure data collection sources:")
            
            sources = ["Azure", "GCP", "FortiGate", "AWS", "On-Premises"]
            
            for source in sources:
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**{source}**")
                
                with col2:
                    enabled = st.checkbox("Enabled", key=f"source_{source.lower()}", value=source in ["Azure", "GCP", "FortiGate"])
                
                with col3:
                    if enabled:
                        st.write("✅ Active")
                    else:
                        st.write("⏸️ Disabled")
            
            if st.button("Save Source Settings"):
                st.success("✅ Data source settings saved!")
        
        # System status
        with st.expander("🔧 System Status"):
            st.write("Analytics engine system information:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Total Data Sources:** {len(self.metrics_data)}")
                st.write(f"**Total Metrics:** {sum(len(data) for data in self.metrics_data.values())}")
                st.write(f"**Generated Reports:** {len(self.reports)}")
            
            with col2:
                st.write(f"**Active Alerts:** {len(self.alerts)}")
                st.write(f"**Engine Status:** ✅ Running")
                st.write(f"**Last Update:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # System actions
            col3, col4 = st.columns(2)
            
            with col3:
                if st.button("Clear All Data", type="secondary"):
                    self.metrics_data.clear()
                    self.reports.clear()
                    self.alerts.clear()
                    st.success("✅ All data cleared!")
                    st.rerun()
            
            with col4:
                if st.button("Regenerate Sample Data", type="secondary"):
                    self.generate_sample_data()
                    st.success("✅ Sample data regenerated!")
                    st.rerun()
