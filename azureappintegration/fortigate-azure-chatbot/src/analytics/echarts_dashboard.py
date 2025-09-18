"""
Apache ECharts Dashboard for FortiGate Azure Chatbot Analytics
Provides comprehensive visualization of system effectiveness metrics
"""

import streamlit as st
import json
from datetime import datetime, timedelta
import random
from typing import Dict, List, Any
import os

class EChartsDashboard:
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state for analytics"""
        if 'analytics_data' not in st.session_state:
            st.session_state.analytics_data = {
                'rag_metrics': [],
                'voice_metrics': [],
                'agent_metrics': [],
                'vector_store_metrics': [],
                'deployment_metrics': [],
                'user_satisfaction': [],
                'system_performance': []
            }
        
        if 'chart_refresh_interval' not in st.session_state:
            st.session_state.chart_refresh_interval = 30  # seconds
    
    def render_dashboard(self):
        """Render the main analytics dashboard"""
        st.markdown("# 📊 Analytics Dashboard")
        st.markdown("**Real-time effectiveness metrics and performance visualization**")
        
        # Dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Overview", 
            "🧠 RAG Analytics", 
            "🎤 Voice Metrics", 
            "🤖 Multi-Agent", 
            "⚡ Performance"
        ])
        
        with tab1:
            self._render_overview_dashboard()
        
        with tab2:
            self._render_rag_analytics()
        
        with tab3:
            self._render_voice_analytics()
        
        with tab4:
            self._render_agent_analytics()
        
        with tab5:
            self._render_performance_analytics()
    
    def _render_overview_dashboard(self):
        """Render overview dashboard with key metrics"""
        st.markdown("### 🎯 System Overview")
        
        # Key Performance Indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self._render_kpi_card("Total Queries", 1247, "📈", "+12%")
        
        with col2:
            self._render_kpi_card("Avg Response Time", "1.2s", "⚡", "-8%")
        
        with col3:
            self._render_kpi_card("User Satisfaction", "94%", "😊", "+5%")
        
        with col4:
            self._render_kpi_card("System Uptime", "99.8%", "🟢", "+0.1%")
        
        # System Health Overview Chart
        st.markdown("#### 📊 System Health Overview")
        self._render_system_health_chart()
        
        # Usage Trends
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Daily Usage Trends")
            self._render_usage_trends_chart()
        
        with col2:
            st.markdown("#### 🔄 Feature Utilization")
            self._render_feature_utilization_chart()
    
    def _render_rag_analytics(self):
        """Render RAG system analytics"""
        st.markdown("### 🧠 RAG System Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Query Response Quality")
            from analytics.chart_components import ChartComponents
            ChartComponents.render_rag_quality_chart()
        
        with col2:
            st.markdown("#### ⚡ Response Time Distribution")
            self._render_rag_response_time_chart()
        
        # Vector Store Performance
        st.markdown("#### 🔍 Vector Store Performance")
        ChartComponents.render_vector_store_performance()
    
    def _render_voice_analytics(self):
        """Render voice processing analytics"""
        st.markdown("### 🎤 Voice Processing Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🗣️ Speech Recognition Accuracy")
            from analytics.chart_components import ChartComponents
            ChartComponents.render_speech_accuracy_chart()
        
        with col2:
            st.markdown("#### 🎵 Voice Quality Metrics")
            ChartComponents.render_voice_quality_chart()
    
    def _render_agent_analytics(self):
        """Render multi-agent system analytics"""
        st.markdown("### 🤖 Multi-Agent System Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Agent Performance Comparison")
            from analytics.chart_components import ChartComponents
            ChartComponents.render_agent_performance_chart()
        
        with col2:
            st.markdown("#### 🔄 Query Routing Effectiveness")
            ChartComponents.render_query_routing_chart()
    
    def _render_performance_analytics(self):
        """Render system performance analytics"""
        st.markdown("### ⚡ System Performance Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💾 Memory Usage")
            from analytics.chart_components import ChartComponents
            ChartComponents.render_memory_usage_chart()
        
        with col2:
            st.markdown("#### 🖥️ CPU Utilization")
            ChartComponents.render_cpu_usage_chart()
    
    def _render_kpi_card(self, title: str, value: str, icon: str, change: str):
        """Render a KPI card"""
        change_color = "green" if change.startswith("+") else "red" if change.startswith("-") else "gray"
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-bottom: 10px;
        ">
            <h3 style="margin: 0; font-size: 14px;">{icon} {title}</h3>
            <h1 style="margin: 10px 0; font-size: 28px;">{value}</h1>
            <p style="margin: 0; color: {change_color}; font-weight: bold;">{change}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_system_health_chart(self):
        """Render system health overview chart"""
        chart_option = {
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["CPU", "Memory", "Response Time", "Success Rate"]},
            "xAxis": {
                "type": "category",
                "data": [f"{i}:00" for i in range(24)]
            },
            "yAxis": {"type": "value"},
            "series": [
                {
                    "name": "CPU",
                    "type": "line",
                    "data": [random.randint(20, 80) for _ in range(24)],
                    "smooth": True
                },
                {
                    "name": "Memory",
                    "type": "line",
                    "data": [random.randint(30, 70) for _ in range(24)],
                    "smooth": True
                },
                {
                    "name": "Response Time",
                    "type": "line",
                    "data": [random.randint(1, 5) for _ in range(24)],
                    "smooth": True
                },
                {
                    "name": "Success Rate",
                    "type": "line",
                    "data": [random.randint(85, 100) for _ in range(24)],
                    "smooth": True
                }
            ]
        }
        
        st.components.v1.html(
            f"""
            <div id="system_health_chart" style="width: 100%; height: 400px;"></div>
            <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.0/dist/echarts.min.js"></script>
            <script>
                var chart = echarts.init(document.getElementById('system_health_chart'));
                chart.setOption({json.dumps(chart_option)});
            </script>
            """,
            height=400
        )
    
    def _render_usage_trends_chart(self):
        """Render daily usage trends"""
        chart_option = {
            "tooltip": {"trigger": "axis"},
            "xAxis": {
                "type": "category",
                "data": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            },
            "yAxis": {"type": "value"},
            "series": [{
                "type": "bar",
                "data": [120, 200, 150, 80, 70, 110, 130],
                "itemStyle": {
                    "color": {
                        "type": "linear",
                        "x": 0, "y": 0, "x2": 0, "y2": 1,
                        "colorStops": [
                            {"offset": 0, "color": "#83bff6"},
                            {"offset": 0.5, "color": "#188df0"},
                            {"offset": 1, "color": "#188df0"}
                        ]
                    }
                }
            }]
        }
        
        st.components.v1.html(
            f"""
            <div id="usage_trends_chart" style="width: 100%; height: 300px;"></div>
            <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.0/dist/echarts.min.js"></script>
            <script>
                var chart = echarts.init(document.getElementById('usage_trends_chart'));
                chart.setOption({json.dumps(chart_option)});
            </script>
            """,
            height=300
        )
    
    def _render_feature_utilization_chart(self):
        """Render feature utilization pie chart"""
        chart_option = {
            "tooltip": {"trigger": "item"},
            "legend": {"orient": "vertical", "left": "left"},
            "series": [{
                "type": "pie",
                "radius": "50%",
                "data": [
                    {"value": 35, "name": "RAG Queries"},
                    {"value": 25, "name": "Voice Commands"},
                    {"value": 20, "name": "Multi-Agent"},
                    {"value": 15, "name": "Deployments"},
                    {"value": 5, "name": "Other"}
                ],
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowOffsetX": 0,
                        "shadowColor": "rgba(0, 0, 0, 0.5)"
                    }
                }
            }]
        }
        
        st.components.v1.html(
            f"""
            <div id="feature_util_chart" style="width: 100%; height: 300px;"></div>
            <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.0/dist/echarts.min.js"></script>
            <script>
                var chart = echarts.init(document.getElementById('feature_util_chart'));
                chart.setOption({json.dumps(chart_option)});
            </script>
            """,
            height=300
        )
    
    def _render_rag_response_time_chart(self):
        """Render RAG response time distribution chart"""
        chart_option = {
            "tooltip": {"trigger": "axis"},
            "xAxis": {
                "type": "category",
                "data": ["<0.5s", "0.5-1s", "1-2s", "2-3s", "3-5s", ">5s"]
            },
            "yAxis": {"type": "value", "name": "Number of Queries"},
            "series": [{
                "type": "bar",
                "data": [45, 120, 85, 35, 15, 5],
                "itemStyle": {
                    "color": {
                        "type": "linear",
                        "x": 0, "y": 0, "x2": 0, "y2": 1,
                        "colorStops": [
                            {"offset": 0, "color": "#73c0de"},
                            {"offset": 0.5, "color": "#3ba0dc"},
                            {"offset": 1, "color": "#5470c6"}
                        ]
                    }
                }
            }]
        }
        
        st.components.v1.html(
            f"""
            <div id="rag_response_time_chart" style="width: 100%; height: 300px;"></div>
            <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.0/dist/echarts.min.js"></script>
            <script>
                var chart = echarts.init(document.getElementById('rag_response_time_chart'));
                chart.setOption({json.dumps(chart_option)});
            </script>
            """,
            height=300
        )
