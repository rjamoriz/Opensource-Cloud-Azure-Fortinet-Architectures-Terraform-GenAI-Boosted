"""
ECharts Chart Components for Analytics Dashboard
Individual chart rendering functions for better modularity
"""

import streamlit as st
import json
import random

class ChartComponents:
    
    @staticmethod
    def render_rag_quality_chart():
        """Render RAG response quality metrics"""
        chart_option = {
            "tooltip": {"trigger": "axis"},
            "radar": {
                "indicator": [
                    {"name": "Relevance", "max": 100},
                    {"name": "Accuracy", "max": 100},
                    {"name": "Completeness", "max": 100},
                    {"name": "Clarity", "max": 100},
                    {"name": "Speed", "max": 100}
                ]
            },
            "series": [{
                "type": "radar",
                "data": [{
                    "value": [85, 92, 78, 88, 95],
                    "name": "RAG Performance"
                }]
            }]
        }
        
        st.components.v1.html(
            f"""
            <div id="rag_quality_chart" style="width: 100%; height: 300px;"></div>
            <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.0/dist/echarts.min.js"></script>
            <script>
                var chart = echarts.init(document.getElementById('rag_quality_chart'));
                chart.setOption({json.dumps(chart_option)});
            </script>
            """,
            height=300
        )
    
    @staticmethod
    def render_vector_store_performance():
        """Render vector store performance metrics"""
        chart_option = {
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["Query Latency", "Similarity Score"]},
            "xAxis": {
                "type": "category",
                "data": [f"Day {i}" for i in range(1, 8)]
            },
            "yAxis": [
                {"type": "value", "name": "Latency (ms)"},
                {"type": "value", "name": "Score", "min": 0, "max": 1}
            ],
            "series": [
                {
                    "name": "Query Latency",
                    "type": "line",
                    "data": [120, 115, 108, 95, 88, 92, 85],
                    "smooth": True
                },
                {
                    "name": "Similarity Score",
                    "type": "line",
                    "yAxisIndex": 1,
                    "data": [0.85, 0.87, 0.89, 0.91, 0.93, 0.92, 0.94],
                    "smooth": True
                }
            ]
        }
        
        st.components.v1.html(
            f"""
            <div id="vector_store_chart" style="width: 100%; height: 350px;"></div>
            <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.0/dist/echarts.min.js"></script>
            <script>
                var chart = echarts.init(document.getElementById('vector_store_chart'));
                chart.setOption({json.dumps(chart_option)});
            </script>
            """,
            height=350
        )
    
    @staticmethod
    def render_voice_quality_chart():
        """Render voice quality metrics"""
        chart_option = {
            "tooltip": {"trigger": "axis"},
            "radar": {
                "indicator": [
                    {"name": "Clarity", "max": 10},
                    {"name": "Naturalness", "max": 10},
                    {"name": "Speed", "max": 10},
                    {"name": "Emotion", "max": 10},
                    {"name": "Pronunciation", "max": 10}
                ]
            },
            "series": [{
                "type": "radar",
                "data": [
                    {"value": [8.5, 9.2, 7.8, 8.8, 9.5], "name": "OpenAI TTS"},
                    {"value": [7.8, 8.5, 8.2, 7.5, 8.8], "name": "ElevenLabs"},
                    {"value": [8.2, 8.8, 9.1, 8.2, 9.2], "name": "Cartesia"}
                ]
            }]
        }
        
        st.components.v1.html(
            f"""
            <div id="voice_quality_chart" style="width: 100%; height: 300px;"></div>
            <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.0/dist/echarts.min.js"></script>
            <script>
                var chart = echarts.init(document.getElementById('voice_quality_chart'));
                chart.setOption({json.dumps(chart_option)});
            </script>
            """,
            height=300
        )
    
    @staticmethod
    def render_agent_performance_chart():
        """Render multi-agent performance comparison"""
        chart_option = {
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["Deployment Agent", "Troubleshooting Agent", "Security Agent"]},
            "radar": {
                "indicator": [
                    {"name": "Response Time", "max": 100},
                    {"name": "Accuracy", "max": 100},
                    {"name": "User Satisfaction", "max": 100},
                    {"name": "Task Completion", "max": 100},
                    {"name": "Confidence", "max": 100}
                ]
            },
            "series": [{
                "type": "radar",
                "data": [
                    {"value": [88, 92, 85, 95, 90], "name": "Deployment Agent"},
                    {"value": [85, 88, 90, 87, 85], "name": "Troubleshooting Agent"},
                    {"value": [90, 95, 88, 92, 93], "name": "Security Agent"}
                ]
            }]
        }
        
        st.components.v1.html(
            f"""
            <div id="agent_performance_chart" style="width: 100%; height: 350px;"></div>
            <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.0/dist/echarts.min.js"></script>
            <script>
                var chart = echarts.init(document.getElementById('agent_performance_chart'));
                chart.setOption({json.dumps(chart_option)});
            </script>
            """,
            height=350
        )
    
    @staticmethod
    def render_memory_usage_chart():
        """Render memory usage over time"""
        chart_option = {
            "tooltip": {"trigger": "axis"},
            "xAxis": {
                "type": "category",
                "data": [f"{i}:00" for i in range(24)]
            },
            "yAxis": {"type": "value", "name": "Memory (GB)"},
            "series": [{
                "type": "line",
                "data": [random.uniform(2.0, 8.0) for _ in range(24)],
                "smooth": True,
                "areaStyle": {},
                "itemStyle": {"color": "#91cc75"}
            }]
        }
        
        st.components.v1.html(
            f"""
            <div id="memory_usage_chart" style="width: 100%; height: 300px;"></div>
            <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.0/dist/echarts.min.js"></script>
            <script>
                var chart = echarts.init(document.getElementById('memory_usage_chart'));
                chart.setOption({json.dumps(chart_option)});
            </script>
            """,
            height=300
        )
    
    @staticmethod
    def render_cpu_usage_chart():
        """Render CPU utilization"""
        chart_option = {
            "tooltip": {"trigger": "axis"},
            "xAxis": {
                "type": "category",
                "data": [f"{i}:00" for i in range(24)]
            },
            "yAxis": {"type": "value", "name": "CPU %"},
            "series": [{
                "type": "line",
                "data": [random.randint(10, 90) for _ in range(24)],
                "smooth": True,
                "itemStyle": {"color": "#fac858"}
            }]
        }
        
        st.components.v1.html(
            f"""
            <div id="cpu_usage_chart" style="width: 100%; height: 300px;"></div>
            <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.0/dist/echarts.min.js"></script>
            <script>
                var chart = echarts.init(document.getElementById('cpu_usage_chart'));
                chart.setOption({json.dumps(chart_option)});
            </script>
            """,
            height=300
        )
    
    @staticmethod
    def render_speech_accuracy_chart():
        """Render speech recognition accuracy"""
        chart_option = {
            "tooltip": {"trigger": "axis"},
            "xAxis": {
                "type": "category",
                "data": ["Clear", "Noisy", "Accented", "Fast", "Slow", "Whisper"]
            },
            "yAxis": {"type": "value", "min": 70, "max": 100},
            "series": [{
                "type": "bar",
                "data": [96, 78, 85, 82, 94, 71],
                "itemStyle": {
                    "color": {
                        "type": "linear",
                        "x": 0, "y": 0, "x2": 0, "y2": 1,
                        "colorStops": [
                            {"offset": 0, "color": "#ee6666"},
                            {"offset": 1, "color": "#fc8452"}
                        ]
                    }
                }
            }]
        }
        
        st.components.v1.html(
            f"""
            <div id="speech_accuracy_chart" style="width: 100%; height: 300px;"></div>
            <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.0/dist/echarts.min.js"></script>
            <script>
                var chart = echarts.init(document.getElementById('speech_accuracy_chart'));
                chart.setOption({json.dumps(chart_option)});
            </script>
            """,
            height=300
        )
    
    @staticmethod
    def render_query_routing_chart():
        """Render query routing effectiveness"""
        chart_option = {
            "tooltip": {"trigger": "item"},
            "series": [{
                "type": "sunburst",
                "data": [
                    {
                        "name": "Deployment",
                        "value": 40,
                        "children": [
                            {"name": "Azure", "value": 25},
                            {"name": "AWS", "value": 10},
                            {"name": "GCP", "value": 5}
                        ]
                    },
                    {
                        "name": "Troubleshooting",
                        "value": 35,
                        "children": [
                            {"name": "Network", "value": 15},
                            {"name": "Config", "value": 12},
                            {"name": "Performance", "value": 8}
                        ]
                    },
                    {
                        "name": "Security",
                        "value": 25,
                        "children": [
                            {"name": "Policies", "value": 15},
                            {"name": "Threats", "value": 10}
                        ]
                    }
                ],
                "radius": [0, "90%"]
            }]
        }
        
        st.components.v1.html(
            f"""
            <div id="query_routing_chart" style="width: 100%; height: 350px;"></div>
            <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.0/dist/echarts.min.js"></script>
            <script>
                var chart = echarts.init(document.getElementById('query_routing_chart'));
                chart.setOption({json.dumps(chart_option)});
            </script>
            """,
            height=350
        )
