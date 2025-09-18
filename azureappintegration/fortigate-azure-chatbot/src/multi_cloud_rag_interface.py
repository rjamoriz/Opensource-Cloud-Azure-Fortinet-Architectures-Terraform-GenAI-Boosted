"""
Streamlit Interface for Multi-Cloud RAG System
Enhanced interface for the multi-cloud VM architecture assistant
"""

import streamlit as st
import asyncio
import json
import yaml
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the multi-cloud RAG system
try:
    from src.multi_cloud_rag import (
        MultiCloudRAGSystem, RAGConfig, CloudProvider, DocumentType
    )
    RAG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Multi-cloud RAG system not available: {e}")
    RAG_AVAILABLE = False

class MultiCloudRAGInterface:
    """Streamlit interface for the multi-cloud RAG system"""
    
    def __init__(self):
        self.rag_system: Optional[MultiCloudRAGSystem] = None
        self.setup_page_config()
        self.setup_session_state()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Multi-Cloud VM Architecture Assistant",
            page_icon="☁️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'rag_initialized' not in st.session_state:
            st.session_state.rag_initialized = False
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'system_stats' not in st.session_state:
            st.session_state.system_stats = {}
        if 'dark_mode' not in st.session_state:
            st.session_state.dark_mode = True  # Default to dark mode
    
    def apply_theme(self):
        """Apply dark or light theme based on session state"""
        if st.session_state.dark_mode:
            # Dark theme
            theme_css = """
            <style>
            /* Dark theme styles */
            .stApp {
                background-color: #0e1117;
                color: #fafafa;
            }
            
            .stSidebar {
                background-color: #262730;
            }
            
            .stSelectbox > div > div {
                background-color: #262730;
                color: #fafafa;
            }
            
            .stTextInput > div > div > input {
                background-color: #262730;
                color: #fafafa;
                border: 1px solid #464852;
            }
            
            .stChatMessage {
                background-color: #1e1e1e;
            }
            
            .stButton > button {
                background-color: #ff4b4b;
                color: white;
                border: none;
                border-radius: 0.25rem;
            }
            
            .stExpander {
                background-color: #262730;
                border: 1px solid #464852;
            }
            
            /* Theme toggle button positioned in top right */
            .theme-toggle {
                position: fixed;
                top: 1rem;
                right: 1rem;
                z-index: 999;
                background-color: #262730;
                border: 1px solid #464852;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                font-size: 1.2rem;
                transition: all 0.3s ease;
            }
            
            .theme-toggle:hover {
                background-color: #ff4b4b;
                transform: scale(1.1);
            }
            </style>
            """
        else:
            # Light theme
            theme_css = """
            <style>
            /* Light theme styles */
            .stApp {
                background-color: #ffffff;
                color: #262730;
            }
            
            .stSidebar {
                background-color: #f0f2f6;
            }
            
            .stSelectbox > div > div {
                background-color: #ffffff;
                color: #262730;
            }
            
            .stTextInput > div > div > input {
                background-color: #ffffff;
                color: #262730;
                border: 1px solid #cccccc;
            }
            
            .stChatMessage {
                background-color: #f8f9fa;
            }
            
            .stButton > button {
                background-color: #ff4b4b;
                color: white;
                border: none;
                border-radius: 0.25rem;
            }
            
            .stExpander {
                background-color: #ffffff;
                border: 1px solid #cccccc;
            }
            
            /* Theme toggle button positioned in top right */
            .theme-toggle {
                position: fixed;
                top: 1rem;
                right: 1rem;
                z-index: 999;
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                font-size: 1.2rem;
                transition: all 0.3s ease;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            
            .theme-toggle:hover {
                background-color: #ff4b4b;
                color: white;
                transform: scale(1.1);
            }
            </style>
            """
        
        st.markdown(theme_css, unsafe_allow_html=True)
    
    def render_theme_toggle(self):
        """Render the theme toggle button in the top right corner"""
        theme_icon = "🌙" if st.session_state.dark_mode else "☀️"
        theme_text = "Light Mode" if st.session_state.dark_mode else "Dark Mode"
        
        # Create the toggle button using HTML/CSS
        toggle_html = f"""
        <div class="theme-toggle" onclick="toggleTheme()" title="Switch to {theme_text}">
            {theme_icon}
        </div>
        
        <script>
        function toggleTheme() {{
            // Use Streamlit's JavaScript API to trigger a callback
            window.parent.postMessage({{
                type: 'streamlit:setComponentValue',
                value: 'theme_toggle_clicked'
            }}, '*');
        }}
        </script>
        """
        
        st.markdown(toggle_html, unsafe_allow_html=True)
        
        # Alternative approach using a sidebar button for the toggle
        with st.sidebar:
            st.markdown("---")
            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button(theme_icon, help=f"Switch to {theme_text}"):
                    st.session_state.dark_mode = not st.session_state.dark_mode
                    st.rerun()
            with col2:
                st.write(f"**{theme_text}**")
    
    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        st.sidebar.title("🔧 Configuration")
        
        # System status
        st.sidebar.subheader("System Status")
        if RAG_AVAILABLE:
            if st.session_state.rag_initialized:
                st.sidebar.success("✅ RAG System Ready")
            else:
                st.sidebar.warning("⚠️ RAG System Not Initialized")
        else:
            st.sidebar.error("❌ RAG System Unavailable")
        
        # Configuration sections
        with st.sidebar.expander("🔑 API Credentials", expanded=False):
            self.render_api_config()
        
        with st.sidebar.expander("⚙️ System Settings", expanded=False):
            self.render_system_config()
        
        with st.sidebar.expander("📊 Cloud Providers", expanded=False):
            self.render_cloud_config()
        
        # System actions
        st.sidebar.subheader("Actions")
        if st.sidebar.button("🚀 Initialize System", disabled=not RAG_AVAILABLE):
            self.initialize_rag_system()
        
        if st.sidebar.button("📈 Show Stats"):
            self.show_system_stats()
        
        if st.sidebar.button("🧹 Clear History"):
            st.session_state.chat_history = []
            st.rerun()
    
    def render_api_config(self):
        """Render API configuration section"""
        st.text_input(
            "OpenAI API Key",
            type="password",
            key="openai_api_key",
            help="Required for embeddings and LLM"
        )
        
        st.text_input(
            "Pinecone API Key",
            type="password",
            key="pinecone_api_key",
            help="Required for vector storage"
        )
        
        st.text_input(
            "Pinecone Environment",
            value="us-west1-gcp-free",
            key="pinecone_environment",
            help="Pinecone environment (e.g., us-west1-gcp-free)"
        )
    
    def render_system_config(self):
        """Render system configuration section"""
        st.selectbox(
            "Vector Store",
            ["pinecone", "azure_ai_search", "weaviate"],
            key="vector_store_type",
            help="Vector database for storing embeddings"
        )
        
        st.selectbox(
            "LLM Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            key="llm_model",
            help="Language model for generating responses"
        )
        
        st.slider(
            "Temperature",
            0.0, 1.0, 0.1,
            key="temperature",
            help="Controls randomness in responses"
        )
        
        st.slider(
            "Max Results",
            5, 20, 10,
            key="max_results",
            help="Maximum number of search results to retrieve"
        )
    
    def render_cloud_config(self):
        """Render cloud provider configuration"""
        st.subheader("Azure Configuration")
        st.text_input("Subscription ID", key="azure_subscription_id")
        st.text_input("Client ID", key="azure_client_id")
        st.text_input("Client Secret", type="password", key="azure_client_secret")
        st.text_input("Tenant ID", key="azure_tenant_id")
        
        st.subheader("GCP Configuration")
        st.text_input("Project ID", key="gcp_project_id")
        st.text_input("Credentials Path", key="gcp_credentials_path")
    
    def initialize_rag_system(self):
        """Initialize the RAG system with current configuration"""
        try:
            if not RAG_AVAILABLE:
                st.error("RAG system is not available. Please check dependencies.")
                return
            
            # Set API keys as environment variables
            if st.session_state.get("openai_api_key"):
                os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
            
            # Create configuration
            config = RAGConfig(
                vector_store_type=st.session_state.get("vector_store_type", "pinecone"),
                vector_store_config={
                    "api_key": st.session_state.get("pinecone_api_key", ""),
                    "environment": st.session_state.get("pinecone_environment", "us-west1-gcp-free"),
                    "index_name": "multi-cloud-vm-rag"
                },
                llm_model=st.session_state.get("llm_model", "gpt-3.5-turbo"),
                temperature=st.session_state.get("temperature", 0.1),
                max_results=st.session_state.get("max_results", 10),
                cloud_configs=self.get_cloud_configs()
            )
            
            # Initialize system
            self.rag_system = MultiCloudRAGSystem(config)
            
            with st.spinner("Initializing RAG system..."):
                # Run async initialization
                asyncio.run(self.rag_system.initialize())
            
            st.session_state.rag_initialized = True
            st.success("✅ RAG system initialized successfully!")
            
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG system: {e}")
            logger.error(f"RAG initialization error: {e}")
    
    def get_cloud_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get cloud provider configurations from session state"""
        configs = {}
        
        # Azure configuration
        if all(st.session_state.get(key) for key in [
            "azure_subscription_id", "azure_client_id", 
            "azure_client_secret", "azure_tenant_id"
        ]):
            configs["azure"] = {
                "subscription_id": st.session_state.azure_subscription_id,
                "client_id": st.session_state.azure_client_id,
                "client_secret": st.session_state.azure_client_secret,
                "tenant_id": st.session_state.azure_tenant_id
            }
        
        # GCP configuration
        if all(st.session_state.get(key) for key in [
            "gcp_project_id", "gcp_credentials_path"
        ]):
            configs["gcp"] = {
                "project_id": st.session_state.gcp_project_id,
                "credentials_path": st.session_state.gcp_credentials_path
            }
        
        return configs
    
    def show_system_stats(self):
        """Show system statistics"""
        if self.rag_system:
            try:
                stats = asyncio.run(self.rag_system.get_system_stats())
                st.session_state.system_stats = stats
                st.success("📊 System statistics updated!")
            except Exception as e:
                st.error(f"Failed to get system stats: {e}")
        else:
            st.warning("RAG system not initialized")
    
    def render_main_interface(self):
        """Render the main chat interface"""
        # Theme toggle at the top right
        col1, col2, col3 = st.columns([6, 1, 1])
        with col1:
            st.title("☁️ Multi-Cloud VM Architecture Assistant")
        with col3:
            theme_icon = "🌙" if st.session_state.dark_mode else "☀️"
            theme_label = "Dark" if not st.session_state.dark_mode else "Light"
            if st.button(f"{theme_icon} {theme_label}", key="theme_toggle_main"):
                st.session_state.dark_mode = not st.session_state.dark_mode
                st.rerun()
        
        st.markdown("Ask questions about Azure, GCP, and multi-cloud VM architecture, deployment, and best practices.")
        
        # Display system stats if available
        if st.session_state.system_stats:
            with st.expander("📊 System Statistics", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Initialized", st.session_state.system_stats.get("initialized", False))
                    st.metric("Vector Store", st.session_state.system_stats.get("vector_store_type", "N/A"))
                
                with col2:
                    kb_stats = st.session_state.system_stats.get("knowledge_base", {})
                    st.metric("Documents", kb_stats.get("total_documents", 0))
                    st.metric("Cloud Providers", len(st.session_state.system_stats.get("cloud_providers", [])))
                
                with col3:
                    vector_stats = st.session_state.system_stats.get("vector_store", {})
                    st.metric("Vector Count", vector_stats.get("total_vectors", 0))
        
        # Chat interface
        self.render_chat_interface()
    
    def render_chat_interface(self):
        """Render the chat interface"""
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    
                    # Display VM recommendations if available
                    if "vm_recommendations" in message and message["vm_recommendations"]:
                        self.render_vm_recommendations(message["vm_recommendations"])
        
        # Query input
        if prompt := st.chat_input("Ask about multi-cloud VM architecture..."):
            self.handle_user_query(prompt)
    
    def handle_user_query(self, query: str):
        """Handle user query and generate response"""
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": query,
            "timestamp": datetime.now().isoformat()
        })
        
        # Check if system is initialized
        if not st.session_state.rag_initialized or not self.rag_system:
            error_msg = "❌ Please initialize the RAG system first using the sidebar configuration."
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_msg,
                "timestamp": datetime.now().isoformat()
            })
            st.rerun()
            return
        
        # Process query
        try:
            with st.spinner("Generating response..."):
                # Get query options from sidebar
                output_format = st.sidebar.selectbox(
                    "Output Format",
                    ["markdown", "json", "yaml"],
                    key="output_format"
                )
                
                include_vm_recommendations = st.sidebar.checkbox(
                    "Include VM Recommendations",
                    value=True,
                    key="include_vm_recommendations"
                )
                
                # Process query asynchronously
                response = asyncio.run(
                    self.rag_system.process_query(
                        query=query,
                        output_format=output_format,
                        include_vm_recommendations=include_vm_recommendations
                    )
                )
                
                # Add response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response.answer,
                    "vm_recommendations": response.vm_recommendations,
                    "source_documents": response.source_documents,
                    "metadata": response.metadata,
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            error_msg = f"❌ Error processing query: {str(e)}"
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_msg,
                "timestamp": datetime.now().isoformat()
            })
            logger.error(f"Query processing error: {e}")
        
        st.rerun()
    
    def render_vm_recommendations(self, recommendations):
        """Render VM recommendations in a structured format"""
        if not recommendations:
            return
        
        st.subheader("💡 VM Recommendations")
        
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"#{i} {rec.cloud_provider.upper()} - {rec.vm_specification.name}", expanded=i==1):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Specifications:**")
                    st.write(f"- CPU Cores: {rec.vm_specification.cpu_cores}")
                    st.write(f"- Memory: {rec.vm_specification.memory_gb} GB")
                    st.write(f"- Storage: {rec.vm_specification.storage_gb} GB ({rec.vm_specification.storage_type.value})")
                    st.write(f"- Network: {rec.vm_specification.network_performance}")
                    
                    st.write("**Pricing:**")
                    cost = rec.cost_estimate
                    st.write(f"- Hourly: ${cost.get('total_hourly', 0):.4f}")
                    st.write(f"- Monthly: ${cost.get('total_monthly', 0):.2f}")
                    st.write(f"- Yearly: ${cost.get('total_yearly', 0):.2f}")
                
                with col2:
                    st.write("**Reasoning:**")
                    st.write(rec.reasoning)
                    
                    st.write("**Confidence Score:**")
                    st.progress(rec.confidence_score)
                    st.write(f"{rec.confidence_score:.2%}")
                    
                    st.write("**Suitable Workloads:**")
                    for workload in rec.vm_specification.suitable_workloads:
                        st.write(f"- {workload}")
                
                # Deployment configuration
                with st.expander("🔧 Deployment Configuration"):
                    st.json(rec.deployment_config)
    
    def run(self):
        """Run the Streamlit interface"""
        # Apply theme first
        self.apply_theme()
        
        # Render theme toggle button
        self.render_theme_toggle()
        
        # Render main interface components
        self.render_sidebar()
        self.render_main_interface()

def main():
    """Main function to run the Streamlit app"""
    interface = MultiCloudRAGInterface()
    interface.run()

if __name__ == "__main__":
    main()
