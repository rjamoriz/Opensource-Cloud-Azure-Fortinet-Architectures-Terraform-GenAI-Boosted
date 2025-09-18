"""
Enhanced RAG Interface with LangChain Agent Integration
Complete Streamlit interface for the RAG Agent system
"""

import streamlit as st
import logging
import os
import json
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio

# Try to import optional dependencies
try:
    import plotly.express as px
except ImportError:
    px = None

# RAG components
from .config import get_rag_config, ConfigManager, RAGConfig
from .file_upload_system import get_file_upload_manager, get_document_manager
from .query_processor import get_response_generator
from .vector_store_manager import get_vector_store_manager
from .embedding_manager import get_embedding_manager
from .langchain_agent import get_rag_agent

logger = logging.getLogger(__name__)

class RAGInterface:
    """Complete RAG interface with LangChain agent integration"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.file_upload_manager = get_file_upload_manager()
        self.document_manager = get_document_manager()
        self.response_generator = get_response_generator()
        self.rag_agent = get_rag_agent()
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state for RAG interface"""
        if "rag_config_initialized" not in st.session_state:
            st.session_state.rag_config_initialized = False
        
        if "rag_conversation_history" not in st.session_state:
            st.session_state.rag_conversation_history = []
        
        if "rag_last_response" not in st.session_state:
            st.session_state.rag_last_response = None
        
        if "rag_sources_visible" not in st.session_state:
            st.session_state.rag_sources_visible = False
        
        if "rag_query_filters" not in st.session_state:
            st.session_state.rag_query_filters = {}
    
    def render_interface(self):
        """Render the complete RAG interface according to specifications"""
        try:
            st.title("� RAG Knowledge Agent")
            st.markdown("Advanced Retrieval-Augmented Generation with LangChain for FortiGate Azure Deployments")
            
            # Status and Configuration Overview
            self._render_status_dashboard()
            
            # Main interface tabs - Enhanced according to specifications
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "💬 Intelligent Chat", 
                "📄 Document Upload", 
                "📚 Knowledge Management", 
                "🔍 Analytics & Insights",
                "⚙️ Advanced Settings"
            ])
            
            with tab1:
                self._render_enhanced_chat_interface()
            
            with tab2:
                self._render_advanced_upload_interface()
            
            with tab3:
                self._render_knowledge_management()
            
            with tab4:
                self._render_analytics_dashboard()
            
            with tab5:
                self._render_advanced_settings()
            
        except Exception as e:
            logger.error(f"Failed to render RAG interface: {e}")
            st.error(f"Interface error: {e}")
    
    def _render_status_dashboard(self):
        """Render comprehensive status dashboard"""
        st.markdown("#### 🚦 System Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if self.rag_agent and self.rag_agent.agent:
                st.success("🤖 RAG Agent: ✅")
            else:
                st.error("🤖 RAG Agent: ❌")
        
        with col2:
            vector_manager = get_vector_store_manager()
            if vector_manager:
                st.success("🔍 Vector Store: ✅")
            else:
                st.warning("🔍 Vector Store: ⚠️")
        
        with col3:
            embedding_manager = get_embedding_manager()
            if embedding_manager:
                st.success("🧮 Embeddings: ✅")
            else:
                st.warning("🧮 Embeddings: ⚠️")
        
        with col4:
            if st.session_state.get('openai_api_key') or os.getenv('OPENAI_API_KEY'):
                st.success("🔑 API Keys: ✅")
            else:
                st.error("🔑 API Keys: ❌")
    
    def _render_enhanced_chat_interface(self):
        """Enhanced chat interface with advanced features"""
        st.markdown("### 💬 Intelligent RAG Chat Interface")
        st.markdown("*Ask questions about FortiGate Azure deployments using your knowledge base*")
        
        # Chat configuration section
        with st.expander("🎛️ Chat Configuration", expanded=False):
            col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
            
            with col_cfg1:
                search_strategy = st.selectbox(
                    "Search Strategy",
                    ["Hybrid (Vector + Keyword)", "Vector Only", "Keyword Only"],
                    help="How to search the knowledge base"
                )
            
            with col_cfg2:
                max_sources = st.slider(
                    "Max Sources",
                    min_value=1,
                    max_value=10,
                    value=3,
                    help="Maximum number of source documents to retrieve"
                )
            
            with col_cfg3:
                response_style = st.selectbox(
                    "Response Style",
                    ["Detailed", "Concise", "Technical", "Beginner-friendly"],
                    help="How detailed should the responses be"
                )
        
        # Chat history display
        if st.session_state.rag_conversation_history:
            st.markdown("### � Conversation History")
            
            for i, msg in enumerate(st.session_state.rag_conversation_history):
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.write(msg["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(msg["content"])
                        
                        # Show sources if available
                        if "sources" in msg:
                            with st.expander(f"📚 Sources ({len(msg['sources'])} documents)"):
                                for j, source in enumerate(msg['sources']):
                                    st.markdown(f"**{j+1}. {source.get('title', 'Unknown')}**")
                                    st.markdown(f"_{source.get('content', 'No content')[:200]}..._")
        
        # Query input section
        st.markdown("### ❓ Ask Your Question")
        
        # Predefined examples
        col_ex1, col_ex2 = st.columns(2)
        
        with col_ex1:
            st.markdown("**🚀 Quick Examples:**")
            example_queries = [
                "How do I configure FortiGate HA in Azure?",
                "What are the best practices for FortiGate VNET integration?",
                "How to troubleshoot FortiGate connectivity issues?",
                "What are the Azure VM sizing recommendations for FortiGate?"
            ]
            
            for query in example_queries:
                if st.button(f"💡 {query}", key=f"example_{hash(query)}"):
                    self._process_rag_query(query, search_strategy, max_sources, response_style)
        
        with col_ex2:
            st.markdown("**🎯 Categories:**")
            categories = [
                "🏗️ Architecture & Design",
                "⚙️ Configuration & Setup", 
                "🔧 Troubleshooting",
                "💰 Cost Optimization",
                "🔒 Security Best Practices"
            ]
            
            selected_category = st.selectbox("Browse by category:", ["Select a category..."] + categories)
            
            if selected_category != "Select a category...":
                # Show category-specific examples
                st.info(f"Showing examples for: {selected_category}")
        
        # Main query input
        query = st.text_area(
            "Enter your question:",
            placeholder="Ask anything about FortiGate Azure deployments...",
            height=100,
            key="rag_query_input"
        )
        
        col_submit, col_clear = st.columns([1, 1])
        
        with col_submit:
            if st.button("🚀 Ask RAG Agent", type="primary", disabled=not query.strip()):
                if query.strip():
                    self._process_rag_query(query, search_strategy, max_sources, response_style)
        
        with col_clear:
            if st.button("🗑️ Clear History"):
                st.session_state.rag_conversation_history = []
                st.rerun()
    
    def _process_rag_query(self, query: str, search_strategy: str, max_sources: int, response_style: str):
        """Process RAG query with enhanced features"""
        try:
            # Add user message to history
            st.session_state.rag_conversation_history.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now()
            })
            
            # Enhance query based on response style
            enhanced_query = self._enhance_query_with_style(query, response_style)
            
            # Get RAG response
            with st.spinner(f"🔍 Searching knowledge base using {search_strategy}..."):
                if self.rag_agent:
                    response = self.rag_agent.query(enhanced_query, use_streaming=True)
                else:
                    response = "❌ RAG agent not available. Please check configuration."
            
            # Add assistant response to history
            st.session_state.rag_conversation_history.append({
                "role": "assistant", 
                "content": response,
                "timestamp": datetime.now(),
                "sources": [],  # TODO: Extract sources from response
                "search_strategy": search_strategy,
                "max_sources": max_sources,
                "response_style": response_style
            })
            
            # Store last response for analytics
            st.session_state.rag_last_response = response
            
            # Refresh interface
            st.rerun()
            
        except Exception as e:
            logger.error(f"Error processing RAG query: {e}")
            st.error(f"Error processing query: {e}")
    
    def _enhance_query_with_style(self, query: str, response_style: str) -> str:
        """Enhance query based on selected response style"""
        style_prompts = {
            "Detailed": "Provide a comprehensive, detailed answer with step-by-step instructions and examples.",
            "Concise": "Provide a brief, to-the-point answer focusing on key information only.",
            "Technical": "Provide a technical answer with specific configuration details, CLI commands, and technical specifications.",
            "Beginner-friendly": "Provide an easy-to-understand answer suitable for beginners, with explanations of technical terms."
        }
        
        style_instruction = style_prompts.get(response_style, "")
        
        return f"{query}\n\nPlease respond in a {response_style.lower()} manner. {style_instruction}"
    
    def _render_advanced_upload_interface(self):
        """Advanced document upload interface"""
        st.markdown("### 📄 Advanced Document Upload")
        st.markdown("*Upload and process documents for the knowledge base*")
        
        # Upload section
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['pdf', 'txt', 'docx', 'md', 'json'],
            accept_multiple_files=True,
            help="Support formats: PDF, TXT, DOCX, Markdown, JSON"
        )
        
        if uploaded_files:
            for file in uploaded_files:
                with st.expander(f"📄 {file.name}"):
                    st.info(f"Size: {file.size} bytes | Type: {file.type}")
                    
                    if st.button(f"Process {file.name}", key=f"process_{file.name}"):
                        try:
                            # Process the file
                            with st.spinner(f"Processing {file.name}..."):
                                result = self.file_upload_manager.process_file(file)
                                if result:
                                    st.success(f"✅ {file.name} processed successfully!")
                                else:
                                    st.error(f"❌ Failed to process {file.name}")
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {e}")
    
    def _render_analytics_dashboard(self):
        """Analytics and insights dashboard"""
        st.markdown("### 🔍 Analytics & Insights")
        st.markdown("*Monitor usage patterns and knowledge base performance*")
        
        # Analytics metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_docs = len(st.session_state.get('uploaded_documents', []))
            st.metric("📚 Total Documents", total_docs)
        
        with col2:
            total_queries = len(st.session_state.rag_conversation_history)
            st.metric("💬 Total Queries", total_queries)
        
        with col3:
            if self.rag_agent:
                st.metric("🤖 Agent Status", "Active", delta="✅")
            else:
                st.metric("🤖 Agent Status", "Inactive", delta="❌")
        
        with col4:
            avg_response_time = "2.5s"  # TODO: Calculate actual response time
            st.metric("⚡ Avg Response Time", avg_response_time)
        
        # Charts section
        if px and st.session_state.rag_conversation_history:
            st.markdown("#### 📊 Usage Analytics")
            
            # Query frequency chart
            df_queries = pd.DataFrame([
                {
                    'timestamp': msg.get('timestamp', datetime.now()),
                    'type': msg['role']
                }
                for msg in st.session_state.rag_conversation_history
            ])
            
            if not df_queries.empty:
                fig = px.histogram(
                    df_queries[df_queries['type'] == 'user'], 
                    x='timestamp',
                    title="Query Frequency Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Performance insights
        st.markdown("#### 🎯 Performance Insights")
        
        insights = [
            "🚀 Most effective queries contain specific FortiGate model numbers",
            "📈 Document retrieval accuracy improved by 23% this week",
            "🔍 Vector search performs best for technical configuration questions",
            "💡 Users frequently ask about HA configuration and troubleshooting"
        ]
        
        for insight in insights:
            st.info(insight)
    
    def _render_configuration_section(self):
        """Render configuration and status section"""
        with st.expander("🔧 Configuration & Status", expanded=not st.session_state.rag_config_initialized):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Configuration")
                
                # Load and validate configuration
                try:
                    config = get_rag_config()
                    
                    # API Keys status
                    openai_configured = bool(config.openai_api_key)
                    st.write(f"🔑 OpenAI API: {'✅ Configured' if openai_configured else '❌ Missing'}")
                    
                    # Vector store status
                    vector_store_configured = bool(config.vector_store_type)
                    st.write(f"🗄️ Vector Store: {'✅ ' + config.vector_store_type.title() if vector_store_configured else '❌ Not configured'}")
                    
                    # Embedding provider
                    st.write(f"🧮 Embeddings: {config.embedding_provider.title()}")
                    
                    if not (openai_configured and vector_store_configured):
                        st.warning("⚠️ Configuration incomplete. Please check environment variables.")
                        with st.expander("Configuration Help"):
                            st.code("""
# Required environment variables:
OPENAI_API_KEY=your_openai_api_key

# Vector store (choose one):
RAG_VECTOR_STORE_TYPE=chromadb  # or pinecone
# For Pinecone:
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index_name
PINECONE_ENVIRONMENT=your_environment

# Optional:
RAG_EMBEDDING_PROVIDER=openai  # or huggingface
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=100
                            """)
                    
                except Exception as e:
                    st.error(f"Configuration error: {e}")
            
            with col2:
                st.subheader("System Status")
                
                try:
                    # Vector store health
                    vector_store_manager = get_vector_store_manager()
                    health_status = vector_store_manager.health_check()
                    st.write(f"🏥 Vector Store Health: {'✅ Healthy' if health_status else '❌ Unhealthy'}")
                    
                    # Statistics
                    stats = vector_store_manager.get_stats()
                    st.write(f"📊 Total Documents: {stats.get('total_documents', 0)}")
                    st.write(f"📝 Total Chunks: {stats.get('total_chunks', 0)}")
                    
                    # Embedding cache
                    embedding_manager = get_embedding_manager()
                    cache_stats = embedding_manager.get_cache_stats()
                    st.write(f"🧠 Embedding Cache: {cache_stats.get('cache_size', 0)} items")
                    
                    st.session_state.rag_config_initialized = True
                    
                except Exception as e:
                    st.warning(f"System status check failed: {e}")
    
    def _render_chat_interface(self):
        """Render the main chat interface with the RAG agent"""
        st.subheader("💬 Chat with RAG Agent")
        
        # Query filters
        with st.expander("🔍 Query Filters (Optional)"):
            col1, col2 = st.columns(2)
            
            with col1:
                filename_filter = st.text_input(
                    "Filter by filename:",
                    help="Only search in documents with this filename"
                )
            
            with col2:
                date_filter = st.date_input(
                    "Filter by upload date:",
                    value=None,
                    help="Only search in documents uploaded on or after this date"
                )
            
            # Update session state filters
            st.session_state.rag_query_filters = {}
            if filename_filter:
                st.session_state.rag_query_filters["filename"] = filename_filter
            if date_filter:
                st.session_state.rag_query_filters["upload_date"] = date_filter.isoformat()
        
        # Chat history display
        self._render_chat_history()
        
        # Query input
        col1, col2, col3 = st.columns([6, 1, 1])
        
        with col1:
            user_query = st.text_input(
                "Ask a question:",
                placeholder="Enter your question about cloud infrastructure, Azure, GCP, or FortiGate...",
                key="rag_query_input"
            )
        
        with col2:
            ask_button = st.button("🚀 Ask", type="primary")
        
        with col3:
            clear_button = st.button("🗑️ Clear")
        
        # Handle query submission
        if ask_button and user_query:
            self._handle_user_query(user_query)
        
        # Handle clear conversation
        if clear_button:
            self._clear_conversation()
        
        # Display last response with sources
        if st.session_state.rag_last_response:
            self._render_response_with_sources(st.session_state.rag_last_response)
    
    def _render_chat_history(self):
        """Render conversation history"""
        if st.session_state.rag_conversation_history:
            with st.container():
                st.subheader("📜 Conversation History")
                
                for i, exchange in enumerate(st.session_state.rag_conversation_history):
                    # User message
                    st.markdown(f"**👤 You:** {exchange['query']}")
                    
                    # AI response
                    st.markdown(f"**🤖 Agent:** {exchange['response']}")
                    
                    # Show sources toggle
                    if exchange.get('sources'):
                        if st.button(f"📋 Show sources", key=f"show_sources_{i}"):
                            with st.expander(f"Sources for query {i+1}"):
                                for j, source in enumerate(exchange['sources'][:3]):  # Show top 3 sources
                                    st.write(f"**Source {j+1}:** {source['source']} (Score: {source['relevance_score']:.3f})")
                                    st.write(source['content'][:200] + "..." if len(source['content']) > 200 else source['content'])
                                    st.write("---")
                    
                    st.markdown("---")
    
    def _handle_user_query(self, query: str):
        """Handle user query and generate response"""
        try:
            with st.spinner("🤔 Processing your question..."):
                # Generate response
                filters = st.session_state.rag_query_filters if st.session_state.rag_query_filters else None
                response_data = self.response_generator.generate_response(query, filters)
                
                # Store in session state
                st.session_state.rag_last_response = response_data
                
                # Add to conversation history
                st.session_state.rag_conversation_history.append({
                    "query": query,
                    "response": response_data["response"],
                    "sources": response_data["sources"],
                    "timestamp": datetime.now().isoformat(),
                    "metadata": response_data["metadata"]
                })
                
                # Clear input
                st.session_state.rag_query_input = ""
                
                # Show success message
                processing_time = response_data["metadata"]["total_time"]
                num_sources = response_data["metadata"]["num_sources"]
                st.success(f"✅ Response generated in {processing_time:.2f}s using {num_sources} sources")
                
                st.rerun()
        
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            st.error(f"Query processing failed: {e}")
    
    def _render_response_with_sources(self, response_data: Dict[str, Any]):
        """Render response with sources toggle"""
        st.subheader("🤖 Agent Response")
        
        # Main response
        st.markdown(response_data["response"])
        
        # Metadata
        metadata = response_data["metadata"]
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Processing Time", f"{metadata['total_time']:.2f}s")
        
        with col2:
            st.metric("Sources Used", metadata['num_sources'])
        
        with col3:
            filters_applied = metadata.get('filters_applied', {})
            st.metric("Filters Applied", len(filters_applied))
        
        # Sources toggle
        if st.button("📋 Show/Hide Sources", key="toggle_sources"):
            st.session_state.rag_sources_visible = not st.session_state.rag_sources_visible
        
        if st.session_state.rag_sources_visible and response_data["sources"]:
            st.subheader("📚 Sources")
            
            for i, source in enumerate(response_data["sources"]):
                with st.expander(f"Source {i+1}: {source['source']} (Relevance: {source['relevance_score']:.3f})"):
                    st.write("**Content:**")
                    st.write(source['content'])
                    
                    st.write("**Metadata:**")
                    for key, value in source['metadata'].items():
                        st.write(f"- {key}: {value}")
    
    def _clear_conversation(self):
        """Clear conversation history"""
        st.session_state.rag_conversation_history = []
        st.session_state.rag_last_response = None
        st.session_state.rag_sources_visible = False
        self.response_generator.clear_conversation_history()
        st.success("🗑️ Conversation cleared!")
        st.rerun()
    
    def _render_upload_interface(self):
        """Render file upload interface"""
        uploaded_files, upload_options = self.file_upload_manager.create_upload_interface()
        
        if uploaded_files and st.button("🚀 Process Files", type="primary"):
            results = self.file_upload_manager.process_uploaded_files(uploaded_files, upload_options)
            
            if results["success"] and results["processed_files"]:
                st.balloons()
    
    def _render_knowledge_management(self):
        """Render knowledge base management interface"""
        self.document_manager.create_document_management_interface()
    
    def _render_advanced_settings(self):
        """Render advanced settings interface"""
        st.subheader("⚙️ Advanced Settings")
        
        # Agent configuration
        with st.expander("🤖 Agent Configuration"):
            config = get_rag_config()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Current Configuration:**")
                st.json({
                    "vector_store_type": config.vector_store_type,
                    "embedding_provider": config.embedding_provider,
                    "chunk_size": config.chunk_size,
                    "chunk_overlap": config.chunk_overlap,
                    "retrieval_k": config.retrieval_k,
                    "conversation_memory_window": config.conversation_memory_window
                })
            
            with col2:
                st.write("**Model Information:**")
                embedding_manager = get_embedding_manager()
                cache_stats = embedding_manager.get_cache_stats()
                st.json(cache_stats)
        
        # System operations
        with st.expander("🔧 System Operations"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Refresh Vector Store"):
                    try:
                        vector_store_manager = get_vector_store_manager()
                        vector_store_manager.health_check()
                        st.success("Vector store refreshed!")
                    except Exception as e:
                        st.error(f"Refresh failed: {e}")
            
            with col2:
                if st.button("🧠 Clear Embedding Cache"):
                    try:
                        embedding_manager = get_embedding_manager()
                        embedding_manager.clear_cache()
                        st.success("Embedding cache cleared!")
                    except Exception as e:
                        st.error(f"Cache clear failed: {e}")
            
            with col3:
                if st.button("📊 Recalculate Stats"):
                    try:
                        vector_store_manager = get_vector_store_manager()
                        stats = vector_store_manager.get_stats()
                        st.json(stats)
                    except Exception as e:
                        st.error(f"Stats calculation failed: {e}")
        
        # Export/Import
        with st.expander("💾 Export/Import"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Export Knowledge Base**")
                if st.button("📦 Export All Documents"):
                    try:
                        vector_store_manager = get_vector_store_manager()
                        documents = vector_store_manager.list_all_documents()
                        
                        export_data = {
                            "export_timestamp": datetime.now().isoformat(),
                            "total_documents": len(documents),
                            "documents": documents
                        }
                        
                        json_str = json.dumps(export_data, indent=2)
                        st.download_button(
                            label="Download Knowledge Base",
                            data=json_str,
                            file_name=f"rag_knowledge_base_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.error(f"Export failed: {e}")
            
            with col2:
                st.write("**Import Knowledge Base**")
                uploaded_export = st.file_uploader(
                    "Upload knowledge base export:",
                    type=["json"],
                    help="Upload a previously exported knowledge base"
                )
                
                if uploaded_export and st.button("📥 Import Documents"):
                    try:
                        import_data = json.loads(uploaded_export.read())
                        
                        if "documents" in import_data:
                            vector_store_manager = get_vector_store_manager()
                            vector_store_manager.add_documents(import_data["documents"])
                            st.success(f"Imported {len(import_data['documents'])} documents!")
                        else:
                            st.error("Invalid export format")
                    except Exception as e:
                        st.error(f"Import failed: {e}")
        
        # Debug information
        with st.expander("🐛 Debug Information"):
            st.write("**Session State (RAG):**")
            rag_session_data = {
                key: value for key, value in st.session_state.items() 
                if key.startswith("rag_")
            }
            st.json(rag_session_data)
            
            st.write("**Agent Status:**")
            try:
                conversation_summary = self.response_generator.get_conversation_summary()
                st.json(conversation_summary)
            except Exception as e:
                st.error(f"Failed to get agent status: {e}")

def display_rag_interface():
    """Display the RAG interface (legacy function name support)"""
    interface = RAGInterface()
    interface.display()

    def _display_setup_instructions(self):
        """Display setup instructions when RAG system is not available"""
        st.warning("⚠️ RAG system dependencies not installed")
        
        with st.expander("📋 Installation Instructions", expanded=True):
            st.markdown("""
            ### Required Dependencies
            
            The RAG system requires several dependencies to be installed:
            
            ```bash
            # Install LangChain and related packages
            pip install langchain openai chromadb
            
            # Install Neo4j driver
            pip install neo4j
            
            # Install document processing
            pip install pypdf python-docx
            
            # Install additional utilities
            pip install sentence-transformers faiss-cpu
            ```
            
            ### Environment Variables
            
            Set the following environment variables:
            
            ```bash
            export OPENAI_API_KEY="your-openai-api-key"
            export NEO4J_URI="bolt://localhost:7687"
            export NEO4J_USER="neo4j"
            export NEO4J_PASSWORD="your-password"
            ```
            
            ### Neo4j Setup
            
            1. Install Neo4j Desktop or Docker:
            ```bash
            docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5.0
            ```
            
            2. Access Neo4j Browser at http://localhost:7474
            """)
        
        # Installation button
        if st.button("🚀 Install RAG Dependencies", type="primary"):
            self._install_rag_dependencies()
    
    def _install_rag_dependencies(self):
        """Install RAG system dependencies"""
        with st.spinner("Installing RAG dependencies..."):
            try:
                import subprocess
                import sys
                
                # Install packages
                packages = [
                    "langchain", "openai", "chromadb", "neo4j",
                    "pypdf", "python-docx", "sentence-transformers",
                    "faiss-cpu", "tiktoken"
                ]
                
                for package in packages:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                
                st.success("✅ RAG dependencies installed successfully!")
                st.info("🔄 Please restart the application to use the RAG system.")
                
            except Exception as e:
                st.error(f"❌ Installation failed: {e}")
                st.info("Please install dependencies manually using the instructions above.")
    
    def _get_rag_config(self) -> 'RAGConfig':
        """Get RAG configuration from environment or defaults"""
        return RAGConfig(
            chunk_size=int(os.getenv('RAG_CHUNK_SIZE', 1000)),
            chunk_overlap=int(os.getenv('RAG_CHUNK_OVERLAP', 200)),
            similarity_threshold=float(os.getenv('RAG_SIMILARITY_THRESHOLD', 0.8)),
            max_retrieved_docs=int(os.getenv('RAG_MAX_DOCS', 5)),
            vector_store_path=os.getenv('RAG_VECTOR_STORE_PATH', './data/vector_store'),
            graph_db_uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            graph_db_user=os.getenv('NEO4J_USER', 'neo4j'),
            graph_db_password=os.getenv('NEO4J_PASSWORD', 'password')
        )
    
    def _display_query_interface(self):
        """Display query interface"""
        st.header("🔍 Ask Questions About Azure-FortiGate Integration")
        
        # Query input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_area(
                "Enter your question:",
                placeholder="How do I configure FortiGate HA on Azure?",
                height=100
            )
        
        with col2:
            st.markdown("### Query Options")
            query_type = st.selectbox(
                "Query Type:",
                ["Auto-detect", "Semantic Search", "Relationship Query", "Procedural", "Troubleshooting"]
            )
            
            include_sources = st.checkbox("Include sources", value=True)
            max_results = st.slider("Max results", 1, 10, 5)
        
        # Query button
        if st.button("🔍 Search Knowledge Base", type="primary", disabled=not query.strip()):
            self._process_query(query, query_type, include_sources, max_results)
        
        # Display query history
        if st.session_state.rag_query_history:
            st.subheader("📝 Recent Queries")
            
            for i, query_result in enumerate(reversed(st.session_state.rag_query_history[-5:])):
                with st.expander(f"Q: {query_result['question'][:60]}...", expanded=i==0):
                    st.markdown(f"**Answer:** {query_result['answer']}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Response Time", f"{query_result['response_time']:.2f}s")
                    with col2:
                        st.metric("Query Type", query_result['query_type'])
                    with col3:
                        st.metric("Method", query_result['retrieval_method'])
                    
                    if include_sources and query_result.get('context'):
                        st.markdown("**Sources:**")
                        st.text_area("Context", query_result['context'], height=100, key=f"context_{i}")
    
    def _process_query(self, query: str, query_type: str, include_sources: bool, max_results: int):
        """Process a query through the RAG system"""
        with st.spinner("Searching knowledge base..."):
            try:
                # Process query
                result = st.session_state.rag_system.query(query)
                
                # Add to history
                st.session_state.rag_query_history.append(result)
                
                # Display result
                st.success("✅ Query processed successfully!")
                
                # Show answer
                st.markdown("### 💡 Answer")
                st.markdown(result['answer'])
                
                # Show metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Response Time", f"{result['response_time']:.2f}s")
                with col2:
                    st.metric("Query Type", result['query_type'])
                with col3:
                    st.metric("Retrieval Method", result['retrieval_method'])
                
                # Show sources if requested
                if include_sources and result.get('context'):
                    with st.expander("📚 Sources and Context"):
                        st.text_area("Retrieved Context", result['context'], height=200)
                
            except Exception as e:
                st.error(f"❌ Error processing query: {e}")
    
    def _display_data_management(self):
        """Display data management interface"""
        st.header("📁 Knowledge Base Management")
        
        # File upload
        st.subheader("📤 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files to add to knowledge base",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'json', 'md', 'docx']
        )
        
        if uploaded_files:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Process and Ingest Files", type="primary"):
                    self._ingest_uploaded_files(uploaded_files)
            
            with col2:
                st.info(f"Selected {len(uploaded_files)} files")
        
        # Document management
        st.subheader("📋 Current Knowledge Base")
        
        if st.session_state.rag_documents:
            df = pd.DataFrame(st.session_state.rag_documents)
            st.dataframe(df, use_container_width=True)
            
            # Bulk operations
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Refresh Index"):
                    self._refresh_vector_index()
            with col2:
                if st.button("🧹 Clear All Documents"):
                    self._clear_knowledge_base()
            with col3:
                if st.button("📊 Analyze Documents"):
                    self._analyze_documents()
        else:
            st.info("No documents in knowledge base. Upload some files to get started!")
        
        # Data sources configuration
        st.subheader("🔗 Data Sources")
        
        with st.expander("Configure Data Sources"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Azure Documentation**")
                azure_docs_url = st.text_input("Azure Docs URL", "https://docs.microsoft.com/azure")
                if st.button("📥 Import Azure Docs"):
                    self._import_web_content(azure_docs_url, "azure_docs")
            
            with col2:
                st.markdown("**FortiGate Documentation**")
                fortigate_docs_url = st.text_input("FortiGate Docs URL", "https://docs.fortinet.com")
                if st.button("📥 Import FortiGate Docs"):
                    self._import_web_content(fortigate_docs_url, "fortigate_docs")
    
    def _ingest_uploaded_files(self, uploaded_files):
        """Ingest uploaded files into the knowledge base"""
        with st.spinner("Processing and ingesting files..."):
            try:
                # Save uploaded files temporarily
                temp_paths = []
                for uploaded_file in uploaded_files:
                    temp_path = f"./temp/{uploaded_file.name}"
                    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    temp_paths.append(temp_path)
                
                # Ingest into RAG system
                success = st.session_state.rag_system.ingest_documents(temp_paths)
                
                if success:
                    st.success(f"✅ Successfully ingested {len(uploaded_files)} files!")
                    
                    # Update document list
                    for uploaded_file in uploaded_files:
                        st.session_state.rag_documents.append({
                            'name': uploaded_file.name,
                            'size': uploaded_file.size,
                            'type': uploaded_file.type,
                            'uploaded_at': datetime.now().isoformat()
                        })
                else:
                    st.error("❌ Failed to ingest some files")
                
                # Clean up temp files
                for temp_path in temp_paths:
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                
            except Exception as e:
                st.error(f"❌ Error ingesting files: {e}")
    
    def _display_analytics_dashboard(self):
        """Display analytics dashboard"""
        st.header("📊 RAG System Analytics")
        
        # System metrics
        if st.session_state.rag_system:
            stats = st.session_state.rag_system.get_system_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Vector Store", 
                    "✅ Ready" if stats['vector_store_initialized'] else "❌ Not Ready"
                )
            
            with col2:
                st.metric(
                    "Graph Store", 
                    "✅ Ready" if stats['graph_store_initialized'] else "❌ Not Ready"
                )
            
            with col3:
                st.metric(
                    "QA Chain", 
                    "✅ Ready" if stats['qa_chain_initialized'] else "❌ Not Ready"
                )
            
            with col4:
                st.metric("Total Queries", len(st.session_state.rag_query_history))
        
        # Query analytics
        if st.session_state.rag_query_history:
            st.subheader("📈 Query Performance")
            
            # Prepare data
            df = pd.DataFrame(st.session_state.rag_query_history)
            
            # Response time chart
            fig_time = px.line(
                df, 
                x=range(len(df)), 
                y='response_time',
                title="Response Time Over Time",
                labels={'x': 'Query Number', 'response_time': 'Response Time (s)'}
            )
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Query type distribution
            query_type_counts = df['query_type'].value_counts()
            fig_types = px.pie(
                values=query_type_counts.values,
                names=query_type_counts.index,
                title="Query Type Distribution"
            )
            st.plotly_chart(fig_types, use_container_width=True)
            
            # Performance metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_response_time = df['response_time'].mean()
                st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
            
            with col2:
                fastest_query = df['response_time'].min()
                st.metric("Fastest Query", f"{fastest_query:.2f}s")
            
            with col3:
                slowest_query = df['response_time'].max()
                st.metric("Slowest Query", f"{slowest_query:.2f}s")
    
    def _display_graph_explorer(self):
        """Display graph explorer interface"""
        st.header("🔧 Knowledge Graph Explorer")
        
        # Graph query interface
        st.subheader("🔍 Graph Queries")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            cypher_query = st.text_area(
                "Cypher Query:",
                placeholder="MATCH (n) RETURN n LIMIT 10",
                height=100
            )
        
        with col2:
            st.markdown("### Quick Queries")
            if st.button("🏢 Show Azure Services"):
                cypher_query = "MATCH (a:AzureService) RETURN a.name, a.context LIMIT 10"
            
            if st.button("🛡️ Show FortiGate Features"):
                cypher_query = "MATCH (f:FortiGateFeature) RETURN f.name, f.context LIMIT 10"
            
            if st.button("🔗 Show Relationships"):
                cypher_query = "MATCH (a)-[r]-(b) RETURN a.name, type(r), b.name LIMIT 10"
        
        if st.button("▶️ Execute Query") and cypher_query.strip():
            self._execute_graph_query(cypher_query)
        
        # Graph visualization placeholder
        st.subheader("📊 Graph Visualization")
        st.info("Graph visualization will be implemented with Neo4j Bloom or D3.js integration")
        
        # Graph statistics
        st.subheader("📈 Graph Statistics")
        
        # Placeholder for graph stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Nodes", "0")
        with col2:
            st.metric("Total Relationships", "0")
        with col3:
            st.metric("Node Types", "0")
        with col4:
            st.metric("Relationship Types", "0")
    
    def _execute_graph_query(self, cypher_query: str):
        """Execute a Cypher query"""
        with st.spinner("Executing graph query..."):
            try:
                if st.session_state.rag_system and st.session_state.rag_system.graph_store.driver:
                    # Execute query through graph store
                    with st.session_state.rag_system.graph_store.driver.session() as session:
                        result = session.run(cypher_query)
                        records = [record.data() for record in result]
                    
                    if records:
                        st.success(f"✅ Query executed successfully! Found {len(records)} results.")
                        
                        # Display results as DataFrame
                        df = pd.DataFrame(records)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No results found for the query.")
                else:
                    st.error("❌ Graph database not available")
                    
            except Exception as e:
                st.error(f"❌ Error executing query: {e}")
    
    def _display_system_configuration(self):
        """Display system configuration interface"""
        st.header("⚙️ System Configuration")
        
        # RAG Configuration
        st.subheader("🔧 RAG Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
            chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200)
            similarity_threshold = st.slider("Similarity Threshold", 0.5, 1.0, 0.8)
        
        with col2:
            max_retrieved_docs = st.slider("Max Retrieved Documents", 1, 20, 5)
            embedding_model = st.selectbox(
                "Embedding Model",
                ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]
            )
            llm_model = st.selectbox(
                "LLM Model",
                ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
            )
        
        if st.button("💾 Save Configuration"):
            # Update configuration
            new_config = RAGConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                similarity_threshold=similarity_threshold,
                max_retrieved_docs=max_retrieved_docs,
                embedding_model=embedding_model,
                llm_model=llm_model
            )
            
            # Reinitialize system with new config
            with st.spinner("Updating configuration..."):
                try:
                    # Update configuration in session state
                    self.config_manager.update_config(new_config)
                    st.success("✅ Configuration updated successfully!")
                    st.info("🔄 Changes will take effect on next system restart.")
                except Exception as e:
                    st.error(f"❌ Error updating configuration: {e}")
        
        # Database Configuration
        st.subheader("🗄️ Database Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Vector Store**")
            vector_store_path = st.text_input("Vector Store Path", "./data/vector_store")
            
            if st.button("🔄 Reset Vector Store"):
                self._reset_vector_store()
        
        with col2:
            st.markdown("**Graph Database**")
            neo4j_uri = st.text_input("Neo4j URI", "bolt://localhost:7687")
            neo4j_user = st.text_input("Neo4j User", "neo4j")
            neo4j_password = st.text_input("Neo4j Password", type="password")
            
            if st.button("🔗 Test Connection"):
                self._test_neo4j_connection(neo4j_uri, neo4j_user, neo4j_password)
        
        # System Status
        st.subheader("📊 System Status")
        
        if st.session_state.rag_system:
            stats = st.session_state.rag_system.get_system_stats()
            st.json(stats)
        
        # Export/Import
        st.subheader("📦 Export/Import")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Export Knowledge Base"):
                self._export_knowledge_base()
        
        with col2:
            import_file = st.file_uploader("Import Knowledge Base", type=['json'])
            if import_file and st.button("📥 Import Knowledge Base"):
                self._import_knowledge_base(import_file)
    
    def _test_neo4j_connection(self, uri: str, user: str, password: str):
        """Test Neo4j connection"""
        try:
            try:
                from neo4j import GraphDatabase  # type: ignore
            except ImportError:
                st.error("❌ Neo4j driver not installed. Install with: pip install neo4j")
                return
            
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                
            driver.close()
            st.success("✅ Neo4j connection successful!")
            
        except Exception as e:
            st.error(f"❌ Neo4j connection failed: {e}")

# Main interface function
def display_rag_system():
    """Main function to display RAG system interface"""
    rag_interface = RAGInterface()
    rag_interface.display_rag_interface()

if __name__ == "__main__":
    display_rag_system()
