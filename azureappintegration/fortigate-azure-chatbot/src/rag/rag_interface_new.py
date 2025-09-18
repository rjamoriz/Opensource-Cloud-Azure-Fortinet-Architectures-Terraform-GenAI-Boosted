"""
Enhanced RAG Interface with LangChain Agent Integration
Complete Streamlit interface for the RAG Agent system
"""

import streamlit as st
import logging
from typing import Dict, Any, Optional, List
import json
from datetime import datetime

# RAG components
from .config import get_rag_config, ConfigManager
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
        """Render the complete RAG interface"""
        try:
            st.title("🤖 RAG Knowledge Agent")
            st.markdown("Advanced Retrieval-Augmented Generation with LangChain")
            
            # Configuration and Status
            self._render_configuration_section()
            
            # Main interface tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "💬 Chat with Agent", 
                "📄 Upload Documents", 
                "📚 Manage Knowledge Base", 
                "⚙️ Advanced Settings"
            ])
            
            with tab1:
                self._render_chat_interface()
            
            with tab2:
                self._render_upload_interface()
            
            with tab3:
                self._render_knowledge_management()
            
            with tab4:
                self._render_advanced_settings()
            
        except Exception as e:
            logger.error(f"Failed to render RAG interface: {e}")
            st.error(f"Interface error: {e}")
    
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

# Global instance
_rag_interface = None

def get_rag_interface() -> RAGInterface:
    """Get the global RAG interface instance"""
    global _rag_interface
    if _rag_interface is None:
        _rag_interface = RAGInterface()
    return _rag_interface

def render_rag_knowledge_component():
    """Main function to render the RAG Knowledge component"""
    try:
        interface = get_rag_interface()
        interface.render_interface()
    except Exception as e:
        logger.error(f"Failed to render RAG Knowledge component: {e}")
        st.error(f"RAG Knowledge component error: {e}")
        
        # Show configuration help
        st.subheader("Configuration Help")
        st.code("""
# Set these environment variables:
OPENAI_API_KEY=your_openai_api_key
RAG_VECTOR_STORE_TYPE=chromadb  # or pinecone
        """)

# Legacy support
def display_rag_interface():
    """Legacy function name support"""
    render_rag_knowledge_component()
