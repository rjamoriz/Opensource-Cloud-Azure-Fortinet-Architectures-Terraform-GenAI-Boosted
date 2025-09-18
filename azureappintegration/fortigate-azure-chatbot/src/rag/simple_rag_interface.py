"""
Simple RAG Interface - TypeVar-free implementation
Temporary solution to bypass complex LangChain imports while maintaining functionality
"""

import streamlit as st
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SimpleRAGInterface:
    """Simplified RAG interface without complex type dependencies"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if "rag_conversation_history" not in st.session_state:
            st.session_state.rag_conversation_history = []
        
        if "rag_documents" not in st.session_state:
            st.session_state.rag_documents = []
    
    def render(self):
        """Render the simplified RAG interface"""
        self.display()
    
    def display(self):
        """Display the simplified RAG interface"""
        st.markdown("### 🧠 RAG Knowledge System")
        st.markdown("*Simplified interface - Enhanced version coming soon*")
        
        # Status indicators
        self._display_status()
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs([
            "💬 Chat Interface",
            "📄 Document Upload", 
            "⚙️ Settings"
        ])
        
        with tab1:
            self._display_chat_interface()
        
        with tab2:
            self._display_upload_interface()
        
        with tab3:
            self._display_settings()
    
    def _display_status(self):
        """Display system status"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            openai_key = os.getenv('OPENAI_API_KEY') or st.session_state.get('openai_api_key')
            if openai_key:
                st.success("🔑 OpenAI: ✅")
            else:
                st.error("🔑 OpenAI: ❌")
        
        with col2:
            # Check DataStax connection
            datastax_endpoint = os.getenv('DATASTAX_API_ENDPOINT') or st.session_state.get('datastax_api_endpoint')
            datastax_key = os.getenv('DATASTAX_API_KEY') or st.session_state.get('datastax_api_key')
            if datastax_endpoint and datastax_key:
                st.success("🔍 DataStax: ✅")
            else:
                st.warning("🔍 DataStax: ⚙️ Configure")
        
        with col3:
            if datastax_endpoint and datastax_key and openai_key:
                st.success("🧠 RAG Agent: ✅")
            else:
                st.info("🧠 RAG Agent: 🚧 Setup Required")
        
        with col4:
            doc_count = len(st.session_state.rag_documents)
            st.metric("📚 Documents", doc_count)
    
    def _display_chat_interface(self):
        """Display chat interface"""
        st.markdown("#### 💬 Chat with Knowledge Base")
        
        # Display conversation history
        if st.session_state.rag_conversation_history:
            st.markdown("**Conversation History:**")
            for i, exchange in enumerate(st.session_state.rag_conversation_history):
                with st.container():
                    st.markdown(f"**👤 You:** {exchange['user']}")
                    st.markdown(f"**🤖 Assistant:** {exchange['assistant']}")
                    st.markdown("---")
        
        # Query input
        query = st.text_area(
            "Ask a question about your documents:",
            placeholder="Enter your question here...",
            height=100
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("💬 Simple Query", type="primary", key="rag_simple_query"):
                if query.strip():
                    self._process_query(query, "simple")
                else:
                    st.warning("Please enter a question")
        
        # Add vector search option if DataStax is configured
        datastax_configured = (
            st.session_state.get('datastax_api_endpoint') and 
            st.session_state.get('datastax_api_key')
        )
        
        if datastax_configured:
            if st.button("🔍 Vector Search Query", key="rag_vector_query"):
                if query.strip():
                    self._process_query(query, "vector_search")
                else:
                    st.warning("Please enter a question")
        
        with col2:
            if st.button("🗑️ Clear History", key="rag_clear_history"):
                st.session_state.rag_conversation_history = []
                st.rerun()
    
    def _process_query(self, query, query_type="simple"):
        """Process user query"""
        try:
            # Generate response based on query type
            if query_type == "vector_search":
                response = self._enhanced_query_with_vector_search(query)
            else:
                response = self._generate_simple_response(query)
            
            # Add to conversation history
            st.session_state.rag_conversation_history.append({
                'user': query,
                'assistant': response,
                'timestamp': datetime.now().isoformat(),
                'query_type': query_type
            })
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing query: {e}")
    
    def _generate_simple_response(self, query):
        """Generate a simple response"""
        # Check if OpenAI is available
        openai_key = os.getenv('OPENAI_API_KEY') or st.session_state.get('openai_api_key')
        
        if not openai_key:
            return "❌ OpenAI API key not configured. Please set your API key in the settings."
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            
            # Create context from uploaded documents
            context = self._get_document_context()
            
            system_prompt = f"""You are a helpful assistant for FortiGate Azure deployments. 
            Use the following context from uploaded documents to answer questions.
            
            Context: {context}
            
            If the context doesn't contain relevant information, provide general guidance about FortiGate and Azure."""
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _get_document_context(self):
        """Get context from uploaded documents"""
        if not st.session_state.rag_documents:
            return "No documents uploaded yet."
        
        context = "Uploaded documents:\n"
        for doc in st.session_state.rag_documents[:3]:  # Limit to first 3 docs
            context += f"- {doc['name']}: {doc['content'][:200]}...\n"
        
        return context
    
    def _display_upload_interface(self):
        """Display document upload interface"""
        st.markdown("#### 📄 Upload Documents")
        
        # Vector store options
        col1, col2 = st.columns(2)
        
        with col1:
            use_vector_store = st.checkbox(
                "📊 Use DataStax Vector Store",
                value=False,
                help="Store documents in DataStax for semantic search"
            )
        
        with col2:
            if use_vector_store:
                datastax_configured = (
                    st.session_state.get('datastax_api_endpoint') and 
                    st.session_state.get('datastax_api_key')
                )
                if datastax_configured:
                    st.success("✅ DataStax Ready")
                else:
                    st.error("❌ Configure DataStax in Settings")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['txt', 'md', 'json'],
            accept_multiple_files=True,
            help="Upload text files to add to the knowledge base"
        )
        
        if uploaded_files:
            for file in uploaded_files:
                with st.expander(f"📄 {file.name}"):
                    content = str(file.read(), "utf-8")
                    st.text_area(f"Content preview:", content[:500] + "...", height=100)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button(f"Add {file.name} to Knowledge Base", key=f"add_{file.name}"):
                            # Add to session state
                            st.session_state.rag_documents.append({
                                'name': file.name,
                                'content': content,
                                'uploaded_at': datetime.now().isoformat()
                            })
                            st.success(f"✅ Added {file.name} to knowledge base!")
                            st.rerun()
                    
                    with col2:
                        if use_vector_store and st.button(f"📊 Add to Vector Store", key=f"vector_{file.name}"):
                            self._add_to_vector_store(file.name, content)
    
    def _add_to_vector_store(self, filename: str, content: str):
        """Add document to DataStax vector store"""
        try:
            datastax_configured = (
                st.session_state.get('datastax_api_endpoint') and 
                st.session_state.get('datastax_api_key')
            )
            
            if not datastax_configured:
                st.error("❌ DataStax not configured. Please configure in Settings.")
                return
            
            with st.spinner(f"Adding {filename} to vector store..."):
                # Import DataStax components
                from vector_stores.datastax_vector_store import DataStaxVectorStore, DataStaxEmbeddingManager
                
                # Initialize components
                vector_store = DataStaxVectorStore(
                    st.session_state.get('datastax_api_endpoint'),
                    st.session_state.get('datastax_api_key'),
                    st.session_state.get('datastax_keyspace', 'default_keyspace')
                )
                
                embedding_manager = DataStaxEmbeddingManager(
                    st.session_state.get('openai_api_key')
                )
                
                # Generate embedding
                embedding = embedding_manager.generate_embedding(content)
                
                if not embedding:
                    st.error("❌ Failed to generate embedding. Check OpenAI API key.")
                    return
                
                # Prepare document
                document = {
                    'id': f"{filename}_{datetime.now().timestamp()}",
                    'content': content,
                    'metadata': {
                        'filename': filename,
                        'uploaded_at': datetime.now().isoformat(),
                        'source': 'rag_interface'
                    },
                    'embedding': embedding
                }
                
                # Add to vector store
                success, message = vector_store.add_documents([document])
                
                if success:
                    st.success(f"✅ {message}")
                    # Also add to session state
                    st.session_state.rag_documents.append({
                        'name': filename,
                        'content': content,
                        'uploaded_at': datetime.now().isoformat(),
                        'vector_stored': True
                    })
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
                    
        except Exception as e:
            st.error(f"❌ Error adding to vector store: {e}")
    
    def _enhanced_query_with_vector_search(self, query: str) -> str:
        """Enhanced query processing with vector search"""
        try:
            # Check if DataStax is configured
            datastax_configured = (
                st.session_state.get('datastax_api_endpoint') and 
                st.session_state.get('datastax_api_key')
            )
            
            if not datastax_configured:
                return self._generate_simple_response(query)
            
            # Import DataStax components
            from vector_stores.datastax_vector_store import DataStaxVectorStore, DataStaxEmbeddingManager
            
            # Initialize components
            vector_store = DataStaxVectorStore(
                st.session_state.get('datastax_api_endpoint'),
                st.session_state.get('datastax_api_key'),
                st.session_state.get('datastax_keyspace', 'default_keyspace')
            )
            
            embedding_manager = DataStaxEmbeddingManager(
                st.session_state.get('openai_api_key')
            )
            
            # Generate query embedding
            query_embedding = embedding_manager.generate_embedding(query)
            
            if not query_embedding:
                return self._generate_simple_response(query)
            
            # Perform vector search
            success, results = vector_store.similarity_search(query_embedding, limit=3)
            
            if success and results:
                # Build context from vector search results
                context = "Relevant documents from vector search:\n"
                for i, result in enumerate(results):
                    similarity = result.get('similarity', 0)
                    content = result.get('content', '')[:300]
                    context += f"\n{i+1}. (Similarity: {similarity:.3f}) {content}...\n"
                
                # Generate enhanced response with vector context
                return self._generate_enhanced_response(query, context)
            else:
                return self._generate_simple_response(query)
                
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return self._generate_simple_response(query)
    
    def _generate_enhanced_response(self, query: str, context: str) -> str:
        """Generate enhanced response with vector search context"""
        openai_key = st.session_state.get('openai_api_key')
        
        if not openai_key:
            return "❌ OpenAI API key not configured."
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            
            system_prompt = f"""You are a helpful assistant for FortiGate Azure deployments. 
            Use the following context from the vector database to answer questions accurately.
            
            Vector Search Context:
            {context}
            
            Provide detailed, accurate answers based on the retrieved context. If the context doesn't contain relevant information, provide general FortiGate and Azure guidance."""
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=600,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating enhanced response: {str(e)}"
        
        # Display uploaded documents
        if st.session_state.rag_documents:
            st.markdown("#### 📚 Uploaded Documents")
            for i, doc in enumerate(st.session_state.rag_documents):
                with st.expander(f"📄 {doc['name']} (uploaded: {doc['uploaded_at'][:19]})"):
                    vector_status = "🔍 Vector Stored" if doc.get('vector_stored') else "📄 Local Only"
                    st.markdown(f"**Status:** {vector_status}")
                    st.text_area(f"Content:", doc['content'][:500] + "...", height=100, key=f"doc_content_{i}")
                    if st.button(f"Remove {doc['name']}", key=f"remove_{i}"):
                        st.session_state.rag_documents.pop(i)
                        st.rerun()
    
    def _display_settings(self):
        """Display settings interface"""
        st.markdown("#### ⚙️ RAG System Settings")
        
        # OpenAI API Key
        with st.expander("🔑 OpenAI Configuration", expanded=True):
            current_key = st.session_state.get('openai_api_key', '') or os.getenv('OPENAI_API_KEY', '')
            
            api_key = st.text_input(
                "OpenAI API Key:",
                value=current_key[:20] + "..." if current_key and len(current_key) > 20 else current_key,
                type="password",
                help="Your OpenAI API key for generating responses"
            )
            
            if st.button("💾 Save API Key", key="rag_save_api_key"):
                if api_key and api_key.startswith('sk-'):
                    st.session_state.openai_api_key = api_key
                    st.success("✅ API key saved!")
                    st.rerun()
                else:
                    st.error("❌ Invalid API key format")
        
        # DataStax Configuration
        with st.expander("🔍 DataStax Vector Database Configuration", expanded=True):
            st.markdown("**Connect to DataStax Astra DB for vector storage**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                current_endpoint = st.session_state.get('datastax_api_endpoint', '') or os.getenv('DATASTAX_API_ENDPOINT', '')
                endpoint = st.text_input(
                    "DataStax API Endpoint:",
                    value=current_endpoint,
                    placeholder="https://your-database-id-region.apps.astra.datastax.com",
                    help="Your DataStax Astra DB API endpoint"
                )
                
                current_keyspace = st.session_state.get('datastax_keyspace', 'default_keyspace')
                keyspace = st.text_input(
                    "Keyspace:",
                    value=current_keyspace,
                    help="DataStax keyspace name"
                )
            
            with col2:
                current_token = st.session_state.get('datastax_api_key', '') or os.getenv('DATASTAX_API_KEY', '')
                token = st.text_input(
                    "DataStax API Token:",
                    value=current_token[:20] + "..." if current_token and len(current_token) > 20 else current_token,
                    type="password",
                    help="Your DataStax Astra DB API token"
                )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💾 Save DataStax Config", key="save_datastax_config"):
                    if endpoint and token:
                        st.session_state.datastax_api_endpoint = endpoint
                        st.session_state.datastax_api_key = token
                        st.session_state.datastax_keyspace = keyspace
                        st.success("✅ DataStax configuration saved!")
                        st.rerun()
                    else:
                        st.error("❌ Please provide both endpoint and token")
            
            with col2:
                if st.button("🔍 Test Connection", key="test_datastax_connection"):
                    if endpoint and token:
                        with st.spinner("Testing DataStax connection..."):
                            try:
                                from vector_stores.datastax_vector_store import DataStaxVectorStore
                                vector_store = DataStaxVectorStore(endpoint, token, keyspace)
                                success, message = vector_store.test_connection()
                                
                                if success:
                                    st.success(f"✅ {message}")
                                else:
                                    st.error(f"❌ {message}")
                            except Exception as e:
                                st.error(f"❌ Connection test failed: {e}")
                    else:
                        st.warning("⚠️ Please configure endpoint and token first")
            
            with col3:
                if st.button("📊 Collection Stats", key="datastax_stats"):
                    if endpoint and token:
                        with st.spinner("Getting collection statistics..."):
                            try:
                                from vector_stores.datastax_vector_store import DataStaxVectorStore
                                vector_store = DataStaxVectorStore(endpoint, token, keyspace)
                                stats = vector_store.get_collection_stats()
                                st.json(stats)
                            except Exception as e:
                                st.error(f"❌ Failed to get stats: {e}")
                    else:
                        st.warning("⚠️ Please configure DataStax first")
        
        # System Information
        with st.expander("📊 System Information"):
            st.json({
                "documents_uploaded": len(st.session_state.rag_documents),
                "conversations": len(st.session_state.rag_conversation_history),
                "openai_configured": bool(st.session_state.get('openai_api_key') or os.getenv('OPENAI_API_KEY')),
                "status": "Simple RAG Interface Active"
            })
        
        # Enhanced RAG Setup
        with st.expander("🚀 Enhanced RAG Setup", expanded=True):
            st.markdown("""
            **🔧 Enhanced RAG System Coming Soon!**
            
            The full RAG system with advanced features is being prepared:
            
            - 🤖 **Multi-Agent Architecture** - Specialized agents for different tasks
            - 🔍 **Vector Search** - Semantic similarity search
            - 📊 **Graph RAG** - Relationship-aware knowledge retrieval
            - 🎯 **Smart Routing** - Intelligent query classification
            - 📈 **Analytics** - Performance monitoring and insights
            
            **Current Status:** Simple interface active, enhanced version in development.
            """)
            
            if st.button("🔄 Check for Enhanced RAG Updates", key="rag_check_updates"):
                st.info("Enhanced RAG system is being developed. Check back soon!")

def get_simple_rag_interface():
    """Factory function to get the simple RAG interface"""
    return SimpleRAGInterface()
