"""
LangChain Agent Core for RAG System
Main agent class that orchestrates RAG operations
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st

try:
    from langchain.agents import AgentType, initialize_agent, Tool
    from langchain.agents.agent_types import AgentType
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.schema import Document
    from langchain.callbacks import StreamlitCallbackHandler
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False

from .config import get_rag_config
from .vector_store_manager import get_vector_store_manager

logger = logging.getLogger(__name__)

class RAGAgent:
    """LangChain-based RAG Agent for document Q&A"""
    
    def __init__(self):
        self.config = get_rag_config()
        self.vector_store_manager = get_vector_store_manager()
        self.llm = self._initialize_llm()
        self.embeddings = self._initialize_embeddings()
        self.memory = self._initialize_memory()
        self.tools = self._create_tools()
        self.agent = self._initialize_agent()
    
    def _initialize_llm(self):
        """Initialize the language model"""
        try:
            if not LANGCHAIN_OPENAI_AVAILABLE:
                st.error("LangChain OpenAI not available. Install: pip install langchain-openai")
                return None
            
            # Check for OpenAI API key
            openai_api_key = st.session_state.get('openai_api_key') or st.secrets.get('openai', {}).get('api_key')
            if not openai_api_key:
                st.warning("OpenAI API key not found. Please configure in settings.")
                return None
            
            llm = ChatOpenAI(
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.max_tokens,
                openai_api_key=openai_api_key,
                streaming=True
            )
            
            logger.info(f"Initialized LLM: {self.config.llm_model}")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            st.error(f"Failed to initialize LLM: {e}")
            return None
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        try:
            if not LANGCHAIN_OPENAI_AVAILABLE:
                return None
            
            openai_api_key = st.session_state.get('openai_api_key') or st.secrets.get('openai', {}).get('api_key')
            if not openai_api_key:
                return None
            
            embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                openai_api_key=openai_api_key
            )
            
            logger.info(f"Initialized embeddings: {self.config.embedding_model}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            return None
    
    def _initialize_memory(self):
        """Initialize conversation memory"""
        try:
            if not LANGCHAIN_AVAILABLE:
                return None
            
            memory = ConversationBufferWindowMemory(
                k=5,  # Remember last 5 exchanges
                memory_key="chat_history",
                return_messages=True
            )
            
            logger.info("Initialized conversation memory")
            return memory
            
        except Exception as e:
            logger.error(f"Failed to initialize memory: {e}")
            return None
    
    def _create_tools(self) -> List:
        """Create tools for the agent"""
        tools = []
        
        if not LANGCHAIN_AVAILABLE:
            return tools
        
        try:
            # Document search tool
            search_tool = Tool(
                name="document_search",
                description="Search through uploaded documents to find relevant information. Use this when users ask questions about the content of uploaded files.",
                func=self._document_search_tool
            )
            tools.append(search_tool)
            
            # Vector store stats tool
            stats_tool = Tool(
                name="knowledge_base_stats",
                description="Get statistics about the knowledge base including number of documents and storage info.",
                func=self._knowledge_base_stats_tool
            )
            tools.append(stats_tool)
            
            logger.info(f"Created {len(tools)} tools for agent")
            
        except Exception as e:
            logger.error(f"Failed to create tools: {e}")
        
        return tools
    
    def _document_search_tool(self, query: str) -> str:
        """Tool function for document search"""
        try:
            results = self.vector_store_manager.similarity_search(query, k=self.config.retrieval_k)
            
            if not results:
                return "No relevant documents found in the knowledge base."
            
            # Format results for the agent
            formatted_results = []
            for i, result in enumerate(results, 1):
                content = result.get('content', '')[:500] + "..." if len(result.get('content', '')) > 500 else result.get('content', '')
                metadata = result.get('metadata', {})
                source = metadata.get('source', 'Unknown')
                
                formatted_results.append(f"""
Document {i}:
Source: {source}
Content: {content}
Relevance Score: {result.get('score', 0.0):.2f}
""")
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Document search tool error: {e}")
            return f"Error searching documents: {str(e)}"
    
    def _knowledge_base_stats_tool(self, query: str) -> str:
        """Tool function for knowledge base statistics"""
        try:
            stats = self.vector_store_manager.get_stats()
            
            return f"""
Knowledge Base Statistics:
- Total Documents: {stats.get('total_documents', 0)}
- Vector Store Type: {stats.get('vector_store_type', 'Unknown')}
- Collection/Index Name: {stats.get('collection_name', stats.get('index_name', 'Unknown'))}
"""
        except Exception as e:
            logger.error(f"Knowledge base stats tool error: {e}")
            return f"Error getting knowledge base stats: {str(e)}"
    
    def _initialize_agent(self):
        """Initialize the LangChain agent"""
        try:
            if not LANGCHAIN_AVAILABLE or not self.llm or not self.tools:
                return None
            
            agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=3
            )
            
            logger.info("Initialized LangChain agent")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            return None
    
    def query(self, question: str, use_streaming: bool = True) -> str:
        """Query the RAG agent"""
        try:
            if not self.agent:
                return "❌ Agent not available. Please check your configuration and API keys."
            
            # Prepare the query with context
            enhanced_query = f"""
Based on the uploaded documents in the knowledge base, please answer this question: {question}

Please provide:
1. A direct answer to the question
2. References to the source documents
3. Any relevant context from the documents

If the information is not available in the uploaded documents, please state that clearly.
"""
            
            # Use streaming callback if in Streamlit
            if use_streaming and hasattr(st, 'empty'):
                st_callback = StreamlitCallbackHandler(st.container())
                response = self.agent.run(enhanced_query, callbacks=[st_callback])
            else:
                response = self.agent.run(enhanced_query)
            
            return response
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            return f"❌ Error processing query: {str(e)}"
    
    def add_documents_to_knowledge_base(self, documents: List[str], metadatas: List[Dict], ids: List[str]) -> bool:
        """Add documents to the knowledge base"""
        try:
            self.vector_store_manager.add_documents(documents, metadatas, ids)
            logger.info(f"Added {len(documents)} documents to knowledge base")
            return True
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        return self.vector_store_manager.get_stats()
    
    def clear_memory(self) -> None:
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
            logger.info("Cleared conversation memory")
    
    def health_check(self) -> Dict[str, Any]:
        """Check agent health status"""
        health_status = {
            "langchain_available": LANGCHAIN_AVAILABLE,
            "langchain_openai_available": LANGCHAIN_OPENAI_AVAILABLE,
            "llm_initialized": self.llm is not None,
            "embeddings_initialized": self.embeddings is not None,
            "agent_initialized": self.agent is not None,
            "vector_store_health": self.vector_store_manager.health_check()
        }
        
        overall_health = all([
            health_status["langchain_available"],
            health_status["llm_initialized"],
            health_status["agent_initialized"],
            health_status["vector_store_health"]["status"] == "healthy"
        ])
        
        health_status["overall_status"] = "healthy" if overall_health else "unhealthy"
        
        return health_status

# Global agent instance
_rag_agent = None

def get_rag_agent() -> RAGAgent:
    """Get the global RAG agent instance"""
    global _rag_agent
    if _rag_agent is None:
        _rag_agent = RAGAgent()
    return _rag_agent

def reset_rag_agent():
    """Reset the global RAG agent instance"""
    global _rag_agent
    _rag_agent = None
