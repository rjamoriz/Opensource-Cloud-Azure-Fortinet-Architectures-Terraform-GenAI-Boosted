"""
RAG Agent Configuration Management
Handles all configuration settings for the RAG system
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
import streamlit as st

@dataclass
class RAGConfig:
    """Configuration class for RAG Agent"""
    
    # Vector Database Settings
    vector_db_type: str = "chromadb"  # "chromadb" or "pinecone"
    chromadb_persist_directory: str = "./chromadb_data"
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "us-west1-gcp-free"
    pinecone_index_name: str = "fortigate-rag-knowledge"
    
    # API Keys
    openai_api_key: Optional[str] = None
    
    # Embedding Settings
    embedding_model: str = "text-embedding-ada-002"  # OpenAI
    embedding_provider: str = "openai"  # "openai" or "huggingface"
    huggingface_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # LLM Settings
    llm_provider: str = "openai"
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.1
    max_tokens: int = 2000
    
    # Chunking Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunk_strategy: str = "recursive"  # "recursive", "semantic", "fixed"
    
    # Retrieval Settings
    retrieval_k: int = 5
    retrieval_strategy: str = "similarity"  # "similarity", "mmr", "similarity_score_threshold"
    score_threshold: float = 0.7
    
    # File Processing Settings
    max_file_size_mb: int = 50
    allowed_extensions: list = None
    batch_size: int = 10
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = ['.pdf', '.docx', '.txt', '.md', '.csv', '.json', '.pptx', '.xlsx']

class ConfigManager:
    """Manages RAG configuration from environment variables and Streamlit session"""
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self) -> RAGConfig:
        """Load configuration from environment and session state"""
        
        # Load from environment variables
        config_dict = {
            'vector_db_type': os.getenv('RAG_VECTOR_DB_TYPE', 'chromadb'),
            'chromadb_persist_directory': os.getenv('RAG_CHROMADB_DIR', './chromadb_data'),
            'pinecone_api_key': os.getenv('PINECONE_API_KEY'),
            'pinecone_environment': os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp-free'),
            'pinecone_index_name': os.getenv('PINECONE_INDEX_NAME', 'fortigate-rag-knowledge'),
            
            'embedding_provider': os.getenv('RAG_EMBEDDING_PROVIDER', 'openai'),
            'embedding_model': os.getenv('RAG_EMBEDDING_MODEL', 'text-embedding-ada-002'),
            'huggingface_model': os.getenv('RAG_HF_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
            
            'llm_provider': os.getenv('RAG_LLM_PROVIDER', 'openai'),
            'llm_model': os.getenv('RAG_LLM_MODEL', 'gpt-3.5-turbo'),
            'llm_temperature': float(os.getenv('RAG_LLM_TEMPERATURE', '0.1')),
            'max_tokens': int(os.getenv('RAG_MAX_TOKENS', '2000')),
            
            'chunk_size': int(os.getenv('RAG_CHUNK_SIZE', '1000')),
            'chunk_overlap': int(os.getenv('RAG_CHUNK_OVERLAP', '200')),
            'chunk_strategy': os.getenv('RAG_CHUNK_STRATEGY', 'recursive'),
            
            'retrieval_k': int(os.getenv('RAG_RETRIEVAL_K', '5')),
            'retrieval_strategy': os.getenv('RAG_RETRIEVAL_STRATEGY', 'similarity'),
            'score_threshold': float(os.getenv('RAG_SCORE_THRESHOLD', '0.7')),
            
            'max_file_size_mb': int(os.getenv('RAG_MAX_FILE_SIZE_MB', '50')),
            'batch_size': int(os.getenv('RAG_BATCH_SIZE', '10')),
        }
        
        # Override with Streamlit session state if available
        if hasattr(st, 'session_state'):
            for key, value in config_dict.items():
                if f'rag_{key}' in st.session_state:
                    config_dict[key] = st.session_state[f'rag_{key}']
        
        return RAGConfig(**config_dict)
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                # Also update session state
                if hasattr(st, 'session_state'):
                    st.session_state[f'rag_{key}'] = value
    
    def get_config(self) -> RAGConfig:
        """Get current configuration"""
        return self.config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'vector_db_type': self.config.vector_db_type,
            'embedding_provider': self.config.embedding_provider,
            'embedding_model': self.config.embedding_model,
            'llm_provider': self.config.llm_provider,
            'llm_model': self.config.llm_model,
            'llm_temperature': self.config.llm_temperature,
            'chunk_size': self.config.chunk_size,
            'chunk_overlap': self.config.chunk_overlap,
            'retrieval_k': self.config.retrieval_k,
            'max_file_size_mb': self.config.max_file_size_mb
        }

# Global configuration instance
config_manager = ConfigManager()

def get_rag_config() -> RAGConfig:
    """Get the global RAG configuration"""
    return config_manager.get_config()

def update_rag_config(**kwargs) -> None:
    """Update the global RAG configuration"""
    config_manager.update_config(**kwargs)
