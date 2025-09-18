"""
Vector Store Manager for RAG Agent
Handles ChromaDB and Pinecone vector database operations
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import streamlit as st

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

from .config import get_rag_config

logger = logging.getLogger(__name__)

class VectorStoreBase(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]) -> None:
        """Add documents to the vector store"""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents by IDs"""
        pass
    
    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        pass

class ChromaDBStore(VectorStoreBase):
    """ChromaDB vector store implementation"""
    
    def __init__(self, collection_name: str = "fortigate_knowledge"):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
        
        self.config = get_rag_config()
        self.collection_name = collection_name
        self._setup_client()
    
    def _setup_client(self):
        """Setup ChromaDB client"""
        try:
            # Create persistent directory if it doesn't exist
            os.makedirs(self.config.chromadb_persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.config.chromadb_persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "FortiGate RAG Knowledge Base"}
            )
            
            logger.info(f"ChromaDB initialized with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup ChromaDB: {e}")
            raise
    
    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]) -> None:
        """Add documents to ChromaDB"""
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to ChromaDB")
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents in ChromaDB"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'id': results['ids'][0][i] if results['ids'] else f"doc_{i}",
                        'score': 1.0 - results['distances'][0][i] if results['distances'] else 0.0
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search ChromaDB: {e}")
            return []
    
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents from ChromaDB"""
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from ChromaDB")
        except Exception as e:
            logger.error(f"Failed to delete documents from ChromaDB: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get ChromaDB collection statistics"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "vector_store_type": "ChromaDB"
            }
        except Exception as e:
            logger.error(f"Failed to get ChromaDB stats: {e}")
            return {"error": str(e)}

class PineconeStore(VectorStoreBase):
    """Pinecone vector store implementation"""
    
    def __init__(self, index_name: str = None):
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone not available. Install with: pip install pinecone-client")
        
        self.config = get_rag_config()
        self.index_name = index_name or self.config.pinecone_index_name
        self._setup_client()
    
    def _setup_client(self):
        """Setup Pinecone client"""
        try:
            api_key = self.config.pinecone_api_key or os.getenv('PINECONE_API_KEY')
            if not api_key:
                raise ValueError("Pinecone API key not found. Set PINECONE_API_KEY environment variable.")
            
            # Initialize Pinecone
            pinecone.init(
                api_key=api_key,
                environment=self.config.pinecone_environment
            )
            
            # Create index if it doesn't exist
            if self.index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI embedding dimension
                    metric="cosine"
                )
            
            self.index = pinecone.Index(self.index_name)
            logger.info(f"Pinecone initialized with index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup Pinecone: {e}")
            raise
    
    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]) -> None:
        """Add documents to Pinecone (requires embeddings)"""
        # Note: This is a simplified version. In practice, you'd need to generate embeddings first
        logger.warning("Pinecone add_documents needs embedding generation implementation")
        pass
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents in Pinecone"""
        # Note: This is a simplified version. In practice, you'd need to embed the query first
        logger.warning("Pinecone similarity_search needs embedding generation implementation")
        return []
    
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents from Pinecone"""
        try:
            self.index.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from Pinecone")
        except Exception as e:
            logger.error(f"Failed to delete documents from Pinecone: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_documents": stats.total_vector_count,
                "index_name": self.index_name,
                "vector_store_type": "Pinecone",
                "dimension": stats.dimension
            }
        except Exception as e:
            logger.error(f"Failed to get Pinecone stats: {e}")
            return {"error": str(e)}

class VectorStoreManager:
    """Manager class for vector store operations"""
    
    def __init__(self):
        self.config = get_rag_config()
        self.vector_store = self._initialize_vector_store()
    
    def _initialize_vector_store(self) -> VectorStoreBase:
        """Initialize the appropriate vector store"""
        try:
            if self.config.vector_db_type == "chromadb":
                if not CHROMADB_AVAILABLE:
                    st.error("ChromaDB not available. Please install: pip install chromadb")
                    raise ImportError("ChromaDB not available")
                return ChromaDBStore()
                
            elif self.config.vector_db_type == "pinecone":
                if not PINECONE_AVAILABLE:
                    st.error("Pinecone not available. Please install: pip install pinecone-client")
                    raise ImportError("Pinecone not available")
                return PineconeStore()
                
            else:
                raise ValueError(f"Unsupported vector store type: {self.config.vector_db_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            # Fallback to ChromaDB if available
            if CHROMADB_AVAILABLE and self.config.vector_db_type != "chromadb":
                logger.info("Falling back to ChromaDB")
                return ChromaDBStore()
            raise
    
    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]) -> None:
        """Add documents to the vector store"""
        return self.vector_store.add_documents(documents, metadatas, ids)
    
    def similarity_search(self, query: str, k: int = None) -> List[Dict]:
        """Search for similar documents"""
        k = k or self.config.retrieval_k
        return self.vector_store.similarity_search(query, k)
    
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents from the vector store"""
        return self.vector_store.delete_documents(ids)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return self.vector_store.get_collection_stats()
    
    def health_check(self) -> Dict[str, Any]:
        """Check vector store health"""
        try:
            stats = self.get_stats()
            return {
                "status": "healthy",
                "vector_store_type": self.config.vector_db_type,
                "stats": stats
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "vector_store_type": self.config.vector_db_type,
                "error": str(e)
            }

# Global vector store manager instance
_vector_store_manager = None

def get_vector_store_manager() -> VectorStoreManager:
    """Get the global vector store manager instance"""
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
    return _vector_store_manager
