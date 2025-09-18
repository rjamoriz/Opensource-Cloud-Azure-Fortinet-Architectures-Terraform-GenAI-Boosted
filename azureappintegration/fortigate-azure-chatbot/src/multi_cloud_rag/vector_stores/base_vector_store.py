"""
Base Vector Store Interface for Multi-Cloud RAG System
Provides abstract interface for various vector database implementations
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class CloudProvider(Enum):
    AZURE = "azure"
    GCP = "gcp"
    MULTI_CLOUD = "multi-cloud"

class DocumentType(Enum):
    VM_CONFIG = "vm-config"
    NETWORKING = "networking"
    SECURITY = "security"
    IDENTITY = "identity"
    COST_OPTIMIZATION = "cost-optimization"
    BEST_PRACTICES = "best-practices"
    ARCHITECTURE = "architecture"

@dataclass
class DocumentMetadata:
    """Standardized metadata for multi-cloud documents"""
    cloud: CloudProvider
    topic: DocumentType
    region: str
    complexity: str  # basic, intermediate, advanced
    use_case: str
    last_updated: str
    source_url: Optional[str] = None
    version: Optional[str] = None
    compliance: Optional[List[str]] = None  # SOC2, GDPR, HIPAA, etc.

@dataclass
class QueryResult:
    """Standardized query result from vector store"""
    content: str
    metadata: DocumentMetadata
    similarity_score: float
    document_id: str

class BaseVectorStore(ABC):
    """Abstract base class for vector store implementations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_dimension = config.get('embedding_dimension', 1536)
        self.index_name = config.get('index_name', 'multi-cloud-vm-rag')
        
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to vector database"""
        pass
    
    @abstractmethod
    async def create_index(self, dimension: int) -> bool:
        """Create vector index with specified dimension"""
        pass
    
    @abstractmethod
    async def upsert_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Insert or update documents in vector store"""
        pass
    
    @abstractmethod
    async def search(
        self, 
        query_vector: List[float], 
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> List[QueryResult]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    async def hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> List[QueryResult]:
        """Perform hybrid search (semantic + keyword)"""
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents by IDs"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        pass

class SearchFilter:
    """Helper class for building search filters"""
    
    @staticmethod
    def by_cloud(cloud: CloudProvider) -> Dict[str, Any]:
        return {"cloud": cloud.value}
    
    @staticmethod
    def by_topic(topic: DocumentType) -> Dict[str, Any]:
        return {"topic": topic.value}
    
    @staticmethod
    def by_region(region: str) -> Dict[str, Any]:
        return {"region": region}
    
    @staticmethod
    def by_complexity(complexity: str) -> Dict[str, Any]:
        return {"complexity": complexity}
    
    @staticmethod
    def by_use_case(use_case: str) -> Dict[str, Any]:
        return {"use_case": use_case}
    
    @staticmethod
    def multi_cloud_only() -> Dict[str, Any]:
        return {"cloud": CloudProvider.MULTI_CLOUD.value}
    
    @staticmethod
    def combine_filters(*filters: Dict[str, Any]) -> Dict[str, Any]:
        """Combine multiple filters into one"""
        combined = {}
        for filter_dict in filters:
            combined.update(filter_dict)
        return combined
