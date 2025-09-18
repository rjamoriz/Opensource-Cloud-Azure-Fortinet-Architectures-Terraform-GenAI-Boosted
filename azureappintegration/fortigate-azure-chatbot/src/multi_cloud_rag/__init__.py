"""
__init__.py for multi_cloud_rag package
Main entry point for the multi-cloud RAG system
"""

from .multi_cloud_rag_system import (
    MultiCloudRAGSystem,
    RAGConfig,
    VMRecommendation,
    QueryResponse
)

from .vector_stores import (
    VectorStoreFactory,
    PineconeVectorStore,
    DocumentMetadata,
    CloudProvider,
    DocumentType,
    SearchFilter
)

from .cloud_apis import (
    CloudAPIFactory,
    AzureCloudAPI,
    GCPCloudAPI,
    VMSpecification,
    NetworkConfiguration,
    VMDeploymentRequest
)

from .knowledge_base import (
    KnowledgeBaseManager,
    KnowledgeBaseSeeder
)

__all__ = [
    # Main system
    'MultiCloudRAGSystem',
    'RAGConfig',
    'VMRecommendation',
    'QueryResponse',
    
    # Vector stores
    'VectorStoreFactory',
    'PineconeVectorStore',
    'DocumentMetadata',
    'CloudProvider',
    'DocumentType',
    'SearchFilter',
    
    # Cloud APIs
    'CloudAPIFactory',
    'AzureCloudAPI',
    'GCPCloudAPI',
    'VMSpecification',
    'NetworkConfiguration',
    'VMDeploymentRequest',
    
    # Knowledge base
    'KnowledgeBaseManager',
    'KnowledgeBaseSeeder'
]

# Version info
__version__ = "1.0.0"
__author__ = "Multi-Cloud RAG Team"
__description__ = "Enhanced multi-cloud VM architecture assistant with RAG capabilities"
