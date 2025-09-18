"""
Vector Store Factory
Creates appropriate vector store instances based on configuration
"""

from typing import Dict, Any, Optional
import logging
from .base_vector_store import BaseVectorStore
from .pinecone_store import PineconeVectorStore

logger = logging.getLogger(__name__)

class VectorStoreFactory:
    """Factory for creating vector store instances"""
    
    SUPPORTED_STORES = {
        'pinecone': PineconeVectorStore,
        # 'azure_ai_search': AzureAISearchStore,  # Commented out due to import issues
        # 'weaviate': WeaviateVectorStore,
        # 'milvus': MilvusVectorStore
    }
    
    @classmethod
    def create_vector_store(
        self, 
        store_type: str, 
        config: Dict[str, Any]
    ) -> Optional[BaseVectorStore]:
        """Create vector store instance based on type"""
        
        if store_type not in self.SUPPORTED_STORES:
            logger.error(f"Unsupported vector store type: {store_type}")
            logger.info(f"Supported types: {list(self.SUPPORTED_STORES.keys())}")
            return None
        
        try:
            store_class = self.SUPPORTED_STORES[store_type]
            return store_class(config)
        except Exception as e:
            logger.error(f"Failed to create {store_type} vector store: {e}")
            return None
    
    @classmethod
    def get_default_config(self, store_type: str) -> Dict[str, Any]:
        """Get default configuration for vector store type"""
        
        configs = {
            'pinecone': {
                'api_key': '',
                'environment': 'us-west1-gcp-free',
                'index_name': 'multi-cloud-vm-rag',
                'embedding_dimension': 1536
            },
            'azure_ai_search': {
                'search_endpoint': '',
                'api_key': '',
                'index_name': 'multi-cloud-vm-rag',
                'embedding_dimension': 1536
            },
            'weaviate': {
                'url': 'http://localhost:8080',
                'api_key': '',
                'index_name': 'MultiCloudVmRag',
                'embedding_dimension': 1536
            },
            'milvus': {
                'host': 'localhost',
                'port': 19530,
                'collection_name': 'multi_cloud_vm_rag',
                'embedding_dimension': 1536
            }
        }
        
        return configs.get(store_type, {})
    
    @classmethod
    def validate_config(self, store_type: str, config: Dict[str, Any]) -> bool:
        """Validate configuration for vector store type"""
        
        required_fields = {
            'pinecone': ['api_key'],
            'azure_ai_search': ['search_endpoint', 'api_key'],
            'weaviate': ['url'],
            'milvus': ['host', 'port']
        }
        
        required = required_fields.get(store_type, [])
        
        for field in required:
            if field not in config or not config[field]:
                logger.error(f"Missing required field '{field}' for {store_type}")
                return False
        
        return True
