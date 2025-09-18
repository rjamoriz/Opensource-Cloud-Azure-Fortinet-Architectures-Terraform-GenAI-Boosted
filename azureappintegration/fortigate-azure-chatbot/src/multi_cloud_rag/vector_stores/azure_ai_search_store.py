"""
Azure AI Search Vector Store Implementation
Leverages Azure Cognitive Search for hybrid semantic + keyword search
"""

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    SearchField,
    SearchFieldDataType
)
from azure.core.credentials import AzureKeyCredential
import asyncio
from typing import List, Dict, Any, Optional
import logging
import json

from .base_vector_store import BaseVectorStore, QueryResult, DocumentMetadata, CloudProvider, DocumentType

logger = logging.getLogger(__name__)

class AzureAISearchStore(BaseVectorStore):
    """Azure AI Search implementation with hybrid search capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.search_endpoint = config.get('search_endpoint')
        self.api_key = config.get('api_key')
        self.search_client = None
        self.index_client = None
        
    async def connect(self) -> bool:
        """Connect to Azure AI Search"""
        try:
            credential = AzureKeyCredential(self.api_key)
            
            self.search_client = SearchClient(
                endpoint=self.search_endpoint,
                index_name=self.index_name,
                credential=credential
            )
            
            self.index_client = SearchIndexClient(
                endpoint=self.search_endpoint,
                credential=credential
            )
            
            # Check if index exists
            try:
                index = self.index_client.get_index(self.index_name)
                logger.info(f"Connected to existing Azure AI Search index: {self.index_name}")
            except:
                logger.info(f"Index {self.index_name} not found, will create it")
                await self.create_index(self.embedding_dimension)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Azure AI Search: {e}")
            return False
    
    async def create_index(self, dimension: int) -> bool:
        """Create Azure AI Search index with vector search capabilities"""
        try:
            # Define the index schema
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SearchableField(name="cloud", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="topic", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="region", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="complexity", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="use_case", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="last_updated", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="source_url", type=SearchFieldDataType.String),
                SimpleField(name="version", type=SearchFieldDataType.String),
                SimpleField(name="compliance", type=SearchFieldDataType.String),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=dimension,
                    vector_search_profile_name="default-vector-profile"
                )
            ]
            
            # Configure vector search
            vector_search = VectorSearch(
                profiles=[
                    VectorSearchProfile(
                        name="default-vector-profile",
                        algorithm_configuration_name="default-algorithm"
                    )
                ]
            )
            
            # Create the index
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search
            )
            
            self.index_client.create_index(index)
            logger.info(f"Created Azure AI Search index: {self.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Azure AI Search index: {e}")
            return False
    
    async def upsert_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Upsert documents to Azure AI Search"""
        try:
            search_documents = []
            for doc in documents:
                search_doc = {
                    'id': doc['id'],
                    'content': doc['content'],
                    'content_vector': doc['embedding'],
                    'cloud': doc['metadata']['cloud'],
                    'topic': doc['metadata']['topic'],
                    'region': doc['metadata']['region'],
                    'complexity': doc['metadata']['complexity'],
                    'use_case': doc['metadata']['use_case'],
                    'last_updated': doc['metadata']['last_updated'],
                    'source_url': doc['metadata'].get('source_url', ''),
                    'version': doc['metadata'].get('version', ''),
                    'compliance': json.dumps(doc['metadata'].get('compliance', []))
                }
                search_documents.append(search_doc)
            
            # Batch upload
            result = self.search_client.upload_documents(documents=search_documents)
            logger.info(f"Upserted {len(documents)} documents to Azure AI Search")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert documents: {e}")
            return False
    
    async def search(
        self, 
        query_vector: List[float], 
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> List[QueryResult]:
        """Perform vector search"""
        try:
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )
            
            filter_expression = self._build_filter_expression(filters) if filters else None
            
            results = self.search_client.search(
                search_text="",
                vector_queries=[vector_query],
                filter=filter_expression,
                top=top_k
            )
            
            return self._convert_search_results(results)
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> List[QueryResult]:
        """Perform hybrid search (semantic + keyword)"""
        try:
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )
            
            filter_expression = self._build_filter_expression(filters) if filters else None
            
            results = self.search_client.search(
                search_text=query_text,
                vector_queries=[vector_query],
                filter=filter_expression,
                top=top_k,
                search_mode="all"  # Combines text and vector search
            )
            
            return self._convert_search_results(results)
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from Azure AI Search"""
        try:
            documents = [{"id": doc_id} for doc_id in document_ids]
            self.search_client.delete_documents(documents=documents)
            logger.info(f"Deleted {len(document_ids)} documents from Azure AI Search")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Azure AI Search index statistics"""
        try:
            # Azure AI Search doesn't provide direct document count API
            # We'll do a search with empty query to get approximate count
            results = self.search_client.search(
                search_text="*",
                include_total_count=True,
                top=1
            )
            
            total_count = getattr(results, 'get_count', lambda: 0)()
            
            return {
                'total_documents': total_count,
                'index_name': self.index_name,
                'endpoint': self.search_endpoint
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def _build_filter_expression(self, filters: Dict[str, Any]) -> str:
        """Build OData filter expression for Azure AI Search"""
        filter_parts = []
        
        for key, value in filters.items():
            if isinstance(value, list):
                # Handle multiple values with OR
                or_parts = [f"{key} eq '{v}'" for v in value]
                filter_parts.append(f"({' or '.join(or_parts)})")
            else:
                filter_parts.append(f"{key} eq '{value}'")
        
        return " and ".join(filter_parts)
    
    def _convert_search_results(self, results) -> List[QueryResult]:
        """Convert Azure AI Search results to QueryResult objects"""
        query_results = []
        
        for result in results:
            try:
                # Parse compliance from JSON string
                compliance_str = result.get('compliance', '[]')
                compliance = json.loads(compliance_str) if compliance_str else []
                
                metadata = DocumentMetadata(
                    cloud=CloudProvider(result['cloud']),
                    topic=DocumentType(result['topic']),
                    region=result['region'],
                    complexity=result['complexity'],
                    use_case=result['use_case'],
                    last_updated=result['last_updated'],
                    source_url=result.get('source_url'),
                    version=result.get('version'),
                    compliance=compliance if compliance else None
                )
                
                query_result = QueryResult(
                    content=result['content'],
                    metadata=metadata,
                    similarity_score=result.get('@search.score', 0.0),
                    document_id=result['id']
                )
                query_results.append(query_result)
                
            except Exception as e:
                logger.warning(f"Failed to convert search result: {e}")
                continue
        
        return query_results
