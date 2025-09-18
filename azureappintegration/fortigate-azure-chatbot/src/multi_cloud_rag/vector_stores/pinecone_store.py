"""
Pinecone Vector Store Implementation
Production-ready implementation with advanced filtering and hybrid search
"""

import pinecone
from pinecone import Pinecone
import asyncio
from typing import List, Dict, Any, Optional
import logging
import json
from datetime import datetime

from .base_vector_store import BaseVectorStore, QueryResult, DocumentMetadata, CloudProvider, DocumentType

logger = logging.getLogger(__name__)

class PineconeVectorStore(BaseVectorStore):
    """Pinecone implementation of vector store"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.environment = config.get('environment', 'us-west1-gcp-free')
        self.pc = None
        self.index = None
        
    async def connect(self) -> bool:
        """Connect to Pinecone"""
        try:
            self.pc = Pinecone(api_key=self.api_key)
            
            # List existing indexes
            indexes = self.pc.list_indexes()
            if self.index_name not in [idx.name for idx in indexes]:
                logger.info(f"Index {self.index_name} not found, will create it")
                return await self.create_index(self.embedding_dimension)
            
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            return False
    
    async def create_index(self, dimension: int) -> bool:
        """Create Pinecone index"""
        try:
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric='cosine',
                spec=pinecone.ServerlessSpec(
                    cloud='aws',
                    region=self.environment
                )
            )
            
            # Wait for index to be ready
            await asyncio.sleep(10)
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Created Pinecone index: {self.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Pinecone index: {e}")
            return False
    
    async def upsert_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Upsert documents to Pinecone"""
        try:
            vectors = []
            for doc in documents:
                vector_data = {
                    'id': doc['id'],
                    'values': doc['embedding'],
                    'metadata': {
                        'content': doc['content'],
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
                }
                vectors.append(vector_data)
            
            # Batch upsert
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Upserted {len(documents)} documents to Pinecone")
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
        """Search Pinecone index"""
        try:
            # Convert filters to Pinecone format
            pinecone_filter = self._convert_filters(filters) if filters else None
            
            response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=pinecone_filter
            )
            
            results = []
            for match in response['matches']:
                metadata_dict = match['metadata']
                
                # Parse compliance from JSON string
                compliance = json.loads(metadata_dict.get('compliance', '[]'))
                
                metadata = DocumentMetadata(
                    cloud=CloudProvider(metadata_dict['cloud']),
                    topic=DocumentType(metadata_dict['topic']),
                    region=metadata_dict['region'],
                    complexity=metadata_dict['complexity'],
                    use_case=metadata_dict['use_case'],
                    last_updated=metadata_dict['last_updated'],
                    source_url=metadata_dict.get('source_url'),
                    version=metadata_dict.get('version'),
                    compliance=compliance if compliance else None
                )
                
                result = QueryResult(
                    content=metadata_dict['content'],
                    metadata=metadata,
                    similarity_score=match['score'],
                    document_id=match['id']
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> List[QueryResult]:
        """
        Hybrid search combining semantic and keyword search
        Note: Pinecone doesn't have native hybrid search, so we'll use semantic search
        with enhanced metadata filtering based on query text
        """
        # For now, use semantic search with text-based filter enhancement
        enhanced_filters = self._enhance_filters_with_text(query_text, filters)
        return await self.search(query_vector, enhanced_filters, top_k)
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from Pinecone"""
        try:
            self.index.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents from Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.get('total_vector_count', 0),
                'dimension': stats.get('dimension', self.embedding_dimension),
                'index_fullness': stats.get('index_fullness', 0.0),
                'namespaces': stats.get('namespaces', {})
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def _convert_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert generic filters to Pinecone filter format"""
        pinecone_filter = {}
        for key, value in filters.items():
            if isinstance(value, list):
                pinecone_filter[key] = {"$in": value}
            else:
                pinecone_filter[key] = {"$eq": value}
        return pinecone_filter
    
    def _enhance_filters_with_text(
        self, 
        query_text: str, 
        existing_filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Enhance filters based on query text analysis"""
        enhanced_filters = existing_filters.copy() if existing_filters else {}
        
        # Simple keyword-based filter enhancement
        query_lower = query_text.lower()
        
        # Detect cloud providers in query
        if 'azure' in query_lower and 'gcp' not in query_lower:
            enhanced_filters['cloud'] = 'azure'
        elif 'gcp' in query_lower and 'azure' not in query_lower:
            enhanced_filters['cloud'] = 'gcp'
        elif 'google cloud' in query_lower and 'azure' not in query_lower:
            enhanced_filters['cloud'] = 'gcp'
        
        # Detect topics
        if any(word in query_lower for word in ['vm', 'virtual machine', 'instance']):
            enhanced_filters['topic'] = 'vm-config'
        elif any(word in query_lower for word in ['network', 'vpc', 'subnet']):
            enhanced_filters['topic'] = 'networking'
        elif any(word in query_lower for word in ['security', 'firewall', 'nsg']):
            enhanced_filters['topic'] = 'security'
        elif any(word in query_lower for word in ['cost', 'pricing', 'billing']):
            enhanced_filters['topic'] = 'cost-optimization'
        
        # Detect complexity
        if any(word in query_lower for word in ['basic', 'simple', 'beginner']):
            enhanced_filters['complexity'] = 'basic'
        elif any(word in query_lower for word in ['advanced', 'complex', 'enterprise']):
            enhanced_filters['complexity'] = 'advanced'
        
        return enhanced_filters
